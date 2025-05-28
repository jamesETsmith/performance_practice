"""
Benchmark absolute performance of complex parameter approaches with torch.compile
Measures forward + backward pass times to find the fastest overall approach
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
import time
import pandas as pd
import numpy as np
import sys

# Import models early
sys.path.append(".")
from complex_alternatives import (
    ComplexNetNative,
    ComplexNetCustomParam,
    ComplexNetFunctional,
)
from complex_advanced import ComplexNetVmap, ComplexNetStructured, ComplexNetHybrid
from complex_compile import ComplexNet as ComplexNetOriginal


# Setup
N_FEATURES = 100
N_CLASSES = 2
BATCH_SIZE = 1000
N_WARMUP = 10
N_RUNS = 50
N_TRIALS = 5  # Number of independent trials for std dev calculation

# Create dataset
X, y = make_classification(n_samples=10000, n_features=N_FEATURES, n_classes=N_CLASSES)
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X).float(), torch.from_numpy(y).long()
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def benchmark_forward_backward(model, input_data, target, device="cuda"):
    """Benchmark forward + backward pass time"""
    model = model.to(device)
    input_data = input_data.to(device)
    target = target.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(N_WARMUP):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(N_RUNS):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    total_time = time.time() - start

    return total_time / N_RUNS * 1000  # Return milliseconds per iteration


def benchmark_with_trials(
    model_class, test_input, test_target, device="cuda", compiled=False
):
    """Run multiple trials and return mean and std dev"""
    times = []

    for trial in range(N_TRIALS):
        try:
            model = model_class()
            if compiled:
                model = torch.compile(model, mode="reduce-overhead")

            time_ms = benchmark_forward_backward(model, test_input, test_target, device)
            times.append(time_ms)

            # Clean up to ensure fresh model each trial
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Trial {trial + 1} failed: {e}")
            return None, None

    if times:
        return np.mean(times), np.std(times)
    else:
        return None, None


def test_model_correctness(model_class, test_input, test_target, device="cuda"):
    """Test if model works and produces valid gradients"""
    try:
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()

        # Test forward
        output = model(test_input)
        loss = criterion(output, test_target)

        # Test backward
        loss.backward()

        # Check if gradients exist
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )

        return True, has_grads
    except Exception as e:
        return False, str(e)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    print(
        f"Configuration: {N_TRIALS} trials, {N_RUNS} runs per trial, {N_WARMUP} warmup runs"
    )
    print("=" * 100)

    # Test data
    test_input = torch.randn(BATCH_SIZE, N_FEATURES)
    test_target = torch.randint(0, N_CLASSES, (BATCH_SIZE,))

    # Models to test
    models = [
        ("Original (Verbose)", ComplexNetOriginal),
        ("Native Complex", ComplexNetNative),
        ("Custom Parameter", ComplexNetCustomParam),
        ("Functional", ComplexNetFunctional),
        ("Vmap-based", ComplexNetVmap),
        ("Structured", ComplexNetStructured),
        ("Hybrid", ComplexNetHybrid),
    ]

    results = []

    for name, model_class in models:
        print(f"\nTesting {name}...")

        # Test if model works
        works, has_grads = test_model_correctness(
            model_class, test_input.to(device), test_target.to(device), device
        )

        if not works:
            print(f"  ✗ Failed: {has_grads}")
            results.append(
                {
                    "Model": name,
                    "Works": False,
                    "Uncompiled Mean (ms)": None,
                    "Uncompiled Std (ms)": None,
                    "Compiled Mean (ms)": None,
                    "Compiled Std (ms)": None,
                    "Speedup": None,
                    "Fastest": None,
                    "Error": has_grads,
                }
            )
            continue

        print(f"  ✓ Model works, gradients: {'✓' if has_grads else '✗'}")

        # Benchmark uncompiled
        print("  Running uncompiled trials...")
        mean_uncompiled, std_uncompiled = benchmark_with_trials(
            model_class, test_input, test_target, device, compiled=False
        )

        if mean_uncompiled:
            print(f"  Uncompiled: {mean_uncompiled:.2f} ± {std_uncompiled:.2f} ms")
        else:
            print("  Uncompiled benchmark failed")

        # Benchmark compiled
        print("  Running compiled trials...")
        mean_compiled, std_compiled = benchmark_with_trials(
            model_class, test_input, test_target, device, compiled=True
        )

        if mean_compiled:
            print(f"  Compiled:   {mean_compiled:.2f} ± {std_compiled:.2f} ms")
        else:
            print("  Compiled benchmark failed")

        # Calculate speedup and determine fastest
        if mean_uncompiled and mean_compiled:
            speedup = mean_uncompiled / mean_compiled
            fastest = "Compiled" if mean_compiled < mean_uncompiled else "Uncompiled"
            print(f"  Speedup:    {speedup:.2f}x")
            print(f"  Fastest:    {fastest}")
        else:
            speedup = None
            fastest = None

        results.append(
            {
                "Model": name,
                "Works": works,
                "Uncompiled Mean (ms)": mean_uncompiled,
                "Uncompiled Std (ms)": std_uncompiled,
                "Compiled Mean (ms)": mean_compiled,
                "Compiled Std (ms)": std_compiled,
                "Speedup": speedup,
                "Fastest": fastest,
                "Error": None,
            }
        )

    # Create results table
    print("\n" + "=" * 100)
    print("ABSOLUTE PERFORMANCE RESULTS (Forward + Backward Pass)")
    print("=" * 100)

    df = pd.DataFrame(results)

    # Sort by fastest time (considering both compiled and uncompiled)
    df["Best Time (ms)"] = df.apply(
        lambda row: min(
            filter(
                lambda x: x is not None,
                [row["Uncompiled Mean (ms)"], row["Compiled Mean (ms)"]],
            )
        )
        if row["Works"]
        else float("inf"),
        axis=1,
    )
    df = df.sort_values("Best Time (ms)")

    # Format for display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option(
        "display.float_format", lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    # Create formatted output
    print("\nDetailed Results (mean ± std dev):")
    print("-" * 100)
    for _, row in df.iterrows():
        if row["Works"]:
            print(f"\n{row['Model']}:")
            if pd.notna(row["Uncompiled Mean (ms)"]):
                print(
                    f"  Uncompiled: {row['Uncompiled Mean (ms)']:.2f} ± {row['Uncompiled Std (ms)']:.2f} ms"
                )
            if pd.notna(row["Compiled Mean (ms)"]):
                print(
                    f"  Compiled:   {row['Compiled Mean (ms)']:.2f} ± {row['Compiled Std (ms)']:.2f} ms"
                )
            if pd.notna(row["Speedup"]):
                print(f"  Speedup:    {row['Speedup']:.2f}x")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    summary_df = df[
        [
            "Model",
            "Uncompiled Mean (ms)",
            "Compiled Mean (ms)",
            "Speedup",
            "Best Time (ms)",
        ]
    ].copy()
    print(summary_df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)

    # Find overall fastest
    fastest_row = df[df["Works"]].iloc[0]
    print(f"\n1. FASTEST OVERALL: {fastest_row['Model']}")
    print(f"   - Best time: {fastest_row['Best Time (ms)']:.2f} ms")
    print(f"   - Achieved with: {fastest_row['Fastest']} version")

    # Find best compiled model
    compiled_df = df[df["Compiled Mean (ms)"].notna()].sort_values("Compiled Mean (ms)")
    if not compiled_df.empty:
        best_compiled = compiled_df.iloc[0]
        print(f"\n2. FASTEST COMPILED: {best_compiled['Model']}")
        print(
            f"   - Time: {best_compiled['Compiled Mean (ms)']:.2f} ± {best_compiled['Compiled Std (ms)']:.2f} ms"
        )

    # Find best uncompiled model
    uncompiled_df = df[df["Uncompiled Mean (ms)"].notna()].sort_values(
        "Uncompiled Mean (ms)"
    )
    if not uncompiled_df.empty:
        best_uncompiled = uncompiled_df.iloc[0]
        print(f"\n3. FASTEST UNCOMPILED: {best_uncompiled['Model']}")
        print(
            f"   - Time: {best_uncompiled['Uncompiled Mean (ms)']:.2f} ± {best_uncompiled['Uncompiled Std (ms)']:.2f} ms"
        )

    # Models where compilation helps
    print("\n4. MODELS WHERE COMPILATION HELPS:")
    helps_df = df[(df["Speedup"].notna()) & (df["Speedup"] > 1.0)]
    for _, row in helps_df.iterrows():
        print(f"   - {row['Model']}: {row['Speedup']:.2f}x speedup")

    # Models where compilation hurts
    print("\n5. MODELS WHERE COMPILATION HURTS:")
    hurts_df = df[(df["Speedup"].notna()) & (df["Speedup"] < 1.0)]
    for _, row in hurts_df.iterrows():
        print(f"   - {row['Model']}: {row['Speedup']:.2f}x (slower when compiled)")


if __name__ == "__main__":
    main()
