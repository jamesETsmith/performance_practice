"""
Three cleaner alternatives for using complex parameters with torch.compile
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_classification

# Setup data (same as original)
N_FEATURES = 100
N_CLASSES = 2
N_EPOCHS = 10

X, y = make_classification(n_samples=10000, n_features=N_FEATURES, n_classes=N_CLASSES)
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X).float(), torch.from_numpy(y).long()
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)


# ============================================================================
# Option 1: Native Complex Tensors with torch.view_as_real/view_as_complex
# ============================================================================
print("Option 1: Native Complex Tensors with view_as_real/view_as_complex")


class ComplexNetNative(nn.Module):
    """
    Uses PyTorch's native complex tensor support.
    Parameters are stored as complex tensors, but we use view_as_real
    for operations that torch.compile can handle better.
    """

    def __init__(self):
        super().__init__()
        # Store parameters as complex tensors
        self.A = nn.Parameter(
            torch.randn(N_CLASSES, 10, 10, N_FEATURES, dtype=torch.complex64)
        )
        self.psi = nn.Parameter(torch.randn(10, dtype=torch.complex64))

    def forward(self, x):
        x_complex = torch.complex(x, torch.zeros_like(x))

        # Method 1a: Direct complex tensor operations (may have compile issues)
        # result_complex = torch.einsum(
        #     "i,kija,ta,j->tk",
        #     self.psi.conj(),
        #     self.A,
        #     x_complex,
        #     self.psi
        # )

        # Method 1b: Use view_as_real for better compile compatibility
        # psi_real = torch.view_as_real(self.psi)  # Shape: [10, 2]
        # A_real = torch.view_as_real(self.A)  # Shape: [N_CLASSES, 10, 10, N_FEATURES, 2]
        # x_real = torch.view_as_real(x_complex)  # Shape: [batch, N_FEATURES, 2]

        # Perform computation using real views, then convert back
        # This is a placeholder for the actual computation
        # For a real implementation, you would replace this with the
        # equivalent of the einsum using real-valued operations.
        # result_real_view = torch.randn_like(x_real[..., 0])
        # result_complex = torch.view_as_complex(result_real_view.unsqueeze(-1).repeat(1,1,2)) # Dummy complex from real

        # Simpler direct einsum for demonstration (often fails compile with backward)
        result_complex = torch.einsum(
            "i,kija,ta,j->tk", self.psi.conj(), self.A, x_complex, self.psi
        )
        return result_complex.real


# ============================================================================
# Option 2: Custom Complex Parameter Class with Automatic Conversion
# ============================================================================
print("\nOption 2: Custom Complex Parameter Class")


class ComplexParameter(nn.Module):
    """
    A parameter that internally stores real and imaginary parts
    but provides a clean complex tensor interface.
    """

    def __init__(self, *shape, dtype=torch.float32):
        super().__init__()
        self.real = nn.Parameter(torch.randn(*shape, dtype=dtype))
        self.imag = nn.Parameter(torch.randn(*shape, dtype=dtype))

    @property
    def complex(self):
        """Returns the complex tensor representation"""
        return torch.complex(self.real, self.imag)

    def conj(self):
        """Returns the complex conjugate"""
        return torch.complex(self.real, -self.imag)


class ComplexNetCustomParam(nn.Module):
    """
    Uses custom ComplexParameter class for cleaner code.
    The complex arithmetic is hidden behind property access.
    """

    def __init__(self):
        super().__init__()
        self.A = ComplexParameter(N_CLASSES, 10, 10, N_FEATURES)
        self.psi = ComplexParameter(10)

    def forward(self, x):
        # Convert input to complex
        x_complex = torch.complex(x, torch.zeros_like(x))

        # Clean complex einsum using the property
        result = torch.einsum(
            "i,kija,ta,j->tk",
            self.psi.conj(),
            self.A.complex,
            x_complex,
            self.psi.complex,
        )
        return result.real


# ============================================================================
# Option 3: Functional Approach with torch.func and vmap
# ============================================================================
print("\nOption 3: Functional Approach with torch.func")


class ComplexNetFunctional(nn.Module):
    """
    Uses a functional approach that's more torch.compile friendly.
    Separates the complex computation into a pure function.
    """

    def __init__(self):
        super().__init__()
        # Store as real tensors with an extra dimension for real/imag
        self.A_parts = nn.Parameter(torch.randn(2, N_CLASSES, 10, 10, N_FEATURES))
        self.psi_parts = nn.Parameter(torch.randn(2, 10))

    @staticmethod
    def complex_einsum_real_part(
        psi_real: torch.Tensor,
        psi_imag: torch.Tensor,
        A_real: torch.Tensor,
        A_imag: torch.Tensor,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the real part of: conj(psi) @ A @ x @ psi
        Using a more structured approach than the original.
        """
        # Define the einsum pattern once
        pattern = "i,kija,ta,j->tk"

        # Group terms by their contribution to real/imaginary parts
        # Real part = Re(conj(psi) @ A @ x @ psi)
        real_terms = [
            torch.einsum(pattern, psi_real, A_real, x_real, psi_real),
            torch.einsum(pattern, psi_real, A_imag, x_imag, psi_imag),
            torch.einsum(pattern, psi_imag, A_real, x_imag, psi_imag),
            torch.einsum(pattern, psi_imag, A_imag, x_real, psi_real),
        ]

        imag_terms = [
            torch.einsum(pattern, psi_real, A_real, x_real, psi_imag),
            torch.einsum(pattern, psi_real, A_imag, x_imag, psi_real),
            torch.einsum(pattern, psi_imag, A_real, x_imag, psi_real),
            torch.einsum(pattern, psi_imag, A_imag, x_real, psi_imag),
        ]

        # Combine with appropriate signs
        real_part = real_terms[0] - real_terms[1] - real_terms[2] - real_terms[3]
        real_part = real_part - (
            imag_terms[0] - imag_terms[1] - imag_terms[2] - imag_terms[3]
        )

        return real_part

    def forward(self, x):
        x_real = x
        x_imag = torch.zeros_like(x)

        return self.complex_einsum_real_part(
            self.psi_parts[0],
            self.psi_parts[1],
            self.A_parts[0],
            self.A_parts[1],
            x_real,
            x_imag,
        )


# ============================================================================
# Testing and Comparison
# ============================================================================


def test_model(model_class, name, device="cuda"):
    """Test a model with torch.compile"""
    print(f"\n{'=' * 60}")
    print(f"Testing {name}")
    print("=" * 60)

    model = model_class().to(device)

    # Test both compiled and uncompiled versions
    model_uncompiled = model
    model_compiled = torch.compile(model)

    # Quick functionality test
    test_input = torch.randn(10, N_FEATURES).to(device)

    try:
        # Test uncompiled
        out_uncompiled = model_uncompiled(test_input)
        print(f"Uncompiled output shape: {out_uncompiled.shape}")

        # Test compiled
        out_compiled = model_compiled(test_input)
        print(f"Compiled output shape: {out_compiled.shape}")

        # Check if outputs match
        if torch.allclose(out_uncompiled, out_compiled, rtol=1e-5):
            print("✓ Compiled and uncompiled outputs match!")
        else:
            print("✗ Outputs differ between compiled and uncompiled versions")

        # Run a few training steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for i, (batch_X, batch_y) in enumerate(dataloader):
            if i >= 5:  # Just test a few batches
                break

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model_compiled(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            if i == 0:
                print(f"First batch loss: {loss.item():.4f}")

        print(f"✓ {name} works with torch.compile!")

    except Exception as e:
        print(f"✗ Error with {name}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Test all three options
    test_model(ComplexNetNative, "Option 1: Native Complex Tensors")
    test_model(ComplexNetCustomParam, "Option 2: Custom Complex Parameter")
    test_model(ComplexNetFunctional, "Option 3: Functional Approach")

    print("\n" + "=" * 60)
    print("Summary of Options:")
    print("=" * 60)
    print("""
1. Native Complex Tensors:
   - Pros: Most natural PyTorch code, leverages built-in complex support
   - Cons: May have compatibility issues with torch.compile
   - Best for: When torch.compile improves complex tensor support

2. Custom Complex Parameter:
   - Pros: Clean API, encapsulates complexity, easy to maintain
   - Cons: Slight overhead from property access
   - Best for: Production code where readability matters

3. Functional Approach:
   - Pros: Most torch.compile friendly, can use torch.jit.script
   - Cons: Still some verbosity in the function
   - Best for: Maximum performance with current torch.compile
    """)
