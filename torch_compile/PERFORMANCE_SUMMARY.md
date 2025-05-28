# Complex Parameters with torch.compile - Performance Summary

## Absolute Performance Results (Forward + Backward Pass)

Based on benchmarking with batch size 1000, 100 features, measuring combined forward and backward pass times.
Results show mean ± standard deviation from 5 independent trials (50 runs per trial).

### 🏆 Performance Rankings

| Rank | Model                   | Compiled Time (ms) | Uncompiled Time (ms) | Speedup | Status                |
| ---- | ----------------------- | ------------------ | -------------------- | ------- | --------------------- |
| 1    | **Functional Approach** | **0.44 ± 0.04** ✅  | 2.32 ± 0.07          | 5.31x   | Best Overall          |
| 2    | Vmap-based              | 0.45 ± 0.13 ✅      | 0.80 ± 0.07          | 1.77x   | Excellent Performance |
| 3    | Custom Parameter Class  | 0.58 ± 0.10 ✅      | 0.91 ± 0.06          | 1.56x   | Recommended           |
| 4    | Native Complex          | ❌ (fails)          | 0.82 ± 0.09          | N/A     | Compile Issues        |
| -    | Original (Verbose)      | ❌                  | ❌                    | N/A     | Implementation Error  |
| -    | Structured              | ❌                  | ❌                    | N/A     | Shape Mismatch        |
| -    | Hybrid                  | ❌                  | ❌                    | N/A     | Shape Mismatch        |

### Key Findings

1.  **Fastest Overall: Functional Approach (Compiled)**
    *   0.44 ± 0.04 ms per iteration (forward + backward).
    *   5.31x speedup with compilation.
    *   Most stable compiled performance (lowest standard deviation).

2.  **Vmap-based Approach: A Close Second**
    *   0.45 ± 0.13 ms compiled.
    *   1.77x speedup.
    *   Slightly cleaner than the pure Functional approach while offering similar top-tier speed.

3.  **Most Practical for General Use: Custom Parameter Class**
    *   0.58 ± 0.10 ms compiled (32% slower than best, but still very fast).
    *   Cleanest API and excellent maintainability.
    *   1.56x speedup with compilation.
    *   **Recommended for most use cases due to its balance of speed and clarity.**

4.  **Compilation Benefits are Significant**
    *   All working, compilable models benefit significantly from `torch.compile`.
    *   Speedups range from 1.56x to 5.31x.
    *   No models that successfully compiled showed performance degradation.

5.  **Native Complex Tensor Issues Persist**
    *   Fastest uncompiled (0.80 ms for Vmap-based, 0.82 ms for Native Complex itself), but `ComplexNetNative` fails with `torch.compile`.
    *   Error: "torch.Tensor.view is not supported for conjugate view tensors" remains an issue.

## Recommendations

*   **For Absolute Maximum Performance:** The **Functional Approach** is the winner, delivering 0.44 ± 0.04 ms when compiled. It requires more manual implementation of complex arithmetic but yields the highest speedup and most stable compiled performance.
*   **For a Balance of Speed and Readability:** The **Vmap-based** approach is an excellent choice, nearly matching the Functional approach in speed (0.45 ± 0.13 ms) with potentially more intuitive code for some users.
*   **For Most New Projects (Best Overall Balance):** The **Custom Parameter Class** is highly recommended. While ~32% slower than the absolute fastest, its 0.58 ± 0.10 ms compiled time is excellent, and it offers a much cleaner, more maintainable, and easier-to-understand codebase.
*   **Avoid For Now:** Native complex tensors (like in `ComplexNetNative`) for `torch.compile` due to ongoing compatibility issues. The original verbose implementation in `complex_compile.py` also has implementation errors and is unnecessarily complex.

## Code Examples

### Fastest: Functional Approach
```python
# From complex_alternatives.py / ComplexNetFunctional
@staticmethod
def complex_einsum_real_part(psi_real, psi_imag, A_real, A_imag, x_real, x_imag):
    # Manual expansion of complex arithmetic
    # More verbose but torch.compile optimizes it very well
    term1_rr = torch.einsum("i,kija,ta,j->tk", psi_real, A_real, x_real, psi_real)
    # ... (7 more terms for real part)
    return term1_rr - term1_ii - term2_ri - term2_ir - term3_rr + term3_ii + term4_ri + term4_ir
```

### Recommended for General Use: Custom Parameter Class
```python
# From complex_alternatives.py / ComplexNetCustomParam
class ComplexParameter(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.real = nn.Parameter(torch.randn(*shape))
        self.imag = nn.Parameter(torch.randn(*shape))

    def forward(self):
        return torch.complex(self.real, self.imag)

# Usage in your model:
# self.A = ComplexParameter(N_CLASSES, 10, 10, N_FEATURES)
# A_complex = self.A() # Get the complex tensor
```

## Conclusion

The verbose manual expansion seen in the initial `complex_compile.py` is **not necessary** and error-prone. Cleaner abstractions, particularly the **Custom Parameter Class**, provide a great balance of very good performance, excellent maintainability, and robust `torch.compile` compatibility. If chasing the absolute last few percent of speed, the **Functional Approach** offers the best raw numbers at the cost of increased code complexity.s