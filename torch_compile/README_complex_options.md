# Complex Parameters with torch.compile: Clean Implementation Options

This document outlines several clean and maintainable approaches for using complex-valued parameters with `torch.compile`, avoiding the verbose manual expansion of complex arithmetic seen in the original implementation.

## Summary of Options

### 1. **Native Complex Tensors** (`complex_alternatives.py` - Option 1)
- **Approach**: Use PyTorch's built-in complex tensor support (`torch.complex64/128`)
- **Pros**:
  - Most natural and readable code
  - Leverages PyTorch's native complex arithmetic
  - No manual real/imaginary bookkeeping
- **Cons**:
  - May have compatibility issues with current torch.compile
  - Performance can vary depending on PyTorch version
- **Best for**: Future-proof code when torch.compile improves complex support

```python
class ComplexNetNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(..., dtype=torch.complex64))
        self.psi = nn.Parameter(torch.randn(..., dtype=torch.complex64))

    def forward(self, x):
        x_complex = torch.complex(x, torch.zeros_like(x))
        result = torch.einsum("i,kija,ta,j->tk",
                             self.psi.conj(), self.A, x_complex, self.psi)
        return result.real
```

### 2. **Custom Complex Parameter Class** (`complex_alternatives.py` - Option 2)
- **Approach**: Encapsulate real/imaginary parts in a custom parameter class
- **Pros**:
  - Clean API with property-based access
  - Encapsulates complexity management
  - Easy to maintain and extend
  - Good torch.compile compatibility
- **Cons**:
  - Slight overhead from property access
  - Requires custom class definition
- **Best for**: Production code where readability and maintainability are priorities

```python
class ComplexParameter(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.real = nn.Parameter(torch.randn(*shape))
        self.imag = nn.Parameter(torch.randn(*shape))

    @property
    def complex(self):
        return torch.complex(self.real, self.imag)
```

### 3. **Functional Approach with torch.jit.script** (`complex_alternatives.py` - Option 3)
- **Approach**: Separate complex operations into JIT-compiled functions
- **Pros**:
  - Excellent torch.compile compatibility
  - Can leverage torch.jit optimizations
  - More structured than manual expansion
- **Cons**:
  - Still requires some manual complex arithmetic
  - Less readable than native complex tensors
- **Best for**: Maximum performance with current torch.compile

### 4. **Custom Operators with torch.library** (`complex_advanced.py` - Option 2)
- **Approach**: Register custom complex operations as torch operators
- **Pros**:
  - First-class torch.compile support
  - Can provide optimized implementations
  - Reusable across models
- **Cons**:
  - Requires understanding of torch.library API
  - More setup code needed
- **Best for**: Library code or when you need specific optimizations

```python
@custom_op("mylib::complex_einsum_real", mutates_args=())
def complex_einsum_real(psi_real, psi_imag, A_real, A_imag, x_real, x_imag):
    # Implementation using native complex tensors internally
    psi = torch.complex(psi_real, psi_imag)
    # ... perform operation ...
    return result.real
```

### 5. **Structured Complex Operations** (`complex_advanced.py` - Option 3)
- **Approach**: Break down complex operations into smaller, optimizable pieces
- **Pros**:
  - Better optimization opportunities for compiler
  - Can cache intermediate results
  - More control over computation flow
- **Cons**:
  - Requires careful operation decomposition
  - More verbose than native approach
- **Best for**: Performance-critical code with specific patterns

## Recommendations

1. **For new projects**: Start with Option 2 (Custom Complex Parameter) as it provides the best balance of readability and torch.compile compatibility.

2. **For maximum performance**: Use Option 4 (Custom Operators) or Option 5 (Structured Operations) if you have specific performance requirements.

3. **For future compatibility**: Keep an eye on Option 1 (Native Complex) as torch.compile support for complex tensors improves.

4. **For existing code migration**: Option 3 (Functional Approach) provides a good migration path from manual implementations.

## Running the Examples

```bash
# Test all basic options
python torch_compile/complex_alternatives.py

# Test advanced options
python torch_compile/complex_advanced.py

# Compare with original verbose implementation
python torch_compile/complex_compile.py
```

## Key Insights

1. **Avoid manual expansion**: The original approach of manually expanding all complex arithmetic terms is error-prone and hard to maintain.

2. **Encapsulation is key**: Whether through custom classes, functions, or operators, encapsulating complex operations makes code more maintainable.

3. **torch.compile compatibility**: Current versions of torch.compile work better with real-valued tensors, so approaches that internally use real representations tend to compile better.

4. **Performance varies**: Always benchmark your specific use case, as performance can vary significantly based on tensor sizes and operation patterns.

## Future Considerations

As PyTorch and torch.compile evolve, native complex tensor support (Option 1) will likely become the preferred approach. Until then, the custom parameter class (Option 2) or custom operators (Option 4) provide the best balance of usability and performance.