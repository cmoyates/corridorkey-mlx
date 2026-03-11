# Compound: backbone-8bit-quantization

**Date:** 2026-03-11
**Search area:** backbone-quantization
**Verdict:** ERROR

## Hypothesis

Quantize backbone's 144 Linear layers to 8-bit int to reduce memory bandwidth and speed up matmuls — backbone attention/MLP dominates compute

## Result

**FIDELITY FAILURE** — changes break numerical parity with golden reference.

### Takeaway

This approach (backbone-quantization) is NOT safe for the modified files.
Do NOT retry the same change without a different mitigation strategy.

## Files changed

- `src/corridorkey_mlx/model/corridorkey.py`

## Why it failed

Runtime error — experiment crashed before producing results

## Error log

```
Traceback (most recent call last):
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/scripts/run_research_experiment.py", line 286, in <module>
    main()
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/scripts/run_research_experiment.py", line 228, in main
    model.load_checkpoint(ckpt)
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/src/corridorkey_mlx/model/corridorkey.py", line 219, in load_checkpoint
    nn.quantize(
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/nn/layers/quantized.py", line 94, in quantize
    leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=Module.is_module)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 108, in tree_map_with_path
    k: tree_map_with_path(
       ^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 99, in tree_map_with_path
    return TreeType(
           ^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 100, in <genexpr>
    tree_map_with_path(
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 108, in tree_map_with_path
    k: tree_map_with_path(
       ^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 108, in tree_map_with_path
    k: tree_map_with_path(
       ^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/utils.py", line 95, in tree_map_with_path
    return fn(path, tree, *rest)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/nn/layers/quantized.py", line 76, in _maybe_quantize
    return m.to_quantized(**kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/nn/layers/linear.py", line 108, in to_quantized
    return QuantizedLinear.from_linear(self, group_size, bits, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/nn/layers/quantized.py", line 290, in from_linear
    ql = cls(input_dims, output_dims, False, group_size, bits, mode=mode)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx/.venv/lib/python3.12/site-packages/mlx/nn/layers/quantized.py", line 245, in __init__
    self.weight, self.scales, *biases = mx.quantize(
                                        ^^^^^^^^^^^^
ValueError: [quantize] The last dimension of the matrix needs to be divisible by the quantization group size 64. However the provided  matrix has shape (336,112)
```

### Takeaway

This runtime error must be addressed before retrying this approach.
The proposer should read this traceback and fix the root cause.

## Notes

test
