# Int8 Backbone Quantization — Reverted (11% Slower)

## Finding
Int8 quantization of backbone stages 1-3 is 11% SLOWER at production resolution on Apple Silicon.

## Evidence
| Config | Latency (1920x1080 tiled) |
|---|---|
| int8 quant | 2796ms |
| fp32 (no quant) | 2517ms |

Quality impact: negligible (max 1e-7 diff).

## Why it's slower
Apple Silicon unified memory eliminates the CPU↔GPU bandwidth bottleneck that makes quantization profitable on discrete GPUs. The dequantize-multiply overhead (unpack int8 → fp32 → matmul) exceeds the bandwidth savings when memory is already shared. MLX's int8 path may also have less optimized Metal kernels than the fp32 path.

## Implication
On Apple Silicon, prefer fp32 weights over quantized for inference-only workloads. Quantization only helps when memory bandwidth is the bottleneck (discrete GPUs, very large models exceeding unified memory).
