# V7 Decoupled Backbone Resolution — Rejected

## Finding
Running backbone at lower resolution fails fidelity at ALL tested ratios for matting.

## Evidence (real frame, 1920x1080, tiled 512px)
| Backbone | Speedup | Alpha max err (uint8) | Alpha mean err |
|---|---|---|---|
| full (512) | baseline | 0 | 0 |
| @448 | 12% | 91 | 0.59 |
| @384 | 21% | 143 | 0.74 |
| @320 | 28% | 226 | 0.96 |
| @256 | 38% | 192 | 1.37 |

Errors concentrate on edges (hair, fingers, silhouettes). Interior regions near-perfect.

## Why it fails
Matting is fundamentally edge-sensitive. The backbone provides spatial features the decoder needs for edge localization. Downscaling destroys sub-tile edge detail. The refiner (dilated convolutions, RF=65px) can sharpen but cannot hallucinate spatial detail that was never in the coarse features.

Note: random noise input makes this MUCH worse (max_err=0.95 even at @448) because every pixel is independent. Real images have spatial coherence, so mean error is low — but edges are still unacceptable.

## Code disposition
Kept as opt-in (`backbone_size=None` = no change). Useful for:
- Future "fast preview" mode (if user opts into quality tradeoff)
- Infrastructure for V5 (needs the downsampler/upsampler plumbing)
