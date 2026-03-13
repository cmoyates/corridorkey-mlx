# V5 Partial Backbone Feature Reuse — Rejected

## Finding
Caching backbone features between frames has an unfavorable quality/speed tradeoff. The only quality-safe option (S3-only) saves too little compute to matter.

## Evidence (real consecutive video frames at 512px)

### Per-stage backbone timing
| Stage | Blocks | Time (ms) | % of backbone |
|---|---|---|---|
| S0 | 0-1 | 4.6 | 13% |
| S1 | 2-4 | 5.1 | 14% |
| S2 | 5-20 | 23.2 | 63% |
| S3 | 21-23 | 3.8 | 10% |

### Caching quality (frame N+1 S0/S1 fresh + frame N S2/S3 cached)
| Strategy | Alpha max err | Alpha mean err | Blocks saved | Pipeline speedup |
|---|---|---|---|---|
| S3 only | 27.7/255 | 0.14/255 | 3 | ~1.6% |
| S2+S3 | 247.5/255 | 3.65/255 | 19 | ~12% |
| S1+S2+S3 | 249.4/255 | 6.28/255 | 22 | ~14% |

## Why S2 caching fails
S2 features (stride-16, 448 channels, 16 blocks) encode spatial detail at a resolution that changes significantly between real consecutive frames. The "semantic stability" hypothesis holds for relative changes (3.2% rel_max) but absolute differences cascade through the SegFormer decoder's cross-scale fusion into large output errors.

S0/S1 features are even MORE volatile (>100% rel_max) but are run fresh, so that's fine. The problem is S2 sits at the intersection: too much compute to skip, too much spatial detail to cache.

## Implication
Without feature warping (V8 — optical flow to spatially transform cached features), direct caching of S2+ is not viable for matting. S3-only is quality-safe but the 3 blocks saved translate to ~1.6% pipeline improvement — complexity not justified.
