# Research Analysis Index — FrozenGroupNorm Implementation

## Document Guide

### 1. FROZENGROTNORM_QUICK_REFERENCE.md
**Purpose**: Quick lookup guide with exact code locations
**Best for**: Finding specific files, line numbers, implementation patterns
**Contents**:
- Critical code locations (refiner, tiling, pipeline, tests, benchmarks)
- Code patterns (parameter threading, precompute + flag check, test fixtures)
- Implementation checklist (7 steps from infrastructure to logging)
- Expected outcomes and failure modes
- Related experiments (exp #32, #41, #42, #46)

**When to use**: Before implementing, to find exact locations

---

### 2. RESEARCH_FINDINGS_FROZEN_GROUPNORM.md
**Purpose**: Deep analysis of codebase patterns and conventions
**Best for**: Understanding project structure, conventions, and design patterns
**Contents**:
- Custom nn.Module subclass patterns (9 pages)
- Test structure patterns (fixtures, naming, tolerances)
- Pipeline parameter flow (load_model → GreenFormer → attributes)
- Experiment log format (JSON structure)
- Existing normalization classes (findings)
- Video pipeline integration
- Key architecture insights for FrozenGroupNorm
- Implementation reference points (DecoderHead.fold_bn pattern)
- Unresolved questions from handoff
- Summary checklist

**When to use**: To understand how to structure code consistently with project

---

## Research Methodology

### Questions Answered
1. How are custom nn.Module subclasses structured?
   - Answer in section 1 of RESEARCH_FINDINGS
   - Examples: RefinerBlock, DecoderHead, GreenFormer

2. How are tests structured?
   - Answer in section 2 of RESEARCH_FINDINGS
   - Files: tests/{conftest.py, test_model_contract.py, test_parity.py}

3. How are model parameters passed through the pipeline?
   - Answer in section 3 of RESEARCH_FINDINGS
   - Pattern: load_model() → GreenFormer.__init__() → self._param

4. What is the experiment log format?
   - Answer in section 4 of RESEARCH_FINDINGS
   - File: research/experiments.jsonl (46 entries)

5. Are there existing custom normalization classes?
   - Answer in section 5 of RESEARCH_FINDINGS
   - Finding: None exist; all use MLX built-ins

---

## Implementation Workflow

### Step 1: Review Patterns (30 min)
- Read FROZENGROTNORM_QUICK_REFERENCE.md (5 min)
- Review critical patterns in code (25 min):
  - DecoderHead.fold_bn() for precompute pattern
  - CNNRefinerModule.prepare_inference() for optimization pattern
  - test_model_contract.py for test patterns

### Step 2: Design (30 min)
- Decide: Separate FrozenGroupNorm class vs mode flag
  - Recommended: Separate class (cleaner interface)
- Plan collect_stats() signature (9 GroupNorm layers → 9 (mean, var) tuples)
- Plan set_frozen_stats() storage
- Design __call__ flag check logic

### Step 3: Implement (2-4 hours)
Use FROZENGROTNORM_QUICK_REFERENCE.md "Implementation Checklist"
1. Create FrozenGroupNorm infrastructure
2. Integrate into Refiner
3. Wire into GreenFormer
4. Update Pipeline
5. Add Tests
6. Benchmark
7. Log Results

### Step 4: Validate (1-2 hours)
- Run all 94 existing tests (regression check)
- Run frozen GN unit tests (collect_stats shape validation)
- Run integration test (tile_size=512 vs 1024 accuracy)
- Benchmark tile_size=512 with frozen GN

---

## Key Files Referenced

### Model Code
- `src/corridorkey_mlx/model/refiner.py` — 9x GroupNorm (focus area)
- `src/corridorkey_mlx/model/corridorkey.py` — GreenFormer, _refiner_tiled
- `src/corridorkey_mlx/model/decoder.py` — DecoderHead.fold_bn() pattern

### Inference
- `src/corridorkey_mlx/inference/pipeline.py` — load_model signature
- `src/corridorkey_mlx/inference/video.py` — VideoProcessor

### Testing
- `tests/conftest.py` — Fixtures, tolerances
- `tests/test_model_contract.py` — Output contracts
- `tests/test_parity.py` — Parity vs PyTorch

### Research & Documentation
- `research/experiments.jsonl` — 46 experiment log entries
- `research/handoff-2026-03-13-v3-groupnorm-tiling.md` — Problem statement
- `research/handoff-2026-03-13-post-v1v2.md` — Previous context
- `scripts/bench_video.py` — Benchmark CLI

---

## Key Findings Summary

### Architecture
- Refiner: CNN with 9 GroupNorm layers (1 stem + 8 in 4 ResBlocks)
- Tiling: Divides image into tile_size x tile_size tiles with 32px overlap
- Problem: Each tile's GroupNorm computes independent stats → divergent outputs
- Solution: Precompute full-image stats, freeze them during tiling

### Critical Discovery
- GroupNorm pytorch_compatible=True is REQUIRED for parity
- Without it: catastrophic failure (exp #32: alpha_final=0.987 error)
- All 9 instances in refiner use pytorch_compatible=True

### Current State
- Single-frame: 422ms @ 1024 (44 kept experiments, optimization plateau)
- Video: 1.87 FPS (V2 async decode, +7% vs V0 baseline)
- Fidelity gates: Tier 1 (max_abs < 5e-3), Tier 2 (PSNR/SSIM/dtSSD)
- V3 tile_size=512: 33% skip rate, 8% latency gain, BUT fidelity FAILS

### Success Criteria
1. tile_size=512 with frozen GN passes fidelity (max_abs < 5e-3)
2. Output matches tile_size=1024 within numerical precision
3. No regression on 94 existing tests
4. Latency improvement >= 5% vs baseline at tile_size=1024

---

## Questions & Decisions

### Design Decisions Needed
1. Separate FrozenGroupNorm class or mode flag?
   - Recommended: Separate class (cleaner, explicit intent)
2. Where to collect stats?
   - Before _refiner_tiled() tile loop (in _refiner_tiled method)
3. How to capture stats?
   - Run full-image forward, intercept each GroupNorm output
4. How many stat collections?
   - One per image (full-image forward with all metadata)

### Unresolved Questions
1. Can we avoid full-image stats pass?
2. MLX nn.GroupNorm internals allow stat injection?
3. tile_size=512 net benefit after stats overhead?
4. Better alternative: larger overlap instead of frozen stats?

---

## Timeline Estimate

- Reading & design: 1 hour
- Implementation: 2-4 hours
- Testing & validation: 1-2 hours
- Benchmarking: 30 min
- Total: 5-8 hours

---

## Contact Points

For questions about:
- Refiner architecture: See refiner.py, RefinerBlock + CNNRefinerModule
- Tiling logic: See corridorkey.py, _refiner_tiled method (~line 280)
- Parameter threading: See pipeline.py, load_model function
- Test patterns: See conftest.py, test_model_contract.py
- Parity standards: See test_parity.py, OUTPUT_TOLERANCES dict
- Previous experiments: See experiments.jsonl (46 entries)

---

## How to Navigate

**I want to...**

...understand the overall architecture
→ Read RESEARCH_FINDINGS_FROZEN_GROUPNORM.md sections 1, 7

...find specific code locations
→ Read FROZENGROTNORM_QUICK_REFERENCE.md "Critical Code Locations"

...understand test patterns
→ Read RESEARCH_FINDINGS_FROZEN_GROUPNORM.md section 2

...see implementation patterns
→ Read FROZENGROTNORM_QUICK_REFERENCE.md "Critical Patterns to Follow"

...get started implementing
→ Read FROZENGROTNORM_QUICK_REFERENCE.md "Implementation Checklist"

...understand the problem context
→ Read research/handoff-2026-03-13-v3-groupnorm-tiling.md

...see what experiments have been tried
→ Read research/experiments.jsonl (46 entries)

