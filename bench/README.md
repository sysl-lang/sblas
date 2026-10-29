# bench

Times every contiguous Level 1 routine of `sh.sysl.blas` against Accelerate and OpenBLAS at three
sizes — L1-resident (1024 elements), L2-resident (262144) and main memory (4194304) — and prints GB/s
of vector data each call reads and writes. Within a trial the three implementations run in turn, so
a change in the machine's load lands on all three alike; each figure is the median of seven trials
of about 1e8 elements.

```
cd bench
sysl build . -o bench
caffeinate -dimsu env VECLIB_MAXIMUM_THREADS=1 ./bench
```

OpenBLAS is loaded at run time from `$SYSL_OPENBLAS`, else Homebrew's
`/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, and pinned to one thread
(`oracle/sh/sysl/cblas/openblas.c` says why it is not linked).

## Apple M5 Pro, sysl 0.8.0, OpenBLAS 0.3.34, 2026-10-29

PENDING: the machine's load stayed far above 10 for the whole session, which makes wall-clock
figures meaningless. The table goes here from the first quiet run.

## What the code says, independent of load

- **`dot`, `axpy`, `scal`, `copy`, `swap`, `rot`**: one step is 128 bytes — eight `ldp q` pairs into
  eight independent `fmla.2d` (f64) or `fmla.4s` (f32) accumulators, the shape OpenBLAS's kernels
  have. Two compare-and-branch pairs per step are the vector loads' run checks, which sysl's
  `@unchecked` cannot remove (DESIGN.md, F1). OpenBLAS's `daxpy` steps 32 elements with software
  prefetch and no checks.
- **`asum`, `nrm2`, `iamax`**: scalar walks, one element per step and a dependent add chain, until a
  vector has a lane-wise `abs` (DESIGN.md, F2).
- **On M4 and later Accelerate uses the SME matrix unit**, which streams from L2 faster than one
  core's NEON pipes can issue (its f64 `dot` reached 88 GFLOPS on L2-resident data in the 0.7.10
  run); OpenBLAS is the like-for-like baseline.
