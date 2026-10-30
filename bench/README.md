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
caffeinate -dimsu env VECLIB_MAXIMUM_THREADS=1 ./bench gemm
caffeinate -dimsu env VECLIB_MAXIMUM_THREADS=1 ./bench trsm
```

With `gemm` it times gemm alone instead: square sizes 64 to 2048, GFLOPS. With `trsm` it times
`B := A⁻¹ B` (left, lower, non-unit, B square) at the same sizes, GFLOPS counted as n³ per call; each
call starts from a fresh copy of B, timed alike for all three.

OpenBLAS is loaded at run time from `$SYSL_OPENBLAS`, else Homebrew's
`/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, and pinned to one thread
(`oracle/sh/sysl/cblas/openblas.c` says why it is not linked).

## Apple M5 Pro, sysl 0.8.0, OpenBLAS 0.3.34, 2026-10-29

Load average 6.0 (1 min) when it started, single-threaded throughout. GB/s; the last two columns are
sysl as a percentage of each library.

| type | routine | size | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | dot | L1 1024 | 182.9 | 150.3 | 146.6 | 122 | 125 |
| f64 | axpy | L1 1024 | 254.2 | 224.8 | 255.0 | 113 | 100 |
| f64 | scal | L1 1024 | 223.5 | 158.7 | 210.1 | 141 | 106 |
| f64 | asum | L1 1024 | 18.3 | 117.2 | 132.2 | 16 | 14 |
| f64 | nrm2 | L1 1024 | 2.1 | 38.4 | 9.4 | 5 | 22 |
| f64 | iamax | L1 1024 | 17.6 | 55.6 | 108.4 | 32 | 16 |
| f64 | copy | L1 1024 | 216.7 | 146.3 | 119.7 | 148 | 181 |
| f64 | swap | L1 1024 | 240.2 | 271.5 | 171.9 | 88 | 140 |
| f64 | rot | L1 1024 | 223.0 | 243.6 | 203.0 | 92 | 110 |
| f64 | dot | L2 262144 | 104.2 | 955.4 | 94.7 | 11 | 110 |
| f64 | axpy | L2 262144 | 145.4 | 687.8 | 135.7 | 21 | 107 |
| f64 | scal | L2 262144 | 149.1 | 532.3 | 151.3 | 28 | 99 |
| f64 | asum | L2 262144 | 15.6 | 264.6 | 114.4 | 6 | 14 |
| f64 | nrm2 | L2 262144 | 2.1 | 37.1 | 9.1 | 6 | 23 |
| f64 | iamax | L2 262144 | 17.1 | 57.7 | 102.5 | 30 | 17 |
| f64 | copy | L2 262144 | 108.5 | 533.0 | 124.2 | 20 | 87 |
| f64 | swap | L2 262144 | 143.7 | 521.1 | 106.2 | 28 | 135 |
| f64 | rot | L2 262144 | 144.2 | 530.4 | 140.4 | 27 | 103 |
| f64 | dot | DRAM 4194304 | 109.8 | 169.0 | 114.4 | 65 | 96 |
| f64 | axpy | DRAM 4194304 | 142.5 | 168.6 | 146.7 | 85 | 97 |
| f64 | scal | DRAM 4194304 | 143.0 | 250.7 | 142.1 | 57 | 101 |
| f64 | asum | DRAM 4194304 | 15.6 | 194.2 | 93.9 | 8 | 17 |
| f64 | nrm2 | DRAM 4194304 | 2.1 | 36.9 | 9.1 | 6 | 23 |
| f64 | iamax | DRAM 4194304 | 17.2 | 57.1 | 90.0 | 30 | 19 |
| f64 | copy | DRAM 4194304 | 103.8 | 136.4 | 121.9 | 76 | 85 |
| f64 | swap | DRAM 4194304 | 145.0 | 206.6 | 83.6 | 70 | 174 |
| f64 | rot | DRAM 4194304 | 145.4 | 201.6 | 149.1 | 72 | 98 |
| f32 | dot | L1 1024 | 173.3 | 160.5 | 137.4 | 108 | 126 |
| f32 | axpy | L1 1024 | 231.1 | 113.5 | 173.9 | 204 | 133 |
| f32 | scal | L1 1024 | 192.7 | 82.0 | 178.2 | 235 | 108 |
| f32 | asum | L1 1024 | 9.5 | 115.7 | 125.4 | 8 | 8 |
| f32 | nrm2 | L1 1024 | 2.2 | 41.5 | 4.7 | 5 | 46 |
| f32 | iamax | L1 1024 | 8.5 | 52.4 | 71.8 | 16 | 12 |
| f32 | copy | L1 1024 | 194.7 | 64.3 | 103.5 | 303 | 188 |
| f32 | swap | L1 1024 | 220.3 | 148.8 | 162.3 | 148 | 136 |
| f32 | rot | L1 1024 | 206.0 | 143.0 | 186.5 | 144 | 110 |
| f32 | dot | L2 262144 | 111.0 | 985.4 | 111.7 | 11 | 99 |
| f32 | axpy | L2 262144 | 149.5 | 682.0 | 134.6 | 22 | 111 |
| f32 | scal | L2 262144 | 149.4 | 528.4 | 148.6 | 28 | 101 |
| f32 | asum | L2 262144 | 8.0 | 260.6 | 130.3 | 3 | 6 |
| f32 | nrm2 | L2 262144 | 2.1 | 36.5 | 4.5 | 6 | 47 |
| f32 | iamax | L2 262144 | 8.6 | 57.4 | 123.1 | 15 | 7 |
| f32 | copy | L2 262144 | 108.3 | 535.1 | 127.7 | 20 | 85 |
| f32 | swap | L2 262144 | 145.8 | 534.6 | 89.3 | 27 | 163 |
| f32 | rot | L2 262144 | 146.6 | 529.0 | 143.7 | 28 | 102 |
| f32 | dot | DRAM 4194304 | 108.4 | 174.6 | 112.2 | 62 | 97 |
| f32 | axpy | DRAM 4194304 | 148.1 | 236.9 | 137.3 | 63 | 108 |
| f32 | scal | DRAM 4194304 | 148.0 | 309.5 | 147.5 | 48 | 100 |
| f32 | asum | DRAM 4194304 | 8.0 | 213.4 | 102.8 | 4 | 8 |
| f32 | nrm2 | DRAM 4194304 | 2.1 | 36.5 | 4.5 | 6 | 47 |
| f32 | iamax | DRAM 4194304 | 8.6 | 57.2 | 89.8 | 15 | 10 |
| f32 | copy | DRAM 4194304 | 98.7 | 262.3 | 122.0 | 38 | 81 |
| f32 | swap | DRAM 4194304 | 139.7 | 236.3 | 83.1 | 59 | 168 |
| f32 | rot | DRAM 4194304 | 140.3 | 236.3 | 138.0 | 59 | 102 |

What the numbers say:

- **The six kernel routines are at OpenBLAS parity or better**: 81–188% of it at every size and both
  widths; `copy` in L2 and DRAM (81–87%) is the one place below. In L1 they also beat Accelerate in
  most cells.
- **Against Accelerate in L2 they reach 11–28%.** On M4 and later Accelerate runs these on the SME
  matrix unit, which streams from L2 faster than one core's NEON pipes can issue (955 GB/s for f64
  `dot`, about 240 GFLOPS); OpenBLAS is NEON and the like-for-like baseline.
- **`asum`, `nrm2` and `iamax` are scalar walks** waiting for a lane-wise `abs` (DESIGN.md, F2), and
  are 6–47% of OpenBLAS. Measured on the same run: a vector `asum` with `|v|` spelled as a `select`
  runs 3.6–4.4x the scalar loop (68 GB/s against 16–19), and the `nrm2` kernel with its band test
  spelled `(v > t) | (v < -t)` runs 8.7x (18.3 GB/s against 2.1, which is 2x OpenBLAS's and half
  Accelerate's). Neither longhand is shipped; both are the gain sysl 0.8.1's `abs` unlocks.
- **The vector loads' run checks cost nothing measurable here** (DESIGN.md, F1): a C replica of the
  `dot` and `axpy` kernels with and without the two compare-and-branch pairs per step ran at
  100–101% at 1024 and 262144 elements. The checks stay a finding for smaller cores and for the
  micro-kernel, not a cost on this one.

## gemm: Apple M5 Pro, sysl 0.8.0, OpenBLAS 0.3.34, 2026-10-29

`./bench gemm`: square column-major `C += A * B`, GFLOPS (2n³ per call), median of seven
interleaved trials of about 4e9 flops, single-threaded, load average 8.9 (1 min) at start, 8.2 at
the end. sysl's packing buffers are allocated once outside the timing.

| type | routine | n | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | gemm | 64 | 51.4 | 204.5 | 25.5 | 25 | 202 |
| f64 | gemm | 128 | 55.9 | 293.5 | 43.6 | 19 | 128 |
| f64 | gemm | 256 | 60.4 | 315.7 | 54.4 | 19 | 111 |
| f64 | gemm | 512 | 59.7 | 344.3 | 58.2 | 17 | 102 |
| f64 | gemm | 1024 | 61.0 | 200.0 | 59.2 | 30 | 103 |
| f64 | gemm | 2048 | 61.4 | 374.2 | 59.4 | 16 | 104 |
| f32 | gemm | 64 | 97.3 | 678.9 | 111.4 | 14 | 87 |
| f32 | gemm | 128 | 110.6 | 866.1 | 114.9 | 13 | 96 |
| f32 | gemm | 256 | 119.0 | 1190.9 | 116.5 | 10 | 102 |
| f32 | gemm | 512 | 124.4 | 1303.5 | 118.2 | 10 | 105 |
| f32 | gemm | 1024 | 123.3 | 1406.3 | 119.1 | 9 | 104 |
| f32 | gemm | 2048 | 126.7 | 1455.8 | 119.3 | 9 | 106 |

- **102–111% of OpenBLAS from n = 256 up, at both widths**; ahead of it at small f64 sizes, where
  OpenBLAS's own setup dominates, and 87–96% of it at small f32 sizes. Accelerate is SME (8–24%),
  out of scope as the target. The untuned structure was 93–97% from n = 512 up (55.6 / 115.4 GFLOPS
  at n = 2048).
- **What the tuning changed, measured one change at a time** (n = 2048, f64 / f32): the k loop taking
  two steps per trip, 54.8 → 61.8 / 113.1 → 125.9 — the inner loop goes from 47 instructions per step
  to 88 per two; unrolling by three or four is slower, as LLVM then loads every step's A column ahead
  and spills (57 instructions a step). A whole column-major A panel packed by vectors: +1–3% at
  n = 64–128, nothing measurable above. Kept out: writing a full tile straight from the accumulators
  to C (within noise at every size), and passing the panels as offsets instead of sub-slices (the
  reference counts stay; DESIGN.md).
- **B's row read as vectors** (`for const` over its lanes) against six scalar loads and a `mov` a
  step: +1–3% at both widths and every size, measured back to back. The loop takes more instructions
  per two steps (f64 101, f32 95, was 88), all of them run checks on the extra vector loads; the
  rest fall from 32.5 to 30.5 a step.
- **What is left in the inner loop**: of 101 instructions per two steps (f64), 48 are `fmla` by
  element and 40 are the vector loads' run checks (F1, gone in sysl 0.8.1). There is no `prfm`
  ahead in either panel (F4); packing B stays a scalar gather.
