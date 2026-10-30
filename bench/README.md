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
```

With `gemm` it times gemm alone instead: square sizes 64 to 2048, GFLOPS.

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
interleaved trials of about 4e9 flops, single-threaded, load average 7.1 (1 min) at start and end.
sysl's packing buffers are allocated once outside the timing. This is the untuned blocked structure.

| type | routine | n | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | gemm | 64 | 45.7 | 204.4 | 25.6 | 22 | 179 |
| f64 | gemm | 128 | 49.5 | 296.2 | 44.2 | 17 | 112 |
| f64 | gemm | 256 | 54.4 | 350.6 | 54.5 | 16 | 100 |
| f64 | gemm | 512 | 55.4 | 382.9 | 59.2 | 14 | 94 |
| f64 | gemm | 1024 | 55.1 | 241.3 | 59.2 | 23 | 93 |
| f64 | gemm | 2048 | 55.6 | 368.8 | 59.6 | 15 | 93 |
| f32 | gemm | 64 | 90.5 | 688.4 | 111.2 | 13 | 81 |
| f32 | gemm | 128 | 100.4 | 861.1 | 114.8 | 12 | 87 |
| f32 | gemm | 256 | 108.3 | 1199.4 | 114.7 | 9 | 94 |
| f32 | gemm | 512 | 113.3 | 1388.5 | 119.1 | 8 | 95 |
| f32 | gemm | 1024 | 113.5 | 1465.9 | 119.7 | 8 | 95 |
| f32 | gemm | 2048 | 115.4 | 1403.1 | 118.6 | 8 | 97 |

- **93–97% of OpenBLAS from n = 512 up, at both widths**; ahead of it at small f64 sizes, where
  OpenBLAS's own setup dominates. Accelerate is SME (8–23%), out of scope as the target.
- **The micro-kernel's inner loop** (one k step, from the bench binary's disassembly; f32 has the same
  shape with `.4s`): 47 instructions — 24 `fmla` by element, 2 `ldp q` for A's four vectors, 5 scalar
  loads (`ldp d`, `ldur d`, `ld1r`, 2 `ldr d`) plus a `mov` for B's six values, 4 `cmp` + `b.hi` + 3
  `add` that are the four A loads' run checks (F1; 11 of 47), and 3 for the loop. The accumulators
  stay in v0–v31 with **no spills**, and `a * b + c` did contract to `fmla`.
- **The likely causes of the shortfall, for the tuning item**: (1) F1's 11 instructions per step —
  removable when `@unchecked` reaches a vector load (0.8.1); (2) B read as six scalars instead of
  three `ldr q` used by lane (`fmla.2d v, a, b.d[1]`); (3) no unrolling of the k loop and no `prfm`
  ahead in A and B, both of which OpenBLAS's kernel has; (4) per tile, the 24 accumulators go to a
  stack tile and back (24 stores, then the write-back reloads them, branching on `beta`), and the
  panel sub-slices take and drop an ARC reference each (`ldr`/`add`/`str` pairs); (5) packing is a
  scalar element copy. (1)–(3) are per k step and so the large-n cost; (4)–(5) weigh at small n.
