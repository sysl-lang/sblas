# bench

Times every contiguous Level 1 routine of `sh.sysl.sblas` against Accelerate and OpenBLAS at three
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

## Apple M5 Pro, sysl 0.8.1, OpenBLAS 0.3.34, 2026-10-30

Load average 5.5 (1 min) when it started, 6.6 at the end of Level 1, 6.9 at the end; single-threaded
throughout. GB/s; the last two columns are sysl as a percentage of each library.

| type | routine | size | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | dot | L1 1024 | 175.8 | 144.3 | 140.9 | 122 | 125 |
| f64 | axpy | L1 1024 | 244.3 | 222.9 | 242.1 | 110 | 101 |
| f64 | scal | L1 1024 | 203.5 | 155.1 | 205.2 | 131 | 99 |
| f64 | asum | L1 1024 | 122.9 | 112.0 | 123.1 | 110 | 100 |
| f64 | nrm2 | L1 1024 | 20.6 | 36.5 | 9.0 | 56 | 228 |
| f64 | iamax | L1 1024 | 79.0 | 52.8 | 103.1 | 150 | 77 |
| f64 | copy | L1 1024 | 197.8 | 119.0 | 111.8 | 166 | 177 |
| f64 | swap | L1 1024 | 220.4 | 263.5 | 153.1 | 84 | 144 |
| f64 | rot | L1 1024 | 207.3 | 233.2 | 183.3 | 89 | 113 |
| f64 | dot | L2 262144 | 100.6 | 742.5 | 87.3 | 14 | 115 |
| f64 | axpy | L2 262144 | 133.4 | 579.1 | 123.2 | 23 | 108 |
| f64 | scal | L2 262144 | 135.8 | 486.4 | 134.8 | 28 | 101 |
| f64 | asum | L2 262144 | 103.5 | 247.6 | 100.2 | 42 | 103 |
| f64 | nrm2 | L2 262144 | 20.3 | 34.5 | 8.7 | 59 | 234 |
| f64 | iamax | L2 262144 | 79.8 | 51.9 | 91.2 | 154 | 87 |
| f64 | copy | L2 262144 | 98.5 | 476.1 | 115.9 | 21 | 85 |
| f64 | swap | L2 262144 | 123.2 | 444.4 | 72.4 | 28 | 170 |
| f64 | rot | L2 262144 | 119.8 | 386.1 | 111.5 | 31 | 107 |
| f64 | dot | DRAM 4194304 | 90.5 | 106.5 | 92.6 | 85 | 98 |
| f64 | axpy | DRAM 4194304 | 125.3 | 151.5 | 122.8 | 83 | 102 |
| f64 | scal | DRAM 4194304 | 122.4 | 191.1 | 113.2 | 64 | 108 |
| f64 | asum | DRAM 4194304 | 79.9 | 166.1 | 80.8 | 48 | 99 |
| f64 | nrm2 | DRAM 4194304 | 19.8 | 33.9 | 8.6 | 58 | 231 |
| f64 | iamax | DRAM 4194304 | 67.0 | 48.7 | 72.4 | 138 | 93 |
| f64 | copy | DRAM 4194304 | 77.9 | 89.0 | 103.7 | 88 | 75 |
| f64 | swap | DRAM 4194304 | 123.6 | 142.2 | 80.6 | 87 | 153 |
| f64 | rot | DRAM 4194304 | 123.6 | 140.9 | 114.5 | 88 | 108 |
| f32 | dot | L1 1024 | 166.8 | 152.1 | 128.1 | 110 | 130 |
| f32 | axpy | L1 1024 | 218.8 | 113.6 | 170.6 | 193 | 128 |
| f32 | scal | L1 1024 | 200.7 | 107.0 | 174.8 | 188 | 115 |
| f32 | asum | L1 1024 | 103.7 | 92.7 | 100.6 | 112 | 103 |
| f32 | nrm2 | L1 1024 | 19.1 | 38.8 | 4.5 | 49 | 427 |
| f32 | iamax | L1 1024 | 64.7 | 47.4 | 61.1 | 136 | 106 |
| f32 | copy | L1 1024 | 185.0 | 61.4 | 90.9 | 301 | 204 |
| f32 | swap | L1 1024 | 207.6 | 107.2 | 154.4 | 194 | 134 |
| f32 | rot | L1 1024 | 193.1 | 124.5 | 180.6 | 155 | 107 |
| f32 | dot | L2 262144 | 114.0 | 882.3 | 103.5 | 13 | 110 |
| f32 | axpy | L2 262144 | 143.8 | 658.1 | 127.7 | 22 | 113 |
| f32 | scal | L2 262144 | 140.9 | 492.7 | 140.8 | 29 | 100 |
| f32 | asum | L2 262144 | 93.7 | 251.0 | 105.3 | 37 | 89 |
| f32 | nrm2 | L2 262144 | 20.2 | 34.8 | 4.3 | 58 | 467 |
| f32 | iamax | L2 262144 | 71.2 | 53.8 | 93.1 | 132 | 76 |
| f32 | copy | L2 262144 | 102.0 | 504.2 | 120.1 | 20 | 85 |
| f32 | swap | L2 262144 | 142.3 | 508.5 | 90.8 | 28 | 157 |
| f32 | rot | L2 262144 | 138.8 | 505.9 | 136.8 | 27 | 101 |
| f32 | dot | DRAM 4194304 | 99.4 | 134.1 | 104.9 | 74 | 95 |
| f32 | axpy | DRAM 4194304 | 136.3 | 181.9 | 128.0 | 75 | 106 |
| f32 | scal | DRAM 4194304 | 138.9 | 265.2 | 140.5 | 52 | 99 |
| f32 | asum | DRAM 4194304 | 75.6 | 134.3 | 78.0 | 56 | 97 |
| f32 | nrm2 | DRAM 4194304 | 20.4 | 34.5 | 4.3 | 59 | 471 |
| f32 | iamax | DRAM 4194304 | 73.8 | 50.6 | 77.9 | 146 | 95 |
| f32 | copy | DRAM 4194304 | 92.6 | 204.3 | 115.0 | 45 | 81 |
| f32 | swap | DRAM 4194304 | 138.8 | 191.9 | 80.3 | 72 | 173 |
| f32 | rot | DRAM 4194304 | 133.9 | 178.7 | 125.8 | 75 | 106 |

Level 2 on the same run, GFLOPS (2mn per call) on square column-major matrices; `gemv T` is `gemv`
on `a.t()`, CBLAS's `trans = 'T'`.

| type | routine | size | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | gemv N | L1 48x48 | 26.7 | 30.1 | 12.4 | 89 | 216 |
| f64 | gemv T | L1 48x48 | 26.5 | 19.5 | 17.3 | 136 | 153 |
| f64 | ger | L1 48x48 | 16.3 | 17.5 | 13.3 | 93 | 123 |
| f64 | gemv N | L2 256x256 | 26.1 | 175.3 | 15.9 | 15 | 164 |
| f64 | gemv T | L2 256x256 | 26.6 | 68.5 | 17.9 | 39 | 148 |
| f64 | ger | L2 256x256 | 15.3 | 29.0 | 16.2 | 53 | 94 |
| f64 | gemv N | DRAM 2048x2048 | 17.1 | 34.9 | 16.0 | 49 | 107 |
| f64 | gemv T | DRAM 2048x2048 | 17.7 | 41.0 | 16.7 | 43 | 106 |
| f64 | ger | DRAM 2048x2048 | 14.2 | 15.7 | 15.6 | 90 | 91 |
| f32 | gemv N | L1 48x48 | 39.7 | 52.0 | 47.9 | 76 | 83 |
| f32 | gemv T | L1 48x48 | 51.0 | 46.5 | 30.3 | 110 | 168 |
| f32 | ger | L1 48x48 | 29.2 | 32.4 | 26.2 | 90 | 112 |
| f32 | gemv N | L2 256x256 | 55.7 | 246.0 | 55.4 | 23 | 101 |
| f32 | gemv T | L2 256x256 | 49.7 | 176.2 | 31.2 | 28 | 159 |
| f32 | ger | L2 256x256 | 29.7 | 94.5 | 24.4 | 31 | 122 |
| f32 | gemv N | DRAM 2048x2048 | 36.7 | 116.5 | 38.8 | 31 | 95 |
| f32 | gemv T | DRAM 2048x2048 | 40.4 | 98.8 | 35.2 | 41 | 115 |
| f32 | ger | DRAM 2048x2048 | 30.3 | 44.2 | 24.8 | 69 | 122 |

What the numbers say:

- **Every Level 1 routine but `iamax` is at OpenBLAS parity or better** at every size and both widths;
  `copy` in L2 and DRAM (75–85%) and f32 `asum` in L2 (89%) are the places below. `nrm2` runs 2.3x
  OpenBLAS's at f64 and 4.3–4.7x at f32, and 49–59% of Accelerate's, with Blue's scaling kept.
- **`iamax` is 76–106% of OpenBLAS.** It reads each chunk of 16 KB once for the largest magnitude, lane
  by lane, and a second time from L1 only when that beats every chunk before it.
- **Level 2 is at OpenBLAS parity or better** except `ger` in f64 L2 and DRAM (91–94%) and f32 `gemv N`
  at 48 (83%, where OpenBLAS's small-matrix path wins), and 15–136% of Accelerate.
- **Against Accelerate in L2 the streaming routines reach 13–42%.** On M4 and later Accelerate runs
  these on the SME matrix unit, which streams from L2 faster than one core's NEON pipes can issue (742
  GB/s for f64 `dot`); OpenBLAS is NEON and the like-for-like baseline.
- **No kernel loop carries a bounds check.** Each kernel checks its slices' lengths once and runs
  `@unchecked`, so the only compare in a main loop is the loop's own: counted in the disassembly,
  `dot` is 23 instructions per 16 f64, `axpy` 27, `gemv`'s four-column pass 67 and its four-row dot
  pass 37. A run check per vector load would add a compare and a branch each — two pairs in `dot`
  and `axpy`, five in each `gemv` pass — which on this core is within noise, since the loops are
  memory-bound or close to it.
- **`sysl.hint.prefetch` is not used.** Measured on this machine at load under 10: 1 KB ahead in `dot`
  and `axpy` was 0–10% slower in L1 and L2 and level in DRAM; 512 bytes ahead on each of `gemv`'s four
  matrix streams moved its DRAM figures up 5–12% of OpenBLAS's but its L2 ones down 3–10%. The
  hardware prefetcher already follows these sequential streams.

## gemm and trsm: Apple M5 Pro, sysl 0.8.1, OpenBLAS 0.3.34, 2026-10-30

`./bench gemm`: square column-major `C += A * B`, GFLOPS (2n³ per call). `./bench trsm`: square
column-major `B := A⁻¹ B`, A lower and non-unit, GFLOPS (n³ per call). Median of seven interleaved
trials of about 4e9 flops, single-threaded; load average 7.1–8.4 (1 min) across the two runs. sysl's
packing buffers are allocated once outside the timing.

| type | routine | n | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | gemm | 64 | 48.2 | 162.4 | 24.5 | 30 | 197 |
| f64 | gemm | 128 | 52.4 | 282.4 | 42.1 | 19 | 124 |
| f64 | gemm | 256 | 57.0 | 250.7 | 50.6 | 23 | 113 |
| f64 | gemm | 512 | 56.7 | 280.4 | 53.4 | 20 | 106 |
| f64 | gemm | 1024 | 54.7 | 106.9 | 53.1 | 51 | 103 |
| f64 | gemm | 2048 | 57.4 | 344.5 | 54.1 | 17 | 106 |
| f32 | gemm | 64 | 98.3 | 661.6 | 101.9 | 15 | 96 |
| f32 | gemm | 128 | 107.8 | 802.4 | 106.5 | 13 | 101 |
| f32 | gemm | 256 | 114.1 | 1137.4 | 107.8 | 10 | 106 |
| f32 | gemm | 512 | 117.5 | 1098.0 | 109.5 | 11 | 107 |
| f32 | gemm | 1024 | 113.6 | 511.4 | 106.9 | 22 | 106 |
| f32 | gemm | 2048 | 117.7 | 1181.6 | 108.7 | 10 | 108 |
| f64 | trsm | 64 | 28.3 | 30.1 | 11.5 | 94 | 246 |
| f64 | trsm | 128 | 35.8 | 57.6 | 22.2 | 62 | 161 |
| f64 | trsm | 256 | 42.8 | 92.2 | 33.2 | 46 | 129 |
| f64 | trsm | 512 | 47.6 | 127.1 | 41.6 | 37 | 114 |
| f64 | trsm | 1024 | 49.6 | 163.5 | 45.9 | 30 | 108 |
| f64 | trsm | 2048 | 51.4 | 216.6 | 49.9 | 24 | 103 |
| f32 | trsm | 64 | 39.4 | 58.2 | 16.5 | 68 | 239 |
| f32 | trsm | 128 | 58.5 | 107.0 | 29.4 | 55 | 199 |
| f32 | trsm | 256 | 75.5 | 199.2 | 47.2 | 38 | 160 |
| f32 | trsm | 512 | 91.3 | 321.0 | 65.4 | 28 | 139 |
| f32 | trsm | 1024 | 99.1 | 457.6 | 80.8 | 22 | 123 |
| f32 | trsm | 2048 | 104.1 | 601.2 | 92.5 | 17 | 113 |

The machine was shared, so absolute figures sit a few percent under a quiet one's (OpenBLAS's f64
gemm reaches 59–60 there); % OB, timed in the same trials, is the comparable column. Accelerate is
SME, out of scope as the target.

- **gemm: 103–124% of OpenBLAS from n = 128 at f64, 101–108% at f32**, and 96% at f32 n = 64 (was
  87–90%). Against the 0.8.1 build of the previous code, three back-to-back rounds each: f64 faster at
  n ≤ 256 (best of three +5–8%) and level within noise above; f32 n = 64 +3–4 points of % OB, level
  within noise above.
- **trsm: 103–246% of OpenBLAS at every size, was 88–112%.** The same comparison: f64 n = 64
  13.5 → 30.0, n = 2048 49.5 → 55.2; f32 n = 64 15.4 → 42.5, n = 2048 91.5 → 114.3. A profile of the
  old build put ~45% of trsm's time in the scalar substitution leaf; it is now solved in registers.
- **The micro-kernel's loop on 0.8.1**: 59 instructions per two steps at both widths — 48 `fmla` by
  element, 6 `ldp q`, two adds, the count and the branch — with no run check (it was 101 / 95 on
  0.8.0; 0.8.1's own `@unchecked` reaching vector loads took them out before any change here). Asking
  for `mul_add` gives the same `fmla`, now promised.
- **Measured and kept out**: the k loop unrolled four times (no longer spills: 28.5 instructions a
  step against 29.5, but no faster); `prfm` ahead in the A and B panels (512 / 384 and 1024 / 768
  bytes) and on the C tile before the write-back (no faster at any size, slower at f64 n ≤ 256).
- **Kept**: B packed by vectors — a row-major B a row at a time, a column-major B V rows at a time
  transposed in registers (`zip1`/`zip2`); a whole column-major tile written back by vectors; trsm's
  leaf held a column at a time in registers.
