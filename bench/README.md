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
caffeinate -dimsu env VECLIB_MAXIMUM_THREADS=1 ./bench l3sym
caffeinate -dimsu ./bench gemm_par
caffeinate -dimsu ./bench each
```

With `gemm` it times gemm alone instead: square sizes 64 to 2048, GFLOPS. With `trsm` it times
`B := A⁻¹ B` (left, lower, non-unit, B square) at the same sizes, GFLOPS counted as n³ per call; each
call starts from a fresh copy of B, timed alike for all three. With `l3sym` it times `syrk` (`C -= A Aᵀ`
on the lower triangle, Cholesky's update) and `trmm` (`B := A B` with A lower on the left, `B := B A`
with A upper on the right) the same way, n³ flops per call each. With `gemm_par` it times `gemm_par` on a
`sh.sysl.sblas.threads.team` and on `sh.sysl.sblas_pool.PoolWorkers` over a standing pool, at 1, 2, 4
and 6 workers, against OpenBLAS set to the same number of threads (`openblas_set_num_threads`),
n = 128 to 2048. With `each` it times one `Workers.each` of empty jobs on `Spawned`, a Team and the
pool.

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
| f64 | iamax | L1 1024 | 118.2 | 54.0 | 106.9 | 219 | 111 |
| f64 | copy | L1 1024 | 197.8 | 119.0 | 111.8 | 166 | 177 |
| f64 | swap | L1 1024 | 220.4 | 263.5 | 153.1 | 84 | 144 |
| f64 | rot | L1 1024 | 207.3 | 233.2 | 183.3 | 89 | 113 |
| f64 | dot | L2 262144 | 100.6 | 742.5 | 87.3 | 14 | 115 |
| f64 | axpy | L2 262144 | 133.4 | 579.1 | 123.2 | 23 | 108 |
| f64 | scal | L2 262144 | 135.8 | 486.4 | 134.8 | 28 | 101 |
| f64 | asum | L2 262144 | 103.5 | 247.6 | 100.2 | 42 | 103 |
| f64 | nrm2 | L2 262144 | 20.3 | 34.5 | 8.7 | 59 | 234 |
| f64 | iamax | L2 262144 | 110.9 | 55.8 | 97.7 | 199 | 114 |
| f64 | copy | L2 262144 | 98.5 | 476.1 | 115.9 | 21 | 85 |
| f64 | swap | L2 262144 | 123.2 | 444.4 | 72.4 | 28 | 170 |
| f64 | rot | L2 262144 | 119.8 | 386.1 | 111.5 | 31 | 107 |
| f64 | dot | DRAM 4194304 | 90.5 | 106.5 | 92.6 | 85 | 98 |
| f64 | axpy | DRAM 4194304 | 125.3 | 151.5 | 122.8 | 83 | 102 |
| f64 | scal | DRAM 4194304 | 122.4 | 191.1 | 113.2 | 64 | 108 |
| f64 | asum | DRAM 4194304 | 79.9 | 166.1 | 80.8 | 48 | 99 |
| f64 | nrm2 | DRAM 4194304 | 19.8 | 33.9 | 8.6 | 58 | 231 |
| f64 | iamax | DRAM 4194304 | 84.8 | 55.0 | 85.8 | 154 | 99 |
| f64 | copy | DRAM 4194304 | 77.9 | 89.0 | 103.7 | 88 | 75 |
| f64 | swap | DRAM 4194304 | 123.6 | 142.2 | 80.6 | 87 | 153 |
| f64 | rot | DRAM 4194304 | 123.6 | 140.9 | 114.5 | 88 | 108 |
| f32 | dot | L1 1024 | 166.8 | 152.1 | 128.1 | 110 | 130 |
| f32 | axpy | L1 1024 | 218.8 | 113.6 | 170.6 | 193 | 128 |
| f32 | scal | L1 1024 | 200.7 | 107.0 | 174.8 | 188 | 115 |
| f32 | asum | L1 1024 | 103.7 | 92.7 | 100.6 | 112 | 103 |
| f32 | nrm2 | L1 1024 | 19.1 | 38.8 | 4.5 | 49 | 427 |
| f32 | iamax | L1 1024 | 108.0 | 49.8 | 70.5 | 217 | 153 |
| f32 | copy | L1 1024 | 185.0 | 61.4 | 90.9 | 301 | 204 |
| f32 | swap | L1 1024 | 207.6 | 107.2 | 154.4 | 194 | 134 |
| f32 | rot | L1 1024 | 193.1 | 124.5 | 180.6 | 155 | 107 |
| f32 | dot | L2 262144 | 114.0 | 882.3 | 103.5 | 13 | 110 |
| f32 | axpy | L2 262144 | 143.8 | 658.1 | 127.7 | 22 | 113 |
| f32 | scal | L2 262144 | 140.9 | 492.7 | 140.8 | 29 | 100 |
| f32 | asum | L2 262144 | 93.7 | 251.0 | 105.3 | 37 | 89 |
| f32 | nrm2 | L2 262144 | 20.2 | 34.8 | 4.3 | 58 | 467 |
| f32 | iamax | L2 262144 | 122.9 | 55.0 | 120.8 | 224 | 102 |
| f32 | copy | L2 262144 | 102.0 | 504.2 | 120.1 | 20 | 85 |
| f32 | swap | L2 262144 | 142.3 | 508.5 | 90.8 | 28 | 157 |
| f32 | rot | L2 262144 | 138.8 | 505.9 | 136.8 | 27 | 101 |
| f32 | dot | DRAM 4194304 | 99.4 | 134.1 | 104.9 | 74 | 95 |
| f32 | axpy | DRAM 4194304 | 136.3 | 181.9 | 128.0 | 75 | 106 |
| f32 | scal | DRAM 4194304 | 138.9 | 265.2 | 140.5 | 52 | 99 |
| f32 | asum | DRAM 4194304 | 75.6 | 134.3 | 78.0 | 56 | 97 |
| f32 | nrm2 | DRAM 4194304 | 20.4 | 34.5 | 4.3 | 59 | 471 |
| f32 | iamax | DRAM 4194304 | 88.9 | 54.4 | 95.1 | 163 | 93 |
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

- **Every Level 1 routine is at OpenBLAS parity or better** at every size and both widths, bar `iamax`
  f32 DRAM (93%);
  `copy` in L2 and DRAM (75–85%) and f32 `asum` in L2 (89%) are the places below. `nrm2` runs 2.3x
  OpenBLAS's at f64 and 4.3–4.7x at f32, and 49–59% of Accelerate's, with Blue's scaling kept.
- **`iamax` is 93–153% of OpenBLAS** in the rows above (the `iamax` rows were re-measured with the
  `maximum`-based kernel at load 9.3 at the start and 8.1 at the end). It reads each
  chunk of 16 KB once for the NaN-propagating largest magnitude, one `fmax` per register, and a second
  time from L1 only when that beats every chunk before it.
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

## syrk, trmm, symv, syr2: Apple M5 Pro, sysl 0.8.3, OpenBLAS 0.3.34, 2026-10-30

`./bench l3sym` (n³ flops per call) and the symmetric rows of `./bench` (2n² per call), with
`VECLIB_MAXIMUM_THREADS=1`. The machine was heavily shared (load average 39–70) and sysl ran under
`nice`, so the absolute figures are low and noisy; % OB, from the same interleaved trials, is the
comparable column.

| type | routine | n | sysl | Accelerate | OpenBLAS | % Acc | % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | syrk L N | 64 | 28.4 | 66.7 | 12.7 | 43 | 224 |
| f64 | syrk L N | 256 | 36.9 | 124.2 | 32.2 | 30 | 115 |
| f64 | syrk L N | 1024 | 49.9 | 223.6 | 48.6 | 22 | 103 |
| f64 | syrk L N | 2048 | 47.9 | 200.0 | 47.0 | 24 | 102 |
| f64 | trmm L L | 64 | 22.6 | 50.9 | 29.3 | 44 | 77 |
| f64 | trmm L L | 256 | 43.4 | 207.9 | 46.2 | 21 | 94 |
| f64 | trmm L L | 1024 | 49.2 | 276.5 | 49.2 | 18 | 100 |
| f64 | trmm L L | 2048 | 51.4 | 331.1 | 52.2 | 16 | 98 |
| f64 | trmm R U | 64 | 23.3 | 70.5 | 34.0 | 33 | 68 |
| f64 | trmm R U | 256 | 38.7 | 166.9 | 43.8 | 23 | 88 |
| f64 | trmm R U | 1024 | 37.0 | 156.7 | 42.6 | 24 | 87 |
| f64 | trmm R U | 2048 | 38.5 | 189.8 | 43.7 | 20 | 88 |
| f32 | syrk L N | 64 | 48.7 | 89.1 | 47.8 | 55 | 102 |
| f32 | syrk L N | 256 | 81.0 | 344.2 | 75.0 | 24 | 108 |
| f32 | syrk L N | 1024 | 94.3 | 509.5 | 91.1 | 19 | 104 |
| f32 | syrk L N | 2048 | 100.4 | 692.2 | 95.4 | 14 | 105 |
| f32 | trmm L L | 64 | 37.3 | 71.3 | 46.1 | 52 | 81 |
| f32 | trmm L L | 256 | 77.6 | 376.9 | 83.7 | 21 | 93 |
| f32 | trmm L L | 1024 | 100.1 | 859.2 | 97.0 | 12 | 103 |
| f32 | trmm L L | 2048 | 103.5 | 1065.7 | 104.3 | 10 | 99 |
| f32 | trmm R U | 64 | 37.6 | 100.7 | 56.2 | 37 | 67 |
| f32 | trmm R U | 256 | 78.5 | 355.3 | 88.4 | 22 | 89 |
| f32 | trmm R U | 1024 | 102.2 | 853.7 | 101.6 | 12 | 101 |
| f32 | trmm R U | 2048 | 108.7 | 986.0 | 105.4 | 11 | 103 |
| f64 | symv L | L1 48x48 | 25.1 | 7.6 | 39.5 | 331 | 63 |
| f64 | symv U | L1 48x48 | 24.4 | 8.9 | 26.0 | 276 | 94 |
| f64 | syr2 L | L1 48x48 | 21.2 | 37.4 | 9.0 | 57 | 235 |
| f64 | symv L | L2 256x256 | 32.7 | 12.0 | 36.3 | 273 | 90 |
| f64 | symv U | L2 256x256 | 35.7 | 24.3 | 35.1 | 147 | 102 |
| f64 | syr2 L | L2 256x256 | 15.3 | 20.4 | 11.7 | 75 | 131 |
| f64 | symv L | DRAM 2048x2048 | 29.4 | 15.5 | 30.3 | 190 | 97 |
| f64 | symv U | DRAM 2048x2048 | 30.8 | 15.4 | 30.5 | 200 | 101 |
| f64 | syr2 L | DRAM 2048x2048 | 21.2 | 8.9 | 16.2 | 237 | 131 |
| f32 | symv L | L1 48x48 | 27.3 | 26.2 | 39.1 | 104 | 70 |
| f32 | symv U | L1 48x48 | 25.9 | 27.8 | 32.3 | 93 | 80 |
| f32 | syr2 L | L1 48x48 | 26.1 | 57.9 | 15.9 | 45 | 164 |
| f32 | symv L | L2 256x256 | 55.9 | 52.2 | 73.8 | 107 | 76 |
| f32 | symv U | L2 256x256 | 61.3 | 58.7 | 69.2 | 104 | 89 |
| f32 | syr2 L | L2 256x256 | 36.3 | 47.1 | 24.3 | 77 | 149 |
| f32 | symv L | DRAM 2048x2048 | 68.2 | 45.5 | 67.5 | 150 | 101 |
| f32 | symv U | DRAM 2048x2048 | 62.7 | 47.7 | 59.5 | 131 | 105 |
| f32 | syr2 L | DRAM 2048x2048 | 38.9 | 35.6 | 26.1 | 109 | 149 |

- **syrk: 102–224% of OpenBLAS at f64, 102–111% at f32.** gemm's own loops, with the tiles above (or
  below) the diagonal skipped.
- **trmm: level with OpenBLAS from n = 512 (87–103%), 67–81% at n = 64.** The right side was 84–92%
  when it ran as the left on the transposes, which hands gemm a row-major C; with its own recursion
  an earlier, quieter round gave 96–106% from n = 512 at both widths. At n = 64 the 16-row diagonal
  blocks are a quarter of the work.
- **syr2: 131–235% of OpenBLAS. symv: level with OpenBLAS in main memory (97–105%), behind it in L1 and
  L2 (63–102%)**, where OpenBLAS's lower-triangle kernel is fastest.

## gemm_par: Apple M5 Pro (6P + 12E), sysl 0.8.5, OpenBLAS 0.3.34, 2026-10-30

`./bench gemm_par`: square column-major `C += A * B`, GFLOPS, median of seven interleaved trials of
about 4e9 flops; sysl on `team(threads, …)` (`sh.sysl.sblas.threads`: threads started once and parked
on a `sysl.threads.Barrier` between calls) and on `PoolWorkers(Pool.new(threads - 1), threads)` (a
standing `sysl-lang/pool` 0.1.4), the calling thread running the first job in both, OpenBLAS on the
same thread count. The second of two runs back to back; load average 9.4 (1 min) at the start and 9.1
at the end, after an afternoon at 30–150, so the 5-minute average was still 35–60.

| type | n | threads | Team | pool | OpenBLAS | Team % OB | pool % OB |
|---|---|---|---:|---:|---:|---:|---:|
| f64 | 128 | 1 | 53.3 | 54.7 | 43.7 | 122 | 125 |
| f64 | 128 | 2 | 77.6 | 84.5 | 46.5 | 167 | 182 |
| f64 | 128 | 4 | 63.9 | 106.6 | 49.4 | 129 | 216 |
| f64 | 128 | 6 | 51.3 | 68.4 | 40.9 | 125 | 167 |
| f64 | 256 | 1 | 59.3 | 60.2 | 52.8 | 112 | 114 |
| f64 | 256 | 2 | 111.5 | 112.3 | 91.5 | 122 | 123 |
| f64 | 256 | 4 | 142.3 | 144.8 | 135.0 | 105 | 107 |
| f64 | 256 | 6 | 138.0 | 151.8 | 147.0 | 94 | 103 |
| f64 | 512 | 1 | 59.5 | 60.3 | 56.7 | 105 | 106 |
| f64 | 512 | 2 | 117.7 | 117.7 | 108.1 | 109 | 109 |
| f64 | 512 | 4 | 221.7 | 216.5 | 199.3 | 111 | 109 |
| f64 | 512 | 6 | 207.7 | 226.9 | 251.2 | 83 | 90 |
| f64 | 1024 | 1 | 61.7 | 62.7 | 59.5 | 104 | 105 |
| f64 | 1024 | 2 | 119.6 | 119.6 | 112.1 | 107 | 107 |
| f64 | 1024 | 4 | 220.3 | 222.1 | 209.6 | 105 | 106 |
| f64 | 1024 | 6 | 257.1 | 263.5 | 281.3 | 91 | 94 |
| f64 | 2048 | 1 | 62.3 | 62.8 | 59.0 | 106 | 106 |
| f64 | 2048 | 2 | 120.1 | 122.0 | 113.8 | 105 | 107 |
| f64 | 2048 | 4 | 216.9 | 222.8 | 216.7 | 100 | 103 |
| f64 | 2048 | 6 | 278.7 | 282.8 | 305.9 | 91 | 92 |
| f32 | 128 | 1 | 114.5 | 113.9 | 115.0 | 100 | 99 |
| f32 | 128 | 2 | 139.0 | 160.4 | 64.4 | 216 | 249 |
| f32 | 128 | 4 | 131.6 | 179.7 | 67.0 | 196 | 268 |
| f32 | 128 | 6 | 68.9 | 112.3 | 56.5 | 122 | 199 |
| f32 | 256 | 1 | 121.4 | 122.2 | 114.5 | 106 | 107 |
| f32 | 256 | 2 | 219.0 | 217.3 | 172.8 | 127 | 126 |
| f32 | 256 | 4 | 338.5 | 272.2 | 244.2 | 139 | 111 |
| f32 | 256 | 6 | 246.2 | 277.2 | 201.7 | 122 | 137 |
| f32 | 512 | 1 | 128.0 | 127.4 | 119.2 | 107 | 107 |
| f32 | 512 | 2 | 245.3 | 245.8 | 212.8 | 115 | 116 |
| f32 | 512 | 4 | 364.8 | 353.3 | 383.4 | 95 | 92 |
| f32 | 512 | 6 | 441.0 | 485.6 | 442.6 | 100 | 110 |
| f32 | 1024 | 1 | 127.1 | 126.6 | 117.4 | 108 | 108 |
| f32 | 1024 | 2 | 249.9 | 249.9 | 229.3 | 109 | 109 |
| f32 | 1024 | 4 | 465.8 | 473.0 | 429.9 | 108 | 110 |
| f32 | 1024 | 6 | 515.7 | 536.2 | 530.6 | 97 | 101 |
| f32 | 2048 | 1 | 121.9 | 122.3 | 110.5 | 110 | 111 |
| f32 | 2048 | 2 | 242.9 | 246.0 | 224.9 | 108 | 109 |
| f32 | 2048 | 4 | 389.1 | 425.6 | 418.3 | 93 | 102 |
| f32 | 2048 | 6 | 570.1 | 570.9 | 574.3 | 99 | 99 |

`./bench each`: microseconds per `Workers.each` of empty jobs, median of seven runs of 20000 calls,
two runs (load 7.5 and 9.7 at their starts):

| workers | Spawned | Team | pool |
|---:|---:|---:|---:|
| 1 | 0.02–0.03 | 0.02 | 0.01 |
| 2 | 11.6–17.0 | 2.4–3.7 | 2.9–3.0 |
| 4 | 24.6–43.4 | 9.9–18.7 | 4.6–5.1 |
| 6 | 36.6–61.3 | 23.8–25.5 | 12.0–12.3 |

- **The Team does not match the pool yet, and the Barrier is why.** Its `each` is two meetings of a
  `sysl.threads.Barrier`, which sleeps every early arrival on a condition variable and wakes them all
  with `notify_all`. Two meetings of it cost 6.3–7.5 / 12.2–15.3 / 18.7–26.0 µs at 2 / 4 / 6 threads,
  where two meetings of a spinning barrier on two atomics cost 0.1–0.3 / 0.5–0.6 / 0.7–0.8 µs (a probe
  outside this repository, two runs at load 9). So the Team's `each` is the barrier's cost, and it is
  2–4x the pool's at 4–6 workers. The finding is reported to sysl; the Team stays on the library's
  barrier rather than carrying a barrier of its own.
- **At n = 128 and 4–6 threads the pool is 1.2–1.8x the Team** over the two runs; at n = 256 they are
  level at 2–4 threads and the pool 8–21% ahead at 6, and at n = 512 and 6 threads still 7–10% ahead
  (f32 n = 256 at four threads read Team 139%, pool 111% of OpenBLAS in this run and 78 / 79% in the
  first: the small sizes move by up to half from run to run).
- **From n = 1024 the two are level**, Team 91–110% and pool 90–111% of OpenBLAS at 2–6 threads over
  both runs, and one worker runs gemm's own speed on both (104–111%).
- **Both are far cheaper than `Spawned`**: a Team's `each` is a fifth of a `Spawned` one at two
  workers and a half to two-fifths at four to six.
