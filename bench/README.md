# bench

Times `dot` and `axpy` from `sh.sysl.blas` against the platform's CBLAS (Accelerate on macOS) at
four sizes — L1-resident, L2-resident, and two in main memory — and prints GFLOPS (2n operations per
call), each the best of seven trials of about 4e8 operations.

```
cd bench
sysl build . -o bench
caffeinate -dimsu env VECLIB_MAXIMUM_THREADS=1 ./bench
```

## Apple M5 Pro, sysl 0.7.10, 2026-10-29

Single-threaded Accelerate (`VECLIB_MAXIMUM_THREADS=1`; the figures were the same without it).
**Measured under load** — load average 38–41, with a compiler release building on the same machine and
this run niced — so treat the absolute figures as a floor and the ratios as indicative; re-run quiet.

| type | n | dot sysl | dot Accelerate | % | axpy sysl | axpy Accelerate | % |
|---|---:|---:|---:|---:|---:|---:|---:|
| f64 | 1024 | 15.77 | 23.36 | 68 | 14.84 | 39.96 | 37 |
| f64 | 16384 | 13.67 | 77.06 | 18 | 9.76 | 45.47 | 21 |
| f64 | 262144 | 11.96 | 88.37 | 14 | 9.21 | 44.88 | 21 |
| f64 | 4194304 | 9.26 | 10.21 | 91 | 8.17 | 9.34 | 87 |
| f32 | 1024 | 31.69 | 31.70 | 100 | 29.13 | 45.55 | 64 |
| f32 | 16384 | 27.22 | 135.15 | 20 | 23.61 | 95.22 | 25 |
| f32 | 262144 | 28.46 | 181.09 | 16 | 19.87 | 92.62 | 21 |
| f32 | 4194304 | 19.96 | 21.74 | 92 | 17.26 | 19.60 | 88 |

What the numbers say:

- **In main memory the two are within 10%**: both are waiting on DRAM.
- **On L2-resident data Accelerate is 5–7x faster**, beyond what one core's NEON pipes can issue
  (88 GFLOPS of f64 `dot`). On M4 and later Accelerate runs these on the SME matrix unit, which
  streams from L2; NEON code cannot match it. OpenBLAS (NEON) is the like-for-like baseline and is the
  next column to add.
- **In L1, f32 `dot` matches Accelerate; f64 `dot` is at 68% and `axpy` at 37–64%.** The kernel loop is
  already vectorized and fused (four `ldp q` pairs into four `fmla`, four accumulators), but each step
  also carries two bounds checks that the loop condition already guarantees. sysl 0.8.0's unchecked
  access removes them; that is the first thing to re-measure.
