# sysl-lang/blas — design

A BLAS written in sysl, laid out the way BLIS lays one out: everything that is the same on every
machine is written once, generically, and each architecture contributes one small micro-kernel. The
package is also a performance stressor for the language: it puts hot loops, bounds checks, vector
code, generics, threads and float semantics under a measured load, and a shortfall against a vendor
BLAS is a measured gap to fix in sysl.

Readable model: Rust's `matrixmultiply`. Evidence that a high-level language can reach vendor speed:
faer (Rust) and Octavian.jl.

## Scope

| level | routines | state |
|---|---|---|
| 1 | axpy, dot, nrm2, asum, scal, copy, swap, iamax, rot, rotg | done, f32 and f64 |
| 2 | gemv, ger | next |
| 3 | gemm (packing, cache blocking, threads), then trsm | after Level 2 |

Complex element types, `rotm`/`rotmg`, and the banded and packed Level 2 forms are not scheduled.

## Layers

```
 public API        gemm(alpha, a, b, beta, c)        views, transposes, checks
     |
 macro-kernel      five loops: NC, KC, MC, NR, MR    generic sysl, written once
     |             packing of A into MR-row panels
     |             packing of B into NR-column panels
     |
 micro-kernel      MR x NR block of C += A~ * B~     one per architecture
```

**The macro-kernel and the packing are generic and written once.** The five BLIS loops partition C
into NC-wide column blocks, the shared dimension into KC-deep slices, and A into MC-tall row blocks;
each MC x KC block of A and KC x NC block of B is copied into contiguous panels laid out in exactly
the order the micro-kernel reads them. Packing is where strides, transposes and edge fringes are
resolved, so the micro-kernel only ever sees unit-stride panels and a full MR x NR tile (fringes are
packed with zeros and the result is written back through a small edge path).

**Each architecture supplies one micro-kernel.** It holds an MR x NR block of C in registers and walks
the KC dimension doing rank-1 updates: broadcast one element of A~, multiply by a row of B~, fuse
into the accumulators.

| target | micro-kernel | registers, and what decides MR x NR |
|---|---|---|
| Apple silicon (AArch64 NEON) | `<N>T` vectors, 128-bit lanes | 32 vector registers: 8x12 (f64) or 16x12 (f32) style tiles, the accumulators taking 24 |
| Cortex-M7 (FPv5, single and double) | scalar FPU, `vfma` | 32 single / 16 double registers: 4x4 f32 |
| Cortex-M4 (FPv4-SP, single only) | scalar FPU, `vfma.f32` | 32 single registers: 4x4 f32; f64 goes through softfloat |
| Cortex-M33 (FPU + DSP extension) | float as M4; the DSP's 16-bit SIMD (`smlad`) for a q15 kernel | 4x4 f32; a q15 kernel 2x4 using paired MACs |
| anything else | the portable `<N>T` kernel, which scalarizes where there is no vector unit | the NEON tile, compiled to whatever the target has |

The portable kernel is the first one written and the one every other is tested against; an
architecture's kernel is an optimization that must agree with it bit for bit on the same summation
order, or within a stated tolerance where it fuses differently.

**Level 1 and Level 2 have no micro-kernel layer.** Their work is O(n) or O(n^2) on as much data, so
they are memory-bound and a contiguous slice kernel over `<W>T` vectors (W = 64 bytes of lanes: four
128-bit registers, so four independent accumulators) is the whole design. Each routine has the
contiguous kernel and a scalar walk that follows arbitrary strides.

## Element types and tile sizes

**Element types are a trait, `Real: Float`, implemented for f32 and f64.** It adds the constants that
depend on the format — Blue's thresholds and scale factors for `nrm2`, the safe range for `rotg` —
each an exact power of two written as a hex float literal. A routine is `axpy[T: Real](…)` and is
monomorphized per width; the vector kernels are `[T: Real, const W: usize]` and receive
`W = 64 / sizeof(T)` from the routine.

**Complex** would be a second trait over `sysl.math.complex.Complex[F]` (conjugation, `|re| + |im|`
for `iamax`, the complex `rotg`), with the real routines unchanged. Level 1 complex is not cheap — it
doubles the routine list (`dotu`/`dotc`, `scnrm2`, `icamax`, `crotg`, the real-scalar `csscal`) — so
it waits for the decision below.

**Tile sizes are const generics**: `gemm_kernel[T, const MR: usize, const NR: usize, const KC: usize,
const MC: usize, const NC: usize]`. A function's value parameter cannot carry a default (probed: "'W'
is a value parameter of the function …, so '= …' has nothing to stand in for"), so the public `gemm`
names the instantiation itself, per target, through `#if`. The tile is then a compile-time constant
everywhere: loop bounds fold, the accumulator array is `[MR][NR]` of registers, and packing loops
unroll.

## The API

**A vector is a strided view over a slice.** `Strided[T]` (read) and `StridedMut[T]` (write) hold the
slice, a length and a signed stride, with BLAS's meaning: a negative stride walks the storage
backwards, so element 0 is the last element the view covers. `view(xs)` and `view_mut(xs)` are the
contiguous case; `strided(xs, n, inc)` checks that the slice holds all `n` elements and traps if it
does not, so every later access is in range.

```sysl
axpy(2.0, view(xs[..]), view_mut(ys[..]))
val d = dot(strided(a[..], n, -2), view(b[..]))
val g = rotg(a, b)        // Givens(c, s, r, z)
val k = iamax(view(xs[..]))   // Option[usize], 0-based; None when empty
```

Two views of different lengths trap: BLAS passes one `n`, and two views that disagree about it are a
bug in the caller.

**A matrix will be a view with two strides**, `Matrix[T] { data, rows, cols, row_stride, col_stride }`
and `MatrixMut[T]`, as BLIS has it: row-major, column-major and any general stride are the same type,
and `a.t()` is a view that swaps the two strides rather than a `trans` flag. `gemm(alpha, a, b, beta,
c)` then needs no `transa`/`transb`/`lda` parameters; packing absorbs whatever strides it is given.

**A C-ABI export layer is possible and not decided** (see the questions). It would be `@export`
definitions named `cblas_dgemm` and so on, converting CBLAS's `(order, trans, n, ptr, inc)` arguments
into views. It would follow the reference's semantics exactly where they differ from the sysl API (a
non-positive stride on a one-vector routine is a quick return there, a view here).

## Where sysl 0.8.0 matters

Written against the installed 0.7.10; correct now, and these are the places 0.8.0 changes:

- **Unchecked access in the kernels.** The disassembly of the contiguous `dot` and `axpy` kernels
  shows the loads already vectorized and fused (`ldp q` pairs into four `fmla.2d`), with **two bounds
  checks per step** — one per `load`/`store` — that the loop condition `i + W <= len` already
  guarantees. The scalar strided walks check every element. `get_unchecked` / `@unchecked` removes
  both, after the view constructor has checked the extent once.
- **Packing and the micro-kernel** index computed offsets inside the innermost loops, where a check is
  a compare and a branch per element; those loops are written for unchecked access from the start.
- **Build modes.** The tests run in debug, which from 0.8.0 traps on integer overflow; nothing here
  relies on wrapping (the test data generator is a xorshift, which only shifts and xors). `+%` is not
  needed.

## Threading

Level 3 only; Level 1 and 2 are memory-bound and run on one core.

The parallel loop is BLIS's: the MC loop (row blocks of A) for tall problems, the NC/JR loop for wide
ones. All threads share one packed panel of B, which is read-only once packed, and each packs its own
blocks of A. The packed B panel crosses threads as a `&sync Frozen[T]` (`buf.freeze()`); the C blocks
each thread writes are disjoint and cross as a `*T` and its extents, since a struct holding a `[]T`
may not cross a domain. A barrier separates packing B from using it. The pool is the caller's or a
package's, not one this library starts per call.

## Oracles

- **Accelerate** (macOS): `libcblas` in the SDK resolves into vecLib, so `@link("cblas")` reaches it
  with no framework flag. The `oracle/` workspace member (`sh.sysl.cblas`) declares the CBLAS entry
  points behind a trait, `Oracle`, implemented for f32 and f64, so each check is written once for both
  widths. It is its own project so that the `blas` package links nothing. CBLAS refuses a negative
  stride on a one-vector routine, so those checks walk the oracle forwards and mirror `iamax`'s index.
- **OpenBLAS**: `sudo -n -u work -H /opt/homebrew/bin/brew install openblas`, then the same `Oracle`
  trait over its `libopenblas` (`--link-path /opt/homebrew/opt/openblas/lib`). It is the like-for-like
  speed baseline (NEON, no matrix unit), and on Linux the only oracle.
- **netlib**: the reference test programs (`dblat1.f`, `sblat1.f`, `dblat2.f`, `dblat3.f`) carry
  hand-derived expected values for small fixed inputs, including every stride sign; porting their
  tables gives a check that depends on no vendor. Level 1's netlib edges — `rotg`'s sign conventions,
  `iamax`'s ties and NaNs, `nrm2`'s scaling — are already tested against the definitions and against
  Accelerate.

Every test runs under ASan as well (`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`).

## Benchmark method

`bench/` times sysl against the oracle at sizes chosen to live in L1, in L2, and in main memory, and
prints GFLOPS (2n operations for dot and axpy; 2mnk for gemm). Each figure is the best of seven
trials, each trial long enough (about 4e8 operations) that the microsecond clock is not what is
measured. Runs are `caffeinate -dimsu` on a quiet machine, with `VECLIB_MAXIMUM_THREADS=1` (and
`OPENBLAS_NUM_THREADS=1`) for single-thread figures, and record the chip and the load average beside
the numbers.

**On M4 and later, Accelerate uses the SME matrix unit**, which NEON code cannot match: on the M5 Pro
Accelerate's f64 `dot` reaches about 88 GFLOPS on L2-resident data single-threaded, several times what
one core's NEON pipes can issue. The target for sysl is therefore **OpenBLAS single-thread** (at
least 80% of its DGEMM), with Accelerate reported alongside.

## Gaps found so far

- **No fused multiply-add on a vector.** `a.mul_add(b, c)` exists for `f32`/`f64` and is refused on
  `<W>T`: `type '<8>real' has no method 'mul_add'`. At `optimization = "2"` the back end contracts
  `a * b + c` into `fmla` anyway, so Level 1 is not slowed; but a micro-kernel cannot *ask* for the
  fused form, and whether it gets one is a back-end choice rather than the program's.
- **No lane-wise absolute value.** `type '<2>real' has no method 'abs'`. `asum`, `nrm2` and `iamax`
  are scalar loops because of it; the alternative, `(v < 0).select(-v, v)`, is the longhand a missing
  operation forces.
- **An `impl` in a `@tests` file is refused for a builtin type even when the trait is the file's
  own.** The stated reason — a trait answering differently in tests and in the shipped program — does
  not arise for a trait that exists only in tests. This is why the oracle trait lives in the `oracle/`
  project, which turned out to be the better structure anyway.

## Open questions for the user

1. **The C-ABI layer**: provide `cblas_*` exports (and the Fortran `dgemm_` symbols)? In this package
   or a separate one (`blas-cblas`)?
2. **Negative stride on a one-vector routine** (`nrm2`, `asum`, `scal`, `iamax`): the sysl API treats
   it as a view walked backwards; the reference returns immediately (and Accelerate reports a
   parameter error). Keep the view semantics?
3. **Mismatched lengths**: trap (now), use the shorter, or return a `Result`?
4. **`iamax`'s result**: `Option[usize]`, 0-based (now), or CBLAS's plain index with 0 for empty?
5. **Level 3 workspace**: caller-supplied packing buffers (keeps the package `@no_alloc` and usable on
   a board), or allocated per call, or both?
6. **Complex**: on `sysl.math.complex.Complex[F]`, and at which level does it start?
7. **Threads**: whose pool — `sysl.threads`, a package, or the caller's?
8. **Vector `mul_add` and `abs`**: add them to `<N>T` before the micro-kernels are written?
9. **The `@tests` impl rule**: allow an `impl` of a test-file trait for any type?
10. **Tile sizes**: fixed per target at compile time (now), or chosen at run time from the cache sizes
    the OS reports?
11. **The speed target on Apple silicon**: OpenBLAS (NEON) as proposed, or Accelerate — which would
    mean SME, and so streaming-mode support in sysl?
12. **The oracle on Linux**: `libcblas` is not a standard name there; link OpenBLAS directly?
