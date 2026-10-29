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

The package needs 0.8.0 (`sysl = "0.8.0"`):

- **Unchecked access in the walks.** A strided walk checks once, before it starts, that the view's
  last element is inside its data (`walkable`), then reads and writes inside an `@unchecked` block.
  Before, every element carried a compare and a branch the optimizer could not remove, since the
  stride arithmetic is beyond what it proves. A view built by hand rather than by `strided` is caught
  by the same check, and the tests trap one in each direction.
- **Not in the kernels, because 0.8.0 cannot reach them.** A contiguous kernel's cost is its vector
  `load`/`store`, each checked as a run (`i <= len - W`), and **`@unchecked` does not cover a vector
  load** — it is a method, not a subscript — and there is no `load_unchecked`. So `dot` and `axpy` each
  keep two compare-and-branch pairs per step (`axpy`'s load and store of `y` share one). A kernel's scalar tail needs nothing: the
  optimizer proves its subscripts itself. This is sysl finding F1 below.
- **Packing and the micro-kernel** index computed offsets inside the innermost loops, where a check is
  a compare and a branch per element; those loops are written for unchecked access from the start.
- **Build modes.** The tests run in debug, which traps on integer overflow. The one trap the 0.7.10
  code met was a real bug, in the oracle check: `n - 1 - forwards` was computed for an empty vector,
  where it is meaningless and underflows; the empty case is now checked on its own. Nothing here
  relies on wrapping (the test data generator is a xorshift, which only shifts and xors).

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
- **OpenBLAS**: `sudo -n -u work -H /opt/homebrew/bin/brew install openblas`. It exports the same
  `cblas_*` names as Accelerate, and a Mach-O link binds each name to one library, so it is not
  linked: `oracle/sh/sysl/cblas/openblas.c` `dlopen`s it (`$SYSL_OPENBLAS`, else Homebrew's path)
  with `RTLD_LOCAL`, pins it to one thread, and forwards `sysl_ob_*` to its entry points. Every
  `Oracle` method takes a `Lib`, and every check runs against both; one test proves the two
  `cblas_ddot`s live in different images. OpenBLAS is the like-for-like speed baseline (NEON, no
  matrix unit), and on Linux the only oracle.
- **netlib**: the reference test programs (`dblat1.f`, `sblat1.f`, `dblat2.f`, `dblat3.f`) carry
  hand-derived expected values for small fixed inputs, including every stride sign; porting their
  tables gives a check that depends on no vendor. Level 1's netlib edges — `rotg`'s sign conventions,
  `iamax`'s ties and NaNs, `nrm2`'s scaling — are already tested against the definitions and against
  Accelerate.

Every test runs under ASan as well (`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`).

## Benchmark method

`bench/` times every contiguous Level 1 routine of sysl, Accelerate and OpenBLAS at sizes chosen to
live in L1, in L2, and in main memory, and prints GB/s of vector data read and written (2mnk GFLOPS
for gemm, when it exists). Within each trial the three run in turn, so a change in the machine's
load lands on all three alike; each figure is the median of seven trials of about 1e8 elements.
Runs are `caffeinate -dimsu` with `VECLIB_MAXIMUM_THREADS=1` (OpenBLAS is pinned to one thread when
it is loaded), and record the chip and the load average beside the numbers.

**On M4 and later, Accelerate uses the SME matrix unit**, which NEON code cannot match: on the M5 Pro
Accelerate's f64 `dot` reaches about 88 GFLOPS on L2-resident data single-threaded, several times what
one core's NEON pipes can issue. The target for sysl is therefore **OpenBLAS single-thread** (at
least 80% of its DGEMM), with Accelerate reported alongside.

## Gaps found so far

- **F1: a vector load or store cannot be unchecked.** `@unchecked` covers subscripts, and `x.load(i)`
  / `y.store(i, v)` are methods, so inside the block they still compare `i` against `len - W` and
  branch to `sysl_bounds_fail` (checked in the emitted IR of a three-line probe: `x[i]` loses its
  check, `x.load(i)` keeps `icmp ule` + `br` to `bounds.bad`). There is no `load_unchecked` either.
  The check is not removable by the optimizer, because the loop's `i + W <= len` is not known not to
  wrap. In the `dot` kernel's loop that is 4 of 27 instructions (`cmp`/`b.hi` twice) beside 8 `ldp q`
  and 8 `fmla.2d`; OpenBLAS's `daxpy` loop has none, and unrolls 32 elements with `prfm` 0x380 ahead.
  The fix in sysl: `@unchecked` reaching a vector load and store (the reference's "every subscript
  written inside it" read to include them), or `load_unchecked`/`store_unchecked`. Its cost is
  measured by a C replica of the same kernel with and without the checks (`bench/README.md`).
- **F2: no lane-wise absolute value** (`type '<2>real' has no method 'abs'`; sysl 0.8.1 adds it).
  `asum`, `nrm2` and `iamax` are scalar walks because of it: `asum` and `iamax` need `|v|`, and
  `nrm2`'s band test is `|v| > t`. The longhands — `(v < 0).select(-v, v)`, or `(v > t) | (v < -t)` —
  compile well (LLVM turns the second into `fabs.2d` + `fcmgt.2d`), which is exactly why they would
  be copied; they are not written here. With it, all three take a contiguous kernel; a scalar
  reduction cannot be vectorized by the back end, since it may not reassociate a float sum.
- **Vector `mul_add`** (sysl 0.8.1). `a * b + c` already contracts to `fmla` at `optimization = "2"`
  (`dot`, `axpy`, `rot` all use `fmla.2d`/`fmla.4s`), so Level 1 is not slowed; it is for the
  micro-kernel, which must be able to ask for the fused form rather than hope for it.
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
