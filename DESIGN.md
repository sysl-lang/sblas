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
| 2 | gemv, ger | done, f32 and f64 |
| 3 | gemm (packing, cache blocking, threads), trsm | gemm: the blocked structure, f32 and f64, single-threaded; trsm: recursive on gemm, f32 and f64, every side, triangle and diagonal |

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
they are memory-bound and a contiguous slice kernel over `<W>T` vectors (W = 128 bytes of lanes:
eight 128-bit registers, so eight independent accumulators) is the whole design. Each routine has the
contiguous kernel and a scalar walk that follows arbitrary strides. `nrm2` keeps three accumulators
per lane, one per Blue band, at 64 bytes of lanes (128 spills); its band is chosen per lane by mask on
`|v|`, a NaN falling in neither outer band and so in the middle sum, as the scalar rule puts it.
`iamax` takes the kernel only at stride 1, where BLAS order is storage order: per 16 KB chunk, the
lane-wise largest magnitude with NaN lanes never chosen (a lane mask records the NaNs, which costs
one compare and one OR per vector), then, only when that beats every earlier chunk, a second pass from
L1 for the first element holding it, so ties go to the earliest index. A chunk holding a NaN ends the
search at the first one, which is the first NaN overall.

Level 2 builds on the Level 1 kernels. `gemv` on a matrix with contiguous columns is column axpys,
four columns per pass so `y` is loaded and stored once per four; on contiguous rows (a transpose) it
is dot products, four rows per pass sharing each load of `x`, at 64 bytes of lanes per row so the four
accumulators and `x` fit the register file. Both take the vector the matrix sweeps past a 16 KB panel
at a time, so it stays in L1. `ger` is column axpys of `x`, or of `y` over `a.t()` when the rows are
the contiguous side. A matrix with neither stride 1, or a vector off stride 1, takes a walk.

## Element types and tile sizes

**Element types are a trait, `Real: Float`, implemented for f32 and f64.** It adds the constants that
depend on the format — Blue's thresholds and scale factors for `nrm2`, the safe range for `rotg` —
each an exact power of two written as a hex float literal. A routine is `axpy[T: Real](…)` and is
monomorphized per width; the vector kernels are `[T: Real, const W: usize]` and receive
`W = 128 / sizeof(T)` from the routine.

**Complex** would be a second trait over `sysl.math.complex.Complex[F]` (conjugation, `|re| + |im|`
for `iamax`, the complex `rotg`), with the real routines unchanged. Level 1 complex is not cheap — it
doubles the routine list (`dotu`/`dotc`, `scnrm2`, `icamax`, `crotg`, the real-scalar `csscal`) — so
it waits for the decision below.

**Tile sizes are const generics**: `gemm_blocked[T, const V, const MV, const NR, const TILE, const KC,
const MC, const NC]`, MR being `MV` vectors of `V` lanes. A function's value parameter cannot carry a
default (probed: "'W' is a value parameter of the function …, so '= …' has nothing to stand in for"),
so the public `gemm` names the instantiation itself, per target, through `#if`. The tile is then a
compile-time constant everywhere: loop bounds fold, the accumulator array `[NR][MV]<V>T` lives in
registers (24 of them on AArch64, checked in the emitted loop), and packing loops unroll. `TILE` (MR x
NR, the scratch tile's length) is passed on its own because an array's length may name a value
parameter but not compute with one — type-level arithmetic, which the reference states is a separate
feature; a body computes with them freely.

**The starting blocking** (`level3.sysl`), to be swept when the kernel is tuned:

| | AArch64 (32 q registers) | anything else (16) |
|---|---|---|
| MR x NR, f64 | 8 x 6 (4 vectors x 6 lanes of B's row, 24 accumulators) | 4 x 4 |
| MR x NR, f32 | 16 x 6 | 8 x 4 |
| KC | 2 KB of depth: 256 f64, 512 f32 | same |
| MC | 960 bytes of height: 120 f64, 240 f32 | same |
| NC | 3072 | same |

The 8x6 shape is BLIS's armv8a dgemm 6x8 turned to accumulate along C's columns (contiguous here,
rows there); OpenBLAS's ARMv8/VORTEX kernels are 8x4 (f64) and 16x4 (f32), which use fewer
accumulators than the register file holds. KC keeps an NR-wide B panel (12 KB) in L1 beside the A
panel being streamed; MC x KC of A (240 KB) sits in L2; KC x NC of B (6 MB) in the last level. KC and
MC are written in bytes so both widths have the same footprint.

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

**A matrix is a view with two strides**, `Matrix[T] { data, rows, cols, row_stride, col_stride }`
and `MatrixMut[T]`, as BLIS has it: row-major, column-major and any general stride are the same type,
and `a.t()` is a view that swaps the two strides rather than a `trans` flag. `matrix(data, rows, cols,
ld)` is BLAS's column-major layout and traps when `ld < max(1, rows)` or the data is short; a
hand-built view is checked by the routine before it reads. `gemm` therefore needs no
`transa`/`transb`/`lda` parameters; packing absorbs whatever strides it is given.

```sysl
gemv(alpha, a, x, beta, y)        // y := alpha a x + beta y;  a.t() for trans = 'T'
ger(alpha, x, y, a)               // a := alpha x yᵀ + a
val len = gemm_work_len[f64](m, n, k)              // GemmWorkLen { a, b }: elements per packing buffer
gemm(alpha, a, b.t(), beta, c, GemmWork(pa, pb))   // c := alpha a bᵀ + beta c, in the caller's buffers
gemm_alloc(alpha, a, b.t(), beta, c)               // sh.sysl.blas.heap: allocates them per call
```

Level 2 keeps the reference's quick returns: an empty matrix leaves `y` alone whatever `beta` (so
`beta` is not applied when there are no columns), `beta = 0` writes `y` without reading it, and
`alpha = 0` reads neither the matrix nor `x`.

`gemm` does the same where the reference is specific: `beta = 0` writes C without reading it (a NaN
there does not survive), `alpha = 0` or `k = 0` reads neither A nor B and only scales C, and
`beta = 1` with `alpha = 0` touches nothing. Shapes that disagree, and buffers shorter than
`gemm_work_len` says, trap. The allocating form lives in its own module because `@no_alloc` in a
file's header binds the whole module.

**`trsm` names its triangle with three enums**, since a view says where the elements are but not
which of them to read:

```sysl
trsm(Side.Left, Uplo.Lower, Diag.NonUnit, alpha, a, b, work)      // b := alpha a⁻¹ b
trsm(Side.Right, Uplo.Lower, Diag.Unit, alpha, u.t(), b, work)    // b := alpha b (uᵀ)⁻¹, u upper
```

`Uplo` describes the view as passed, not the storage: `a.t()` of an upper-triangular matrix is a
lower-triangular view, so CBLAS's `(uplo = U, trans = T)` is `Uplo.Lower, a.t()` here. Only the
named triangle is read, and with `Diag.Unit` not the diagonal either. `trsm_work_len(side, m, n)` sizes
the same `GemmWork` buffers; a triangle within the substitution leaf needs none. `alpha = 0` sets B to
zero reading neither B nor A, as the reference does; a singular A is not detected (a zero on the
diagonal gives infinities), also as the reference does.

The solve is recursive: A is halved (on a multiple of the 16-row leaf), the first diagonal block's rows
of B are solved, the rest are updated by `gemm` (`B₂ -= A₂₁ X₁`, in the caller's packing buffers), and
the second block is solved — top first for a lower triangle, bottom first for an upper. Halving keeps
gemm's shared dimension as deep as the problem allows (an n = 2048 solve's first update is a k = 1024
gemm), so nearly all of a large solve's flops are gemm's; only the 16-row diagonal blocks are done by
substitution, with the inner loop along whichever of B's dimensions is contiguous. `Right` is `Left`
on the transposes (`X A = B` is `Aᵀ Xᵀ = Bᵀ`, the triangle flipped), which the views make free.

**A C-ABI export layer is possible and not decided** (see the questions). It would be `@export`
definitions named `cblas_dgemm` and so on, converting CBLAS's `(order, trans, n, ptr, inc)` arguments
into views. It would follow the reference's semantics exactly where they differ from the sysl API (a
non-positive stride on a one-vector routine is a quick return there, a view here).

## Where sysl 0.8.1 matters

The package needs 0.8.1 (`sysl = "0.8.1"`):

- **Unchecked access in the walks.** A strided walk checks once, before it starts, that the view's
  last element is inside its data (`walkable`), then reads and writes inside an `@unchecked` block,
  since the stride arithmetic is beyond what the optimizer proves. A view built by hand rather than by
  `strided` is caught by the same check, and the tests trap one in each direction.
- **Unchecked vector loads and stores in the kernels.** `@unchecked` reaches a vector `load` and
  `store` (0.8.1). Each kernel checks once that its slices have the same length (`runs`), after which
  the loop condition `i + W <= len` keeps every load, store and tail subscript in bounds, and the whole
  kernel runs `@unchecked`: no main loop of Level 1 or 2 carries a compare beside its own (counted in
  the disassembly; `bench/README.md`). Tests hand a kernel slices of different lengths and see it trap
  before it reads.
- **Lane-wise `abs` and `mul_add`** (0.8.1) are what `asum`, `nrm2` and `iamax`'s kernels and the
  fused accumulations are written with.
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
- **Where the two disagree, the reference decides.** Accelerate's `cblas_?gemv` with `trans` and
  `M = 0` scales `y` by `beta`; the reference and OpenBLAS return at once, as Accelerate itself does
  for `N = 0` without `trans`. `sh.sysl.blas` follows the reference, the oracle check compares that
  one case with `y`'s starting value, and a test pins Accelerate's behaviour so a change in it shows.
- **netlib**: the reference test programs (`dblat1.f`, `sblat1.f`, `dblat2.f`, `dblat3.f`) carry
  hand-derived expected values for small fixed inputs, including every stride sign; porting their
  tables gives a check that depends on no vendor. Level 1's tables are ported in `netlib_tests.sysl`
  (dot, axpy, rotg, rot, copy, swap, nrm2, asum, scal, iamax, at both widths, with netlib's
  `SFAC`-scaled comparison and the scaling stress test for `nrm2`); its header records the source URLs
  and SHA-256s. They agree with the package everywhere, `iamax` with a NaN included: netlib's
  `i?amax` returns the first NaN wherever it is, and so does the package. Negative strides agree
  too: netlib's two-vector routines and its `nrm2` test walk them backwards as the views do.
- **`iamax` with a NaN is the one place the vendor libraries are not the oracle.** Accelerate never
  answers a NaN (not even at element 0, nor when every other element is smaller); OpenBLAS answers one
  only at element 0. Neither is netlib's rule, which the package follows, so
  `oracle_iamax_nan` pins what each library does (as the Accelerate `gemv` M = 0 note does for that
  case) and `netlib_tests.sysl` is the oracle.

Every test runs under ASan as well (`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`).

## Benchmark method

`bench/` times every contiguous Level 1 routine of sysl, Accelerate and OpenBLAS at sizes chosen to
live in L1, in L2, and in main memory, and prints GB/s of vector data read and written (2mnk GFLOPS
for gemm, when it exists); `gemv` and `ger` are reported in GFLOPS (2mn per call) on square
column-major matrices of 48, 256 and 2048, whose f64 storage fits L1, fits L2, and does not. Within each trial the three run in turn, so a change in the machine's
load lands on all three alike; each figure is the median of seven trials of about 1e8 elements.
Runs are `caffeinate -dimsu` with `VECLIB_MAXIMUM_THREADS=1` (OpenBLAS is pinned to one thread when
it is loaded), and record the chip and the load average beside the numbers.

**On M4 and later, Accelerate uses the SME matrix unit**, which NEON code cannot match: on the M5 Pro
Accelerate's f64 `dot` reaches about 88 GFLOPS on L2-resident data single-threaded, several times what
one core's NEON pipes can issue. The target for sysl is therefore **OpenBLAS single-thread** (at
least 80% of its DGEMM), with Accelerate reported alongside.

## Gaps found so far

- **F1: a vector load or store could not be unchecked — fixed in sysl 0.8.1**, where `@unchecked`
  reaches `load` and `store`. Every Level 1 and 2 kernel now runs unchecked after one length check;
  `dot`'s loop is 23 instructions per 16 f64 with no compare beside its own. gemm's micro-kernel loop
  is 59 instructions per two steps at both widths (48 `fmla` by element, 6 `ldp q`, the counter and
  the branch) with no check in it or in the tile's store, and B's packing, the write-back and trsm's
  leaf move vectors unchecked.
- **F2: no lane-wise absolute value — fixed in sysl 0.8.1**, where `v.abs()` is one `fabs` per
  register. `asum`, `nrm2` and `iamax` each take a contiguous kernel with it, and none spells `|v|` as
  a `select`: `asum` is at OpenBLAS parity, `nrm2` 2.3–4.7x it, `iamax` 76–106% (`bench/README.md`).
- **Vector `mul_add` — added in sysl 0.8.1.** The kernels' accumulations ask for the fused form
  (`a.mul_add(b, acc)`) rather than relying on `a * b + c` contracting at `optimization = "2"`; so do
  gemm's micro-kernel (`a[v].mul_add(b[l], acc)`, the same `fmla` by element as before) and trsm's
  leaf.
- **F5: a vector `load` as an argument of `mul_add`, `select` or `shuffle` took no width — fixed in
  sysl 0.8.2**, where it takes the receiver's, as an operator's other operand gives it
  (`reference/vectors.md`, "Where the width comes from"). `acc = a.mul_add(y.load(i), acc)` compiles, and
  `dot` and `axpy` write it so; the receiver itself is still bound to a typed `val`, since nothing gives
  a receiver's load its width.
- **An `impl` in a `@tests` file is refused for a builtin type even when the trait is the file's
  own.** The stated reason — a trait answering differently in tests and in the shipped program — does
  not arise for a trait that exists only in tests. This is why the oracle trait lives in the `oracle/`
  project, which turned out to be the better structure anyway.
- **Not a gap: reading a lane by a loop's variable.** A lane index must be a constant, and the
  variable of a `for const` over a constant range is one, so the micro-kernel reads B's row as
  vectors and feeds each lane to `fmla` by element (`fmla.2d v, a, b.d[1]`, as OpenBLAS does) with
  `for const jv in 0..<NR / V` and `for const l in 0..<V`, generically over the tile. What was
  missing is a vector-to-array conversion (`val arr: [N]f64 = v` is refused in sysl 0.8.0, *declared
  […]f64 but the value is <N>f64*); sysl 0.8.1 makes it implicit at an assignment, an argument and a
  return, where the element type and the length agree.
- **F4: prefetch is available from sysl 0.8.1** (`sysl.hint.prefetch(p, access, locality)`, one
  `prfm`). Level 1 and 2 do not use it: measured 1 KB ahead in `dot` and `axpy` and 512 bytes ahead on
  `gemv`'s four matrix streams, it was no faster than the hardware prefetcher on these sequential
  streams outside DRAM-sized `gemv`, and slower in L1 and L2. gemm does not use it either:
  prefetching both panels ahead in the k loop (A 512 or 1024 bytes, B 384 or 768) and the C tile
  before its write-back measured no faster at any size on an M5 Pro, whose hardware prefetcher
  already follows the unit-stride panel walks.
- **A slice argument is retained and released around an inlined call.** Each tile's `micro` call
  takes and drops a reference count on `work.a` and `work.b` (`ldr`/`add`/`str` + `cbz` each way)
  even with the panels passed as offsets into the whole buffers; a borrowed parameter needs neither.
  About 16 instructions per tile, under 1% at n = 64, so it is noted rather than chased.

## Open questions for the user

1. **The C-ABI layer**: provide `cblas_*` exports (and the Fortran `dgemm_` symbols)? In this package
   or a separate one (`blas-cblas`)?
2. **Negative stride on a one-vector routine** (`nrm2`, `asum`, `scal`, `iamax`): the sysl API treats
   it as a view walked backwards; the reference returns immediately (and Accelerate reports a
   parameter error). Keep the view semantics?
3. **Mismatched lengths**: trap (now), use the shorter, or return a `Result`?
4. **`iamax`'s result**: `Option[usize]`, 0-based (now), or CBLAS's plain index with 0 for empty?
5. **Level 3 workspace** — *decided*: caller-supplied packing buffers in the core (`GemmWork`, sized
   by `gemm_work_len` and `trsm_work_len`; it stays `@no_alloc` and usable on a board), plus
   `sh.sysl.blas.heap.gemm_alloc` and `trsm_alloc`.
6. **Complex**: on `sysl.math.complex.Complex[F]`, and at which level does it start?
7. **Threads**: whose pool — `sysl.threads`, a package, or the caller's?
8. **Vector `mul_add` and `abs`**: add them to `<N>T` before the micro-kernels are written?
9. **The `@tests` impl rule**: allow an `impl` of a test-file trait for any type?
10. **Tile sizes** — *decided*: fixed per target at compile time.
11. **The speed target on Apple silicon** — *decided*: OpenBLAS's NEON path. Accelerate's SME is out
    of scope (it would need streaming-mode support in sysl) and is reported alongside, not chased.
12. **The oracle on Linux**: `libcblas` is not a standard name there; link OpenBLAS directly?
13. **How a triangle is named** — *decided, may be overruled*: enum parameters `Side`, `Uplo`, `Diag`,
    with `Uplo` describing the view as passed (so `a.t()` of an upper triangle is `Lower`), rather than
    CBLAS's triangle-of-the-storage or a triangular view type. A view type (`Lower(a)`) would carry the
    triangle through `t()` itself; it waits for the routines that would share it (`trmm`, `symm`,
    `syrk`, `trsv`) to show whether that is worth a type per triangle.
14. **`iamax` and NaN** — *decided*: the first NaN wins over everything, infinities included; with no
    NaN, the first element of largest magnitude (reference LAPACK 3.10's `i?amax`). Accelerate and
    OpenBLAS differ (see Oracles).
