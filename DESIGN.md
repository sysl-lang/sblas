# sysl-lang/sblas — design

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
| 2 | gemv, ger, symv, syr2 | done, f32 and f64 |
| 3 | gemm (packing, cache blocking, threads), trsm, trmm, syrk | gemm: the blocked structure, f32 and f64, and `gemm_par` on the caller's workers, bitwise equal to it; trsm and trmm: recursive on gemm, f32 and f64, every side, triangle and diagonal; syrk: gemm's five loops writing one triangle |

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
lane-wise NaN-propagating `maximum` of the magnitudes (one `fmax` per register, and a NaN anywhere
makes the chunk's largest a NaN), then, only when that beats every earlier chunk, a second pass from
L1 for the first element holding it, so ties go to the earliest index. A chunk holding a NaN ends the
search at the first one, which is the first NaN overall.

Level 2 builds on the Level 1 kernels. `gemv` on a matrix with contiguous columns is column axpys,
four columns per pass so `y` is loaded and stored once per four; on contiguous rows (a transpose) it
is dot products, four rows per pass sharing each load of `x`, at 64 bytes of lanes per row so the four
accumulators and `x` fit the register file. Both take the vector the matrix sweeps past a 16 KB panel
at a time, so it stays in L1. `ger` is column axpys of `x`, or of `y` over `a.t()` when the rows are
the contiguous side. A matrix with neither stride 1, or a vector off stride 1, takes a walk.

`symv` reads each element of its triangle once, for two uses: column `j` off the diagonal is added
into `y` times `alpha x[j]` and dotted with `x` for `y[j]` (the row of the other triangle it mirrors),
four columns per pass so that `x` and `y` are loaded, and `y` stored, once per four; the four by four
diagonal blocks and the up to three columns left over (the last of a lower triangle, the first of an
upper one) take the reference's loop. `syr2` is two axpys into each column's part of the triangle in
one pass. A triangle whose rows are contiguous is the other triangle of the transpose, whose columns
are — a symmetric matrix is its own transpose, and the `syr2` update is symmetric — so both run on
`a.t()` with the triangle flipped there.

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

**A view comes apart into views.** `a.block(i, j, rows, cols)`, `a.col(j)` and `a.row(i)` (on both
`Matrix` and `MatrixMut`) are views of the same storage with the parent's strides, so a blocked
algorithm — slapack's `getrf` is the first — passes its panels and trailing blocks to `trsm` and
`gemm`, and its columns and rows to `iamax`, `scal`, `swap` and `ger`, with no copy. Each traps if it
reaches past the parent; an empty block may sit at the far edges (`i == rows`, `j == cols`), where it
starts at the end of the storage and reads nothing.

```sysl
gemv(alpha, a, x, beta, y)        // y := alpha a x + beta y;  a.t() for trans = 'T'
ger(alpha, x, y, a)               // a := alpha x yᵀ + a
val len = gemm_work_len[f64](m, n, k)              // GemmWorkLen { a, b }: elements per packing buffer
gemm(alpha, a, b.t(), beta, c, GemmWork(pa, pb))   // c := alpha a bᵀ + beta c, in the caller's buffers
gemm_alloc(alpha, a, b.t(), beta, c)               // sh.sysl.sblas.heap: allocates them per call
var pool = Spawned(6)                              // sh.sysl.sblas.threads: any `impl Workers`
gemm_par_alloc(&pool, alpha, a, b.t(), beta, c)    // the same C, bitwise, on six threads
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

**`trmm` takes trsm's arguments** and multiplies in place: `b := alpha op(A) b` on the left,
`b := alpha b op(A)` on the right, with the transpose again a view (`Uplo.Lower, u.t()` for CBLAS's
`uplo = U, trans = T`). Its recursion is trsm's run the other way: for a lower A on the left, B₂ is
multiplied first, then takes `alpha A₂₁ B₁` from gemm while B₁ is still the caller's, and B₁ is
multiplied last; an upper A is the mirror. The right side splits B's columns the same way. Either
side is the other on the transposes, and the one run is the one that leaves gemm's C with its rows
closer together than its columns — a column-major B on the right stays column-major, where turning
it would put every write-back on gemm's element-at-a-time path. The 16-row diagonal blocks are
multiplied directly: a column-major B a column at a time in registers, against the triangle packed
times alpha with zeros elsewhere, the lanes of each column's own vector done one at a time so an
infinity in B reaches exactly the rows the reference product sends it to; any other B row by row (or
column by column on the right) in the order that reads only what is not yet overwritten.
`trmm_work_len(side, m, n)` covers gemm's updates for B split either way. `alpha = 0` sets B to zero
reading neither B nor A.

```sysl
trmm(Side.Right, Uplo.Upper, Diag.NonUnit, 1.0, t, v, work)    // v := v t, QR's block reflector
syrk(Uplo.Lower, -1.0, a21, 1.0, a22, work)                    // a22 -= a21 a21ᵀ, Cholesky's update
symv(Uplo.Lower, 1.0, a, x, 0.0, y)                            // y := a x, a symmetric
syr2(Uplo.Lower, -1.0, v, w, a)                                // a -= v wᵀ + w vᵀ
```

**`syrk` is gemm's five loops writing one triangle.** `c := alpha a aᵀ + beta c` (`a.t()` for
`trans = 'T'`) runs `gemm_blocked` on `a` and `a.t()` with the triangle a compile-time parameter
(`tri_all` for gemm, where every check folds away): a block of A whose rows lie wholly outside the
triangle against the current panel of B is not packed, a tile wholly outside it is not multiplied,
and a tile the diagonal crosses is computed whole and written back only inside it. So syrk does half
of gemm's flops at gemm's speed, and the other triangle of C is neither read nor written — bit for
bit as it was. `syrk_work_len(n, k)` is gemm's for an `n` x `n` C. `beta = 0` writes the triangle
without reading it; `alpha = 0` or `k = 0` only scales the triangle.

`symv` and `syr2` read (write) only the `uplo` triangle, diagonal included; the other triangle may hold
anything. They keep Level 2's quick returns: an empty matrix leaves `y` alone, `beta = 0` writes `y`
without reading it, `alpha = 0` reads neither the matrix nor `x` (and `syr2` returns at once). A
matrix that is not square, or vectors of the wrong length, trap.

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

Level 3 only; Level 1 and 2 are memory-bound and run on one core. `gemm_par` is built, f32 and f64,
on sysl 0.8.3 (which lends a closure at a `@scoped` parameter a struct holding views, F6 below).

**The pool is the caller's; the core never starts a thread** (decided). The core stays `@no_alloc`
and imports nothing that requires `os`, so it states what it needs as a trait, and the caller's pool
implements it:

```sysl
trait Workers
    count(*self) -> usize                       // how many jobs `each` runs at once
    each(*self, job: &sync Fn(usize) -> unit)   // job(0) ..< job(count - 1), all finished on return
```

An `impl` marks its `each` `@scoped(job)` (a trait requirement may not carry it: *"'@scoped'
promises what a body does with a parameter, and 'each' is a requirement with no body"*), which is
what lets the job, a closure written at the call inside gemm, borrow the matrices' views rather than
share them. `sysl.threads` has no pool, so `sh.sysl.sblas.threads.Spawned(n)` is one over `spawn`:
each `each` starts a thread per job past the first, runs the first itself, and joins the rest (a
fan-out joined by a loop over the same slice). It is a module of its own because it requires `os`,
which neither the core nor `heap` does. It stays as the dependency-free example.

**A standing pool is a workspace member, `pool-workers/`** (package `sblas-pool`, module
`sh.sysl.sblas_pool`), so the core never depends on `sysl-lang/pool` and only a program that wants
the pool names it. Its module cannot be `sh.sysl.sblas.pool`: a name inside a path another package
offers is refused (*"'sh.sysl.sblas' is both a module of this project and one sblas offers"*,
`reference/packages.md` § two packages cannot quietly share a name). `PoolWorkers(pool: &Pool, n)`
holds the program's pool by reference — shared, not owned, so the program runs its own tasks on it
between products — and its `each` opens one `Pool.scope`, spawns `job(1)` … `job(n - 1)` into it and
runs `job(0)` on the calling thread, as `Spawned` does; `on(pool)` is `pool.threads() + 1` workers.
One `scope` of direct spawns rather than `parallel_for`, since `parallel_for` adds a level of halving
tasks for what is a handful of jobs. Measured (`./bench each`, empty jobs, median of 7 x 20000):

| workers | `Spawned` | pool |
|---:|---:|---:|
| 2 | 12.4 µs | 2.9 µs |
| 4 | 24.2 | 5.3 |
| 6 | 34.4 | 9.1 |

Running job 0 on the caller matters: the first version handed every job to the pool and slept in
`scope`, and a single worker then ran at 85–91% of `Spawned(1)` at n = 128–512 f64 (a task handoff
and a condvar wake per `each`); with job 0 here, and one worker running inline with no task at all,
the two are level at every size.

**What is left at small n is the two `each` per panel, and it is small.** An n = 256 product is one
KC x NC panel, so two `each`: 2 x 5.3 µs at 4 workers and 2 x 9.1 µs at 6, against a product of about
0.25 and 0.22 ms — 4% and 8%. The grid adds 0% at 4 workers (`par_grid(4, 32, 43)` = 4 x 1, largest
block 344 tiles, the mean exactly) and 4.6% at 6 (2 x 3, 240 against a mean of 229) for f64. Fusing
the two `each` would need a barrier inside the jobs, which a `Workers` that runs jobs in turn (more
jobs than threads, `Backwards`) would deadlock on; packing B per job instead duplicates the packing
`tm` times and changes `gemm_par_work_len`. Neither is worth 2–4% at one size. Run to run, the n =
256 figures move by ±20% on this machine at load 7–10, more than either cost. P-core / E-core placement
is not measured: macOS has no public call that pins a thread to a core class, so it is not claimed as
a cause.

**f64 at four threads and n = 512–2048 is level with OpenBLAS** over three clean runs (pool 99–111%
at n = 1024, median about 104% over n = 512–1024 in ten rounds below load 10). Single rounds dip to
83–96%, but the dips hit OpenBLAS's own timings in the same round, so they are load, not sblas.
Against the ideal of one thread's speed times four it runs at 84–93%.

**The entry is a second routine, `gemm_par(workers, alpha, a, b, beta, c, work)`**, so `gemm` and its
single-thread speed are untouched, and `gemm_par_work_len[T](m, n, k, workers)` sizes a `GemmWork`
whose `a` holds one packed block of A per worker (`gemm_work_len`'s `a` times the workers; `b` is
shared). `sh.sysl.sblas.heap.gemm_par_alloc(workers, …)` allocates them per call.

**The partition is BLIS's IC/JR.** For each KC x NC panel: one `each` packs B, each job a run of its
NR-wide panels into the shared buffer; its return is the barrier. A second `each` computes: the
panel's tiles are split into a `tm x tn` grid of row runs (whole MR panels) by column runs (whole NR
panels), `tm * tn <= count` chosen to minimise the largest run, rows first, so a tall problem splits
IC and a short one also splits JR; each job packs its own rows of A into its own slice of `work.a`
and runs the MC/JR/IR loops over its block with a tile of its own. Jobs past `tm * tn` return at once.

**Determinism by construction**: every run starts on a multiple of MR or NR, and MC and NC are
multiples of both, so each tile of C is the same tile, packed the same way and summed over k in the
same order, as single-threaded gemm makes it; the result is bitwise identical for any worker count.
The tests hold it to that: every element of C's storage, margins included, equal to gemm's on values
whose sums round, across every blocking fringe, transposes, block views, `alpha = 0`, `beta` of zero
(C seeded with NaN) and one, at 1, 2, 3, 5 and 64 workers on a pool that runs its jobs last first, and
at 1, 2, 3, 6 and 7 real threads, and at 1, 2, 3 and 6 workers of a standing pool across both
widths' fringes (`pool-workers/`); and once against OpenBLAS on three threads.

## Oracles

- **Accelerate** (macOS): `libcblas` in the SDK resolves into vecLib, so `@link("cblas")` reaches it
  with no framework flag. The `oracle/` workspace member (`sh.sysl.cblas`) declares the CBLAS entry
  points behind a trait, `Oracle`, implemented for f32 and f64, so each check is written once for both
  widths. It is its own project so that the `sblas` package links nothing. CBLAS refuses a negative
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
  for `N = 0` without `trans`. `sh.sysl.sblas` follows the reference, the oracle check compares that
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
- **F6: a closure at a `@scoped` parameter may borrow a view, but not a struct that holds one —
  fixed in sysl 0.8.3**, where a struct or tuple of lendable views and shareable scalars is lent as
  its views are; `gemm_par`'s jobs capture the `Matrix[T]` `a` and `b`, the `MatrixMut[T]` `c` and
  the `GemmWork[T]` whole, with no view rebuilt from its fields inside the closure.

- **F7: no vector form of the NaN-propagating maximum — fixed in sysl 0.8.3**, where
  `a.maximum(b)` and `a.maximum()` are IEEE 754-2019's maximum per lane and across lanes (`fmax`, one
  instruction per register on AArch64). `iamax` keeps one running `maximum` of the magnitudes per chunk
  and tests the result against itself for a NaN, with no per-lane compare or sum in the loop.

## Open questions for the user

1. **The C-ABI layer**: provide `cblas_*` exports (and the Fortran `dgemm_` symbols)? In this package
   or a separate one (`sblas-cblas`)?
2. **Negative stride on a one-vector routine** (`nrm2`, `asum`, `scal`, `iamax`): the sysl API treats
   it as a view walked backwards; the reference returns immediately (and Accelerate reports a
   parameter error). Keep the view semantics?
3. **Mismatched lengths**: trap (now), use the shorter, or return a `Result`?
4. **`iamax`'s result**: `Option[usize]`, 0-based (now), or CBLAS's plain index with 0 for empty?
5. **Level 3 workspace** — *decided*: caller-supplied packing buffers in the core (`GemmWork`, sized
   by `gemm_work_len` and `trsm_work_len`; it stays `@no_alloc` and usable on a board), plus
   `sh.sysl.sblas.heap.gemm_alloc` and `trsm_alloc`.
6. **Complex**: on `sysl.math.complex.Complex[F]`, and at which level does it start?
7. **Threads** — *decided*: the caller's pool, through the core's `Workers` trait; the core starts no
   thread. A standing pool is `sysl-lang/pool`, behind `PoolWorkers` in the `pool-workers/` member
   (decided); `Spawned` stays as the dependency-free example. Open: a coordinate fetches only a
   repository's root package, so a program outside this repository cannot depend on `sblas-pool` —
   give it a repository of its own (`sysl-lang/sblas-pool`), or should a coordinate be able to name a
   workspace member?
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
    OpenBLAS differ (see Oracles). The kernel carries a running NaN-propagating lane-wise `maximum`
    (F7), so detecting a NaN costs nothing beyond the max itself.
