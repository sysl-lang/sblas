# sblas

A BLAS written in sysl, laid out the way BLIS lays one out: the blocking loops and packing written
once, generically, and one small micro-kernel per architecture. Module `sh.sysl.sblas`; no C, no
allocation, and it builds for bare-metal targets.

Level 1 is here, generic over `f32` and `f64`: `axpy`, `dot`, `nrm2` (Blue's overflow- and
underflow-safe algorithm), `asum`, `scal`, `copy`, `swap`, `iamax`, `rot`, `rotg`. Level 2 has `gemv` (and, through `a.t()`, its transpose), `ger`, `symv` and `syr2`.
Level 3 has `gemm`, its threaded form `gemm_par`, `trsm`, `trmm` and `syrk`. The rest of Level 2 and Level 3 are designed in [DESIGN.md](DESIGN.md).

```sysl
import sh.sysl.sblas.*

var xs: [4]f64 = [1, 2, 3, 4]
var ys: [4]f64 = [1, 1, 1, 1]

axpy(2.0, view(xs[..]), view_mut(ys[..]))                 // ys = [3, 5, 7, 9]
print(dot(view(xs[..]), strided(ys[..], 4, -1)))          // ys walked backwards
print(nrm2(view(xs[..])), iamax(view(xs[..])))            // 5.477…, Some(3)
```

A vector is a strided view with BLAS's stride rules: a negative stride walks the storage backwards,
so element 0 is the last element the view covers. `strided` checks that the slice holds every element,
and two views of different lengths trap.

A matrix is a view with a stride per dimension. `matrix(data, rows, cols, ld)` is column-major with
leading dimension `ld`, and `a.t()` is its transpose over the same storage, so no routine takes a
transpose flag:

```sysl
val m: [6]f64 = [1, 4, 2, 5, 3, 6]
var b: [6]f64
val a = matrix(m[..], 2, 3, 2)                              // [[1, 2, 3], [4, 5, 6]]
gemv(1.0, a, view(xs[0..<3]), 0.0, view_mut(ys[0..<2]))     // ys[0..<2] = a xs = [14, 32]
gemv(1.0, a.t(), view(ys[0..<2]), 1.0, view_mut(xs[0..<3]))  // xs[0..<3] += aᵀ ys
ger(2.0, view(ys[0..<2]), view(xs[0..<3]), matrix_mut(b[..], 2, 3, 2))  // b += 2 ys xsᵀ
```

A matrix comes apart into views of the same storage: `a.block(i, j, rows, cols)` is the block whose
first element is `(i, j)`, and `a.col(j)` and `a.row(i)` are vectors, each trapping if it reaches past
`a`. A blocked algorithm hands its pieces to the routines this way, as LAPACK hands `A(i, j)` and `lda`:

```sysl
gemv(1.0, a.block(0, 1, 2, 2), view(xs[0..<2]), 0.0, view_mut(ys[0..<2]))  // the right 2 x 2 of a
print(dot(a.row(0), a.row(1)))                                            // 1·4 + 2·5 + 3·6 = 32
```

`gemm` is `c := alpha * a * b + beta * c` over the same views. Its packing buffers are the caller's,
so the package still allocates nothing; `gemm_work_len` says how long they must be, and
`sh.sysl.sblas.heap.gemm_alloc` is the form that allocates them per call:

```sysl
var c: [4]f64
var pa: [64]f64
var pb: [64]f64
gemm(1.0, a, a.t(), 0.0, matrix_mut(c[..], 2, 2, 2), GemmWork(pa[..], pb[..]))  // c = a aᵀ = [14, 32, 32, 77]
```

`gemm_par` is the same product on the caller's workers, and gives the same C bit for bit whatever
their number. The package starts no thread: it takes any value implementing `Workers` (`count`, and
`each`, which runs `job(0)` … `job(count - 1)` and returns when all are done), and
`gemm_par_work_len` sizes the buffers, a block of A per worker. Two implementations come with it:

- **`sh.sysl.sblas_pool.PoolWorkers`** (`pool-workers/`) stands on a work-stealing pool of
  [`sysl-lang/pool`](https://github.com/sysl-lang/pool) that the program keeps: no thread is started
  per call, an `each` costs 3–9 µs at 2–6 workers, and small products run at OpenBLAS's speed. It is
  a workspace member of its own, so the sblas package itself still depends on nothing. A coordinate
  fetches a repository's root package only, so another program copies its one file
  (`pool-workers/sh/sysl/sblas_pool/pool.sysl`, 15 lines of code) for now.
- **`sh.sysl.sblas.threads.Spawned(n)`** is the dependency-free example over `sysl.threads`: it starts
  its threads on every call (12–34 µs per `each`), which a large product does not notice.

```sysl
import sh.sysl.sblas.heap.gemm_par_alloc
import sh.sysl.sblas_pool.on
import sh.sysl.sblas.threads.Spawned
import sh.sysl.pool.Pool

val pool = Pool.new(5)                    // kept for the program's life, and shared with its own work
var six = on(pool)                        // the calling thread and the pool's five
gemm_par_alloc(&six, 1.0, a, a.t(), 0.0, matrix_mut(c[..], 2, 2, 2))  // the same c, on six threads

var spawned = Spawned(6)                  // sh.sysl.sblas.threads: six threads started for the call
gemm_par_alloc(&spawned, 1.0, a, a.t(), 0.0, matrix_mut(c[..], 2, 2, 2))
```

`trsm` solves a triangular system in place for every column of B at once — `b := alpha a⁻¹ b` on the
`Left`, `b := alpha b a⁻¹` on the `Right` — with the same packing buffers (`trsm_work_len`). `Uplo`
names the triangle of the view as passed, so the transpose of an upper triangle is `Lower`:

```sysl
val l: [4]f64 = [2, 1, 0, 4]                                // [[2, 0], [1, 4]]
var rhs: [2]f64 = [2, 9]
trsm(Side.Left, Uplo.Lower, Diag.NonUnit, 1.0, matrix(l[..], 2, 2, 2), matrix_mut(rhs[..], 2, 1, 2), GemmWork(pa[..], pb[..]))  // rhs = [1, 2]
trsm(Side.Left, Uplo.Upper, Diag.NonUnit, 1.0, matrix(l[..], 2, 2, 2).t(), matrix_mut(rhs[..], 2, 1, 2), GemmWork(pa[..], pb[..]))  // lᵀ x = rhs
```

`trmm` multiplies by a triangle in place with trsm's arguments (`trmm_work_len`), and `syrk` updates
one triangle of a symmetric C, leaving the other bit for bit (`syrk_work_len`); `symv` and `syr2` are
the Level 2 routines on a symmetric matrix held in one triangle. Each reads only the `uplo` triangle,
and a transpose is again a view:

```sysl
trmm(Side.Right, Uplo.Upper, Diag.NonUnit, 1.0, t, v, work)   // v := v t
syrk(Uplo.Lower, -1.0, a21, 1.0, a22, work)                   // a22 -= a21 a21ᵀ, lower triangle only
syrk(Uplo.Upper, 1.0, a.t(), 0.0, c, work)                    // c := aᵀ a, upper triangle only
symv(Uplo.Lower, 1.0, s, x, 0.0, y)                           // y := s x, s symmetric
syr2(Uplo.Lower, -1.0, v, w, s)                               // s -= v wᵀ + w vᵀ
```

```
dependencies {
  sblas { git = "github.com/sysl-lang/sblas", version = "0.4.0" }
}
```

Needs sysl 0.8.3.

## Layout

| directory | what |
|---|---|
| `sh/sysl/sblas/` | the package — `sblas.sysl` (element types, vector and matrix views), `level1.sysl`, `level2.sysl`, `level3.sysl` — and its tests against the definitions (`nrm2`'s range and band edges, `rotg`'s signs, `iamax`'s ties and NaNs across lanes and chunks, every length around the vector width, `gemv`'s and `ger`'s scalings and strides, `gemm` at every blocking boundary, `gemm_par` bitwise equal to it at every boundary and worker count, `trsm` solved by hand and multiplied back across them, and `syrk`, `trmm`, `symv` and `syr2` exactly equal to their definitions on integer data across the same boundaries, three layouts and every stride, with NaN or a sentinel in what they must not read or write) |
| `sh/sysl/sblas/heap/` | `gemm_alloc`, `gemm_par_alloc`, `trsm_alloc`, `trmm_alloc`, `syrk_alloc`: the Level 3 routines with their packing buffers allocated per call (the main module is `@no_alloc`) |
| `sh/sysl/sblas/threads/` | `Spawned(n)`: `Workers` for `gemm_par` over `sysl.threads`, a thread per job per call, and its tests on real threads (it requires `os`, which the rest does not) |
| `sh/sysl/sblas/netlib_tests.sysl` | netlib's own Level 1 test tables (`dblat1.f`, `sblat1.f`) ported: the fixed vectors and expected results at every stride in them, with netlib's `SFAC`-scaled comparison |
| `pool-workers/` | `PoolWorkers`: `Workers` on a standing `sysl-lang/pool`, a workspace member so that only a program that wants the pool depends on it, and its tests: bitwise `gemm` at 1, 2, 3 and 6 workers across every fringe, one pool reused for hundreds of calls, more workers than blocks and than threads |
| `oracle/` | Accelerate and OpenBLAS declared behind one trait for both widths, and the tests comparing every routine with both on generated data at every stride sign, transpose and shape |
| `bench/` | GB/s of every contiguous Level 1 routine, GFLOPS of `gemv`, `ger`, `symv` and `syr2`, and with `gemm`, `gemm_par`, `trsm` or `l3sym` (`syrk` and `trmm`) as its argument those routines' GFLOPS, against Accelerate and OpenBLAS; figures in [bench/README.md](bench/README.md) |

## Speed

Every Level 1 routine runs a contiguous kernel over 128-bit registers when its views are contiguous:
`asum`, `nrm2` and `iamax` take each lane's magnitude, `nrm2` choosing each lane's Blue band by mask.
`gemv` runs four columns per pass of `y` (or four rows' dot products per pass of `x`, for a
transpose), and `ger` column axpys; both take the vector a 16 KB panel at a time, so it stays in L1.
Each kernel checks its slices once and then loads and stores `@unchecked`, so no loop carries a
bounds check. Single-threaded on an M5 Pro, every Level 1 and 2 routine is at OpenBLAS parity or
better except `iamax` (76–106%), `copy` beyond L1 (75–85%), `ger` in f64 (91–94%), f32 `asum` in L2 (89%), f32 `gemv` at 48 (83%) and `symv` in L1 and L2 (63–102%; level in main memory);
`nrm2` is 2.3x
OpenBLAS at f64 and 4.3–4.7x at f32. Figures in [bench/README.md](bench/README.md).

`gemm` is BLIS's five loops around an 8x6 (f64) or 16x6 (f32) micro-kernel of `fmla` by element,
with both panels packed by vectors; single-threaded on an M5 Pro it runs at 103–124% of OpenBLAS's
NEON gemm from n = 128 at f64 and 96–108% at f32. `trsm` recurses onto gemm and solves its 16-row
leaves in registers: 103–246% of OpenBLAS at every size. `syrk` is gemm's loops skipping the tiles
outside its triangle (102–224% of OpenBLAS), and `trmm` recurses onto gemm as trsm does, level with
OpenBLAS from n = 512. `gemm_par` on a standing pool runs at
84–131% of OpenBLAS on the same thread count at n = 256 and 4–6 threads, where `Spawned` gives
71–97% because starting threads per call is a large share of the time; from n = 1024 the two are
level (85–108% at 2–6 threads), f64 at four threads trailing OpenBLAS by up to a sixth with either.

`sysl test .` runs the package, the pool member and the oracle; under ASan,
`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`.
