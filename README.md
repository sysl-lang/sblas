# sblas

A BLAS written in sysl, laid out the way BLIS lays one out: the blocking loops and packing written
once, generically, and one small micro-kernel per architecture. Module `sh.sysl.sblas`; no C, no
allocation, and it builds for bare-metal targets.

Level 1 is here, generic over `f32` and `f64`: `axpy`, `dot`, `nrm2` (Blue's overflow- and
underflow-safe algorithm), `asum`, `scal`, `copy`, `swap`, `iamax`, `rot`, `rotg`. Level 2 has begun: `gemv` (and, through `a.t()`, its transpose) and `ger`.
Level 3 has `gemm` and `trsm`. The rest of Level 2 and Level 3 are designed in [DESIGN.md](DESIGN.md).

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

`trsm` solves a triangular system in place for every column of B at once — `b := alpha a⁻¹ b` on the
`Left`, `b := alpha b a⁻¹` on the `Right` — with the same packing buffers (`trsm_work_len`). `Uplo`
names the triangle of the view as passed, so the transpose of an upper triangle is `Lower`:

```sysl
val l: [4]f64 = [2, 1, 0, 4]                                // [[2, 0], [1, 4]]
var rhs: [2]f64 = [2, 9]
trsm(Side.Left, Uplo.Lower, Diag.NonUnit, 1.0, matrix(l[..], 2, 2, 2), matrix_mut(rhs[..], 2, 1, 2), GemmWork(pa[..], pb[..]))  // rhs = [1, 2]
trsm(Side.Left, Uplo.Upper, Diag.NonUnit, 1.0, matrix(l[..], 2, 2, 2).t(), matrix_mut(rhs[..], 2, 1, 2), GemmWork(pa[..], pb[..]))  // lᵀ x = rhs
```

```
dependencies {
  sblas { git = "github.com/sysl-lang/sblas", version = "0.3.0" }
}
```

Needs sysl 0.8.2.

## Layout

| directory | what |
|---|---|
| `sh/sysl/sblas/` | the package — `sblas.sysl` (element types, vector and matrix views), `level1.sysl`, `level2.sysl`, `level3.sysl` — and its tests against the definitions (`nrm2`'s range and band edges, `rotg`'s signs, `iamax`'s ties and NaNs across lanes and chunks, every length around the vector width, `gemv`'s and `ger`'s scalings and strides, `gemm` at every blocking boundary, `trsm` solved by hand and multiplied back across them) |
| `sh/sysl/sblas/heap/` | `gemm_alloc`, `trsm_alloc`: gemm and trsm with their packing buffers allocated per call (the main module is `@no_alloc`) |
| `sh/sysl/sblas/netlib_tests.sysl` | netlib's own Level 1 test tables (`dblat1.f`, `sblat1.f`) ported: the fixed vectors and expected results at every stride in them, with netlib's `SFAC`-scaled comparison |
| `oracle/` | Accelerate and OpenBLAS declared behind one trait for both widths, and the tests comparing every routine with both on generated data at every stride sign, transpose and shape |
| `bench/` | GB/s of every contiguous Level 1 routine, GFLOPS of `gemv` and `ger`, and with `gemm` or `trsm` as its argument that routine's GFLOPS, against Accelerate and OpenBLAS; figures in [bench/README.md](bench/README.md) |

## Speed

Every Level 1 routine runs a contiguous kernel over 128-bit registers when its views are contiguous:
`asum`, `nrm2` and `iamax` take each lane's magnitude, `nrm2` choosing each lane's Blue band by mask.
`gemv` runs four columns per pass of `y` (or four rows' dot products per pass of `x`, for a
transpose), and `ger` column axpys; both take the vector a 16 KB panel at a time, so it stays in L1.
Each kernel checks its slices once and then loads and stores `@unchecked`, so no loop carries a
bounds check. Single-threaded on an M5 Pro, every Level 1 and 2 routine is at OpenBLAS parity or
better except `iamax` (76–106%), `copy` beyond L1 (75–85%), `ger` in f64 (91–94%), f32 `asum` in L2 (89%) and f32 `gemv` at 48 (83%);
`nrm2` is 2.3x
OpenBLAS at f64 and 4.3–4.7x at f32. Figures in [bench/README.md](bench/README.md).

`gemm` is BLIS's five loops around an 8x6 (f64) or 16x6 (f32) micro-kernel of `fmla` by element,
with both panels packed by vectors; single-threaded on an M5 Pro it runs at 103–124% of OpenBLAS's
NEON gemm from n = 128 at f64 and 96–108% at f32. `trsm` recurses onto gemm and solves its 16-row
leaves in registers: 103–246% of OpenBLAS at every size.

`sysl test .` runs the package and the oracle; under ASan,
`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`.
