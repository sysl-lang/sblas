# blas

A BLAS written in sysl, laid out the way BLIS lays one out: the blocking loops and packing written
once, generically, and one small micro-kernel per architecture. Module `sh.sysl.blas`; no C, no
allocation, and it builds for bare-metal targets.

Level 1 is here, generic over `f32` and `f64`: `axpy`, `dot`, `nrm2` (Blue's overflow- and
underflow-safe algorithm), `asum`, `scal`, `copy`, `swap`, `iamax`, `rot`, `rotg`. Level 2 has begun: `gemv` (and, through `a.t()`, its transpose) and `ger`. The rest
of Level 2 and Level 3 are designed in [DESIGN.md](DESIGN.md).

```sysl
import sh.sysl.blas.*

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

```
dependencies {
  blas { git = "github.com/sysl-lang/blas", version = "0.1.0" }
}
```

Needs sysl 0.8.0.

## Layout

| directory | what |
|---|---|
| `sh/sysl/blas/` | the package — `blas.sysl` (element types, vector and matrix views), `level1.sysl`, `level2.sysl` — and its tests against the definitions (`nrm2`'s range, `rotg`'s signs, `iamax`'s ties, `gemv`'s and `ger`'s scalings and strides) |
| `sh/sysl/blas/netlib_tests.sysl` | netlib's own Level 1 test tables (`dblat1.f`, `sblat1.f`) ported: the fixed vectors and expected results at every stride in them, with netlib's `SFAC`-scaled comparison |
| `oracle/` | Accelerate and OpenBLAS declared behind one trait for both widths, and the tests comparing every routine with both on generated data at every stride sign, transpose and shape |
| `bench/` | GB/s of every contiguous Level 1 routine and GFLOPS of `gemv` and `ger` against Accelerate and OpenBLAS; figures in [bench/README.md](bench/README.md) |

## Speed

`dot`, `axpy`, `scal`, `copy`, `swap` and `rot` run a contiguous kernel over eight 128-bit registers
per step. `asum`, `nrm2` and `iamax` are scalar until sysl 0.8.1 gives a vector a lane-wise `abs`;
every vector load is still bounds-checked as a run, since sysl's `@unchecked` does not reach one.
`gemv` runs four columns per pass of `y` (or four rows' dot products per pass of `x`, for a
transpose), and `ger` column axpys; both take the vector a 16 KB panel at a time, so it stays in L1.
Both are written up as sysl findings in [DESIGN.md](DESIGN.md#gaps-found-so-far).

`sysl test .` runs the package and the oracle; under ASan,
`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`.
