# blas

A BLAS written in sysl, laid out the way BLIS lays one out: the blocking loops and packing written
once, generically, and one small micro-kernel per architecture. Module `sh.sysl.blas`; no C, no
allocation, and it builds for bare-metal targets.

Level 1 is here, generic over `f32` and `f64`: `axpy`, `dot`, `nrm2` (Blue's overflow- and
underflow-safe algorithm), `asum`, `scal`, `copy`, `swap`, `iamax`, `rot`, `rotg`.

Level 3 has `gemm` over matrix views: `c := alpha * a * b + beta * c`, where a transpose is `a.t()`
(the same storage, strides swapped) rather than a flag. The packing buffers are the caller's, so the
package still allocates nothing; `gemm_work_len` says how long they must be, and
`sh.sysl.blas.heap.gemm_alloc` is the form that allocates them per call.

```sysl
import sh.sysl.blas.*

val a: [6]f64 = [1, 4, 2, 5, 3, 6]               // 2 x 3, column-major
var c: [4]f64
var pa: [64]f64
var pb: [64]f64
gemm(1.0, col_major(a[..], 2, 3, 2), col_major(a[..], 2, 3, 2).t(), 0.0, col_major_mut(c[..], 2, 2, 2), GemmWork(pa[..], pb[..]))
// c = A * A^T = [14, 32, 32, 77]
```

The rest of Level 2 and 3 is designed in [DESIGN.md](DESIGN.md).

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

```
dependencies {
  blas { git = "github.com/sysl-lang/blas", version = "0.1.0" }
}
```

Needs sysl 0.8.0.

## Layout

| directory | what |
|---|---|
| `sh/sysl/blas/` | the package, and its tests against the definitions (`nrm2`'s range, `rotg`'s signs, `iamax`'s ties, `gemm` at every blocking boundary) |
| `sh/sysl/blas/heap/` | `gemm_alloc`: gemm with its packing buffers allocated per call |
| `oracle/` | Accelerate and OpenBLAS declared behind one trait for both widths, and the tests comparing every routine with both on generated data at every stride sign |
| `bench/` | GB/s of every contiguous routine, and gemm's GFLOPS (`sysl run . -- gemm`), against Accelerate and OpenBLAS; figures in [bench/README.md](bench/README.md) |

## Speed

`dot`, `axpy`, `scal`, `copy`, `swap` and `rot` run a contiguous kernel over eight 128-bit registers
per step. `asum`, `nrm2` and `iamax` are scalar until sysl 0.8.1 gives a vector a lane-wise `abs`;
every vector load is still bounds-checked as a run, since sysl's `@unchecked` does not reach one.
Both are written up as sysl findings in [DESIGN.md](DESIGN.md#gaps-found-so-far).

`sysl test .` runs the package and the oracle; under ASan,
`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`.
