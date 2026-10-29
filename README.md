# blas

A BLAS written in sysl, laid out the way BLIS lays one out: the blocking loops and packing written
once, generically, and one small micro-kernel per architecture. Module `sh.sysl.blas`; no C, no
allocation, and it builds for bare-metal targets.

Level 1 is here, generic over `f32` and `f64`: `axpy`, `dot`, `nrm2` (Blue's overflow- and
underflow-safe algorithm), `asum`, `scal`, `copy`, `swap`, `iamax`, `rot`, `rotg`. Levels 2 and 3 are
designed in [DESIGN.md](DESIGN.md).

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
dependencies { blas = "github.com/sysl-lang/blas" }
```

## Layout

| directory | what |
|---|---|
| `sh/sysl/blas/` | the package, and its tests against the definitions (`nrm2`'s range, `rotg`'s signs, `iamax`'s ties) |
| `oracle/` | the platform's CBLAS declared behind one trait for both widths, and the tests comparing every routine with it on generated data at every stride sign |
| `bench/` | GFLOPS against the platform's CBLAS; figures in [bench/README.md](bench/README.md) |

`sysl test .` runs the package and the oracle; under ASan,
`SYSL_EXTRA_CFLAGS="-fsanitize=address -g" sysl test .`.
