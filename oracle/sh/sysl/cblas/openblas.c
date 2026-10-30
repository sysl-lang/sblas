/* OpenBLAS, reached at run time rather than at the link.
 *
 * OpenBLAS and Accelerate export the same names (`cblas_ddot`, `ddot_`, ...), and a Mach-O link binds
 * each undefined name to ONE library, so a binary that links both gets one of them twice and never
 * knows. Loading OpenBLAS with `dlopen(RTLD_LOCAL)` and asking that handle for each entry point keeps
 * the two apart: every `sysl_ob_*` below calls OpenBLAS's own routine, and the `cblas_*` names the
 * sysl side declares directly stay Accelerate's.
 *
 * The library is `$SYSL_OPENBLAS` if that is set, otherwise Homebrew's. It is pinned to one thread on
 * load, since the comparison is single-threaded. A missing library is a trap at the first call, with
 * the path on stderr: an oracle that silently answered nothing would pass every check.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void *handle;

static void *entry(const char *name) {
    if (handle == NULL) {
        const char *lib = getenv("SYSL_OPENBLAS");

        if (lib == NULL) lib = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib";
        handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (handle == NULL) {
            fprintf(stderr, "cblas-oracle: cannot load OpenBLAS from %s: %s\n", lib, dlerror());
            abort();
        }

        void (*threads)(int) = (void (*)(int))dlsym(handle, "openblas_set_num_threads");

        if (threads != NULL) threads(1);
    }

    void *f = dlsym(handle, name);

    if (f == NULL) {
        fprintf(stderr, "cblas-oracle: OpenBLAS has no %s\n", name);
        abort();
    }

    return f;
}

/* One lazily resolved pointer per entry point, so a call costs an indirect branch and nothing more. */
#define RESOLVE(ty, name) static ty fn; if (fn == NULL) fn = (ty)entry(name)

/* Sets how many threads OpenBLAS's own routines use from here on: the threaded comparison. */
typedef void (*set_threads)(int);

void sysl_ob_set_threads(int n) {
    RESOLVE(set_threads, "openblas_set_num_threads");
    fn(n);
}

typedef double (*dot_d)(int, const double *, int, const double *, int);
typedef float (*dot_s)(int, const float *, int, const float *, int);
typedef void (*axpy_d)(int, double, const double *, int, double *, int);
typedef void (*axpy_s)(int, float, const float *, int, float *, int);
typedef double (*red_d)(int, const double *, int);
typedef float (*red_s)(int, const float *, int);
typedef void (*scal_d)(int, double, double *, int);
typedef void (*scal_s)(int, float, float *, int);
typedef void (*pair_d)(int, double *, int, double *, int);
typedef void (*pair_s)(int, float *, int, float *, int);
typedef void (*rot_d)(int, double *, int, double *, int, double, double);
typedef void (*rot_s)(int, float *, int, float *, int, float, float);
typedef void (*rotg_d)(double *, double *, double *, double *);
typedef void (*rotg_s)(float *, float *, float *, float *);
typedef size_t (*imax_d)(int, const double *, int);
typedef size_t (*imax_s)(int, const float *, int);

double sysl_ob_ddot(int n, const double *x, int incx, const double *y, int incy) {
    RESOLVE(dot_d, "cblas_ddot");
    return fn(n, x, incx, y, incy);
}

float sysl_ob_sdot(int n, const float *x, int incx, const float *y, int incy) {
    RESOLVE(dot_s, "cblas_sdot");
    return fn(n, x, incx, y, incy);
}

void sysl_ob_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy) {
    RESOLVE(axpy_d, "cblas_daxpy");
    fn(n, alpha, x, incx, y, incy);
}

void sysl_ob_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    RESOLVE(axpy_s, "cblas_saxpy");
    fn(n, alpha, x, incx, y, incy);
}

double sysl_ob_dnrm2(int n, const double *x, int incx) {
    RESOLVE(red_d, "cblas_dnrm2");
    return fn(n, x, incx);
}

float sysl_ob_snrm2(int n, const float *x, int incx) {
    RESOLVE(red_s, "cblas_snrm2");
    return fn(n, x, incx);
}

double sysl_ob_dasum(int n, const double *x, int incx) {
    RESOLVE(red_d, "cblas_dasum");
    return fn(n, x, incx);
}

float sysl_ob_sasum(int n, const float *x, int incx) {
    RESOLVE(red_s, "cblas_sasum");
    return fn(n, x, incx);
}

void sysl_ob_dscal(int n, double alpha, double *x, int incx) {
    RESOLVE(scal_d, "cblas_dscal");
    fn(n, alpha, x, incx);
}

void sysl_ob_sscal(int n, float alpha, float *x, int incx) {
    RESOLVE(scal_s, "cblas_sscal");
    fn(n, alpha, x, incx);
}

void sysl_ob_dcopy(int n, double *x, int incx, double *y, int incy) {
    RESOLVE(pair_d, "cblas_dcopy");
    fn(n, x, incx, y, incy);
}

void sysl_ob_scopy(int n, float *x, int incx, float *y, int incy) {
    RESOLVE(pair_s, "cblas_scopy");
    fn(n, x, incx, y, incy);
}

void sysl_ob_dswap(int n, double *x, int incx, double *y, int incy) {
    RESOLVE(pair_d, "cblas_dswap");
    fn(n, x, incx, y, incy);
}

void sysl_ob_sswap(int n, float *x, int incx, float *y, int incy) {
    RESOLVE(pair_s, "cblas_sswap");
    fn(n, x, incx, y, incy);
}

void sysl_ob_drot(int n, double *x, int incx, double *y, int incy, double c, double s) {
    RESOLVE(rot_d, "cblas_drot");
    fn(n, x, incx, y, incy, c, s);
}

void sysl_ob_srot(int n, float *x, int incx, float *y, int incy, float c, float s) {
    RESOLVE(rot_s, "cblas_srot");
    fn(n, x, incx, y, incy, c, s);
}

void sysl_ob_drotg(double *a, double *b, double *c, double *s) {
    RESOLVE(rotg_d, "cblas_drotg");
    fn(a, b, c, s);
}

void sysl_ob_srotg(float *a, float *b, float *c, float *s) {
    RESOLVE(rotg_s, "cblas_srotg");
    fn(a, b, c, s);
}

/* Level 2. CBLAS's `order` and `trans` are enums, passed as the ints they are. */
typedef void (*gemv_d)(int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef void (*gemv_s)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
typedef void (*ger_d)(int, int, int, double, const double *, int, const double *, int, double *, int);
typedef void (*ger_s)(int, int, int, float, const float *, int, const float *, int, float *, int);

void sysl_ob_dgemv(int order, int trans, int m, int n, double alpha, const double *a, int lda, const double *x, int incx,
                   double beta, double *y, int incy) {
    RESOLVE(gemv_d, "cblas_dgemv");
    fn(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void sysl_ob_sgemv(int order, int trans, int m, int n, float alpha, const float *a, int lda, const float *x, int incx,
                   float beta, float *y, int incy) {
    RESOLVE(gemv_s, "cblas_sgemv");
    fn(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void sysl_ob_dger(int order, int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a,
                  int lda) {
    RESOLVE(ger_d, "cblas_dger");
    fn(order, m, n, alpha, x, incx, y, incy, a, lda);
}

void sysl_ob_sger(int order, int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a,
                  int lda) {
    RESOLVE(ger_s, "cblas_sger");
    fn(order, m, n, alpha, x, incx, y, incy, a, lda);
}

/* 1 when the `cblas_ddot` this file reaches lives in a different image from the one the link bound,
 * which is the whole point of the arrangement and the one thing a mistake in it would quietly undo.
 */
double cblas_ddot(int n, const double *x, int incx, const double *y, int incy);

int sysl_ob_distinct(void) {
    Dl_info ours, linked;

    if (!dladdr(entry("cblas_ddot"), &ours) || !dladdr((void *)cblas_ddot, &linked)) return 0;
    return ours.dli_fbase != linked.dli_fbase;
}

/* OpenBLAS's `CBLAS_INDEX` is a `size_t`, where Accelerate's `cblas_i?amax` answers an `int`. */
int sysl_ob_idamax(int n, const double *x, int incx) {
    RESOLVE(imax_d, "cblas_idamax");
    return (int)fn(n, x, incx);
}

int sysl_ob_isamax(int n, const float *x, int incx) {
    RESOLVE(imax_s, "cblas_isamax");
    return (int)fn(n, x, incx);
}

/* CBLAS's order and transpose enums are `int`-sized, passed here as their values (102, 111, 112). */
typedef void (*gemm_d)(int, int, int, int, int, int, double, const double *, int, const double *, int,
                       double, double *, int);
typedef void (*gemm_s)(int, int, int, int, int, int, float, const float *, int, const float *, int,
                       float, float *, int);

void sysl_ob_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha,
                   const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    RESOLVE(gemm_d, "cblas_dgemm");
    fn(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sysl_ob_sgemm(int order, int transa, int transb, int m, int n, int k, float alpha,
                   const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    RESOLVE(gemm_s, "cblas_sgemm");
    fn(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/* Side, uplo and diag are CBLAS enums too: 141/142, 121/122, 131/132. */
typedef void (*trsm_d)(int, int, int, int, int, int, int, double, const double *, int, double *, int);
typedef void (*trsm_s)(int, int, int, int, int, int, int, float, const float *, int, float *, int);

void sysl_ob_dtrsm(int order, int side, int uplo, int trans, int diag, int m, int n, double alpha,
                   const double *a, int lda, double *b, int ldb) {
    RESOLVE(trsm_d, "cblas_dtrsm");
    fn(order, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

void sysl_ob_strsm(int order, int side, int uplo, int trans, int diag, int m, int n, float alpha,
                   const float *a, int lda, float *b, int ldb) {
    RESOLVE(trsm_s, "cblas_strsm");
    fn(order, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}
