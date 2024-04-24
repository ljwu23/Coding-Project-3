#include <immintrin.h>
const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   // insert your code here: implementation of vectorized vector-matrix multiply
 for (int i = 0; i < n; i++) {
        __m512d sum = _mm512_setzero_pd(); // Initialize sum to zero using SIMD intrinsic
        for (int j = 0; j < n; j += 8) { // Process 8 elements at a time for better vectorization
            __m512d a_vec = _mm512_loadu_pd(&A[i * n + j]); // Load 8 elements of A into a SIMD register
            __m512d x_vec = _mm512_loadu_pd(&x[j]); // Load 8 elements of x into a SIMD register
            sum = _mm512_fmadd_pd(a_vec, x_vec, sum); // Multiply and accumulate using SIMD intrinsics
        }
        // Horizontal sum of the elements in sum vector
        sum = _mm512_add_pd(sum, _mm512_permutex_pd(sum, 0b00011011));
        sum = _mm512_add_pd(sum, _mm512_permutex_pd(sum, 0b00001110));
        // Extract the result and store it in y
        double result[8];
        _mm512_storeu_pd(result, sum);
        y[i] += result[0] + result[1] + result[2] + result[3];
    }
}
