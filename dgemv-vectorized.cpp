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
  __m256d y_vec, a_vec, x_vec;
  for (int i = 0; i < n; i++) {
      y_vec = _mm256_loadu_pd(&y[i]); // Load y[i] into y_vec
      for (int j = 0; j < n; j += 4) {
         a_vec = _mm256_loadu_pd(&A[i * n + j]); // Load four consecutive elements of A into a_vec
         x_vec = _mm256_loadu_pd(&x[j]); // Load four consecutive elements of x into x_vec
         y_vec = _mm256_fmadd_pd(a_vec, x_vec, y_vec); // Multiply-add operation: y_vec = a_vec * x_vec + y_vec
      }
      _mm256_storeu_pd(&y[i], y_vec); // Store the result back to y
   }
}
