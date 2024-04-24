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
      __m256d result = _mm256_setzero_pd(); 
      for (int j = 0; j < n; j += 4) { 
         __m256d A = _mm256_loadu_pd(&A[i * n + j]); 
         __m256d x = _mm256_loadu_pd(&x[j]); 
         result = _mm256_fmadd_pd(A, x, result); 
      }
      y[i] += result[0] + result[1] + result[2] + result[3];
   }
}
