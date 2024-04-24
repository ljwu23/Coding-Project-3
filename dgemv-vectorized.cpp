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
      __m256d ymm_result = _mm256_setzero_pd(); 
      for (int j = 0; j < n; j += 4) { 
         __m256d ymm_A = _mm256_loadu_pd(&A[i * n + j]); 
         __m256d ymm_x = _mm256_loadu_pd(&x[j]); 
         ymm_result = _mm256_fmadd_pd(ymm_A, ymm_x, ymm_result); 
      }
      y[i] += ymm_result[0] + ymm_result[1] + ymm_result[2] + ymm_result[3];
   }
}
