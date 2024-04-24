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
      __m256d ymm_result = _mm256_setzero_pd(); // Initialize result register to zero
      for (int j = 0; j < n; j += 4) { // Process 4 elements at a time
         __m256d ymm_A = _mm256_loadu_pd(&A[i * n + j]); // Load 4 elements from A
         __m256d ymm_x = _mm256_loadu_pd(&x[j]); // Load 4 elements from x
         ymm_result = _mm256_add_pd(ymm_result, _mm256_mul_pd(ymm_A, ymm_x)); // Multiply and add
      }
      // Horizontal sum of the 4 elements in ymm_result
      double result[4];
      _mm256_storeu_pd(result, ymm_result);
      y[i] += result[0] + result[1] + result[2] + result[3];
   }
}
