package matmul

// MatMul computes C = A × B using SIMD acceleration.
// A is M×K, B is K×N, C is M×N
// All matrices are in row-major order (contiguous rows)
func MatMul(a, b, c []float64, m, k, n int) {
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		panic("matmul: insufficient buffer size")
	}

	matmulAVX2(a, b, c, m, k, n)
}

// MatMulNaive is the baseline implementation for comparison
func MatMulNaive(a, b, c []float64, m, k, n int) {
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		panic("matmul: insufficient buffer size")
	}

	for i := 0; i < m*n; i++ {
		c[i] = 0
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// MatMulBlocked uses cache-friendly blocking
func MatMulBlocked(a, b, c []float64, m, k, n, blockSize int) {
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		panic("matmul: insufficient buffer size")
	}

	for i := 0; i < m*n; i++ {
		c[i] = 0
	}

	for i0 := 0; i0 < m; i0 += blockSize {
		for j0 := 0; j0 < n; j0 += blockSize {
			for k0 := 0; k0 < k; k0 += blockSize {
				iMax := min(i0+blockSize, m)
				jMax := min(j0+blockSize, n)
				kMax := min(k0+blockSize, k)

				for i := i0; i < iMax; i++ {
					for kk := k0; kk < kMax; kk++ {
						aik := a[i*k+kk]
						bRowStart := kk * n
						cRowStart := i * n
						for j := j0; j < jMax; j++ {
							c[cRowStart+j] += aik * b[bRowStart+j]
						}
					}
				}
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
