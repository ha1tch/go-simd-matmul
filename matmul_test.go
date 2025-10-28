package matmul

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestMatMulCorrectness(t *testing.T) {
	rand.Seed(42)

	tests := []struct {
		m, k, n int
	}{
		{4, 4, 4},
		{8, 8, 8},
		{10, 10, 10},
		{16, 16, 16},
		{32, 32, 32},
		{7, 11, 13},
		{64, 64, 64},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("%dx%dx%d", tt.m, tt.k, tt.n), func(t *testing.T) {
			a := make([]float64, tt.m*tt.k)
			b := make([]float64, tt.k*tt.n)
			cNaive := make([]float64, tt.m*tt.n)
			cSIMD := make([]float64, tt.m*tt.n)

			for i := range a {
				a[i] = rand.Float64()*10 - 5
			}
			for i := range b {
				b[i] = rand.Float64()*10 - 5
			}

			MatMulNaive(a, b, cNaive, tt.m, tt.k, tt.n)
			MatMul(a, b, cSIMD, tt.m, tt.k, tt.n)

			maxDiff := 0.0
			for i := 0; i < tt.m*tt.n; i++ {
				diff := math.Abs(cNaive[i] - cSIMD[i])
				if diff > maxDiff {
					maxDiff = diff
				}

				if cNaive[i] != 0 {
					relErr := diff / math.Abs(cNaive[i])
					if relErr > 1e-10 {
						t.Errorf("Element %d: naive=%v, simd=%v, relErr=%e",
							i, cNaive[i], cSIMD[i], relErr)
					}
				} else if diff > 1e-10 {
					t.Errorf("Element %d: naive=%v, simd=%v, diff=%e",
						i, cNaive[i], cSIMD[i], diff)
				}
			}

			t.Logf("Max absolute difference: %e", maxDiff)
		})
	}
}

func TestMatMulIdentity(t *testing.T) {
	n := 8
	a := make([]float64, n*n)
	b := make([]float64, n*n)
	c := make([]float64, n*n)

	rand.Seed(42)
	for i := range a {
		a[i] = rand.Float64() * 10
	}

	for i := 0; i < n; i++ {
		b[i*n+i] = 1.0
	}

	MatMul(a, b, c, n, n, n)

	for i := 0; i < n*n; i++ {
		if math.Abs(a[i]-c[i]) > 1e-10 {
			t.Errorf("Element %d: expected %v, got %v", i, a[i], c[i])
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		a := make([]float64, size*size)
		bMat := make([]float64, size*size)
		c := make([]float64, size*size)

		rand.Seed(42)
		for i := range a {
			a[i] = rand.Float64()
		}
		for i := range bMat {
			bMat[i] = rand.Float64()
		}

		b.Run(fmt.Sprintf("Naive/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatMulNaive(a, bMat, c, size, size, size)
			}
		})

		b.Run(fmt.Sprintf("Blocked32/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatMulBlocked(a, bMat, c, size, size, size, 32)
			}
		})

		b.Run(fmt.Sprintf("SIMD/%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, c, size, size, size)
			}
		})
	}
}

func BenchmarkMatMulAttention(b *testing.B) {
	configs := []struct {
		name    string
		seqLen  int
		headDim int
	}{
		{"BERT-128x64", 128, 64},
		{"BERT-256x64", 256, 64},
		{"GPT-512x64", 512, 64},
		{"GPT-1024x64", 1024, 64},
	}

	for _, cfg := range configs {
		a := make([]float64, cfg.seqLen*cfg.headDim)
		bMat := make([]float64, cfg.headDim*cfg.seqLen)
		c := make([]float64, cfg.seqLen*cfg.seqLen)

		rand.Seed(42)
		for i := range a {
			a[i] = rand.Float64()
		}
		for i := range bMat {
			bMat[i] = rand.Float64()
		}

		b.Run(fmt.Sprintf("Naive/%s", cfg.name), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatMulNaive(a, bMat, c, cfg.seqLen, cfg.headDim, cfg.seqLen)
			}
		})

		b.Run(fmt.Sprintf("SIMD/%s", cfg.name), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatMul(a, bMat, c, cfg.seqLen, cfg.headDim, cfg.seqLen)
			}
		})
	}
}
