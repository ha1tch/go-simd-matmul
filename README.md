# go-simd-matmul

High-performance SIMD matrix multiplication for Go using AVX2/FMA instructions.

## Performance

- 7-9x faster than naive Go implementation
- Zero allocations
- Works with any matrix dimensions
- < 1e-13 floating-point accuracy

## Benchmarks

| Size | Naive | SIMD | Speedup |
|------|-------|------|---------|
| 32×32 | 58.4 µs | 6.5 µs | 8.96x |
| 64×64 | 482.8 µs | 56.8 µs | 8.50x |
| 128×128 | 4.1 ms | 551 µs | 7.42x |
| 256×256 | 34.3 ms | 5.0 ms | 6.90x |
| 512×512 | 309.8 ms | 65.6 ms | 4.72x |

### Transformer Attention Sizes

| Configuration | Naive | SIMD | Speedup |
|--------------|-------|------|---------|
| BERT 128×64 | 2.0 ms | 244 µs | 8.17x |
| BERT 256×64 | 8.0 ms | 952 µs | 8.45x |
| GPT 512×64 | 32.1 ms | 4.0 ms | 8.08x |
| GPT 1024×64 | 130.9 ms | 17.9 ms | 7.32x |

## Installation

```bash
go get github.com/ha1tch/go-simd-matmul
```

## Requirements

- Go 1.21 or later
- x86-64 CPU with AVX2/FMA support
  - Intel: Haswell (2013) or newer
  - AMD: Excavator (2015) or newer

Check CPU support:
```bash
grep -E 'avx2|fma' /proc/cpuinfo
```

## Usage

```go
package main

import "github.com/ha1tch/go-simd-matmul"

func main() {
    // Compute C = A × B
    m, k, n := 128, 64, 128
    a := make([]float64, m*k)  // Row-major
    b := make([]float64, k*n)  // Row-major
    c := make([]float64, m*n)  // Output

    // Fill a and b with data...

    matmul.MatMul(a, b, c, m, k, n)
}
```

## API

```go
// MatMul computes C = A × B using SIMD acceleration
// A is M×K, B is K×N, C is M×N
// All matrices are in row-major order
func MatMul(a, b, c []float64, m, k, n int)

// MatMulNaive is the baseline implementation for comparison
func MatMulNaive(a, b, c []float64, m, k, n int)

// MatMulBlocked uses cache-friendly blocking
func MatMulBlocked(a, b, c []float64, m, k, n, blockSize int)
```

## When to Use

✓ Good fit:
- Pure Go requirement (no CGO)
- Transformer inference workloads
- 7-9x speedup is sufficient
- Single-threaded workloads

✗ Consider alternatives:
- Need maximum performance (use OpenBLAS)
- Need multi-threading (use gonum/mat with BLAS)
- Many matrix operations (use full BLAS library)

## Benchmarks

Run benchmarks:
```bash
go test -bench=. -benchtime=3s
```

## Tests

Run tests:
```bash
go test -v
```

## Companion Package

For transformer inference, pair with:
```go
import (
    "github.com/ha1tch/go-simd-matmul"
    "github.com/ha1tch/go-simd-softmax"
)
```

## License
Apache 2.0

## Author
haitch <h@ual.fi>

https://oldbytes.space/@haitchfive

