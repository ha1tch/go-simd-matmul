// SIMD-accelerated matrix multiplication using AVX2
// Computes C = A × B where:
//   A is M×K (row-major)
//   B is K×N (row-major)
//   C is M×N (row-major)

#include "textflag.h"

// func matmulAVX2(a, b, c []float64, m, k, n int)
TEXT ·matmulAVX2(SB), NOSPLIT, $0-72
	MOVQ a_base+0(FP), SI    // SI = &A[0]
	MOVQ b_base+24(FP), DI   // DI = &B[0]
	MOVQ c_base+48(FP), BX   // BX = &C[0]
	MOVQ m+72(FP), R8        // R8 = m
	MOVQ k+80(FP), R9        // R9 = k
	MOVQ n+88(FP), R10       // R10 = n
	
	// Zero out C matrix
	MOVQ BX, R11             // R11 = &C[0]
	MOVQ R8, R12
	IMULQ R10, R12           // R12 = m * n (total elements)
	XORQ R13, R13
	
zero_loop:
	CMPQ R13, R12
	JGE zero_done
	VXORPD Y0, Y0, Y0
	
	// Zero 4 elements at a time if possible
	MOVQ R12, R14
	SUBQ R13, R14            // R14 = remaining elements
	CMPQ R14, $4
	JL zero_scalar
	
	VMOVUPD Y0, (R11)(R13*8)
	ADDQ $4, R13
	JMP zero_loop
	
zero_scalar:
	VXORPD X1, X1, X1
	VMOVSD X1, (R11)(R13*8)
	INCQ R13
	JMP zero_loop
	
zero_done:
	// Main computation: C[i,j] = sum_k A[i,k] * B[k,j]
	XORQ R11, R11            // R11 = i (row of A and C)
	
row_loop:
	CMPQ R11, R8             // if i >= m, done
	JGE done
	
	// Calculate A row pointer: &A[i * k]
	MOVQ R11, R12
	IMULQ R9, R12
	LEAQ (SI)(R12*8), R13    // R13 = &A[i,0]
	
	// Calculate C row pointer: &C[i * n]
	MOVQ R11, R12
	IMULQ R10, R12
	LEAQ (BX)(R12*8), R14    // R14 = &C[i,0]
	
	XORQ R15, R15            // R15 = j (column index)
	
col_loop:
	MOVQ R10, AX
	SUBQ R15, AX             // AX = n - j (remaining columns)
	CMPQ AX, $4
	JL col_scalar
	
	// Process 4 columns at once with SIMD
	VXORPD Y0, Y0, Y0        // Y0 = accumulator for C[i,j:j+4]
	
	XORQ CX, CX              // CX = kk (inner loop)
	
inner_simd:
	CMPQ CX, R9              // if kk >= k, done
	JGE inner_simd_done
	
	// Load A[i, kk] and broadcast
	MOVQ CX, DX
	VBROADCASTSD (R13)(DX*8), Y1  // Y1 = [A[i,kk], A[i,kk], A[i,kk], A[i,kk]]
	
	// Calculate &B[kk, j]
	MOVQ CX, DX
	IMULQ R10, DX            // DX = kk * n
	ADDQ R15, DX             // DX = kk * n + j
	
	// Load B[kk, j:j+4]
	VMOVUPD (DI)(DX*8), Y2   // Y2 = [B[kk,j], B[kk,j+1], B[kk,j+2], B[kk,j+3]]
	
	// Multiply and accumulate
	VFMADD231PD Y1, Y2, Y0   // Y0 += Y1 * Y2 (FMA instruction)
	
	INCQ CX
	JMP inner_simd
	
inner_simd_done:
	// Store result to C[i, j:j+4]
	MOVQ R15, DX
	VMOVUPD Y0, (R14)(DX*8)
	
	ADDQ $4, R15
	JMP col_loop
	
col_scalar:
	// Process remaining columns one at a time
	CMPQ R15, R10
	JGE row_done
	
	VXORPD X0, X0, X0        // X0 = accumulator for C[i,j]
	XORQ CX, CX              // CX = kk
	
inner_scalar:
	CMPQ CX, R9
	JGE inner_scalar_done
	
	// Load A[i, kk]
	MOVQ CX, DX
	VMOVSD (R13)(DX*8), X1
	
	// Calculate &B[kk, j]
	MOVQ CX, DX
	IMULQ R10, DX
	ADDQ R15, DX
	
	// Load B[kk, j]
	VMOVSD (DI)(DX*8), X2
	
	// Multiply and accumulate
	VFMADD231SD X1, X2, X0   // X0 += X1 * X2
	
	INCQ CX
	JMP inner_scalar
	
inner_scalar_done:
	// Store C[i, j]
	MOVQ R15, DX
	VMOVSD X0, (R14)(DX*8)
	
	INCQ R15
	JMP col_scalar
	
row_done:
	INCQ R11
	JMP row_loop
	
done:
	VZEROUPPER
	RET
