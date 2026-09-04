package model

// wyf_recurrence.go implements Woodbury-Yamamoto-Fernandes (WYF) chunkwise
// parallel recurrence for GatedDeltaNet linear attention (e.g. in Qwen3.8 and GLM-5.3).
//
// In sequential token-by-token recurrence, the state update:
//   S_t = diag(alpha_t) * S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) * k_t^T
// requires loading and storing the full Dim x Dim state matrix at every token,
// creating an arithmetic intensity bottleneck (~0.5 flop/byte).
//
// Woodbury-Yamamoto-Fernandes (WYF) chunkwise parallel block processing computes
// intra-chunk token dependencies in chunks of size C (e.g. 16 or 32 tokens) via
// parallel triangular block substitution (A * U = V'). This converts the token loop
// into chunked matrix operations (K^T * K Gram matrix, triangular solve, and block GEMMs),
// raising arithmetic intensity and enabling SIMD/accelerator hardware parallelism.

// WYFChunkConfig holds chunked execution parameters for WYF recurrence.
type WYFChunkConfig struct {
	ChunkSize int
	Dim       int
}

// NewWYFChunkConfig returns a validated WYFChunkConfig with sensible defaults.
func NewWYFChunkConfig(chunkSize, dim int) WYFChunkConfig {
	if chunkSize <= 0 {
		chunkSize = 16
	}
	return WYFChunkConfig{
		ChunkSize: chunkSize,
		Dim:       dim,
	}
}

// Recurrence executes WYF chunkwise parallel recurrence with this configuration.
func (cfg WYFChunkConfig) Recurrence(k, v, beta, alpha [][]float32, s0 [][]float32) ([][]float32, [][][]float32) {
	return WYFChunkwiseRecurrence(k, v, beta, alpha, s0, cfg.ChunkSize)
}

func extractAlpha(alpha [][]float32, t, dim int) (float32, []float32, bool) {
	if alpha == nil || t >= len(alpha) || len(alpha[t]) == 0 {
		return 1.0, nil, true
	}
	row := alpha[t]
	if len(row) == 1 {
		return row[0], nil, true
	}
	if len(row) >= dim {
		val := row[0]
		uniform := true
		for i := 1; i < dim; i++ {
			if row[i] != val {
				uniform = false
				break
			}
		}
		if uniform {
			return val, nil, true
		}
		return 0, row[:dim], false
	}
	return row[0], nil, true
}

func extractBeta(beta [][]float32, t, dim int) (float32, []float32, bool) {
	if beta == nil || t >= len(beta) || len(beta[t]) == 0 {
		return 1.0, nil, true
	}
	row := beta[t]
	if len(row) == 1 {
		return row[0], nil, true
	}
	if len(row) >= dim {
		val := row[0]
		uniform := true
		for i := 1; i < dim; i++ {
			if row[i] != val {
				uniform = false
				break
			}
		}
		if uniform {
			return val, nil, true
		}
		return 0, row[:dim], false
	}
	return row[0], nil, true
}

// SequentialGDN computes the reference token-by-token recurrence:
//
//	S_t = diag(alpha_t) * S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) * k_t^T
//
// with token output y_t = S_t * k_t.
// Returns outputs [T][Dim] and states [T][Dim][Dim].
func SequentialGDN(k, v, beta, alpha [][]float32, s0 [][]float32) ([][]float32, [][][]float32) {
	T := len(k)
	if T == 0 {
		return nil, nil
	}
	dim := len(k[0])
	if dim == 0 {
		return nil, nil
	}

	// Initialize S from s0 (or zeros if nil).
	S := make([][]float32, dim)
	for i := 0; i < dim; i++ {
		S[i] = make([]float32, dim)
		if s0 != nil && i < len(s0) {
			copy(S[i], s0[i])
		}
	}

	outputs := make([][]float32, T)
	states := make([][][]float32, T)

	p := make([]float32, dim)
	u := make([]float32, dim)

	for t := 0; t < T; t++ {
		kt := k[t]
		vt := v[t]
		aScalar, aVec, aUniform := extractAlpha(alpha, t, dim)
		bScalar, bVec, bUniform := extractBeta(beta, t, dim)

		// 1. p = S_{t-1} * k_t
		for i := 0; i < dim; i++ {
			var sum float32
			row := S[i]
			for j := 0; j < dim; j++ {
				sum += row[j] * kt[j]
			}
			p[i] = sum
		}

		// 2. Decay state: S_decayed = diag(alpha_t) * S_{t-1}
		if aUniform {
			for i := 0; i < dim; i++ {
				row := S[i]
				for j := 0; j < dim; j++ {
					row[j] *= aScalar
				}
			}
		} else {
			for i := 0; i < dim; i++ {
				ai := aVec[i]
				row := S[i]
				for j := 0; j < dim; j++ {
					row[j] *= ai
				}
			}
		}

		// 3. Error vector u_t = beta_t * (v_t - p)
		if bUniform {
			for i := 0; i < dim; i++ {
				u[i] = bScalar * (vt[i] - p[i])
			}
		} else {
			for i := 0; i < dim; i++ {
				u[i] = bVec[i] * (vt[i] - p[i])
			}
		}

		// 4. Update state: S_t = S_decayed + u_t * k_t^T
		for i := 0; i < dim; i++ {
			ui := u[i]
			row := S[i]
			for j := 0; j < dim; j++ {
				row[j] += ui * kt[j]
			}
		}

		// 5. Output: y_t = S_t * k_t
		yt := make([]float32, dim)
		for i := 0; i < dim; i++ {
			var sum float32
			row := S[i]
			for j := 0; j < dim; j++ {
				sum += row[j] * kt[j]
			}
			yt[i] = sum
		}
		outputs[t] = yt

		// 6. Snapshot state
		stCopy := make([][]float32, dim)
		for i := 0; i < dim; i++ {
			stCopy[i] = make([]float32, dim)
			copy(stCopy[i], S[i])
		}
		states[t] = stCopy
	}

	return outputs, states
}

// WYFChunkwiseRecurrence computes the chunkwise parallel block recurrence using
// intra-chunk triangular block substitution, matching SequentialGDN within Delta < 1e-4.
func WYFChunkwiseRecurrence(k, v, beta, alpha [][]float32, s0 [][]float32, chunkSize int) ([][]float32, [][][]float32) {
	T := len(k)
	if T == 0 {
		return nil, nil
	}
	dim := len(k[0])
	if dim == 0 {
		return nil, nil
	}
	if chunkSize <= 0 {
		chunkSize = 16
	}

	// Working state S (size dim x dim)
	S := make([][]float32, dim)
	for i := 0; i < dim; i++ {
		S[i] = make([]float32, dim)
		if s0 != nil && i < len(s0) {
			copy(S[i], s0[i])
		}
	}

	outputs := make([][]float32, T)
	states := make([][][]float32, T)

	// Pre-allocated chunk workspace buffers to avoid GC pressure.
	p0 := make([][]float32, chunkSize)
	Vprime := make([][]float32, chunkSize)
	U := make([][]float32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		p0[i] = make([]float32, dim)
		Vprime[i] = make([]float32, dim)
		U[i] = make([]float32, dim)
	}
	G := make([][]float32, chunkSize)
	A := make([][]float32, chunkSize)
	AInv := make([][]float32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		G[i] = make([]float32, chunkSize)
		A[i] = make([]float32, chunkSize)
		AInv[i] = make([]float32, chunkSize)
	}
	gamma := make([][]float32, chunkSize+1)
	for i := 0; i <= chunkSize; i++ {
		gamma[i] = make([]float32, chunkSize+1)
	}
	aScalars := make([]float32, chunkSize)
	bScalars := make([]float32, chunkSize)
	aVecs := make([][]float32, chunkSize)
	bVecs := make([][]float32, chunkSize)

	for offset := 0; offset < T; offset += chunkSize {
		cLen := chunkSize
		if offset+cLen > T {
			cLen = T - offset
		}

		allAlphaUniform := true
		allBetaUniform := true

		for t := 0; t < cLen; t++ {
			globalT := offset + t
			as, av, au := extractAlpha(alpha, globalT, dim)
			aScalars[t] = as
			aVecs[t] = av
			if !au {
				allAlphaUniform = false
			}

			bs, bv, bu := extractBeta(beta, globalT, dim)
			bScalars[t] = bs
			bVecs[t] = bv
			if !bu {
				allBetaUniform = false
			}
		}

		kChunk := k[offset : offset+cLen]
		vChunk := v[offset : offset+cLen]

		// Precompute S_0 * k_t for all t in chunk
		for i := 0; i < dim; i++ {
			rowS := S[i]
			for t := 0; t < cLen; t++ {
				p0[t][i] = fdot(rowS, kChunk[t])
			}
		}

		// Gram matrix G[j][t] = k_j^T * k_t
		for j := 0; j < cLen; j++ {
			kj := kChunk[j]
			for t := j; t < cLen; t++ {
				dot := fdot(kj, kChunk[t])
				G[j][t] = dot
				G[t][j] = dot
			}
		}

		if allAlphaUniform && allBetaUniform {
			// Fast path: uniform scalars across dim
			// 1. Compute gamma decay table: gamma[j][t] = prod_{m=j}^{t-1} a_m
			for j := 0; j <= cLen; j++ {
				gamma[j][j] = 1.0
				for t := j + 1; t <= cLen; t++ {
					gamma[j][t] = gamma[j][t-1] * aScalars[t-1]
				}
			}

			// 2. Compute V'[t][d] = beta_t * (v_t[d] - gamma[0][t] * p0[t][d])
			for t := 0; t < cLen; t++ {
				rowV := Vprime[t]
				vt := vChunk[t]
				pt := p0[t]
				bt := bScalars[t]
				g0 := gamma[0][t]
				for d := 0; d < dim; d++ {
					rowV[d] = bt * (vt[d] - g0*pt[d])
				}
			}

			// 3. Construct unit upper-triangular matrix A (cLen x cLen)
			for j := 0; j < cLen; j++ {
				A[j][j] = 1.0
				for t := j + 1; t < cLen; t++ {
					A[j][t] = bScalars[t] * gamma[j+1][t] * G[j][t]
				}
			}

			// 4. Invert unit upper-triangular A -> AInv
			for j := 0; j < cLen; j++ {
				AInv[j][j] = 1.0
				for t := j + 1; t < cLen; t++ {
					var sum float32
					for m := j; m < t; m++ {
						sum += AInv[j][m] * A[m][t]
					}
					AInv[j][t] = -sum
				}
			}

			// 5. U = V' * AInv  (U is [cLen][dim])
			for t := 0; t < cLen; t++ {
				rowU := U[t]
				for d := 0; d < dim; d++ {
					rowU[d] = 0
				}
				for j := 0; j <= t; j++ {
					aij := AInv[j][t]
					vRow := Vprime[j]
					for d := 0; d < dim; d++ {
						rowU[d] += vRow[d] * aij
					}
				}
			}

			// 6. Compute outputs y_t
			// y_t = gamma[0][t+1] * p0[t] + sum_{j=0}^t gamma[j+1][t+1] * G[j][t] * u_j
			for t := 0; t < cLen; t++ {
				globalT := offset + t
				yt := make([]float32, dim)
				g0 := gamma[0][t+1]
				pt := p0[t]
				for d := 0; d < dim; d++ {
					yt[d] = g0 * pt[d]
				}
				for j := 0; j <= t; j++ {
					coeff := gamma[j+1][t+1] * G[j][t]
					uj := U[j]
					for d := 0; d < dim; d++ {
						yt[d] += coeff * uj[d]
					}
				}
				outputs[globalT] = yt
			}

			// 7. Intermediate states within chunk: S_{t+1} = alpha_t * S_t + u_t * k_t^T
			currS := S
			for t := 0; t < cLen; t++ {
				globalT := offset + t
				nextS := make([][]float32, dim)
				backing := make([]float32, dim*dim)
				at := aScalars[t]
				ut := U[t]
				kt := kChunk[t]
				for i := 0; i < dim; i++ {
					row := backing[i*dim : (i+1)*dim]
					currRow := currS[i]
					ui := ut[i]
					for d := 0; d < dim; d++ {
						row[d] = at*currRow[d] + ui*kt[d]
					}
					nextS[i] = row
				}
				states[globalT] = nextS
				currS = nextS
			}
			S = currS

		} else {
			// Per-dimension general path (handles non-uniform alpha/beta across dim)
			U := make([][]float32, cLen)
			for t := 0; t < cLen; t++ {
				U[t] = make([]float32, dim)
			}

			// Solve per-dimension
			for d := 0; d < dim; d++ {
				// 1. gamma table for dimension d
				gamma := make([][]float32, cLen+1)
				for j := 0; j <= cLen; j++ {
					gamma[j] = make([]float32, cLen+1)
					gamma[j][j] = 1.0
					for t := j + 1; t <= cLen; t++ {
						ad := aScalars[t-1]
						if aVecs[t-1] != nil {
							ad = aVecs[t-1][d]
						}
						gamma[j][t] = gamma[j][t-1] * ad
					}
				}

				// 2. Vprime for dimension d
				vPrimeD := make([]float32, cLen)
				for t := 0; t < cLen; t++ {
					bd := bScalars[t]
					if bVecs[t] != nil {
						bd = bVecs[t][d]
					}
					vPrimeD[t] = bd * (vChunk[t][d] - gamma[0][t]*p0[t][d])
				}

				// 3. Matrix A for dimension d
				A := make([][]float32, cLen)
				for j := 0; j < cLen; j++ {
					A[j] = make([]float32, cLen)
					A[j][j] = 1.0
					for t := j + 1; t < cLen; t++ {
						bd := bScalars[t]
						if bVecs[t] != nil {
							bd = bVecs[t][d]
						}
						A[j][t] = bd * gamma[j+1][t] * G[j][t]
					}
				}

				// 4. Invert A -> AInv
				AInv := make([][]float32, cLen)
				for j := 0; j < cLen; j++ {
					AInv[j] = make([]float32, cLen)
					AInv[j][j] = 1.0
					for t := j + 1; t < cLen; t++ {
						var sum float32
						for m := j; m < t; m++ {
							sum += AInv[j][m] * A[m][t]
						}
						AInv[j][t] = -sum
					}
				}

				// 5. U for dimension d
				for t := 0; t < cLen; t++ {
					var sum float32
					for j := 0; j <= t; j++ {
						sum += vPrimeD[j] * AInv[j][t]
					}
					U[t][d] = sum
				}
			}

			// 6. Compute outputs y_t
			for t := 0; t < cLen; t++ {
				globalT := offset + t
				yt := make([]float32, dim)
				for d := 0; d < dim; d++ {
					g0 := float32(1.0)
					for m := 0; m <= t; m++ {
						ad := aScalars[m]
						if aVecs[m] != nil {
							ad = aVecs[m][d]
						}
						g0 *= ad
					}
					var yVal float32 = g0 * p0[t][d]
					for j := 0; j <= t; j++ {
						var gj float32 = 1.0
						for m := j + 1; m <= t; m++ {
							ad := aScalars[m]
							if aVecs[m] != nil {
								ad = aVecs[m][d]
							}
							gj *= ad
						}
						yVal += gj * G[j][t] * U[j][d]
					}
					yt[d] = yVal
				}
				outputs[globalT] = yt
			}

			// 7. Intermediate states within chunk
			currS := S
			for t := 0; t < cLen; t++ {
				globalT := offset + t
				nextS := make([][]float32, dim)
				backing := make([]float32, dim*dim)
				ut := U[t]
				kt := kChunk[t]
				for i := 0; i < dim; i++ {
					row := backing[i*dim : (i+1)*dim]
					currRow := currS[i]
					ai := aScalars[t]
					if aVecs[t] != nil {
						ai = aVecs[t][i]
					}
					ui := ut[i]
					for d := 0; d < dim; d++ {
						row[d] = ai*currRow[d] + ui*kt[d]
					}
					nextS[i] = row
				}
				states[globalT] = nextS
				currS = nextS
			}
			S = currS
		}
	}

	return outputs, states
}

// SequentialGDNPrefill computes reference prefill outputs and the final state matrix S_T.
func SequentialGDNPrefill(k, v, beta, alpha [][]float32, s0 [][]float32) ([][]float32, [][]float32) {
	T := len(k)
	if T == 0 {
		return nil, nil
	}
	dim := len(k[0])
	if dim == 0 {
		return nil, nil
	}

	S := make([][]float32, dim)
	for i := 0; i < dim; i++ {
		S[i] = make([]float32, dim)
		if s0 != nil && i < len(s0) {
			copy(S[i], s0[i])
		}
	}

	outputs := make([][]float32, T)
	p := make([]float32, dim)
	u := make([]float32, dim)

	for t := 0; t < T; t++ {
		kt := k[t]
		vt := v[t]
		aScalar, aVec, aUniform := extractAlpha(alpha, t, dim)
		bScalar, bVec, bUniform := extractBeta(beta, t, dim)

		for i := 0; i < dim; i++ {
			p[i] = fdot(S[i], kt)
		}

		if aUniform {
			for i := 0; i < dim; i++ {
				row := S[i]
				for j := 0; j < dim; j++ {
					row[j] *= aScalar
				}
			}
		} else {
			for i := 0; i < dim; i++ {
				ai := aVec[i]
				row := S[i]
				for j := 0; j < dim; j++ {
					row[j] *= ai
				}
			}
		}

		if bUniform {
			for i := 0; i < dim; i++ {
				u[i] = bScalar * (vt[i] - p[i])
			}
		} else {
			for i := 0; i < dim; i++ {
				u[i] = bVec[i] * (vt[i] - p[i])
			}
		}

		for i := 0; i < dim; i++ {
			ui := u[i]
			row := S[i]
			for j := 0; j < dim; j++ {
				row[j] += ui * kt[j]
			}
		}

		yt := make([]float32, dim)
		for i := 0; i < dim; i++ {
			yt[i] = fdot(S[i], kt)
		}
		outputs[t] = yt
	}

	return outputs, S
}

// WYFChunkwisePrefill computes chunkwise parallel block recurrence outputs and the final state
// matrix S_T, skipping intermediate state allocations to achieve maximal throughput.
func WYFChunkwisePrefill(k, v, beta, alpha [][]float32, s0 [][]float32, chunkSize int) ([][]float32, [][]float32) {
	T := len(k)
	if T == 0 {
		return nil, nil
	}
	dim := len(k[0])
	if dim == 0 {
		return nil, nil
	}
	if chunkSize <= 0 {
		chunkSize = 16
	}

	S := make([][]float32, dim)
	for i := 0; i < dim; i++ {
		S[i] = make([]float32, dim)
		if s0 != nil && i < len(s0) {
			copy(S[i], s0[i])
		}
	}

	outputs := make([][]float32, T)

	p0 := make([][]float32, chunkSize)
	Vprime := make([][]float32, chunkSize)
	U := make([][]float32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		p0[i] = make([]float32, dim)
		Vprime[i] = make([]float32, dim)
		U[i] = make([]float32, dim)
	}
	G := make([][]float32, chunkSize)
	A := make([][]float32, chunkSize)
	AInv := make([][]float32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		G[i] = make([]float32, chunkSize)
		A[i] = make([]float32, chunkSize)
		AInv[i] = make([]float32, chunkSize)
	}
	gamma := make([][]float32, chunkSize+1)
	for i := 0; i <= chunkSize; i++ {
		gamma[i] = make([]float32, chunkSize+1)
	}
	aScalars := make([]float32, chunkSize)
	bScalars := make([]float32, chunkSize)

	for offset := 0; offset < T; offset += chunkSize {
		cLen := chunkSize
		if offset+cLen > T {
			cLen = T - offset
		}

		allAlphaUniform := true
		allBetaUniform := true
		for t := 0; t < cLen; t++ {
			globalT := offset + t
			as, _, au := extractAlpha(alpha, globalT, dim)
			aScalars[t] = as
			if !au {
				allAlphaUniform = false
			}
			bs, _, bu := extractBeta(beta, globalT, dim)
			bScalars[t] = bs
			if !bu {
				allBetaUniform = false
			}
		}

		if !allAlphaUniform || !allBetaUniform {
			// Fallback to full implementation when non-uniform
			out, states := WYFChunkwiseRecurrence(k[offset:offset+cLen], v[offset:offset+cLen],
				beta[offset:offset+cLen], alpha[offset:offset+cLen], S, chunkSize)
			for t := 0; t < cLen; t++ {
				outputs[offset+t] = out[t]
			}
			S = states[cLen-1]
			continue
		}

		kChunk := k[offset : offset+cLen]
		vChunk := v[offset : offset+cLen]

		// 1. S_0 * K
		for i := 0; i < dim; i++ {
			rowS := S[i]
			for t := 0; t < cLen; t++ {
				p0[t][i] = fdot(rowS, kChunk[t])
			}
		}

		// 2. Gram matrix G = K * K^T
		for j := 0; j < cLen; j++ {
			kj := kChunk[j]
			for t := j; t < cLen; t++ {
				dot := fdot(kj, kChunk[t])
				G[j][t] = dot
				G[t][j] = dot
			}
		}

		// 3. Gamma table
		for j := 0; j <= cLen; j++ {
			gamma[j][j] = 1.0
			for t := j + 1; t <= cLen; t++ {
				gamma[j][t] = gamma[j][t-1] * aScalars[t-1]
			}
		}

		// 4. V' = beta * (v - gamma * S0*k)
		for t := 0; t < cLen; t++ {
			rowV := Vprime[t]
			vt := vChunk[t]
			pt := p0[t]
			bt := bScalars[t]
			g0 := gamma[0][t]
			for d := 0; d < dim; d++ {
				rowV[d] = bt * (vt[d] - g0*pt[d])
			}
		}

		// 5. Matrix A
		for j := 0; j < cLen; j++ {
			A[j][j] = 1.0
			for t := j + 1; t < cLen; t++ {
				A[j][t] = bScalars[t] * gamma[j+1][t] * G[j][t]
			}
		}

		// 6. Invert A -> AInv
		for j := 0; j < cLen; j++ {
			AInv[j][j] = 1.0
			for t := j + 1; t < cLen; t++ {
				var sum float32
				for m := j; m < t; m++ {
					sum += AInv[j][m] * A[m][t]
				}
				AInv[j][t] = -sum
			}
		}

		// 7. U = V' * AInv
		for t := 0; t < cLen; t++ {
			rowU := U[t]
			for d := 0; d < dim; d++ {
				rowU[d] = 0
			}
			for j := 0; j <= t; j++ {
				aij := AInv[j][t]
				vRow := Vprime[j]
				for d := 0; d < dim; d++ {
					rowU[d] += vRow[d] * aij
				}
			}
		}

		// 8. Outputs y_t
		for t := 0; t < cLen; t++ {
			globalT := offset + t
			yt := make([]float32, dim)
			g0 := gamma[0][t+1]
			pt := p0[t]
			for d := 0; d < dim; d++ {
				yt[d] = g0 * pt[d]
			}
			for j := 0; j <= t; j++ {
				coeff := gamma[j+1][t+1] * G[j][t]
				uj := U[j]
				for d := 0; d < dim; d++ {
					yt[d] += coeff * uj[d]
				}
			}
			outputs[globalT] = yt
		}

		// 9. End-of-chunk state update: S = gamma[0][cLen] * S0 + sum_{j=0}^{cLen-1} gamma[j+1][cLen] * (u_j * k_j^T)
		gChunkEnd := gamma[0][cLen]
		for i := 0; i < dim; i++ {
			row := S[i]
			for d := 0; d < dim; d++ {
				row[d] *= gChunkEnd
			}
		}
		for j := 0; j < cLen; j++ {
			coeff := gamma[j+1][cLen]
			uj := U[j]
			kj := kChunk[j]
			for i := 0; i < dim; i++ {
				uScale := coeff * uj[i]
				row := S[i]
				for d := 0; d < dim; d++ {
					row[d] += uScale * kj[d]
				}
			}
		}
	}

	return outputs, S
}
