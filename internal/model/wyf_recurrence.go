package model

import (
	"errors"
	"fmt"
	"math"
)

// Default configurations for WYF chunkwise GatedDeltaNet recurrence.
const (
	DefaultWYFChunkSize = 32
	DefaultWYFHeadDim   = 128
)

// WyfRecurrenceConfig defines execution geometry for GatedDeltaNet recurrence.
type WyfRecurrenceConfig struct {
	BatchSize int // Number of batch sequences (default: 1)
	NumHeads  int // Number of attention heads (default: 1)
	SeqLen    int // Sequence length N
	HeadDim   int // Head dimension D (default: 128)
	ChunkSize int // Chunk size C (default: 32)
}

// Validate validates and fills default parameters for WyfRecurrenceConfig.
func (c *WyfRecurrenceConfig) Validate() error {
	if c.BatchSize <= 0 {
		c.BatchSize = 1
	}
	if c.NumHeads <= 0 {
		c.NumHeads = 1
	}
	if c.SeqLen <= 0 {
		return errors.New("wyf_recurrence: SeqLen must be positive")
	}
	if c.HeadDim <= 0 {
		c.HeadDim = DefaultWYFHeadDim
	}
	if c.ChunkSize <= 0 {
		c.ChunkSize = DefaultWYFChunkSize
	}
	return nil
}

// WyfChunkwiseRecurrence executes chunkwise parallel recurrence for GatedDeltaNet linear attention.
//
// Arguments:
//   - q, k, v: query, key, and value tensors of dimension [totalTokens, D]
//   - beta: input update gate tensor of length [totalTokens]
//   - gate: decay factor tensor in (0, 1] of length [totalTokens]
//   - n: sequence length N (tokens per sequence)
//   - d: head dimension D (defaults to 128 if <= 0)
//   - c: chunk size C (defaults to 32 if <= 0)
//   - initState: optional initial recurrent state of size [numSeqs, D, D] (or nil for zero state)
//
// Returns:
//   - output: output tensor of size [totalTokens, D]
//   - finalState: final recurrent state of size [numSeqs, D, D]
//   - err: non-nil if input dimensions are mismatched
func WyfChunkwiseRecurrence(
	q, k, v, beta, gate []float32,
	n, d, c int,
	initState []float32,
) ([]float32, []float32, error) {
	if d < 0 {
		return nil, nil, errors.New("wyf_recurrence: head dimension d cannot be negative")
	}
	if d == 0 {
		d = DefaultWYFHeadDim
	}
	if c < 0 {
		return nil, nil, errors.New("wyf_recurrence: chunk size c cannot be negative")
	}
	if c == 0 {
		c = DefaultWYFChunkSize
	}
	if n < 0 {
		return nil, nil, errors.New("wyf_recurrence: sequence length n cannot be negative")
	}
	if n == 0 {
		if d > 0 && len(q) > 0 {
			n = len(q) / d
		} else {
			return nil, nil, errors.New("wyf_recurrence: sequence length n must be positive")
		}
	}

	totalTokens := len(beta)
	if totalTokens == 0 {
		return nil, nil, errors.New("wyf_recurrence: empty input tensors")
	}
	if totalTokens%n != 0 {
		return nil, nil, fmt.Errorf("wyf_recurrence: total tokens %d not divisible by sequence length %d", totalTokens, n)
	}
	numSeqs := totalTokens / n

	cfg := WyfRecurrenceConfig{
		BatchSize: 1,
		NumHeads:  numSeqs,
		SeqLen:    n,
		HeadDim:   d,
		ChunkSize: c,
	}
	return WyfChunkwiseRecurrenceConfig(q, k, v, beta, gate, cfg, initState)
}

// WyfChunkwiseRecurrenceConfig executes WyfChunkwiseRecurrence using an explicit WyfRecurrenceConfig.
func WyfChunkwiseRecurrenceConfig(
	q, k, v, beta, gate []float32,
	cfg WyfRecurrenceConfig,
	initState []float32,
) ([]float32, []float32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}

	n := cfg.SeqLen
	d := cfg.HeadDim
	c := cfg.ChunkSize
	numSeqs := cfg.BatchSize * cfg.NumHeads
	totalTokens := numSeqs * n
	expectedElems := totalTokens * d
	stateSizePerSeq := d * d

	if len(q) < expectedElems || len(k) < expectedElems || len(v) < expectedElems {
		return nil, nil, fmt.Errorf("wyf_recurrence: q/k/v length (%d/%d/%d) smaller than expected %d",
			len(q), len(k), len(v), expectedElems)
	}
	if len(beta) < totalTokens || len(gate) < totalTokens {
		return nil, nil, fmt.Errorf("wyf_recurrence: beta/gate length (%d/%d) smaller than expected %d",
			len(beta), len(gate), totalTokens)
	}

	output := make([]float32, expectedElems)
	finalState := make([]float32, numSeqs*stateSizePerSeq)

	// Pre-allocate chunk scratch buffers
	bScratch := make([]float32, c*d)
	outInterScratch := make([]float32, c*d)
	rScratch := make([]float32, c*d)
	mScratch := make([]float32, c*c)
	pScratch := make([]float32, c*c)
	deltaScratch := make([]float32, c*d)
	prefixLog := make([]float64, c)
	gamma := make([]float32, c)

	curState := make([]float32, stateSizePerSeq)

	for seq := 0; seq < numSeqs; seq++ {
		// Initialize state for this sequence
		if len(initState) >= (seq+1)*stateSizePerSeq {
			copy(curState, initState[seq*stateSizePerSeq:(seq+1)*stateSizePerSeq])
		} else if len(initState) >= stateSizePerSeq && numSeqs == 1 {
			copy(curState, initState[:stateSizePerSeq])
		} else {
			for i := range curState {
				curState[i] = 0.0
			}
		}

		qSeq := q[seq*n*d : (seq+1)*n*d]
		kSeq := k[seq*n*d : (seq+1)*n*d]
		vSeq := v[seq*n*d : (seq+1)*n*d]
		betaSeq := beta[seq*n : (seq+1)*n]
		gateSeq := gate[seq*n : (seq+1)*n]
		outSeq := output[seq*n*d : (seq+1)*n*d]

		for t0 := 0; t0 < n; t0 += c {
			curC := c
			if t0+curC > n {
				curC = n - t0
			}

			// 1. Prefix scan of cumulative decays across the chunk
			var sumLog float64
			for t := 0; t < curC; t++ {
				gVal := gateSeq[t0+t]
				if gVal <= 0.0 {
					gVal = 1e-7
				}
				sumLog += math.Log(float64(gVal))
				prefixLog[t] = sumLog
				gamma[t] = float32(math.Exp(sumLog))
			}

			// 2. Inter-chunk projection from incoming state S_0:
			//    b_t = gamma_t * (S_0 * k_t)
			//    out_inter_t = gamma_t * (S_0 * q_t)
			for t := 0; t < curC; t++ {
				kRow := kSeq[(t0+t)*d : (t0+t+1)*d]
				qRow := qSeq[(t0+t)*d : (t0+t+1)*d]
				bRow := bScratch[t*d : (t+1)*d]
				oiRow := outInterScratch[t*d : (t+1)*d]
				gT := gamma[t]

				for dv := 0; dv < d; dv++ {
					stRow := curState[dv*d : (dv+1)*d]
					var sumK, sumQ float32
					for dk := 0; dk < d; dk++ {
						sumK += stRow[dk] * kRow[dk]
						sumQ += stRow[dk] * qRow[dk]
					}
					bRow[dv] = gT * sumK
					oiRow[dv] = gT * sumQ
				}
			}

			// 3. Right-hand side: R_t = beta_t * (v_t - b_t)
			for t := 0; t < curC; t++ {
				betaT := betaSeq[t0+t]
				vRow := vSeq[(t0+t)*d : (t0+t+1)*d]
				bRow := bScratch[t*d : (t+1)*d]
				rRow := rScratch[t*d : (t+1)*d]
				for dv := 0; dv < d; dv++ {
					rRow[dv] = betaT * (vRow[dv] - bRow[dv])
				}
			}

			// 4. Lower-triangular cross-token kernel M_{t, s} and causal attention P_{t, s}
			for t := 0; t < curC; t++ {
				kRowT := kSeq[(t0+t)*d : (t0+t+1)*d]
				qRowT := qSeq[(t0+t)*d : (t0+t+1)*d]
				betaT := betaSeq[t0+t]
				mScratch[t*curC+t] = 1.0

				for s := 0; s <= t; s++ {
					kRowS := kSeq[(t0+s)*d : (t0+s+1)*d]
					var dotKK, dotQK float32
					for dk := 0; dk < d; dk++ {
						dotKK += kRowT[dk] * kRowS[dk]
						dotQK += qRowT[dk] * kRowS[dk]
					}
					decayST := float32(1.0)
					if s < t {
						decayST = float32(math.Exp(prefixLog[t] - prefixLog[s]))
						cTS := decayST * dotKK
						mScratch[t*curC+s] = betaT * cTS
					}
					pScratch[t*curC+s] = decayST * dotQK
				}
			}

			// 5. Triangular forward substitution: solve M * Delta = R
			for t := 0; t < curC; t++ {
				deltaT := deltaScratch[t*d : (t+1)*d]
				rT := rScratch[t*d : (t+1)*d]
				copy(deltaT, rT)

				for s := 0; s < t; s++ {
					mTS := mScratch[t*curC+s]
					deltaS := deltaScratch[s*d : (s+1)*d]
					for dv := 0; dv < d; dv++ {
						deltaT[dv] -= mTS * deltaS[dv]
					}
				}
			}

			// 6. Chunk readout outputs: out_t = out_inter_t + sum_{s<=t} P_{t, s} * delta_s
			for t := 0; t < curC; t++ {
				outInterT := outInterScratch[t*d : (t+1)*d]
				outT := outSeq[(t0+t)*d : (t0+t+1)*d]
				copy(outT, outInterT)

				for s := 0; s <= t; s++ {
					pTS := pScratch[t*curC+s]
					deltaS := deltaScratch[s*d : (s+1)*d]
					for dv := 0; dv < d; dv++ {
						outT[dv] += pTS * deltaS[dv]
					}
				}
			}

			// 7. Update chunk-end recurrent state with rank-1 outer products:
			//    S_end = gamma_{C-1} * S_0 + sum_{s=0}^{C-1} delta_s * tilde_k_s^T
			decayEnd := gamma[curC-1]
			for i := 0; i < stateSizePerSeq; i++ {
				curState[i] *= decayEnd
			}
			for s := 0; s < curC; s++ {
				decaySK := float32(math.Exp(prefixLog[curC-1] - prefixLog[s]))
				kRowS := kSeq[(t0+s)*d : (t0+s+1)*d]
				deltaS := deltaScratch[s*d : (s+1)*d]
				for dv := 0; dv < d; dv++ {
					dVal := deltaS[dv]
					stRow := curState[dv*d : (dv+1)*d]
					for dk := 0; dk < d; dk++ {
						stRow[dk] += dVal * (decaySK * kRowS[dk])
					}
				}
			}
		}

		copy(finalState[seq*stateSizePerSeq:(seq+1)*stateSizePerSeq], curState)
	}

	return output, finalState, nil
}

// SequentialGatedDeltaNet executes the exact sequential reference GatedDeltaNet recurrence.
// It processes tokens one by one for numerical verification against WyfChunkwiseRecurrence.
//
// Arguments:
//   - q, k, v: query, key, and value tensors of dimension [totalTokens, D]
//   - beta: input update gate tensor of length [totalTokens]
//   - gate: decay factor tensor in (0, 1] of length [totalTokens]
//   - n: sequence length N (tokens per sequence)
//   - d: head dimension D (defaults to 128 if <= 0)
//   - initState: optional initial recurrent state of size [numSeqs, D, D] (or nil for zero state)
//
// Returns:
//   - output: output tensor of size [totalTokens, D]
//   - finalState: final recurrent state of size [numSeqs, D, D]
//   - err: non-nil if input dimensions are mismatched
func SequentialGatedDeltaNet(
	q, k, v, beta, gate []float32,
	n, d int,
	initState []float32,
) ([]float32, []float32, error) {
	if d < 0 {
		return nil, nil, errors.New("sequential_gdn: head dimension d cannot be negative")
	}
	if d == 0 {
		d = DefaultWYFHeadDim
	}
	if n < 0 {
		return nil, nil, errors.New("sequential_gdn: sequence length n cannot be negative")
	}
	if n == 0 {
		if d > 0 && len(q) > 0 {
			n = len(q) / d
		} else {
			return nil, nil, errors.New("sequential_gdn: sequence length n must be positive")
		}
	}

	totalTokens := len(beta)
	if totalTokens == 0 {
		return nil, nil, errors.New("sequential_gdn: empty input tensors")
	}
	if totalTokens%n != 0 {
		return nil, nil, fmt.Errorf("sequential_gdn: total tokens %d not divisible by sequence length %d", totalTokens, n)
	}
	numSeqs := totalTokens / n

	cfg := WyfRecurrenceConfig{
		BatchSize: 1,
		NumHeads:  numSeqs,
		SeqLen:    n,
		HeadDim:   d,
		ChunkSize: DefaultWYFChunkSize,
	}
	return SequentialGatedDeltaNetConfig(q, k, v, beta, gate, cfg, initState)
}

// SequentialGatedDeltaNetConfig executes SequentialGatedDeltaNet using an explicit WyfRecurrenceConfig.
func SequentialGatedDeltaNetConfig(
	q, k, v, beta, gate []float32,
	cfg WyfRecurrenceConfig,
	initState []float32,
) ([]float32, []float32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}

	n := cfg.SeqLen
	d := cfg.HeadDim
	numSeqs := cfg.BatchSize * cfg.NumHeads
	totalTokens := numSeqs * n
	expectedElems := totalTokens * d
	stateSizePerSeq := d * d

	if len(q) < expectedElems || len(k) < expectedElems || len(v) < expectedElems {
		return nil, nil, fmt.Errorf("sequential_gdn: q/k/v length (%d/%d/%d) smaller than expected %d",
			len(q), len(k), len(v), expectedElems)
	}
	if len(beta) < totalTokens || len(gate) < totalTokens {
		return nil, nil, fmt.Errorf("sequential_gdn: beta/gate length (%d/%d) smaller than expected %d",
			len(beta), len(gate), totalTokens)
	}

	output := make([]float32, expectedElems)
	finalState := make([]float32, numSeqs*stateSizePerSeq)

	curState := make([]float32, stateSizePerSeq)
	kvmem := make([]float32, d)
	delta := make([]float32, d)

	for seq := 0; seq < numSeqs; seq++ {
		// Initialize state for this sequence
		if len(initState) >= (seq+1)*stateSizePerSeq {
			copy(curState, initState[seq*stateSizePerSeq:(seq+1)*stateSizePerSeq])
		} else if len(initState) >= stateSizePerSeq && numSeqs == 1 {
			copy(curState, initState[:stateSizePerSeq])
		} else {
			for i := range curState {
				curState[i] = 0.0
			}
		}

		qSeq := q[seq*n*d : (seq+1)*n*d]
		kSeq := k[seq*n*d : (seq+1)*n*d]
		vSeq := v[seq*n*d : (seq+1)*n*d]
		betaSeq := beta[seq*n : (seq+1)*n]
		gateSeq := gate[seq*n : (seq+1)*n]
		outSeq := output[seq*n*d : (seq+1)*n*d]

		for t := 0; t < n; t++ {
			gVal := gateSeq[t]
			betaVal := betaSeq[t]
			qRow := qSeq[t*d : (t+1)*d]
			kRow := kSeq[t*d : (t+1)*d]
			vRow := vSeq[t*d : (t+1)*d]
			outRow := outSeq[t*d : (t+1)*d]

			// 1. Decay state: S'_t = g_t * S_{t-1}
			for i := 0; i < stateSizePerSeq; i++ {
				curState[i] *= gVal
			}

			// 2. Compute kvmem = S'_t * k_t
			for dv := 0; dv < d; dv++ {
				stRow := curState[dv*d : (dv+1)*d]
				var sum float32
				for dk := 0; dk < d; dk++ {
					sum += stRow[dk] * kRow[dk]
				}
				kvmem[dv] = sum
			}

			// 3. Compute delta = beta_t * (v_t - kvmem)
			for dv := 0; dv < d; dv++ {
				delta[dv] = betaVal * (vRow[dv] - kvmem[dv])
			}

			// 4. Update state: S_t = S'_t + delta * k_t^T
			for dv := 0; dv < d; dv++ {
				dVal := delta[dv]
				stRow := curState[dv*d : (dv+1)*d]
				for dk := 0; dk < d; dk++ {
					stRow[dk] += dVal * kRow[dk]
				}
			}

			// 5. Readout: out_t = S_t * q_t
			for dv := 0; dv < d; dv++ {
				stRow := curState[dv*d : (dv+1)*d]
				var sum float32
				for dk := 0; dk < d; dk++ {
					sum += stRow[dk] * qRow[dk]
				}
				outRow[dv] = sum
			}
		}

		copy(finalState[seq*stateSizePerSeq:(seq+1)*stateSizePerSeq], curState)
	}

	return output, finalState, nil
}
