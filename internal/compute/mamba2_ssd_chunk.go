package compute

import (
	"errors"
	"math"
)

// Mamba-2 SSD chunked prefill pipeline (chunk cumsum -> chunk state -> state
// passing -> chunk scan) with cu_seqlens varlen packed batches, initial_states,
// and per-sequence final_states. Component and host oracle only: no hardware or
// speed claim.

// SSDChunkScanVarlen runs the 4-stage Mamba-2 SSD chunked scan over packed
// varlen sequences defined by cuSeqlens ([numSeqs+1], starting at 0).
func SSDChunkScanVarlen(
	x, dt, a, b, c []float32,
	cuSeqlens []int,
	initialStates []float32,
	chunkSize, dstate int,
) (out []float32, finalStates []float32, err error) {
	if chunkSize <= 0 || dstate <= 0 || len(cuSeqlens) < 2 || cuSeqlens[0] != 0 {
		return nil, nil, errors.New("compute: invalid SSD chunk scan varlen args")
	}
	totalTokens := cuSeqlens[len(cuSeqlens)-1]
	numSeqs := len(cuSeqlens) - 1
	if totalTokens <= 0 || len(x) != totalTokens || len(dt) != totalTokens ||
		len(a) != totalTokens || len(b) != totalTokens*dstate || len(c) != totalTokens*dstate {
		return nil, nil, errors.New("compute: SSD varlen tensor length mismatch")
	}
	if initialStates != nil && len(initialStates) != numSeqs*dstate {
		return nil, nil, errors.New("compute: SSD initialStates length mismatch")
	}

	out = make([]float32, totalTokens)
	finalStates = make([]float32, numSeqs*dstate)

	for seq := 0; seq < numSeqs; seq++ {
		start, end := cuSeqlens[seq], cuSeqlens[seq+1]
		if end < start {
			return nil, nil, errors.New("compute: non-monotonic cuSeqlens")
		}
		state := make([]float64, dstate)
		if initialStates != nil {
			for s := 0; s < dstate; s++ {
				state[s] = float64(initialStates[seq*dstate+s])
			}
		}

		for cStart := start; cStart < end; cStart += chunkSize {
			cEnd := min(cStart+chunkSize, end)
			clen := cEnd - cStart

			// Stage 1: chunk cumsum of log-decay (a[t]*dt[t]).
			cumsum := make([]float64, clen)
			for i := 0; i < clen; i++ {
				v := float64(a[cStart+i]) * float64(dt[cStart+i])
				if i == 0 {
					cumsum[i] = v
				} else {
					cumsum[i] = cumsum[i-1] + v
				}
			}

			// Stage 2 & 3: chunk state accumulation + state passing from prev chunk.
			chunkEndState := make([]float64, dstate)
			totalDecay := math.Exp(cumsum[clen-1])
			for s := 0; s < dstate; s++ {
				chunkEndState[s] = state[s] * totalDecay
				for i := 0; i < clen; i++ {
					t := cStart + i
					decayToEnd := math.Exp(cumsum[clen-1] - cumsum[i])
					chunkEndState[s] += decayToEnd * float64(dt[t]) * float64(b[t*dstate+s]) * float64(x[t])
				}
			}

			// Stage 4: chunk scan output using incoming chunk state + intra-chunk contribution.
			for i := 0; i < clen; i++ {
				t := cStart + i
				var y float64
				fromPrev := math.Exp(cumsum[i])
				for s := 0; s < dstate; s++ {
					st := state[s] * fromPrev
					for j := 0; j <= i; j++ {
						tj := cStart + j
						st += math.Exp(cumsum[i]-cumsum[j]) * float64(dt[tj]) * float64(b[tj*dstate+s]) * float64(x[tj])
					}
					y += st * float64(c[t*dstate+s])
				}
				out[t] = float32(y)
			}
			state = chunkEndState
		}

		for s := 0; s < dstate; s++ {
			finalStates[seq*dstate+s] = float32(state[s])
		}
	}
	return out, finalStates, nil
}
