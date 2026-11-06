// logprob_topk.go — per-row top-k log-softmax receipt without materializing
// [rows, vocab].
//
// ADAPTED from vllm-project/vllm, vllm/v1/worker/gpu/sample/logprob.py at
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0): _topk_log_softmax_kernel
// (per-row max + log-sum-exp streamed over 1024-wide blocks, output emitted
// only at the gathered ids), _ranks_kernel (chosen-token rank counting ties as
// >=) and _fill_logprob_token_ids_kernel (gather-column layout: chosen token
// first, then the bounded top-k ids). Re-expressed here as the CPU contract
// reference that pins the observable behavior of the CUDA family in
// logprob_topk.cu; no upstream code copied. Ref
// oss-port-vllmtop-logprob-device-topk-20261103.
//
// SEMANTICS (pinned by logprob_topk_test.go):
//   - All value arithmetic is float32, exactly as on device: row max,
//     exp(x-max) sum, log, and the final subtracts. The float64 host oracle
//     (agent.nativeTopLogprobs, internal/agent/native_decode_logprobs.go:57)
//     stays the accuracy reference and fallback; this reference must match it
//     within fp32 tolerance.
//   - Logprobs are emitted only at the caller-gathered ids; a full-vocab
//     logprob vector is never materialized.
//   - A NaN or infinite logit anywhere in the row refuses the whole row with
//     ErrLogprobTopKNonFinite instead of producing garbage.
//   - Ties order by ascending id, matching the Go sort (logprob desc, id asc).

package compute

import (
	"errors"
	"math"
)

// LogprobTopKBlockWidth is the streaming width both the CUDA family and this
// reference step through each logits row with; it mirrors the upstream
// BLOCK_SIZE=1024 / _MAX_TOPK_BLOCK gather cap.
const LogprobTopKBlockWidth = 1024

// Sentinel errors closing the DeviceTopKLogprobs / ArgmaxTopK contracts.
// Callers match these with errors.Is, never on prose.
var (
	// ErrLogprobTopKEmpty rejects a zero-length logits row.
	ErrLogprobTopKEmpty = errors.New("compute: logprob topk: empty logits row")

	// ErrLogprobTopKIDRange rejects a gathered id or chosen token outside
	// [0, len(logits)).
	ErrLogprobTopKIDRange = errors.New("compute: logprob topk: token id out of range")

	// ErrLogprobTopKShape rejects a negative k.
	ErrLogprobTopKShape = errors.New("compute: logprob topk: k must be >= 0")

	// ErrLogprobTopKNonFinite refuses a row containing a NaN or infinite
	// logit; the compute-side twin of agent.ErrNonFiniteLogits. Distinct from
	// ErrNonFiniteLogits, the sampling-truncation family's own sentinel.
	ErrLogprobTopKNonFinite = errors.New("compute: logprob topk: non-finite model logit")
)

// TopKLogprobsRow is one row's computed receipt: the log-softmax value at each
// gathered id plus the chosen token's rank.
type TopKLogprobsRow struct {
	// IDs echoes the gathered ids in call order (conventionally the chosen
	// token first, then the top-k columns), mirroring the upstream
	// [batch, 1+num_cols] gather layout.
	IDs []int32
	// Logprobs[i] is the fp32 log-softmax of logits[IDs[i]].
	Logprobs []float32
	// ChosenRank counts the tokens whose logit is >= the chosen token's
	// logit (upstream _ranks_kernel: ties rank the chosen token worst among
	// equals). Always >= 1 for a finite row.
	ChosenRank int32
}

// DeviceTopKLogprobs computes the log-softmax at each gathered id for one
// fp32 logits row, streaming the row in LogprobTopKBlockWidth-wide blocks
// exactly as k_lptk_row_lse in logprob_topk.cu does: pass one reduces the row
// max (and flags any non-finite entry), pass two sums exp(x - max) into the
// log-sum-exp, and the gather emits only the numIds values. It never
// materializes a [vocab] logprob vector.
func DeviceTopKLogprobs(logits []float32, ids []int32, chosen int32) (*TopKLogprobsRow, error) {
	vocab := len(logits)
	if vocab == 0 {
		return nil, ErrLogprobTopKEmpty
	}
	if chosen < 0 || int(chosen) >= vocab {
		return nil, ErrLogprobTopKIDRange
	}
	for _, id := range ids {
		if id < 0 || int(id) >= vocab {
			return nil, ErrLogprobTopKIDRange
		}
	}

	// Pass 1: row max, block-streamed; a non-finite entry refuses the row.
	maxVal := float32(math.Inf(-1))
	for start := 0; start < vocab; start += LogprobTopKBlockWidth {
		end := start + LogprobTopKBlockWidth
		if end > vocab {
			end = vocab
		}
		partialMax := float32(math.Inf(-1))
		bad := false
		for _, x := range logits[start:end] {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				bad = true
				break
			}
			if x > partialMax {
				partialMax = x
			}
		}
		if bad {
			return nil, ErrLogprobTopKNonFinite
		}
		if partialMax > maxVal {
			maxVal = partialMax
		}
	}

	// Pass 2: se = sum(exp(x - max)) accumulated in fp32, block-streamed.
	var se float32
	for start := 0; start < vocab; start += LogprobTopKBlockWidth {
		end := start + LogprobTopKBlockWidth
		if end > vocab {
			end = vocab
		}
		var partial float32
		for _, x := range logits[start:end] {
			partial += float32(math.Exp(float64(x - maxVal)))
		}
		se += partial
	}
	lse := float32(math.Log(float64(se)))

	row := &TopKLogprobsRow{
		IDs:      append([]int32(nil), ids...),
		Logprobs: make([]float32, len(ids)),
	}
	for i, id := range ids {
		row.Logprobs[i] = logits[id] - maxVal - lse
	}

	// Rank pass (upstream _ranks_kernel): count logits >= the chosen logit.
	chosenLogit := logits[chosen]
	var rank int32
	for _, x := range logits {
		if x >= chosenLogit {
			rank++
		}
	}
	row.ChosenRank = rank
	return row, nil
}

// ArgmaxTopK returns the k ids of the largest logits, ordered by (logit desc,
// id asc), scanning the row once with O(k) state — the bounded argmax-topk
// pass that supplies the gathered ids when no device top-k output is
// available. It mirrors the insertion discipline of the float64 host oracle
// (agent.nativeTopLogprobs): a candidate displaces the current worst only on
// a strictly greater logit, so equal logits keep ascending-id order. k is
// clamped to the row length; k == 0 yields an empty slice.
func ArgmaxTopK(logits []float32, k int) ([]int32, error) {
	vocab := len(logits)
	if vocab == 0 {
		return nil, ErrLogprobTopKEmpty
	}
	if k < 0 {
		return nil, ErrLogprobTopKShape
	}
	if k > vocab {
		k = vocab
	}
	if k == 0 {
		return []int32{}, nil
	}

	top := make([]int32, 0, k)
	for id, raw := range logits {
		if math.IsNaN(float64(raw)) || math.IsInf(float64(raw), 0) {
			return nil, ErrLogprobTopKNonFinite
		}
		if len(top) == k && raw <= logits[top[k-1]] {
			continue
		}
		pos := len(top)
		for pos > 0 && raw > logits[top[pos-1]] {
			pos--
		}
		if len(top) < k {
			top = append(top, 0)
		}
		copy(top[pos+1:], top[pos:len(top)-1])
		top[pos] = int32(id)
	}
	return top, nil
}
