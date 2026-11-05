package compute

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default

// hostFloat64TopLogprobs mirrors agent.nativeTopLogprobs
// (internal/agent/native_decode_logprobs.go:57): the float64 oracle the
// device family must match within fp32 tolerance. It is kept local because
// internal/agent imports internal/compute, so the package under test cannot
// import it back. Same scan order, same O(V*k) insertion discipline, same
// (logit desc, id asc) candidate order.
func hostFloat64TopLogprobs(logits []float32, token, k int) (float64, []int, []float64, error) {
	if token < 0 || token >= len(logits) {
		return 0, nil, nil, errors.New("token out of range")
	}
	if k > len(logits) {
		k = len(logits)
	}
	maxLogit := math.Inf(-1)
	for _, raw := range logits {
		v := float64(raw)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, nil, nil, errors.New("non-finite model logit")
		}
		if v > maxLogit {
			maxLogit = v
		}
	}
	top := make([]int, 0, k)
	var expSum float64
	for id, raw := range logits {
		expSum += math.Exp(float64(raw) - maxLogit)
		if k == 0 {
			continue
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
		top[pos] = id
	}
	logZ := maxLogit + math.Log(expSum)
	lp := func(id int) float64 {
		v := float64(logits[id]) - logZ
		if v > 0 {
			v = 0
		}
		return v
	}
	row := make([]float64, len(top))
	for i, id := range top {
		row[i] = lp(id)
	}
	return lp(token), top, row, nil
}

// fixtureLogits returns a deterministic LCG row in [-8, 8) — wide enough to
// span several streamed blocks without any hand-computed expectations.
func fixtureLogits(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*1664525 + 1013904223
		f := float64((s>>8)&0xFFFF) / 65535.0
		out[i] = float32((f - 0.5) * 16)
	}
	return out
}

func closeF32(got float32, want float64, tol float64) bool {
	return math.Abs(float64(got)-want) <= tol
}

// TestDeviceTopKLogprobsMatchHostFloat64 pins the borrowed contract:
// (a) a k=5 row equals the float64 host oracle within fp32 tolerance,
// including the chosen token's entry;
// (b) tie order matches the Go sort (logprob desc, id asc);
// (c) a NaN/inf logit row returns the ErrLogprobTopKNonFinite-class sentinel
// instead of garbage. Fixtures (a)/(b) are built so the assertions hold by
// construction of the upstream semantics, not by running the implementation.
func TestDeviceTopKLogprobsMatchHostFloat64(t *testing.T) {
	const tol = 1e-4 // fp32 tolerance band for the block-streamed reduction

	t.Run("k5_tail_block", func(t *testing.T) {
		const vocab = 3000 // two full 1024 blocks + a 952-wide tail
		logits := fixtureLogits(vocab, 0x9e3779b9)
		const k = 5

		topk, err := ArgmaxTopK(logits, k)
		if err != nil {
			t.Fatalf("ArgmaxTopK returned %v, want nil", err)
		}
		if len(topk) != k {
			t.Fatalf("ArgmaxTopK returned %d ids, want %d", len(topk), k)
		}

		// Chosen token: the argmin, guaranteed disjoint from the top-k.
		chosen := 0
		for id, x := range logits {
			if x < logits[chosen] {
				chosen = id
			}
		}
		for _, id := range topk {
			if int(id) == chosen {
				t.Fatalf("fixture broken: chosen %d inside top-k %v", chosen, topk)
			}
		}

		ids := append([]int32{int32(chosen)}, topk...)
		got, err := DeviceTopKLogprobs(logits, ids, int32(chosen))
		if err != nil {
			t.Fatalf("DeviceTopKLogprobs returned %v, want nil", err)
		}
		wantChosenLP, wantIDs, wantRowLP, err := hostFloat64TopLogprobs(logits, chosen, k)
		if err != nil {
			t.Fatalf("oracle failed: %v", err)
		}

		// Closed contract on the gathered ids themselves.
		wantIDs32 := make([]int32, len(wantIDs))
		for i, id := range wantIDs {
			wantIDs32[i] = int32(id)
		}
		if !reflect.DeepEqual(got.IDs[1:], wantIDs32) {
			t.Errorf("top-k ids = %v, want %v", got.IDs[1:], wantIDs32)
		}
		if len(got.Logprobs) != len(ids) {
			t.Fatalf("Logprobs length %d, want %d", len(got.Logprobs), len(ids))
		}

		// (a) every emitted value, chosen entry included, within fp32
		// tolerance of the float64 oracle.
		if !closeF32(got.Logprobs[0], wantChosenLP, tol) {
			t.Errorf("chosen logprob = %v, want %v within %v", got.Logprobs[0], wantChosenLP, tol)
		}
		for i := range wantRowLP {
			if !closeF32(got.Logprobs[1+i], wantRowLP[i], tol) {
				t.Errorf("logprob[%d] (id %d) = %v, want %v within %v", i, wantIDs[i], got.Logprobs[1+i], wantRowLP[i], tol)
			}
		}
		for i, lp := range got.Logprobs {
			if lp > 0 {
				t.Errorf("logprob[%d] = %v, want <= 0 (log-softmax property)", i, lp)
			}
		}

		// Rank: count of tokens with logit >= the chosen logit.
		wantRank := 0
		for _, x := range logits {
			if x >= logits[chosen] {
				wantRank++
			}
		}
		if got.ChosenRank != int32(wantRank) {
			t.Errorf("ChosenRank = %d, want %d", got.ChosenRank, wantRank)
		}
	})

	t.Run("tie_order", func(t *testing.T) {
		// Hand-built ties: 5@{0,2}, 2@{1,4,5}. The k=4 cut must keep the two
		// 5s, then 2@1 and 2@4 (ascending id among equals), dropping 2@5.
		logits := []float32{5, 2, 5, 1, 2, 2, 0, -3, 1, 0, -1, -2}
		const k = 4

		topk, err := ArgmaxTopK(logits, k)
		if err != nil {
			t.Fatalf("ArgmaxTopK returned %v, want nil", err)
		}
		wantTopk := []int32{0, 2, 1, 4}
		if !reflect.DeepEqual(topk, wantTopk) {
			t.Fatalf("ArgmaxTopK = %v, want %v (logprob desc, id asc)", topk, wantTopk)
		}

		// Independent cross-check: plain Go sort of the float64 oracle
		// logprobs by (logprob desc, id asc) must agree with the bounded
		// insertion pass.
		_, oracleIDs, oracleLP, err := hostFloat64TopLogprobs(logits, 7, len(logits))
		if err != nil {
			t.Fatalf("oracle failed: %v", err)
		}
		type pair struct {
			lp float64
			id int
		}
		pairs := make([]pair, len(oracleIDs))
		for i, id := range oracleIDs {
			pairs[i] = pair{oracleLP[i], id}
		}
		sort.Slice(pairs, func(a, b int) bool {
			if pairs[a].lp != pairs[b].lp {
				return pairs[a].lp > pairs[b].lp
			}
			return pairs[a].id < pairs[b].id
		})
		sortedTopk := make([]int32, k)
		for i := 0; i < k; i++ {
			sortedTopk[i] = int32(pairs[i].id)
		}
		if !reflect.DeepEqual(topk, sortedTopk) {
			t.Errorf("ArgmaxTopK = %v, want %v from Go sort", topk, sortedTopk)
		}

		// Emitted values stay ordered and match the oracle at the tie cut.
		ids := append([]int32{7}, topk...)
		got, err := DeviceTopKLogprobs(logits, ids, 7)
		if err != nil {
			t.Fatalf("DeviceTopKLogprobs returned %v, want nil", err)
		}
		// Descending order holds across the top-k columns (index 1+); column
		// 0 is the chosen token, outside the top-k cut.
		for i := 2; i < len(got.Logprobs); i++ {
			if got.Logprobs[i-1] < got.Logprobs[i] {
				t.Errorf("top-k logprobs not descending at %d: %v > %v", i, got.Logprobs[i-1], got.Logprobs[i])
			}
		}
		lpByID := make(map[int32]float64, len(oracleIDs))
		for i, id := range oracleIDs {
			lpByID[int32(id)] = oracleLP[i]
		}
		for i, id := range wantTopk {
			wantLP := lpByID[id]
			if !closeF32(got.Logprobs[1+i], wantLP, tol) {
				t.Errorf("logprob[%d] (id %d) = %v, want %v within %v", i, id, got.Logprobs[1+i], wantLP, tol)
			}
		}
		// Ties emit identical fp32 values for the equal logits.
		if got.Logprobs[1] != got.Logprobs[2] {
			t.Errorf("tied ids 0/2 emitted %v vs %v, want equal", got.Logprobs[1], got.Logprobs[2])
		}
	})

	t.Run("non_finite", func(t *testing.T) {
		nan := float32(math.NaN())
		plusInf := float32(math.Inf(1))
		minusInf := float32(math.Inf(-1))
		rows := map[string][]float32{
			"nan":      {1, nan, 2},
			"+inf":     {1, plusInf, 2},
			"-inf":     {minusInf, 1, 2},
			"nan_deep": append(fixtureLogits(2500, 7), nan),
		}
		for name, row := range rows {
			name, row := name, row
			t.Run(name, func(t *testing.T) {
				res, err := DeviceTopKLogprobs(row, []int32{0, 1}, 0)
				if !errors.Is(err, ErrLogprobTopKNonFinite) {
					t.Errorf("DeviceTopKLogprobs error = %v, want ErrLogprobTopKNonFinite", err)
				}
				if res != nil {
					t.Errorf("result = %+v, want nil on refusal", res)
				}
				if _, err := ArgmaxTopK(row, 2); !errors.Is(err, ErrLogprobTopKNonFinite) {
					t.Errorf("ArgmaxTopK error = %v, want ErrLogprobTopKNonFinite", err)
				}
			})
		}
	})

	t.Run("sentinels", func(t *testing.T) {
		checks := []struct {
			name     string
			err      func() error
			sentinel error
		}{
			{"empty_row", func() error { _, err := DeviceTopKLogprobs(nil, nil, 0); return err }, ErrLogprobTopKEmpty},
			{"chosen_out_of_range", func() error {
				_, err := DeviceTopKLogprobs([]float32{1, 2}, []int32{0}, 2)
				return err
			}, ErrLogprobTopKIDRange},
			{"gathered_id_out_of_range", func() error {
				_, err := DeviceTopKLogprobs([]float32{1, 2}, []int32{5}, 0)
				return err
			}, ErrLogprobTopKIDRange},
			{"negative_k", func() error { _, err := ArgmaxTopK([]float32{1, 2}, -1); return err }, ErrLogprobTopKShape},
			{"empty_argmax", func() error { _, err := ArgmaxTopK(nil, 3); return err }, ErrLogprobTopKEmpty},
		}
		for _, c := range checks {
			if !errors.Is(c.err(), c.sentinel) {
				t.Errorf("%s: errors.Is(%v, %v) = false, want true", c.name, c.err(), c.sentinel)
			}
		}
		// k clamps to the row length instead of erroring.
		all, err := ArgmaxTopK([]float32{0.5, 0.25}, 10)
		if err != nil {
			t.Fatalf("ArgmaxTopK(k>vocab) returned %v, want nil", err)
		}
		if !reflect.DeepEqual(all, []int32{0, 1}) {
			t.Errorf("ArgmaxTopK(k>vocab) = %v, want [0 1]", all)
		}
	})

	// Source witness only: the CUDA candidate defines exactly three kernels
	// and one fcuda_* host entry, carries the upstream pin, streams in
	// 1024-wide blocks, and drops any CUB dependency. No compiler or GPU
	// proof; the Go reference above holds the bits.
	t.Run("cuda_source_contract", func(t *testing.T) {
		raw, err := os.ReadFile(filepath.FromSlash("logprob_topk.cu"))
		if err != nil {
			t.Fatal(err)
		}
		rawText := string(raw)
		code := stripCComments(strings.ReplaceAll(rawText, "\r\n", "\n"))
		for token, want := range map[string]int{
			"__global__":                         3,
			"k_lptk_row_lse(":                    1,
			"k_lptk_gather(":                     1,
			"k_lptk_rank(":                       1,
			`extern "C" int fcuda_logprob_topk(`: 1,
			"#define LPTK_BLOCK_WIDTH 1024":      1,
		} {
			if got := strings.Count(code, token); got != want {
				t.Errorf("logprob_topk.cu has %d occurrences of %q, want %d", got, token, want)
			}
		}
		if !strings.Contains(rawText, "ae065918640a198a56f6f17eb45e251c3229fe41") {
			t.Error("logprob_topk.cu lacks the upstream vLLM pin citation")
		}
		if strings.Contains(code, "cub/") || strings.Contains(code, "BlockScan") {
			t.Error("logprob_topk.cu still depends on CUB; the adaptation must reduce by hand")
		}
	})
}
