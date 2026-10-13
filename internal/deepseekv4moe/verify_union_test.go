package deepseekv4moe

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestSimulateVerifyUnionFixture(t *testing.T) {
	tok := func(a, b []int) []ExpertRoute {
		return []ExpertRoute{{Layer: 0, Experts: a}, {Layer: 1, Experts: b}}
	}
	seq := [][]ExpertRoute{
		tok([]int{0, 1}, []int{4, 5}),
		tok([]int{1, 2}, []int{4, 5}),
		tok([]int{0, 3}, []int{6, 7}),
		tok([]int{0, 1}, []int{4, 6}),
		tok([]int{2, 3}, []int{5, 7}),
	}
	got, err := SimulateVerifyUnion([][][]ExpertRoute{seq}, 1, 2, 2, 8, 2, CachePolicyLRU)
	if err != nil {
		t.Fatal(err)
	}
	if got.Window != 2 || got.Windows != 2 {
		t.Fatalf("window=%d windows=%d, want 2/2", got.Window, got.Windows)
	}
	// Unions per (window, layer): {0,1,2}=3, {4,5}=2, {0,1,3}=3, {4,6,7}=3.
	if got.UnionP50 != 3 || got.UnionP95 != 3 || got.UnionMean != 2.75 {
		t.Fatalf("union mean/p50/p95 = %v/%d/%d, want 2.75/3/3", got.UnionMean, got.UnionP50, got.UnionP95)
	}
	if len(got.Layers) != 2 || got.Layers[0].Mean != 3 || got.Layers[1].Mean != 2.5 {
		t.Fatalf("layers = %+v", got.Layers)
	}
	if got.VerifyMisses != 11 || got.SequentialMisses != 16 {
		t.Fatalf("misses verify=%d sequential=%d, want 11/16", got.VerifyMisses, got.SequentialMisses)
	}

	k0, err := SimulateVerifyUnion([][][]ExpertRoute{seq}, 0, 2, 2, 8, 2, CachePolicyLRU)
	if err != nil {
		t.Fatal(err)
	}
	if k0.VerifyMisses != k0.SequentialMisses || k0.UnionMean != 2 || k0.Windows != 5 {
		t.Fatalf("k=0 must equal sequential: %+v", k0)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSimulateVerifyUnionRejects(t *testing.T) {
	seq := [][][]ExpertRoute{{{{Layer: 0, Experts: []int{1, 1}}}}}
	cases := []struct {
		name     string
		k, cap   int
		policy   CachePolicy
		sequence [][][]ExpertRoute
		want     error
	}{
		{"capacity", 0, 0, CachePolicyLRU, nil, ErrInvalidTraceCapacity},
		{"negative k", -1, 4, CachePolicyLRU, nil, ErrInvalidTraceShape},
		{"oracle", 0, 4, CachePolicyOPT, nil, ErrInvalidVerifyPolicy},
		{"duplicate", 0, 4, CachePolicyLRU, seq, ErrDuplicateTraceExpert},
	}
	for _, tc := range cases {
		_, err := SimulateVerifyUnion(tc.sequence, tc.k, tc.cap, 1, 4, 2, tc.policy)
		if !errors.Is(err, tc.want) {
			t.Fatalf("%s: err=%v want %v", tc.name, err, tc.want)
		}
	}
}

// TestVerifyUnionTraceReport replays a captured fak.expert-route.trace/v1 file named by
// FAK_V41_ROUTES_TRACE (decode records only, split at each prefill) and logs the union and
// miss table. It skips when the variable is unset.
// fak-test:runtime fast est=2s lane=default
func TestVerifyUnionTraceReport(t *testing.T) {
	path := os.Getenv("FAK_V41_ROUTES_TRACE")
	if path == "" {
		t.Skip("FAK_V41_ROUTES_TRACE unset")
	}
	sequences, layers, experts, topK := loadDecodeSequences(t, path)
	const groupBytes = 12.2 * (1 << 20)
	for _, ringGiB := range []float64{12, 64.2} {
		capacity := int(ringGiB * (1 << 30) / groupBytes)
		for _, policy := range []CachePolicy{CachePolicyLRU, CachePolicyLFRU} {
			for k := 0; k <= 4; k++ {
				r, err := SimulateVerifyUnion(sequences, k, capacity, layers, experts, topK, policy)
				if err != nil {
					t.Fatal(err)
				}
				if r.Windows == 0 || r.UnionP95 > r.Window*topK || r.UnionMean < float64(topK) {
					t.Fatalf("union out of bounds: %+v", r)
				}
				t.Logf("ring=%.1fGiB cap=%d policy=%s K=%d windows=%d union mean/p50/p95=%.2f/%d/%d seq_miss/step=%.1f verify_miss/step=%.1f",
					ringGiB, capacity, policy, k, r.Windows, r.UnionMean, r.UnionP50, r.UnionP95, r.SequentialPerStep, r.VerifyPerStep)
				if k == 4 && policy == CachePolicyLRU && ringGiB == 12 {
					for _, l := range r.Layers {
						t.Logf("  layer=%d mean=%.2f p50=%d p95=%d", l.Layer, l.Mean, l.P50, l.P95)
					}
				}
			}
		}
	}
}

func loadDecodeSequences(t *testing.T, path string) ([][][]ExpertRoute, int, int, int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	var header struct {
		Layers  int `json:"layers"`
		Experts int `json:"experts"`
		TopK    int `json:"top_k"`
	}
	if !scanner.Scan() || json.Unmarshal(scanner.Bytes(), &header) != nil {
		t.Fatal("missing trace header")
	}
	var sequences [][][]ExpertRoute
	var current [][]ExpertRoute
	lastToken := -1
	flush := func() {
		if len(current) > 0 {
			sequences = append(sequences, current)
		}
		current, lastToken = nil, -1
	}
	for scanner.Scan() {
		var rec struct {
			Layer   int    `json:"layer"`
			Token   int    `json:"token"`
			Phase   string `json:"phase"`
			Experts []int  `json:"experts"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Phase != "decode" {
			flush()
			continue
		}
		if rec.Token != lastToken {
			current = append(current, nil)
			lastToken = rec.Token
		}
		current[len(current)-1] = append(current[len(current)-1], ExpertRoute{Layer: rec.Layer, Experts: rec.Experts})
	}
	flush()
	return sequences, header.Layers, header.Experts, header.TopK
}
