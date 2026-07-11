// Tests for the pure, model-independent accounting helpers in radixbench:
// longest-common-prefix discovery, the declare-one-prefix reuse accounting,
// lexicographic (DFS) ordering, and the simple token tallies. These functions
// are deterministic and depend on no model, GPU, file, or network, so the
// expected values below are computed by hand from the request token streams.
package main

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLongestCommonPrefix(t *testing.T) {
	tests := []struct {
		name string
		reqs [][]int
		want int
	}{
		{"empty", nil, 0},
		{"empty slice", [][]int{}, 0},
		// Single request: the loop over reqs[1:] is empty, so lcp == len(reqs[0]).
		{"single", [][]int{{1, 2, 3, 4}}, 4},
		{"single empty", [][]int{{}}, 0},
		// All three share [9, 8, 7] then diverge.
		{"three-share-3", [][]int{{9, 8, 7, 1}, {9, 8, 7, 2, 2}, {9, 8, 7}}, 3},
		// First token already differs => no shared prefix.
		{"no-share", [][]int{{1, 2, 3}, {4, 5, 6}}, 0},
		// One request is a strict prefix of the other; lcp is bounded by the shorter.
		{"prefix-bound", [][]int{{1, 2, 3, 4, 5}, {1, 2}}, 2},
		// Identical requests share their full (equal) length.
		{"identical", [][]int{{5, 6, 7}, {5, 6, 7}}, 3},
		// One empty request forces lcp to 0 regardless of the others.
		{"one-empty", [][]int{{1, 2, 3}, {}}, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := longestCommonPrefix(tc.reqs); got != tc.want {
				t.Errorf("longestCommonPrefix(%v) = %d, want %d", tc.reqs, got, tc.want)
			}
		})
	}
}

func TestDeclaredMatched(t *testing.T) {
	tests := []struct {
		name        string
		reqs        [][]int
		wantLCP     int
		wantMatched int
	}{
		// matched = lcp * (len(reqs)-1): the prefix is reused by every request after the first.
		{"empty", nil, 0, 0},
		{"single", [][]int{{1, 2, 3}}, 3, 0}, // lcp=3 but (1-1)=0 reuses
		{"three-share-3", [][]int{{9, 8, 7, 1}, {9, 8, 7, 2}, {9, 8, 7}}, 3, 6}, // 3*(3-1)
		{"no-share", [][]int{{1}, {2}, {3}}, 0, 0},
		{"identical-pair", [][]int{{5, 6, 7}, {5, 6, 7}}, 3, 3}, // 3*(2-1)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			lcp, matched := declaredMatched(tc.reqs)
			if lcp != tc.wantLCP || matched != tc.wantMatched {
				t.Errorf("declaredMatched(%v) = (lcp=%d, matched=%d), want (lcp=%d, matched=%d)",
					tc.reqs, lcp, matched, tc.wantLCP, tc.wantMatched)
			}
		})
	}
}

func TestLexLess(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"first-elem-less", []int{1, 9}, []int{2, 0}, true},
		{"first-elem-greater", []int{2, 0}, []int{1, 9}, false},
		// Equal shared portion, a is shorter => a is less.
		{"shorter-is-less", []int{1, 2}, []int{1, 2, 3}, true},
		{"longer-is-not-less", []int{1, 2, 3}, []int{1, 2}, false},
		// Fully equal sequences => not strictly less (len(a) < len(b) is false).
		{"equal-not-less", []int{1, 2, 3}, []int{1, 2, 3}, false},
		// Divergence at a later index decides it.
		{"diverge-late", []int{1, 2, 3}, []int{1, 2, 4}, true},
		// Empty is less than any non-empty.
		{"empty-less", []int{}, []int{0}, true},
		{"empty-vs-empty", []int{}, []int{}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := lexLess(tc.a, tc.b); got != tc.want {
				t.Errorf("lexLess(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestLoadWorkload is the witness for the #322 adoption path: an operator's own
// token-id prompt set, loaded from JSON, becomes a Workload the same accounting runs
// over. It writes a temp file, loads it, and checks the requests + metadata + the
// name-from-filename fallback survive the round trip — no model, file, or network.
func TestLoadWorkload(t *testing.T) {
	dir := t.TempDir()

	// A workload with explicit metadata and a shared 3-token prefix.
	named := filepath.Join(dir, "few-shot.json")
	if err := os.WriteFile(named, []byte(
		`{"name":"few-shot","desc":"shared preamble","sglang_published":"50-99% band",`+
			`"requests":[[1,2,3,10],[1,2,3,11],[1,2,3,12]]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := loadWorkload(named)
	if err != nil {
		t.Fatalf("loadWorkload: %v", err)
	}
	if w.Name != "few-shot" || w.Desc != "shared preamble" || w.SGLang != "50-99% band" {
		t.Errorf("metadata not preserved: %+v", w)
	}
	if len(w.Requests) != 3 || totalTokens(w.Requests) != 12 {
		t.Errorf("requests not loaded verbatim: %v", w.Requests)
	}
	// The shared [1,2,3] prefix must be discoverable by the same radix accounting.
	if matched, _, _ := radixMatched(w.Requests, 0); matched != 6 { // 3 reused by reqs 2 and 3
		t.Errorf("radix reuse over loaded workload = %d, want 6", matched)
	}

	// Name defaults to the base filename when the JSON omits one.
	noName := filepath.Join(dir, "agents.json")
	if err := os.WriteFile(noName, []byte(`{"requests":[[7,8],[7,9]]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	w2, err := loadWorkload(noName)
	if err != nil {
		t.Fatalf("loadWorkload (no name): %v", err)
	}
	if w2.Name != "agents" {
		t.Errorf("name fallback = %q, want %q", w2.Name, "agents")
	}

	// An empty / malformed workload is a clear error, not a silent zero-request run.
	empty := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(empty, []byte(`{"requests":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadWorkload(empty); err == nil {
		t.Error("loadWorkload accepted a zero-request workload, want error")
	}
	if _, err := loadWorkload(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("loadWorkload accepted a missing file, want error")
	}
}

// TestParseTrace is the witness for the #3397 capture format: one request per non-blank
// line, space-separated non-negative integer ids, '#' comments and blank lines skipped —
// round-tripped into the exact [][]int shape radixMatched consumes. Malformed ids and an
// all-comment/blank trace are clear errors, not silent zero-request sweeps.
func TestParseTrace(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    [][]int
		wantErr bool
	}{
		{
			name: "round-trip with comments and blanks",
			in: "# captured 2026-07-11 from the gateway key side-stream\n" +
				"1 2 3 4 10\n" +
				"\n" +
				"   # indented comment between requests\n" +
				"1 2 3 4 11\n" +
				"7\n",
			want: [][]int{{1, 2, 3, 4, 10}, {1, 2, 3, 4, 11}, {7}},
		},
		// CRLF line endings and surrounding whitespace are an export reality, not an error.
		{"crlf and padding", "1 2\r\n  3 4  \r\n", [][]int{{1, 2}, {3, 4}}, false},
		{"zero id is a valid block id", "0 0 1\n", [][]int{{0, 0, 1}}, false},
		{"non-integer token", "1 two 3\n", nil, true},
		{"negative id", "1 -2 3\n", nil, true},
		{"empty input", "", nil, true},
		{"comments and blanks only", "# a\n\n# b\n", nil, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseTrace(strings.NewReader(tc.in))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseTrace(%q) = %v, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseTrace(%q): %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseTrace(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// TestSweepBudgets replays a crafted trace with KNOWN reuse through the sweep and checks
// the exact matched counts by hand. Two 4-token requests A and B alternate 3 times each:
//   - budget 4: only one request fits, so the alternation evicts the other on every
//     insert — every request misses (matched 0) and the last 5 inserts each evict one leaf;
//   - budget 8: both fit (the tree's 8 cached tokens are within budget, eviction fires
//     only ABOVE it), so the 4 repeat arrivals each match in full: 16 of 24 tokens;
//   - budget 0 (unbounded): the ceiling — same 16.
func TestSweepBudgets(t *testing.T) {
	a := []int{1, 1, 1, 1}
	b := []int{2, 2, 2, 2}
	reqs := [][]int{a, b, a, b, a, b} // 24 prompt tokens; 16 reusable when both fit

	rows := sweepBudgets(reqs, []int{4, 8, 0})
	if len(rows) != 3 {
		t.Fatalf("sweepBudgets returned %d rows, want 3", len(rows))
	}
	wantMatched := []int{0, 16, 16}
	wantEvict := []int{5, 0, 0}
	for i, r := range rows {
		if r.MatchedTokens != wantMatched[i] || r.Evictions != wantEvict[i] {
			t.Errorf("budget %d: matched=%d evictions=%d, want matched=%d evictions=%d",
				r.BudgetTokens, r.MatchedTokens, r.Evictions, wantMatched[i], wantEvict[i])
		}
		if r.TotalTokens != 24 {
			t.Errorf("budget %d: total=%d, want 24", r.BudgetTokens, r.TotalTokens)
		}
		if wantHit := float64(wantMatched[i]) / 24; math.Abs(r.HitRate-wantHit) > 1e-12 {
			t.Errorf("budget %d: hitRate=%v, want %v", r.BudgetTokens, r.HitRate, wantHit)
		}
		// The savings proxy is exactly matched tokens priced at dollarsPerMTok per million.
		if wantSave := float64(wantMatched[i]) / 1e6 * dollarsPerMTok; math.Abs(r.SavingsUSD-wantSave) > 1e-15 {
			t.Errorf("budget %d: savings=%v, want %v", r.BudgetTokens, r.SavingsUSD, wantSave)
		}
	}
	// Hit rate must be monotonically non-decreasing as the budget grows (0 = unbounded last).
	for i := 1; i < len(rows); i++ {
		if rows[i].HitRate < rows[i-1].HitRate {
			t.Errorf("hit rate fell as budget grew: budget %d -> %v then budget %d -> %v",
				rows[i-1].BudgetTokens, rows[i-1].HitRate, rows[i].BudgetTokens, rows[i].HitRate)
		}
	}
}

// TestSweepBudgetsSharedPrefix pins the exact matched count on a shared-prefix trace at
// generous budgets (no eviction): three requests share [1,2,3,4], so requests 2 and 3
// each reuse 4 tokens — 8 of 15 total — identically at budget 16 and unbounded. It also
// witnesses the trace-file path end to end: parseTrace output feeds sweepBudgets directly.
func TestSweepBudgetsSharedPrefix(t *testing.T) {
	reqs, err := parseTrace(strings.NewReader(
		"# three requests sharing a 4-token preamble\n" +
			"1 2 3 4 10\n" +
			"1 2 3 4 11\n" +
			"1 2 3 4 12\n"))
	if err != nil {
		t.Fatalf("parseTrace: %v", err)
	}
	rows := sweepBudgets(reqs, []int{16, 0})
	for _, r := range rows {
		if r.MatchedTokens != 8 || r.TotalTokens != 15 || r.Evictions != 0 {
			t.Errorf("budget %d: matched=%d/%d evictions=%d, want 8/15 and 0 evictions",
				r.BudgetTokens, r.MatchedTokens, r.TotalTokens, r.Evictions)
		}
	}
}

// TestSweepBudgetsMonotoneOnInterleavedWorkload sweeps the interleaved agents shape (the
// FCFS-thrash case) across an ascending ladder and asserts the curve an operator reads is
// monotone: more budget never buys a LOWER hit rate. Deterministic — the workload
// generator is seeded and the replay is pure accounting.
func TestSweepBudgetsMonotoneOnInterleavedWorkload(t *testing.T) {
	w := agents(256, 16, 4, 3, 4) // 3 agents, shared 16-token system prefix, interleaved arrivals
	l := maxReqLen(w.Requests)
	ladder := []int{l / 2, l, l + l/2, 2 * l, 4 * l, 0} // ascending; 0 = unbounded last
	rows := sweepBudgets(w.Requests, ladder)
	for i := 1; i < len(rows); i++ {
		if rows[i].HitRate < rows[i-1].HitRate {
			t.Errorf("hit rate fell as budget grew: budget %d -> %v then budget %d -> %v",
				rows[i-1].BudgetTokens, rows[i-1].HitRate, rows[i].BudgetTokens, rows[i].HitRate)
		}
	}
	// The unbounded row must equal the unbounded radixMatched accounting exactly.
	unbounded := rows[len(rows)-1]
	if wantMatched, _, _ := radixMatched(w.Requests, 0); unbounded.MatchedTokens != wantMatched {
		t.Errorf("unbounded row matched=%d, want %d (radixMatched at budget 0)", unbounded.MatchedTokens, wantMatched)
	}
}

// TestParseBudgetsAndDefaultLadder covers the two ways a sweep picks its budgets: the
// explicit -budgets CSV (order preserved, 0 allowed, junk refused) and the default ladder
// (anchored to max request length, deduped, always ending at 0 = unbounded).
func TestParseBudgetsAndDefaultLadder(t *testing.T) {
	got, err := parseBudgets(" 64, 128 ,0,256 ")
	if err != nil {
		t.Fatalf("parseBudgets: %v", err)
	}
	if want := []int{64, 128, 0, 256}; !reflect.DeepEqual(got, want) {
		t.Errorf("parseBudgets = %v, want %v", got, want)
	}
	for _, bad := range []string{"", " , ", "64,abc", "64,-1"} {
		if out, err := parseBudgets(bad); err == nil {
			t.Errorf("parseBudgets(%q) = %v, want error", bad, out)
		}
	}

	// maxReqLen = 8 => ladder 4, 8, 12, 16, 32, 64, then 0 (unbounded ceiling).
	reqs := [][]int{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2}}
	if got, want := defaultBudgetLadder(reqs), []int{4, 8, 12, 16, 32, 64, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("defaultBudgetLadder = %v, want %v", got, want)
	}
	// A 1-token trace collapses the low rungs (L/2 = 0 dropped, duplicates deduped).
	if got, want := defaultBudgetLadder([][]int{{9}}), []int{1, 2, 4, 8, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("defaultBudgetLadder(single token) = %v, want %v", got, want)
	}
}

func TestMaxTokenID(t *testing.T) {
	ws := []Workload{
		{Requests: [][]int{{1, 2}, {3, 255}}},
		{Requests: [][]int{{4}, {300, 5}}},
	}
	if got := maxTokenID(ws); got != 300 {
		t.Errorf("maxTokenID = %d, want 300", got)
	}
	if got := maxTokenID(nil); got != -1 {
		t.Errorf("maxTokenID(nil) = %d, want -1 (no ids)", got)
	}
}

func TestMaxReqLenAndTotalTokens(t *testing.T) {
	tests := []struct {
		name      string
		reqs      [][]int
		wantMax   int
		wantTotal int
	}{
		{"empty", nil, 0, 0},
		{"single", [][]int{{1, 2, 3}}, 3, 3},
		{"mixed", [][]int{{1, 2}, {1, 2, 3, 4}, {1}}, 4, 7},
		{"with-empty", [][]int{{}, {7, 7}, {}}, 2, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := maxReqLen(tc.reqs); got != tc.wantMax {
				t.Errorf("maxReqLen(%v) = %d, want %d", tc.reqs, got, tc.wantMax)
			}
			if got := totalTokens(tc.reqs); got != tc.wantTotal {
				t.Errorf("totalTokens(%v) = %d, want %d", tc.reqs, got, tc.wantTotal)
			}
		})
	}
}
