// Package main tests for the pure slice utilities used by the deletioncert
// demonstrator. These functions are deterministic and resource-free: no model,
// no journal, no I/O — so they are unit-testable in isolation with hand-computed
// expected values.
package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestArgmax pins argmax's exact contract: it scans with a strict '>' from an
// initial best of -1e30 at index 0, so it returns the FIRST index of the maximum
// value and returns 0 for an empty slice.
func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{"single", []float32{0}, 0},
		{"ascending", []float32{1, 3, 2}, 1},
		{"max at end", []float32{1, 2, 9}, 2},
		{"first of ties", []float32{5, 5, 1}, 0}, // strict '>' keeps the earliest max
		{"all negative", []float32{-2, -1, -3}, 1},
		{"single negative below init is still picked", []float32{-5}, 0},
		{"empty returns zero", []float32{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argmax(tt.in); got != tt.want {
				t.Fatalf("argmax(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

// TestMaxAbsIntDelta checks the deletion-property witness helper: equal-length
// slices yield the maximum element-wise absolute difference; a length mismatch
// yields the 1<<30 sentinel.
func TestMaxAbsIntDelta(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want int
	}{
		{"identical is zero", []int{3, 3}, []int{3, 3}, 0},
		{"max in second slot", []int{1, 5}, []int{4, 1}, 4}, // |1-4|=3, |5-1|=4
		{"negative diff abs", []int{0}, []int{7}, 7},
		{"sign both directions", []int{-3, 2}, []int{3, -2}, 6}, // |-3-3|=6, |2-(-2)|=4
		{"length mismatch sentinel", []int{1, 2, 3}, []int{1, 2}, 1 << 30},
		{"both empty is zero", []int{}, []int{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := maxAbsIntDelta(tt.a, tt.b); got != tt.want {
				t.Fatalf("maxAbsIntDelta(%v, %v) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// TestEqualInts verifies element-wise equality with a length guard.
func TestEqualInts(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"equal", []int{1, 2, 3}, []int{1, 2, 3}, true},
		{"differ in middle", []int{1, 9, 3}, []int{1, 2, 3}, false},
		{"length mismatch", []int{1, 2}, []int{1, 2, 3}, false},
		{"both empty", []int{}, []int{}, true},
		{"empty vs nonempty", []int{}, []int{1}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := equalInts(tt.a, tt.b); got != tt.want {
				t.Fatalf("equalInts(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// TestConcat checks the variadic flatten preserves order across all parts.
func TestConcat(t *testing.T) {
	got := concat([]int{3, 17, 5}, []int{41, 2}, []int{23, 11})
	want := []int{3, 17, 5, 41, 2, 23, 11}
	if !equalInts(got, want) {
		t.Fatalf("concat = %v, want %v", got, want)
	}

	// An empty middle part contributes nothing and order is kept.
	got2 := concat([]int{1}, []int{}, []int{2, 3})
	want2 := []int{1, 2, 3}
	if !equalInts(got2, want2) {
		t.Fatalf("concat with empty middle = %v, want %v", got2, want2)
	}

	// No parts yields an empty (length-0) result.
	if got3 := concat(); len(got3) != 0 {
		t.Fatalf("concat() = %v, want empty", got3)
	}
}

func TestDeletionCertificateSelfcheckRuns(t *testing.T) {
	if err := run(""); err != nil {
		t.Fatalf("deletioncert selfcheck failed: %v", err)
	}
}

// TestProveL3DeletionRung (#56) gates the L3 page-key rung wired into -selfcheck: mint +
// verify over the in-process mock pool succeeds, and the three fail-closed tamper checks
// (still-resident page, forged key state, over-claim scope) all return errors. A
// regression that lets any tamper pass reds CI here rather than slipping through the demo.
func TestProveL3DeletionRung(t *testing.T) {
	if err := proveL3DeletionRung("sha256:l3-rung-selfcheck-subject"); err != nil {
		t.Fatalf("L3 deletion rung selfcheck failed: %v", err)
	}
}

// --- isolation benchmark (#1065) --------------------------------------------
// These tests make the per-tenant KV cache-isolation benchmark CI-gradeable under
// `go test ./...` (a `make ci` step): the oracle, the leaky-baseline discrimination,
// and the three honesty fences are all asserted here, so a regression in the gate or
// a dishonest corpus/scope edit reds CI rather than slipping through a self-graded run.

// TestIsolationBenchPasses pins the full oracle: every adversarial read-back case is
// admitted-or-refused exactly as its (scope, owner, reader) tuple requires, so the
// run is valid with no failed cases.
func TestIsolationBenchPasses(t *testing.T) {
	r := computeIsolationBench(42)
	if !r.Valid {
		t.Fatalf("isolation bench not valid: %d/%d passed, failures=%+v", r.PassedCases, r.CorpusSize, r.FailedCases)
	}
	if r.PassedCases != r.CorpusSize {
		t.Fatalf("passed %d of %d cases; want all", r.PassedCases, r.CorpusSize)
	}
	if len(r.FailedCases) != 0 {
		t.Fatalf("unexpected failed cases: %+v", r.FailedCases)
	}
}

// TestIsolationBenchBaselineDiscriminates is the deliverable: the SAME corpus run
// against the leaky (non-isolating) baseline MUST surface leaks. A benchmark that only
// ever passes on its author's happy path measures nothing.
func TestIsolationBenchBaselineDiscriminates(t *testing.T) {
	r := computeIsolationBench(42)
	if r.BaselineFailedCases == 0 {
		t.Fatalf("leaky baseline leaked nothing — metric does not discriminate")
	}
}

// TestIsolationBenchScopeIsHonest enforces AC #6: the emitted scope is the honest
// l3-working-set token and no over-claim string (weights/backups/replicas/embeddings/
// deleted-everywhere) appears anywhere in the artifact.
func TestIsolationBenchScopeIsHonest(t *testing.T) {
	r := computeIsolationBench(42)
	if r.Scope != "l3-working-set" {
		t.Fatalf("scope = %q, want l3-working-set", r.Scope)
	}
	if err := assertHonestArtifact(r); err != nil {
		t.Fatalf("honest-artifact fence failed: %v", err)
	}
	b, _ := json.Marshal(r)
	for _, tok := range bannedScopeTokens {
		if strings.Contains(string(b), tok) {
			t.Fatalf("emitted artifact contains over-claim token %q", tok)
		}
	}
}

// TestIsolationBenchControlPathNoPayload enforces AC #8: no page payload byte from the
// corpus may surface in the emitted artifact — only digests, scopes, tags, and counts.
func TestIsolationBenchControlPathNoPayload(t *testing.T) {
	r := computeIsolationBench(42)
	b, _ := json.Marshal(r)
	blob := string(b)
	for _, tc := range isolationCorpus {
		if strings.Contains(blob, tc.content) {
			t.Fatalf("emitted artifact leaked payload bytes for case %q", tc.name)
		}
	}
}

// TestIsolationBenchBoundaryStated enforces AC #9: the verbatim structural-floor
// boundary travels WITH the score, never buried in adjacent prose.
func TestIsolationBenchBoundaryStated(t *testing.T) {
	r := computeIsolationBench(42)
	if !strings.Contains(r.Boundary, "structural floor") ||
		!strings.Contains(r.Boundary, "NOT a public-leaderboard rank") {
		t.Fatalf("boundary missing the verbatim structural-floor statement: %q", r.Boundary)
	}
}

// TestIsolationBenchHasConcurrentDimension enforces AC #7: the corpus includes
// interleaved multi-tenant delete/read cases (the mem0 concurrent-load gap).
func TestIsolationBenchHasConcurrentDimension(t *testing.T) {
	n := 0
	for _, tc := range isolationCorpus {
		if strings.HasPrefix(tc.name, "concurrent-") {
			n++
		}
	}
	if n == 0 {
		t.Fatalf("corpus has no concurrent-load cases")
	}
}

// TestIsolationBenchDeterministic pins reproducibility: a fixed seed yields a
// byte-identical artifact, so a third party re-runs it exactly.
func TestIsolationBenchDeterministic(t *testing.T) {
	a, _ := json.Marshal(computeIsolationBench(42))
	b, _ := json.Marshal(computeIsolationBench(42))
	if string(a) != string(b) {
		t.Fatalf("isolation bench is not deterministic for a fixed seed")
	}
}

// TestRunIsolationBenchExitsCleanly is the end-to-end smoke: the wrapped runner (with
// its fail-closed checks) completes without error on the shipped corpus.
func TestRunIsolationBenchExitsCleanly(t *testing.T) {
	if err := runIsolationBench("", 42); err != nil {
		t.Fatalf("runIsolationBench failed: %v", err)
	}
}
