package bench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestNegBench exercises all four negation task families twice: once with the
// reference oracle (every family must score a perfect pass rate, pinning the
// scorer's deterministic behaviour) and once with the negation-blind degenerate
// set (the response-driven families must show failures, proving the scorer
// discriminates rather than always passing).
func TestNegBench(t *testing.T) {
	fams := ScoreNegBench(NegBenchReferenceResponses())
	if len(fams) != len(NegBenchFamilies) {
		t.Fatalf("got %d families, want %d", len(fams), len(NegBenchFamilies))
	}
	seen := map[string]NegBenchFamilyResult{}
	for _, f := range fams {
		seen[f.Family] = f
	}
	for _, name := range NegBenchFamilies {
		f, ok := seen[name]
		if !ok {
			t.Fatalf("family %q missing from results", name)
		}
		if f.Total == 0 {
			t.Fatalf("family %q has no fixtures", name)
		}
		if f.Passed != f.Total || f.PassRate != 1.0 {
			for _, it := range f.Items {
				if !it.Pass {
					t.Logf("family %s item %s failed: %s (resp=%q)", name, it.ID, it.Why, it.Response)
				}
			}
			t.Fatalf("family %q: reference oracle scored %d/%d (rate %.3f), want all pass",
				name, f.Passed, f.Total, f.PassRate)
		}
	}

	// The negation-blind trap must fail the three response-driven families.
	deg := ScoreNegBench(NegBenchDegenerateResponses())
	degSeen := map[string]NegBenchFamilyResult{}
	for _, f := range deg {
		degSeen[f.Family] = f
	}
	for _, name := range []string{NegBenchCloze, NegBenchQA, NegBenchAdherence} {
		f := degSeen[name]
		if f.Passed >= f.Total {
			t.Fatalf("family %q: degenerate set scored %d/%d — scorer does not discriminate negation",
				name, f.Passed, f.Total)
		}
	}
	// De Morgan is response-independent: it must remain fully passing regardless.
	if dm := degSeen[NegBenchDeMorgan]; dm.Passed != dm.Total {
		t.Fatalf("de morgan family is response-independent but scored %d/%d", dm.Passed, dm.Total)
	}
}

// TestNegBenchDeMorganEvaluator guards the tiny boolean evaluator that backs the
// De Morgan family: the equivalence checker must accept true De Morgan pairs and
// reject the deliberately-wrong rewrite fixture.
func TestNegBenchDeMorganEvaluator(t *testing.T) {
	ok, err := exprsEquivalent([]string{"a", "b"}, "!(a & b)", "!a | !b")
	if err != nil || !ok {
		t.Fatalf("valid De Morgan pair not equivalent: ok=%v err=%v", ok, err)
	}
	bad, err := exprsEquivalent([]string{"a", "b"}, "!(a & b)", "!a & !b")
	if err != nil || bad {
		t.Fatalf("wrong rewrite scored equivalent: bad=%v err=%v", bad, err)
	}
	if _, err := negBenchEvalBool("a & ", map[string]bool{"a": true}); err == nil {
		t.Fatal("malformed expression accepted")
	}
	if _, err := negBenchEvalBool("a & c", map[string]bool{"a": true}); err == nil {
		t.Fatal("unbound variable accepted")
	}
}

// TestNegBenchArtifact is the env-gated OBSERVED-witness arm. It stays a no-op
// under a bare `go test` (fast) and, when FAK_NEGBENCH_OUT is set, scores the
// named model's responses (or the reference oracle if none supplied) and writes
// a self-verifying JSON witness. It flips no gate.
//
//	FAK_NEGBENCH_OUT=/tmp/negbench-opus.json FAK_NEGBENCH_MODEL=claude-opus-4-8 \
//	FAK_NEGBENCH_RESPONSES=/tmp/responses.json \
//	  go test ./internal/bench -run TestNegBenchArtifact -count=1
func TestNegBenchArtifact(t *testing.T) {
	out := os.Getenv("FAK_NEGBENCH_OUT")
	if out == "" {
		t.Skip("set FAK_NEGBENCH_OUT to write the negbench witness artifact")
	}
	model := os.Getenv("FAK_NEGBENCH_MODEL")
	host := os.Getenv("FAK_NEGBENCH_HW")

	responses := NegBenchReferenceResponses()
	if rf := os.Getenv("FAK_NEGBENCH_RESPONSES"); rf != "" {
		raw, err := os.ReadFile(rf)
		if err != nil {
			t.Fatalf("read responses: %v", err)
		}
		responses = map[string]string{}
		if err := json.Unmarshal(raw, &responses); err != nil {
			t.Fatalf("parse responses json ({id: response}): %v", err)
		}
		if model == "" {
			t.Fatal("FAK_NEGBENCH_MODEL must name the model when supplying live responses")
		}
	}
	if model == "" {
		model = "reference-oracle"
	}

	art := BuildNegBenchArtifact(model, host, responses)
	if art.Provenance != NegBenchObservedProvenance {
		t.Fatalf("provenance=%q want OBSERVED", art.Provenance)
	}
	if art.Enforced {
		t.Fatal("negbench must enforce no floor")
	}
	if len(art.Families) != len(NegBenchFamilies) {
		t.Fatalf("artifact has %d families, want %d", len(art.Families), len(NegBenchFamilies))
	}
	if !VerifyNegBenchArtifact(art) {
		t.Fatal("artifact digest does not self-verify")
	}

	enc, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, enc, 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	// Re-read and re-verify from disk so the witness is trustworthy end-to-end.
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var round NegBenchArtifact
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatal(err)
	}
	if !VerifyNegBenchArtifact(round) {
		t.Fatal("artifact does not self-verify after round-trip")
	}
	t.Logf("NEGBENCH model=%s host=%s -> %s", art.Model, art.Host, filepath.Clean(out))
	for _, f := range art.Families {
		t.Logf("  %-24s %d/%d (%.3f)", f.Family, f.Passed, f.Total, f.PassRate)
	}
}
