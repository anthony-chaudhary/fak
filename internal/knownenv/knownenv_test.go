package knownenv

import (
	"strings"
	"testing"
)

// TestAnnotateRsync23Witness is the issue's (#2144) primary witness: a WSL rsync
// exit-23 failure, run through the default registry, returns annotated
// "known env failure #<id>, not your diff" — the exact inline signal the agent
// reads to skip the dead-end instead of debugging a tool it cannot fix.
func TestAnnotateRsync23Witness(t *testing.T) {
	// The bytes a real WSL go-test capture emits when the rsync sink aborts.
	output := "rsync: [sender] write error: Broken pipe (32)\n" +
		"rsync error: error in socket IO (code 23) at io.c(line 820)\n"
	got := Annotate(output, 23, DefaultRegistry())
	if len(got) != 1 {
		t.Fatalf("rsync-23 output should match exactly one signature, got %d: %+v", len(got), got)
	}
	if got[0].ID != "rsync-23" {
		t.Fatalf("matched the wrong signature: %q", got[0].ID)
	}
	if got[0].Verdict != VerdictKnownEnv {
		t.Fatalf("verdict = %q, want %q", got[0].Verdict, VerdictKnownEnv)
	}
	// The banner MUST carry the witness phrase verbatim so an agent (or a human
	// scanning the transcript) reads "not your diff" inline.
	if !strings.Contains(got[0].Line, "known env failure #rsync-23, not your diff") {
		t.Fatalf("banner missing the witness phrase: %q", got[0].Line)
	}
	t.Logf("WITNESS: %s", got[0].Line)
}

// TestAnnotateTierDeclaredWitness is the issue's second named witness: a
// peer-owned architest TIER_DECLARED red — which reds the trunk for everyone —
// returns annotated as a known env failure, not the reader's diff.
func TestAnnotateTierDeclaredWitness(t *testing.T) {
	output := "--- FAIL: TestEveryPackageDeclaresTier\n" +
		"    architest_test.go:210: ARCH_LAYER_VIOLATION TIER_DECLARED: package \"peerleaf\" declares no tier\n"
	got := Annotate(output, 1, DefaultRegistry())
	if len(got) != 1 || got[0].ID != "architest-tier-drift" {
		t.Fatalf("TIER_DECLARED red should match architest-tier-drift, got %+v", got)
	}
	if !strings.Contains(got[0].Line, "not your diff") {
		t.Fatalf("banner missing not-your-diff signal: %q", got[0].Line)
	}
	t.Logf("WITNESS: %s", got[0].Line)
}

// TestAnnotateCleanOutputIsBlameless is the load-bearing NEGATIVE: an ordinary
// test failure that is NOT a known environment flake returns NO annotation, so the
// registry can never tell an agent "not your fault" about a defect that IS its
// own. A false "not your fault" is worse than no signal — it masks a real bug.
func TestAnnotateCleanOutputIsBlameless(t *testing.T) {
	output := "--- FAIL: TestMyFeature\n    my_test.go:42: got 3, want 4\n"
	if got := Annotate(output, 1, DefaultRegistry()); len(got) != 0 {
		t.Fatalf("a normal test failure must not be annotated environmental, got %+v", got)
	}
}

// TestMatchAndConditions proves the AND semantics: a signature declaring BOTH a
// needle and an exit code matches only when BOTH hold, so a needle appearing under
// a different exit code (or vice versa) does not misfire.
func TestMatchAndConditions(t *testing.T) {
	ec := 23
	sig := Signature{ID: "x", Needle: "code 23", ExitCode: &ec, Verdict: VerdictKnownEnv}
	cases := []struct {
		name    string
		output  string
		exit    int
		wantHit bool
	}{
		{"both hold", "boom code 23 here", 23, true},
		{"needle only, wrong exit", "boom code 23 here", 1, false},
		{"exit only, no needle", "unrelated failure", 23, false},
		{"neither", "unrelated failure", 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sig.Match(tc.output, tc.exit); got != tc.wantHit {
				t.Fatalf("Match(%q, %d) = %v, want %v", tc.output, tc.exit, got, tc.wantHit)
			}
		})
	}
}

// TestUnmatchableSignatureNeverFires guards the catch-all: a signature with
// neither a needle nor an exit code must never match, so a blank/malformed row
// cannot flag every failure as environmental.
func TestUnmatchableSignatureNeverFires(t *testing.T) {
	blank := Signature{ID: "blank", Verdict: VerdictKnownEnv}
	if blank.Matchable() {
		t.Fatal("a signature with no needle and no exit code must not be Matchable")
	}
	if blank.Match("anything at all", 0) {
		t.Fatal("an unmatchable signature must never match")
	}
	if got := Annotate("anything", 0, []Signature{blank}); len(got) != 0 {
		t.Fatalf("an unmatchable signature must produce no annotation, got %+v", got)
	}
}

// TestExitCodeZeroIsDistinctFromUnset proves the pointer design: ExitCode == 0 (a
// real exit code) matches only exit 0, and is not confused with "no exit
// condition". Guards the subtle nil-vs-zero bug the *int design exists to prevent.
func TestExitCodeZeroIsDistinctFromUnset(t *testing.T) {
	zero := 0
	sig := Signature{ID: "z", ExitCode: &zero, Verdict: VerdictKnownEnv}
	if !sig.Match("clean but flagged", 0) {
		t.Fatal("ExitCode=0 must match exit 0")
	}
	if sig.Match("clean but flagged", 1) {
		t.Fatal("ExitCode=0 must NOT match exit 1")
	}
}

// TestParseRegistryRobustness proves the shared-ledger robustness: blank lines,
// non-JSON lines, foreign-schema rows, and unmatchable rows are all dropped, while
// a well-formed row round-trips through MarshalLine -> ParseRegistry.
func TestParseRegistryRobustness(t *testing.T) {
	good := Signature{Schema: Schema, ID: "seat", Needle: "STALE_CRED", Verdict: VerdictKnownEnv}
	line, err := MarshalLine(good)
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	data := strings.Join([]string{
		"",                                     // blank
		"not json at all",                      // torn append
		`{"schema":"other.v1","id":"foreign"}`, // foreign schema
		`{"schema":"fak.known-env.v1","id":"empty"}`, // right schema, unmatchable -> dropped
		line, // the one good row
	}, "\n")
	got := ParseRegistry([]byte(data))
	if len(got) != 1 {
		t.Fatalf("expected exactly the one good row to survive, got %d: %+v", len(got), got)
	}
	if got[0].ID != "seat" || got[0].Needle != "STALE_CRED" {
		t.Fatalf("round-trip corrupted the row: %+v", got[0])
	}
}

// TestDefaultRegistryIsWellFormed keeps the compiled seed honest: every seeded
// signature is matchable, carries a stable id and the schema tag, and has a
// verdict — so DefaultRegistry can never ship a dead or blank row.
func TestDefaultRegistryIsWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, s := range DefaultRegistry() {
		if strings.TrimSpace(s.ID) == "" {
			t.Fatalf("seed signature with empty id: %+v", s)
		}
		if seen[s.ID] {
			t.Fatalf("duplicate seed id %q", s.ID)
		}
		seen[s.ID] = true
		if s.Schema != Schema {
			t.Fatalf("seed %q missing schema tag", s.ID)
		}
		if !s.Matchable() {
			t.Fatalf("seed %q is unmatchable — it would never fire", s.ID)
		}
		if strings.TrimSpace(s.Verdict) == "" {
			t.Fatalf("seed %q missing verdict", s.ID)
		}
	}
}
