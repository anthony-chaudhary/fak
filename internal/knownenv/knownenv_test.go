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

// TestEffectiveRegistrySeedOnly proves the floor: with an empty fleet, the
// effective registry is exactly the matchable seed rows, in seed order — merging
// against nothing changes nothing.
func TestEffectiveRegistrySeedOnly(t *testing.T) {
	seed := DefaultRegistry()
	got := EffectiveRegistry(seed, nil)
	if len(got) != len(seed) {
		t.Fatalf("empty fleet must yield exactly the seed, got %d rows want %d", len(got), len(seed))
	}
	for i := range seed {
		if got[i].ID != seed[i].ID {
			t.Fatalf("seed order disturbed at %d: got %q want %q", i, got[i].ID, seed[i].ID)
		}
	}
}

// TestEffectiveRegistryFleetExtendsWithNewID proves the extension: a fleet row
// with a NEW id is appended after all seed rows, so the fleet can teach the
// registry failures the compiled seed does not know.
func TestEffectiveRegistryFleetExtendsWithNewID(t *testing.T) {
	seed := DefaultRegistry()
	fleet := []Signature{{Schema: Schema, ID: "gpu-tdr", Needle: "TDR_TIMEOUT", Verdict: VerdictKnownEnv}}
	got := EffectiveRegistry(seed, fleet)
	if len(got) != len(seed)+1 {
		t.Fatalf("new-id fleet row must extend the registry: got %d rows want %d", len(got), len(seed)+1)
	}
	if last := got[len(got)-1]; last.ID != "gpu-tdr" {
		t.Fatalf("fleet row must append after the seed, tail = %q", last.ID)
	}
}

// TestEffectiveRegistryFleetOverridesSameID proves the refine semantics: a fleet
// row sharing a seed id replaces that seed row's FIELDS but keeps its POSITION —
// the floor slot is stable, the fleet just updates its contents.
func TestEffectiveRegistryFleetOverridesSameID(t *testing.T) {
	seed := DefaultRegistry()
	slot := -1
	for i, s := range seed {
		if s.ID == "rsync-23" {
			slot = i
		}
	}
	if slot < 0 {
		t.Fatal("seed is missing rsync-23 — the override witness needs it")
	}
	fleet := []Signature{{
		Schema:  Schema,
		ID:      "rsync-23",
		Needle:  "code 23",
		Verdict: VerdictKnownEnv,
		Owner:   "fleet-refreshed",
		Note:    "refined by the fleet ledger",
	}}
	got := EffectiveRegistry(seed, fleet)
	if len(got) != len(seed) {
		t.Fatalf("same-id override must not grow the registry: got %d rows want %d", len(got), len(seed))
	}
	if got[slot].ID != "rsync-23" {
		t.Fatalf("overridden row moved: slot %d holds %q", slot, got[slot].ID)
	}
	if got[slot].Owner != "fleet-refreshed" || got[slot].Note != "refined by the fleet ledger" {
		t.Fatalf("slot %d must carry the FLEET row's fields, got owner=%q note=%q", slot, got[slot].Owner, got[slot].Note)
	}
}

// TestEffectiveRegistryDropsUnmatchableAndBlankID guards the entry gate: a fleet
// row declaring no condition (no needle, nil exit code) and a fleet row with a
// blank id are both refused — neither can enter the effective registry, so a
// blank row can neither fire nor blank out a seed floor entry.
func TestEffectiveRegistryDropsUnmatchableAndBlankID(t *testing.T) {
	seed := DefaultRegistry()
	fleet := []Signature{
		{Schema: Schema, ID: "catch-all", Verdict: VerdictKnownEnv},           // unmatchable: no needle, nil exit
		{Schema: Schema, ID: "   ", Needle: "boom", Verdict: VerdictKnownEnv}, // blank id: banner would cite nothing
		{Schema: Schema, ID: "rsync-23", Verdict: VerdictKnownEnv},            // unmatchable same-id: must NOT displace the seed floor
	}
	got := EffectiveRegistry(seed, fleet)
	if len(got) != len(seed) {
		t.Fatalf("dropped rows must not change the registry size: got %d rows want %d", len(got), len(seed))
	}
	for _, s := range got {
		if s.ID == "catch-all" || strings.TrimSpace(s.ID) == "" {
			t.Fatalf("a dropped row leaked into the effective registry: %+v", s)
		}
		if s.ID == "rsync-23" && !s.Matchable() {
			t.Fatalf("an unmatchable fleet row displaced the seed floor entry: %+v", s)
		}
	}
}

// TestAnnotateFromLedgerMergesSeedAndFleet proves the one-call entrypoint end to
// end: fleet JSONL bytes teach it a NEW signature it then annotates by, and nil
// bytes degrade cleanly to the compiled seed alone.
func TestAnnotateFromLedgerMergesSeedAndFleet(t *testing.T) {
	line, err := MarshalLine(Signature{Schema: Schema, ID: "gpu-tdr", Needle: "TDR_TIMEOUT", Verdict: VerdictKnownEnv})
	if err != nil {
		t.Fatalf("MarshalLine: %v", err)
	}
	got := AnnotateFromLedger("display driver reset: TDR_TIMEOUT during kernel launch", 1, []byte(line+"\n"))
	if len(got) != 1 || got[0].ID != "gpu-tdr" {
		t.Fatalf("fleet-only signature should annotate via the ledger bytes, got %+v", got)
	}
	if !strings.Contains(got[0].Line, "known env failure #gpu-tdr, not your diff") {
		t.Fatalf("banner missing the witness phrase: %q", got[0].Line)
	}
	// Nil ledger bytes: the seed floor alone still recognizes its own witnesses.
	seedOnly := AnnotateFromLedger("rsync error: error in socket IO (code 23)", 23, nil)
	if len(seedOnly) != 1 || seedOnly[0].ID != "rsync-23" {
		t.Fatalf("nil ledger must degrade to the seed alone, got %+v", seedOnly)
	}
}

// TestEffectiveRegistryDoesNotMutateInputs proves the fold is pure: merging
// returns a fresh slice and leaves both input slices byte-identical — a caller
// can reuse its seed and fleet slices without defensive copies.
func TestEffectiveRegistryDoesNotMutateInputs(t *testing.T) {
	snapshot := func(sigs []Signature) []string {
		var lines []string
		for _, s := range sigs {
			line, err := MarshalLine(s)
			if err != nil {
				t.Fatalf("MarshalLine: %v", err)
			}
			lines = append(lines, line)
		}
		return lines
	}
	seed := DefaultRegistry()
	fleet := []Signature{
		{Schema: Schema, ID: "rsync-23", Needle: "code 23", Verdict: VerdictKnownEnv, Note: "override"},
		{Schema: Schema, ID: "gpu-tdr", Needle: "TDR_TIMEOUT", Verdict: VerdictKnownEnv},
	}
	seedBefore, fleetBefore := snapshot(seed), snapshot(fleet)
	_ = EffectiveRegistry(seed, fleet)
	seedAfter, fleetAfter := snapshot(seed), snapshot(fleet)
	if len(seedAfter) != len(seedBefore) || len(fleetAfter) != len(fleetBefore) {
		t.Fatal("EffectiveRegistry changed an input slice's length")
	}
	for i := range seedBefore {
		if seedAfter[i] != seedBefore[i] {
			t.Fatalf("seed[%d] mutated:\n before %s\n after  %s", i, seedBefore[i], seedAfter[i])
		}
	}
	for i := range fleetBefore {
		if fleetAfter[i] != fleetBefore[i] {
			t.Fatalf("fleet[%d] mutated:\n before %s\n after  %s", i, fleetBefore[i], fleetAfter[i])
		}
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
