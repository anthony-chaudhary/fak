package knobcensus

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const fixtureRoot = "testdata/repo"

// TestScanFixture pins the walker against a known fixture tree: three INTENT
// knobs, three HOUSEKEEPING knobs (one folded from #2199's context inventory),
// and none of the excluded plumbing/output knobs.
func TestScanFixture(t *testing.T) {
	census, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if census.Intent != 3 {
		t.Errorf("Intent = %d, want 3", census.Intent)
	}
	if census.Housekeeping != 3 {
		t.Errorf("Housekeeping = %d, want 3", census.Housekeeping)
	}
	// Every scanned knob lands in exactly one verdict bucket (INTENT or
	// HOUSEKEEPING; excluded plumbing/output knobs are dropped before the list),
	// so the total partitions cleanly into the two counts — nothing uncounted or
	// double-counted. Asserted as that partition, not a frozen total.
	if len(census.Knobs) != census.Intent+census.Housekeeping {
		t.Fatalf("len(Knobs) = %d, want intent(%d)+housekeeping(%d): %+v", len(census.Knobs), census.Intent, census.Housekeeping, census.Knobs)
	}

	got := map[string]Knob{}
	for _, k := range census.Knobs {
		got[k.Key()] = k
	}
	for _, want := range []struct {
		key     string
		verdict Verdict
	}{
		{"flag:account", Intent},
		{"flag:account-refresh", Intent}, // strong-intent token overrides "refresh"
		{"env:FAK_GOAL_OBJECTIVE", Intent},
		{"flag:session-cooldown-ttl", Housekeeping},
		{"flag:ctx-view-budget", Housekeeping}, // folded from #2199, not re-derived
		{"env:FAK_GUARD_AUTO_REFRESH", Housekeeping},
	} {
		k, ok := got[want.key]
		if !ok {
			t.Errorf("missing expected knob %q", want.key)
			continue
		}
		if k.Verdict != want.verdict {
			t.Errorf("%q verdict = %q, want %q", want.key, k.Verdict, want.verdict)
		}
		if k.Disposition != k.Verdict.Disposition() || k.OwnerEpic != k.Verdict.OwnerEpic() {
			t.Errorf("%q disposition/epic disagree with verdict: %+v", want.key, k)
		}
		if k.File == "" || k.Line == 0 {
			t.Errorf("%q missing file:line provenance: %+v", want.key, k)
		}
	}
	// The over-match guards: none of these gate user behavior; none may appear.
	for _, bad := range []string{"flag:json", "flag:root", "env:FAK_ADDR"} {
		if _, ok := got[bad]; ok {
			t.Errorf("walker over-matched a non-behavior knob: %q", bad)
		}
	}
}

// TestScanDeterministic is the "run the verb twice → identical output" witness at
// the walker level: two scans of the same tree marshal to byte-identical JSON.
func TestScanDeterministic(t *testing.T) {
	a, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan a: %v", err)
	}
	b, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan b: %v", err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("scan not deterministic:\n a=%s\n b=%s", ja, jb)
	}
}

// TestRouteCoverageFolds is the cross-surface witness: a control exposed as both
// a flag ("model") and an env ("FAK_MODEL") is one knob whose route covers both
// surfaces, while a single-surface control names only its own surface. This is the
// route-coverage column the issue's "How" calls for and its "each INTENT row names
// its route" witness.
func TestRouteCoverageFolds(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"),
		[]byte("[lanes.trees]\ncmd = [\"cmd/fak/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package main\n\nimport (\n\t\"flag\"\n\t\"os\"\n)\n\n" +
		"func sample() {\n" +
		"\tfs := flag.NewFlagSet(\"s\", flag.ContinueOnError)\n" +
		"\tfs.String(\"model\", \"\", \"which model\")\n" + // INTENT, flag surface
		"\t_ = os.Getenv(\"FAK_MODEL\")\n" + // same control, env surface
		"\tfs.String(\"account\", \"\", \"which account\")\n" + // INTENT, single surface
		"\t_ = fs\n}\n"
	if err := os.WriteFile(filepath.Join(dir, "sample.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	census, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	route := map[string][]Surface{}
	for _, k := range census.Knobs {
		route[k.Key()] = k.RouteCoverage
	}
	// "model" folds flag+env into one two-surface route on BOTH rows.
	for _, key := range []string{"flag:model", "env:FAK_MODEL"} {
		r := route[key]
		if len(r) != 2 || r[0] != SurfaceEnv || r[1] != SurfaceFlag {
			t.Errorf("%s route_coverage = %v, want [env flag]", key, r)
		}
	}
	// "account" is exposed on one surface only — the promotion gap #2208 tracks.
	if r := route["flag:account"]; len(r) != 1 || r[0] != SurfaceFlag {
		t.Errorf("flag:account route_coverage = %v, want [flag]", r)
	}
}

// TestVerdictDisposition pins the closed vocabulary → disposition/owner mapping.
func TestVerdictDisposition(t *testing.T) {
	if Intent.Disposition() != "promote" || Intent.OwnerEpic() != "#2208" {
		t.Errorf("INTENT should promote under #2208, got %s / %s", Intent.Disposition(), Intent.OwnerEpic())
	}
	if Housekeeping.Disposition() != "automate" || Housekeeping.OwnerEpic() != "#2198" {
		t.Errorf("HOUSEKEEPING should automate under #2198, got %s / %s", Housekeeping.Disposition(), Housekeeping.OwnerEpic())
	}
}
