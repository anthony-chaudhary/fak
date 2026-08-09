package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeKnobsRepo lays down a minimal dos.toml + a cmd/fak source with one INTENT
// knob (account) and two HOUSEKEEPING knobs (a cooldown-ttl flag + an env), plus a
// plumbing flag the census must exclude, so `fak index knobs` is tested against
// known bytes.
func writeKnobsRepo(t *testing.T) string {
	t.Helper()
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
		"\tfs.String(\"account\", \"\", \"which account\")\n" +
		"\tfs.Duration(\"cooldown-ttl\", 0, \"warmth ttl\")\n" +
		"\tfs.Bool(\"json\", false, \"plumbing\")\n" +
		"\t_ = os.Getenv(\"FAK_GUARD_AUTO_REFRESH\")\n" +
		"\t_ = fs\n}\n"
	if err := os.WriteFile(filepath.Join(dir, "sample.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestIndexKnobsJSON(t *testing.T) {
	root := writeKnobsRepo(t)
	var out, errb bytes.Buffer
	if rc := RunIndex(&out, &errb, []string{"knobs", "--json", "--root", root}); rc != 0 {
		t.Fatalf("runIndex knobs --json rc=%d, stderr=%s", rc, errb.String())
	}
	var census struct {
		Knobs []struct {
			Name          string   `json:"name"`
			Surface       string   `json:"surface"`
			Verdict       string   `json:"verdict"`
			Disposition   string   `json:"disposition"`
			OwnerEpic     string   `json:"owner_epic"`
			RouteCoverage []string `json:"route_coverage"`
		} `json:"knobs"`
		Intent       int `json:"intent"`
		Housekeeping int `json:"housekeeping"`
	}
	if err := json.Unmarshal(out.Bytes(), &census); err != nil {
		t.Fatalf("knobs --json is not valid JSON: %v\n%s", err, out.String())
	}
	if census.Intent != 1 || census.Housekeeping != 2 {
		t.Fatalf("counts = INTENT %d / HOUSEKEEPING %d, want 1 / 2: %+v", census.Intent, census.Housekeeping, census.Knobs)
	}
	got := map[string]string{}
	route := map[string][]string{}
	for _, k := range census.Knobs {
		got[k.Surface+":"+k.Name] = k.Verdict
		route[k.Surface+":"+k.Name] = k.RouteCoverage
	}
	if got["flag:account"] != "INTENT" {
		t.Errorf("flag:account verdict = %q, want INTENT", got["flag:account"])
	}
	// Route-coverage: this fixture exposes each knob on a single surface, so every
	// row names exactly its own surface (an INTENT knob on one surface is the
	// promotion gap #2208 tracks). The cross-surface fold is witnessed in the
	// knobcensus package test.
	if r := route["flag:account"]; len(r) != 1 || r[0] != "flag" {
		t.Errorf("flag:account route_coverage = %v, want [flag]", r)
	}
	if got["flag:cooldown-ttl"] != "HOUSEKEEPING" {
		t.Errorf("flag:cooldown-ttl verdict = %q, want HOUSEKEEPING", got["flag:cooldown-ttl"])
	}
	if got["env:FAK_GUARD_AUTO_REFRESH"] != "HOUSEKEEPING" {
		t.Errorf("env:FAK_GUARD_AUTO_REFRESH verdict = %q, want HOUSEKEEPING", got["env:FAK_GUARD_AUTO_REFRESH"])
	}
	if _, plumbing := got["flag:json"]; plumbing {
		t.Errorf("census over-matched the plumbing flag --json")
	}
}

// TestIndexKnobsTable checks the human table carries the header + the verdict
// footer counts.
func TestIndexKnobsTable(t *testing.T) {
	root := writeKnobsRepo(t)
	var out, errb bytes.Buffer
	if rc := RunIndex(&out, &errb, []string{"knobs", "--root", root}); rc != 0 {
		t.Fatalf("runIndex knobs rc=%d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"VERDICT", "ROUTE", "INTENT", "HOUSEKEEPING", "#2208", "#2198"} {
		if !strings.Contains(got, want) {
			t.Errorf("knobs table missing %q, got:\n%s", want, got)
		}
	}
}
