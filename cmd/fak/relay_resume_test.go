// relay_resume_test.go — the #1876 done-condition witness: `fak relay resume` prints the
// baton's fields, and --json round-trips the canonical wire bytes byte-identically.
package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/relay"
)

// fullBaton builds a baton with EVERY field populated, so the print witness can assert
// each one surfaces and the round-trip witness exercises the whole type tree.
func fullBaton(t *testing.T) relay.Baton {
	t.Helper()
	pin := ctxplan.NewObjectivePin("pin-7", "ship the relay read half", 1)
	return relay.Baton{
		Schema:      relay.Schema,
		RelayID:     "RID-2026-07-01-relaydemo",
		Leg:         3,
		ParentTrace: "trace-abc123",
		Objective:   pin,
		DoneWhen:    "issue #1876 closed by a witnessed commit",
		ProgressCursor: relay.ProgressCursor{
			StartSHA:   "b2926823aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			LedgerRef:  "ledger-row-42",
			HeldRegion: []string{"cmd/fak/relay_resume*.go"},
		},
		NextAction:    "wire the relay verb into main.go dispatch",
		OpenQuestions: []string{"#1908 rotation test scope"},
		Artifacts: []relay.Artifact{
			{Kind: string(relay.ArtifactCommit), Ref: "b2926823"},
			{Kind: string(relay.ArtifactIssue), Ref: "#1876"},
		},
		DoNotRederive: []string{"memory:resume-wave-operations"},
		Tombstone: relay.Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  "b2926823aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Note:   "context ceiling",
		},
	}
}

func writeBatonFile(t *testing.T, b relay.Baton) string {
	t.Helper()
	data, err := relay.Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	path := filepath.Join(t.TempDir(), "leg.baton.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write baton: %v", err)
	}
	return path
}

// The human summary must surface every baton field — an operator inspecting a handoff
// sees the WHOLE thing a successor leg would receive.
func TestRelayResume_PrintsBatonFields(t *testing.T) {
	b := fullBaton(t)
	path := writeBatonFile(t, b)

	var out, errOut bytes.Buffer
	if rc := runRelay(strings.NewReader(""), &out, &errOut, []string{"resume", "--baton", path}); rc != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", rc, errOut.String())
	}
	got := out.String()
	for _, want := range []string{
		relay.Schema,
		"RID-2026-07-01-relaydemo",
		"leg=3",
		"trace-abc123",
		"RELAY_ROTATED",
		"context ceiling",
		"ship the relay read half",
		"pin-7",
		"issue #1876 closed by a witnessed commit",
		"b2926823aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"ledger-row-42",
		"cmd/fak/relay_resume*.go",
		"wire the relay verb into main.go dispatch",
		"#1908 rotation test scope",
		"commit",
		"#1876",
		"memory:resume-wave-operations",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("summary is missing %q\n--- output ---\n%s", want, got)
		}
	}
}

// --json must emit the canonical wire bytes: byte-identical to relay.Marshal over the
// same value, so `resume --json | resume --baton - --json` is a fixed point.
func TestRelayResume_JSONRoundTrips(t *testing.T) {
	b := fullBaton(t)
	canonical, err := relay.Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	path := writeBatonFile(t, b)

	var out, errOut bytes.Buffer
	if rc := runRelay(strings.NewReader(""), &out, &errOut, []string{"resume", "--baton", path, "--json"}); rc != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", rc, errOut.String())
	}
	got := strings.TrimRight(out.String(), "\n")
	if got != string(canonical) {
		t.Fatalf("--json output is not the canonical wire form\n got: %s\nwant: %s", got, canonical)
	}

	// Second hop through stdin: the output feeds back in and re-emits the same bytes.
	var out2, errOut2 bytes.Buffer
	if rc := runRelay(strings.NewReader(out.String()), &out2, &errOut2, []string{"resume", "--baton", "-", "--json"}); rc != 0 {
		t.Fatalf("stdin hop exit = %d, want 0 (stderr: %s)", rc, errOut2.String())
	}
	if out2.String() != out.String() {
		t.Fatalf("round-trip is not a fixed point\n got: %s\nwant: %s", out2.String(), out.String())
	}
}

// A nil-slice baton and its projected form must print the SAME canonical bytes — the
// codec's `[]`-not-`null` projection holds through the CLI.
func TestRelayResume_JSONCanonicalizesNilSlices(t *testing.T) {
	b := fullBaton(t)
	b.OpenQuestions = nil
	b.DoNotRederive = nil
	path := writeBatonFile(t, b)

	var out, errOut bytes.Buffer
	if rc := runRelay(strings.NewReader(""), &out, &errOut, []string{"resume", path, "--json"}); rc != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", rc, errOut.String())
	}
	if strings.Contains(out.String(), "null") {
		t.Fatalf("--json emitted a null slice; want the schema-mandated []\n%s", out.String())
	}
}

// The reader contract's schema gate: a wrong tag (or a bare {}) is refused, exit 1.
func TestRelayResume_RejectsWrongSchema(t *testing.T) {
	for name, body := range map[string]string{
		"wrong-tag": `{"schema":"fak.relay.baton.v0","relay_id":"r"}`,
		"empty-obj": `{}`,
	} {
		var out, errOut bytes.Buffer
		rc := runRelay(strings.NewReader(body), &out, &errOut, []string{"resume", "--baton", "-"})
		if rc != 1 {
			t.Errorf("%s: exit = %d, want 1", name, rc)
		}
		if !strings.Contains(errOut.String(), "schema") {
			t.Errorf("%s: stderr does not name the schema gate: %s", name, errOut.String())
		}
	}
}

// Usage errors are exit 2: no subcommand, an unknown subcommand, no baton, two batons.
func TestRelayResume_UsageErrors(t *testing.T) {
	for name, argv := range map[string][]string{
		"no-subcommand": {},
		"unknown-sub":   {"rotate"},
		"no-baton":      {"resume"},
		"two-batons":    {"resume", "a.json", "b.json"},
	} {
		var out, errOut bytes.Buffer
		if rc := runRelay(strings.NewReader(""), &out, &errOut, argv); rc != 2 {
			t.Errorf("%s: exit = %d, want 2 (stderr: %s)", name, rc, errOut.String())
		}
	}
}
