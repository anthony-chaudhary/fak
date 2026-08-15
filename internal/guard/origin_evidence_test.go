package guard

import (
	"os"
	"path/filepath"
	"testing"
)

// TestPolicyOriginEvidenceWritesZeroBytes is the #6093 witness: resolving and proving
// the policy origin-evidence location for a launch materializes the LOCATION and
// nothing else. Before the fix every launch copied the 34,702-byte embedded capability
// floor into its own <trace-id>-policy.json, so N launches left N byte-identical 34KB
// files that no reader in the tree ever opened.
func TestPolicyOriginEvidenceWritesZeroBytes(t *testing.T) {
	root := t.TempDir()
	traces := []string{"trace-a", "trace-b", "trace-c"}
	for _, traceID := range traces {
		path := PolicyOriginEvidencePath(root, traceID, "")
		if path == "" {
			t.Fatalf("PolicyOriginEvidencePath(%q) returned no location", traceID)
		}
		abs, ok := EnsureOriginEvidence(path)
		if !ok {
			t.Fatalf("EnsureOriginEvidence(%q) refused a fresh location", path)
		}
		info, err := os.Stat(abs)
		if err != nil {
			t.Fatalf("policy origin evidence must exist for PathWitness: %v", err)
		}
		if info.Size() != 0 {
			t.Fatalf("policy origin evidence for %s is %d bytes; want 0", traceID, info.Size())
		}
	}

	// One 0-byte location per trace, and zero bytes across the whole directory: the
	// target operating envelope of #6093.
	entries, err := os.ReadDir(filepath.Join(root, ".fak", "guard-origin"))
	if err != nil {
		t.Fatalf("read guard-origin dir: %v", err)
	}
	if len(entries) != len(traces) {
		t.Fatalf("guard-origin holds %d files; want one per trace (%d)", len(entries), len(traces))
	}
	var total int64
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			t.Fatalf("stat %s: %v", entry.Name(), err)
		}
		total += info.Size()
	}
	if total != 0 {
		t.Fatalf("guard-origin wrote %d bytes for %d launches; want 0", total, len(traces))
	}
}

// TestPolicyOriginEvidencePathHonorsExplicitPolicy pins the operator's own --policy
// file as a location we hand over untouched: never re-derived from the trace id, and
// never rewritten (which would clobber the operator's manifest).
func TestPolicyOriginEvidencePathHonorsExplicitPolicy(t *testing.T) {
	root := t.TempDir()
	explicit := filepath.Join(root, "operator-policy.json")
	const body = `{"schema":"fak.policy/1"}`
	if err := os.WriteFile(explicit, []byte(body), 0o600); err != nil {
		t.Fatalf("seed operator policy: %v", err)
	}
	if got := PolicyOriginEvidencePath(root, "trace-a", explicit); got != explicit {
		t.Fatalf("explicit policy path = %q; want %q", got, explicit)
	}
	if _, ok := EnsureOriginEvidence(explicit); !ok {
		t.Fatalf("EnsureOriginEvidence refused an existing operator policy")
	}
	got, err := os.ReadFile(explicit)
	if err != nil {
		t.Fatalf("read operator policy: %v", err)
	}
	if string(got) != body {
		t.Fatalf("operator policy rewritten: got %q, want %q", got, body)
	}
}

// TestPolicyOriginEvidencePathNeedsRootAndTrace: with no location to name there is no
// evidence to hand over, and the caller must not register a bare ".fak/guard-origin"
// directory ref.
func TestPolicyOriginEvidencePathNeedsRootAndTrace(t *testing.T) {
	for _, tc := range []struct{ name, root, traceID string }{
		{"no trace", t.TempDir(), "  "},
		{"no root", " ", "trace-a"},
		{"neither", "", ""},
	} {
		if got := PolicyOriginEvidencePath(tc.root, tc.traceID, ""); got != "" {
			t.Fatalf("%s: PolicyOriginEvidencePath = %q; want empty", tc.name, got)
		}
	}
}

// TestEnsureOriginEvidenceKeepsProducerRows: the placeholder exists only until the
// producer appends; re-proving a live location must never truncate what it already
// wrote (transcripts and Stop ledgers are appended to across a session).
func TestEnsureOriginEvidenceKeepsProducerRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "trace-a-transcript.jsonl")
	if _, ok := EnsureOriginEvidence(path); !ok {
		t.Fatalf("EnsureOriginEvidence(%q) refused a fresh location", path)
	}
	const row = "{\"turn\":1}\n"
	if err := os.WriteFile(path, []byte(row), 0o600); err != nil {
		t.Fatalf("append producer row: %v", err)
	}
	abs, ok := EnsureOriginEvidence(path)
	if !ok {
		t.Fatalf("EnsureOriginEvidence(%q) refused an existing location", path)
	}
	if !filepath.IsAbs(abs) {
		t.Fatalf("EnsureOriginEvidence returned relative path %q", abs)
	}
	got, err := os.ReadFile(abs)
	if err != nil {
		t.Fatalf("read producer rows: %v", err)
	}
	if string(got) != row {
		t.Fatalf("producer rows clobbered: got %q, want %q", got, row)
	}
	if _, ok := EnsureOriginEvidence("   "); ok {
		t.Fatalf("EnsureOriginEvidence accepted an empty location")
	}
}
