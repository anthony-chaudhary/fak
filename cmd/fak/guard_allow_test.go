package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestGuardAllowOverlayRoundTrip: save then load returns a normalized (deduped, sorted)
// overlay; a save always stamps the current version.
func TestGuardAllowOverlayRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "allow.json") // sub dir must be created by save
	in := guardAllowOverlay{Allow: []string{"Zed", "Read", "Read", " Edit "}, AllowPrefix: []string{"mcp__x__"}}
	if err := saveGuardAllowOverlay(path, in); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Version != guardAllowOverlayVersion {
		t.Errorf("version = %q, want %q", got.Version, guardAllowOverlayVersion)
	}
	wantAllow := []string{"Edit", "Read", "Zed"} // deduped, trimmed, sorted
	if strings.Join(got.Allow, ",") != strings.Join(wantAllow, ",") {
		t.Errorf("allow = %v, want %v", got.Allow, wantAllow)
	}
	if strings.Join(got.AllowPrefix, ",") != "mcp__x__" {
		t.Errorf("allow_prefix = %v", got.AllowPrefix)
	}
}

// TestGuardAllowOverlayMissingIsEmpty: a missing overlay file is the common no-op case
// — an empty overlay, no error (the base floor stands unchanged).
func TestGuardAllowOverlayMissingIsEmpty(t *testing.T) {
	got, err := loadGuardAllowOverlay(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("missing overlay should be no error, got %v", err)
	}
	if len(got.Allow) != 0 || len(got.AllowPrefix) != 0 {
		t.Errorf("missing overlay should be empty, got %+v", got)
	}
}

// TestGuardAllowOverlayMalformedFailsLoud: an unknown field and an unsupported version
// each fail loud, so an operator who believes they widened the floor is never silently
// still enforcing the bare default.
func TestGuardAllowOverlayMalformedFailsLoud(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "typo.json")
	if err := os.WriteFile(bad, []byte(`{"allows":["Read"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGuardAllowOverlay(bad); err == nil {
		t.Error("unknown field should fail loud, got nil")
	}
	ver := filepath.Join(dir, "ver.json")
	if err := os.WriteFile(ver, []byte(`{"version":"fak-guard-allow/v9","allow":["Read"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGuardAllowOverlay(ver); err == nil {
		t.Error("unsupported version should fail loud, got nil")
	}
}

// TestGuardApplyAllowOverlayWidensFloor: the union adds only NEW entries, never
// double-counts one already on the floor, and initializes a nil Allow map.
func TestGuardApplyAllowOverlayWidensFloor(t *testing.T) {
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{
		Allow:       map[string]bool{"Read": true},
		AllowPrefix: []string{"read_"},
	}}
	ov := guardAllowOverlay{
		Allow:       []string{"Read", "custom_tool", "mcp__srv__do"}, // Read already allowed
		AllowPrefix: []string{"read_", "mcp__srv__"},                 // read_ already present
	}
	added := guardApplyAllowOverlay(&rt, ov)
	if added != 3 { // custom_tool, mcp__srv__do, mcp__srv__ (prefix)
		t.Errorf("added = %d, want 3", added)
	}
	if !rt.Adjudicator.Allow["custom_tool"] || !rt.Adjudicator.Allow["mcp__srv__do"] {
		t.Errorf("overlay did not widen Allow: %+v", rt.Adjudicator.Allow)
	}
	var hasPrefix bool
	for _, p := range rt.Adjudicator.AllowPrefix {
		if p == "mcp__srv__" {
			hasPrefix = true
		}
	}
	if !hasPrefix {
		t.Errorf("overlay did not add allow_prefix: %v", rt.Adjudicator.AllowPrefix)
	}
	// Idempotent re-apply adds nothing.
	if again := guardApplyAllowOverlay(&rt, ov); again != 0 {
		t.Errorf("re-apply added = %d, want 0", again)
	}
}

// TestGuardApplyAllowOverlayNilMap: an empty base floor (nil Allow map) is widened
// without a panic.
func TestGuardApplyAllowOverlayNilMap(t *testing.T) {
	rt := policy.Runtime{}
	if added := guardApplyAllowOverlay(&rt, guardAllowOverlay{Allow: []string{"x"}}); added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}
	if !rt.Adjudicator.Allow["x"] {
		t.Error("nil Allow map was not initialized")
	}
}

// TestGuardAllowBlockedTools: only DEFAULT_DENY rows are surfaced (a POLICY_BLOCK danger
// deny and an ALLOW are ignored), counted, and ordered most-blocked first.
func TestGuardAllowBlockedTools(t *testing.T) {
	rows := []journal.Row{
		{Verdict: "DENY", Reason: "DEFAULT_DENY", Tool: "send_email"},
		{Verdict: "DENY", Reason: "DEFAULT_DENY", Tool: "custom_tool"},
		{Verdict: "DENY", Reason: "DEFAULT_DENY", Tool: "custom_tool"},
		{Verdict: "DENY", Reason: "POLICY_BLOCK", Tool: "Bash"}, // danger deny — NOT surfaced
		{Verdict: "ALLOW", Reason: "", Tool: "Read"},            // not a block
		{Verdict: "DENY", Reason: "DEFAULT_DENY", Tool: ""},     // empty tool ignored
	}
	got := guardAllowBlockedTools(rows)
	if len(got) != 2 {
		t.Fatalf("blocked tools = %d (%+v), want 2", len(got), got)
	}
	if got[0].name != "custom_tool" || got[0].count != 2 {
		t.Errorf("first = %+v, want custom_tool x2", got[0])
	}
	if got[1].name != "send_email" || got[1].count != 1 {
		t.Errorf("second = %+v, want send_email x1", got[1])
	}
	for _, b := range got {
		if b.name == "Bash" {
			t.Error("POLICY_BLOCK danger deny must not be surfaced as allowable")
		}
	}
}

// TestRunGuardAllowFromJournalListThenAddAll: --from-journal lists the blocked tools and
// the exact allow command; --add-all records them in the overlay.
func TestRunGuardAllowFromJournalListThenAddAll(t *testing.T) {
	dir := t.TempDir()
	jp := filepath.Join(dir, "audit.jsonl")
	writeGuardAllowTestJournal(t, jp,
		journal.Row{Verdict: "DENY", Reason: "DEFAULT_DENY", Tool: "custom_tool"},
		journal.Row{Verdict: "DENY", Reason: "POLICY_BLOCK", Tool: "Bash"},
	)
	overlayPath := filepath.Join(dir, "allow.json")

	// List mode: prints the tool + the exact command, writes nothing.
	ov, _ := loadGuardAllowOverlay(overlayPath)
	var out, errb bytes.Buffer
	if code := runGuardAllowFromJournal(&out, &errb, overlayPath, &ov, jp, false); code != 0 {
		t.Fatalf("list mode exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "custom_tool") || !strings.Contains(s, "fak guard allow custom_tool") {
		t.Errorf("list output missing tool/command:\n%s", s)
	}
	if strings.Contains(s, "Bash") {
		t.Errorf("danger POLICY_BLOCK must not be offered as allowable:\n%s", s)
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Error("list mode must not write the overlay")
	}

	// Add-all mode: records custom_tool in the overlay.
	ov, _ = loadGuardAllowOverlay(overlayPath)
	out.Reset()
	errb.Reset()
	if code := runGuardAllowFromJournal(&out, &errb, overlayPath, &ov, jp, true); code != 0 {
		t.Fatalf("add-all exit=%d stderr=%s", code, errb.String())
	}
	saved, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatalf("reload overlay: %v", err)
	}
	if len(saved.Allow) != 1 || saved.Allow[0] != "custom_tool" {
		t.Errorf("add-all overlay = %v, want [custom_tool]", saved.Allow)
	}
}

// TestRunGuardAllowFromJournalNoBlocks: a clean journal reports nothing to allow and
// writes nothing.
func TestRunGuardAllowFromJournalNoBlocks(t *testing.T) {
	dir := t.TempDir()
	jp := filepath.Join(dir, "audit.jsonl")
	writeGuardAllowTestJournal(t, jp, journal.Row{Verdict: "ALLOW", Tool: "Read"})
	overlayPath := filepath.Join(dir, "allow.json")
	ov, _ := loadGuardAllowOverlay(overlayPath)
	var out, errb bytes.Buffer
	if code := runGuardAllowFromJournal(&out, &errb, overlayPath, &ov, jp, true); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "nothing to allow") {
		t.Errorf("want 'nothing to allow', got:\n%s", out.String())
	}
}

// TestGuardAllowSummaryHintOnDefaultDeny: the exit summary invites the operator to allow
// a blocked tool ONLY when a DEFAULT_DENY happened — never for a danger POLICY_BLOCK, and
// never on a clean session.
func TestGuardAllowSummaryHintOnDefaultDeny(t *testing.T) {
	withDefaultDeny := gateway.AdjudicationSummary{
		Total: 3, Denied: 1, ByReason: map[string]uint64{"DEFAULT_DENY": 1},
	}
	if s := formatAuditSummary(withDefaultDeny); !strings.Contains(s, "fak guard allow --from-journal") {
		t.Errorf("DEFAULT_DENY session should surface the allow control:\n%s", s)
	}
	onlyDanger := gateway.AdjudicationSummary{
		Total: 3, Denied: 1, ByReason: map[string]uint64{"POLICY_BLOCK": 1},
	}
	if s := formatAuditSummary(onlyDanger); strings.Contains(s, "fak guard allow") {
		t.Errorf("danger-only session must NOT invite widening the floor:\n%s", s)
	}
	clean := gateway.AdjudicationSummary{Total: 2, Allowed: 2, ByReason: map[string]uint64{}}
	if s := formatAuditSummary(clean); strings.Contains(s, "fak guard allow") {
		t.Errorf("clean session must not print the allow hint:\n%s", s)
	}
}

// TestGuardAllowSubtractRemoves: subtract drops named entries and leaves the rest.
func TestGuardAllowSubtractRemoves(t *testing.T) {
	got := guardAllowSubtract([]string{"a", "b", "c"}, []string{"b"})
	if strings.Join(got, ",") != "a,c" {
		t.Errorf("subtract = %v, want [a c]", got)
	}
}

// writeGuardAllowTestJournal writes rows as JSONL that journal.ReadRows can read back.
func writeGuardAllowTestJournal(t *testing.T, path string, rows ...journal.Row) {
	t.Helper()
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			t.Fatalf("encode row: %v", err)
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write journal: %v", err)
	}
}
