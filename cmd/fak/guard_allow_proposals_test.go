package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// proposalsTestEnv points the overlay (and therefore the proposals file, which lives
// in the same directory) at a temp dir, and returns the two paths.
func proposalsTestEnv(t *testing.T) (overlayPath, proposalsPath string) {
	t.Helper()
	dir := t.TempDir()
	overlayPath = filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	return overlayPath, guardAllowProposalsPath()
}

// TestGuardAllowProposalsPathBesideOverlay: the proposals file mirrors the overlay's
// directory — both under the env override and (structurally) the same base dir.
func TestGuardAllowProposalsPathBesideOverlay(t *testing.T) {
	overlayPath, proposalsPath := proposalsTestEnv(t)
	if got, want := filepath.Dir(proposalsPath), filepath.Dir(overlayPath); got != want {
		t.Errorf("proposals dir = %q, want overlay dir %q", got, want)
	}
	if filepath.Base(proposalsPath) != guardAllowProposalsFile {
		t.Errorf("proposals file = %q, want %q", filepath.Base(proposalsPath), guardAllowProposalsFile)
	}
}

// TestGuardAllowProposalsRoundTrip: save then load returns normalized entries (trimmed,
// deduped, sorted lists; trimmed reason), stamps the version, and drops empty entries.
func TestGuardAllowProposalsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "allow-proposals.json") // sub dir must be created by save
	in := guardAllowProposals{Proposals: []guardAllowProposalEntry{
		{Allow: []string{"Zed", "Read", "Read", " Edit "}, Reason: "  needed for triage  "},
		{AllowPrefix: []string{"mcp__x__"}},
		{}, // proposes nothing — must be dropped on load
	}}
	if err := saveGuardAllowProposals(path, in); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := loadGuardAllowProposals(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Version != guardAllowProposalsVersion {
		t.Errorf("version = %q, want %q", got.Version, guardAllowProposalsVersion)
	}
	if len(got.Proposals) != 2 {
		t.Fatalf("proposals = %d, want 2 (empty entry dropped): %+v", len(got.Proposals), got.Proposals)
	}
	if strings.Join(got.Proposals[0].Allow, ",") != "Edit,Read,Zed" {
		t.Errorf("allow = %v, want [Edit Read Zed]", got.Proposals[0].Allow)
	}
	if got.Proposals[0].Reason != "needed for triage" {
		t.Errorf("reason = %q, want trimmed", got.Proposals[0].Reason)
	}
	if strings.Join(got.Proposals[1].AllowPrefix, ",") != "mcp__x__" {
		t.Errorf("allow_prefix = %v", got.Proposals[1].AllowPrefix)
	}
}

// TestGuardAllowProposalsMissingIsEmpty: no proposals file is the common case — an
// empty set, no error.
func TestGuardAllowProposalsMissingIsEmpty(t *testing.T) {
	got, err := loadGuardAllowProposals(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("missing proposals should be no error, got %v", err)
	}
	if len(got.Proposals) != 0 {
		t.Errorf("missing proposals should be empty, got %+v", got)
	}
}

// TestGuardAllowProposalsMalformedFailsLoud: an unknown field and an unsupported
// version each fail loud, so an operator never reviews (or applies) a file that
// parsed differently than it was written.
func TestGuardAllowProposalsMalformedFailsLoud(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "typo.json")
	if err := os.WriteFile(bad, []byte(`{"proposal":[{"allow":["Read"]}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGuardAllowProposals(bad); err == nil {
		t.Error("unknown field should fail loud, got nil")
	}
	ver := filepath.Join(dir, "ver.json")
	if err := os.WriteFile(ver, []byte(`{"version":"fak-guard-allow-proposals/v9"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGuardAllowProposals(ver); err == nil {
		t.Error("unsupported version should fail loud, got nil")
	}
}

// TestGuardAllowProposeNeverTouchesOverlay: --propose records the request in the
// proposals file and the REAL overlay is never created or modified — the agent can
// ask, not grant.
func TestGuardAllowProposeNeverTouchesOverlay(t *testing.T) {
	overlayPath, proposalsPath := proposalsTestEnv(t)
	var out, errOut bytes.Buffer
	if code := runGuardAllowPropose(&out, &errOut, proposalsPath, []string{"WebSearch", "mcp__jira__create"}, false, "issue triage needs these"); code != 0 {
		t.Fatalf("propose exit = %d, stderr: %s", code, errOut.String())
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Errorf("overlay must remain untouched by --propose; stat err = %v", err)
	}
	props, err := loadGuardAllowProposals(proposalsPath)
	if err != nil {
		t.Fatalf("load proposals: %v", err)
	}
	if len(props.Proposals) != 1 || strings.Join(props.Proposals[0].Allow, ",") != "WebSearch,mcp__jira__create" {
		t.Fatalf("proposals = %+v", props.Proposals)
	}
	if props.Proposals[0].Reason != "issue triage needs these" {
		t.Errorf("reason = %q", props.Proposals[0].Reason)
	}
	if !strings.Contains(out.String(), "PROPOSAL") || !strings.Contains(out.String(), "overlay is unchanged") {
		t.Errorf("propose output should state the overlay is unchanged, got: %s", out.String())
	}

	// A second propose call with --prefix appends a distinct entry.
	if code := runGuardAllowPropose(&out, &errOut, proposalsPath, []string{"mcp__jira__"}, true, ""); code != 0 {
		t.Fatalf("propose prefix exit = %d", code)
	}
	props, err = loadGuardAllowProposals(proposalsPath)
	if err != nil {
		t.Fatalf("reload proposals: %v", err)
	}
	if len(props.Proposals) != 2 || strings.Join(props.Proposals[1].AllowPrefix, ",") != "mcp__jira__" {
		t.Fatalf("after prefix propose, proposals = %+v", props.Proposals)
	}
}

// TestGuardAllowFromProposalsListOnly: the default (no --apply) LISTS the pending
// proposals and changes nothing — neither the overlay nor the proposals file.
func TestGuardAllowFromProposalsListOnly(t *testing.T) {
	overlayPath, proposalsPath := proposalsTestEnv(t)
	seed := guardAllowProposals{Proposals: []guardAllowProposalEntry{
		{Allow: []string{"WebSearch"}, Reason: "research"},
		{AllowPrefix: []string{"mcp__jira__"}},
	}}
	if err := saveGuardAllowProposals(proposalsPath, seed); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runGuardAllowFromProposals(&out, &errOut, proposalsPath, false, false); code != 0 {
		t.Fatalf("list exit = %d, stderr: %s", code, errOut.String())
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Errorf("list-only must never write the overlay; stat err = %v", err)
	}
	props, err := loadGuardAllowProposals(proposalsPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(props.Proposals) != 2 {
		t.Errorf("list-only must not clear proposals; got %d pending", len(props.Proposals))
	}
	s := out.String()
	for _, want := range []string{"WebSearch", "mcp__jira__", "research", "UNCHANGED", "--from-proposals --apply"} {
		if !strings.Contains(s, want) {
			t.Errorf("list output missing %q:\n%s", want, s)
		}
	}
}

// TestGuardAllowFromProposalsApplyMergesAndClears: --apply merges the pending entries
// into the real overlay through the existing save path (preserving what was already
// there) and clears the applied proposals so they can never be re-applied.
func TestGuardAllowFromProposalsApplyMergesAndClears(t *testing.T) {
	overlayPath, proposalsPath := proposalsTestEnv(t)
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Allow: []string{"Read"}}); err != nil {
		t.Fatal(err)
	}
	seed := guardAllowProposals{Proposals: []guardAllowProposalEntry{
		{Allow: []string{"WebSearch"}, Reason: "research"},
		{AllowPrefix: []string{"mcp__jira__"}},
	}}
	if err := saveGuardAllowProposals(proposalsPath, seed); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runGuardAllowFromProposals(&out, &errOut, proposalsPath, true, false); code != 0 {
		t.Fatalf("apply exit = %d, stderr: %s", code, errOut.String())
	}
	ov, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(ov.Allow, ",") != "Read,WebSearch" {
		t.Errorf("overlay allow = %v, want [Read WebSearch] (merge preserves existing)", ov.Allow)
	}
	if strings.Join(ov.AllowPrefix, ",") != "mcp__jira__" {
		t.Errorf("overlay allow_prefix = %v, want [mcp__jira__]", ov.AllowPrefix)
	}
	props, err := loadGuardAllowProposals(proposalsPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(props.Proposals) != 0 {
		t.Errorf("applied proposals must be cleared, got %d pending", len(props.Proposals))
	}
	if !strings.Contains(out.String(), "Applied 2 proposal(s)") {
		t.Errorf("apply output missing summary:\n%s", out.String())
	}
}

// TestGuardAllowFromProposalsEmptyIsCalmNoOp: nothing pending is a normal, zero-exit
// state, and it still never creates the overlay.
func TestGuardAllowFromProposalsEmptyIsCalmNoOp(t *testing.T) {
	overlayPath, proposalsPath := proposalsTestEnv(t)
	var out, errOut bytes.Buffer
	if code := runGuardAllowFromProposals(&out, &errOut, proposalsPath, false, false); code != 0 {
		t.Fatalf("empty list exit = %d, stderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "no pending proposals") {
		t.Errorf("output = %s", out.String())
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Errorf("overlay must remain untouched; stat err = %v", err)
	}
}

// TestCmdGuardAllowProposalsFlagDiscipline: --propose and --from-proposals are
// mutually exclusive; --apply cannot ride along with --propose; and --propose with no
// names is a usage error. All misuse is a non-zero exit with the overlay untouched.
func TestCmdGuardAllowProposalsFlagDiscipline(t *testing.T) {
	overlayPath, _ := proposalsTestEnv(t)
	cases := [][]string{
		{"--propose", "--from-proposals", "X"},
		{"--propose", "--apply", "X"},
		{"--propose"},
	}
	for _, argv := range cases {
		var out, errOut bytes.Buffer
		if code := cmdGuardAllowProposals(&out, &errOut, argv); code == 0 {
			t.Errorf("argv %v: want non-zero exit", argv)
		}
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Errorf("overlay must remain untouched on misuse; stat err = %v", err)
	}
}
