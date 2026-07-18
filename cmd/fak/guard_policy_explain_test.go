package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// policyExplainRun renders the explain report against injected overlay layers and
// fails the test on a non-zero exit, returning the report text.
func policyExplainRun(t *testing.T, layers []guardAllowOverlayLayer, denyPath string) string {
	t.Helper()
	var out, errb bytes.Buffer
	if code := runGuardPolicyExplain(&out, &errb, layers, denyPath); code != 0 {
		t.Fatalf("runGuardPolicyExplain exit=%d stderr=%s", code, errb.String())
	}
	return out.String()
}

// policyExplainHeaderIndex asserts the four amendment-class group headers all
// render, in class order, and returns the byte offset of each header.
func policyExplainHeaderIndex(t *testing.T, got string) map[string]int {
	t.Helper()
	headers := []string{"== FROZEN", "== RATCHET", "== GATED-WIDEN", "== SELF-AMENDABLE"}
	idx := make(map[string]int, len(headers))
	last := -1
	for _, h := range headers {
		i := strings.Index(got, h)
		if i < 0 {
			t.Fatalf("missing group header %q in output:\n%s", h, got)
		}
		if i < last {
			t.Fatalf("group header %q out of order in output:\n%s", h, got)
		}
		idx[h] = i
		last = i
	}
	return idx
}

// policyExplainRowLine asserts a knob row renders between the group's header and
// the next header, and returns that row's full line.
func policyExplainRowLine(t *testing.T, got, row string, from, to int) string {
	t.Helper()
	ri := strings.Index(got, row)
	if ri < 0 {
		t.Fatalf("knob row %q missing from output:\n%s", row, got)
	}
	if ri < from || ri > to {
		t.Fatalf("knob row %q outside its group (row@%d, group %d..%d):\n%s", row, ri, from, to, got)
	}
	line := got[ri:]
	if nl := strings.IndexByte(line, '\n'); nl >= 0 {
		line = line[:nl]
	}
	return line
}

// TestPolicyExplainGroupsAndOverlayProvenance is the #5172 witness: all four
// amendment-class groups render, and an operator overlay-added allow entry renders
// under GATED-WIDEN with provenance repo-overlay.
func TestPolicyExplainGroupsAndOverlayProvenance(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	if err := os.WriteFile(overlayPath, []byte(`{"version":"`+guardAllowOverlayVersion+`","allow":["zz_overlay_tool"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := policyExplainRun(t,
		[]guardAllowOverlayLayer{{Name: "repo", Path: overlayPath}},
		filepath.Join(dir, "deny.json")) // deny overlay absent — the common no-op

	idx := policyExplainHeaderIndex(t, got)
	line := policyExplainRowLine(t, got, "allow.tools[zz_overlay_tool]", idx["== GATED-WIDEN"], idx["== SELF-AMENDABLE"])
	for _, want := range []string{"class=GATED-WIDEN", "provenance=repo-overlay", "fak guard allow"} {
		if !strings.Contains(line, want) {
			t.Fatalf("overlay allow row lacks %q: %q", want, line)
		}
	}
}

// TestPolicyExplainUserLayerProvenance pins the layer→provenance mapping: the same
// entry supplied by the user layer renders as user-overlay, not repo-overlay.
func TestPolicyExplainUserLayerProvenance(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	if err := os.WriteFile(overlayPath, []byte(`{"allow":["zz_user_tool"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := policyExplainRun(t,
		[]guardAllowOverlayLayer{{Name: "user", Path: overlayPath}},
		filepath.Join(dir, "deny.json"))

	idx := policyExplainHeaderIndex(t, got)
	line := policyExplainRowLine(t, got, "allow.tools[zz_user_tool]", idx["== GATED-WIDEN"], idx["== SELF-AMENDABLE"])
	if !strings.Contains(line, "provenance=user-overlay") {
		t.Fatalf("user-layer allow row lacks provenance=user-overlay: %q", line)
	}
}

// TestPolicyExplainDenyOverlayUnderRatchet pins the tighten-only side: a deny
// overlay entry renders under RATCHET with repo-overlay provenance.
func TestPolicyExplainDenyOverlayUnderRatchet(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	dir := t.TempDir()
	denyPath := filepath.Join(dir, "deny.json")
	if err := os.WriteFile(denyPath, []byte(`{"deny":["zz_bad_tool"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := policyExplainRun(t, nil, denyPath)

	idx := policyExplainHeaderIndex(t, got)
	line := policyExplainRowLine(t, got, "deny.tools[zz_bad_tool]", idx["== RATCHET"], idx["== GATED-WIDEN"])
	for _, want := range []string{"class=RATCHET", "provenance=repo-overlay"} {
		if !strings.Contains(line, want) {
			t.Fatalf("deny overlay row lacks %q: %q", want, line)
		}
	}
}

// TestPolicyExplainBareCheckout pins the no-overlay case: every group still
// renders (each carries its embedded rows), so the mutability model is visible
// on a checkout with no operator decisions yet.
func TestPolicyExplainBareCheckout(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	got := policyExplainRun(t, nil, filepath.Join(t.TempDir(), "deny.json"))
	idx := policyExplainHeaderIndex(t, got)
	policyExplainRowLine(t, got, "danger.arg_rules", idx["== FROZEN"], idx["== RATCHET"])
	policyExplainRowLine(t, got, "allow.tools", idx["== GATED-WIDEN"], idx["== SELF-AMENDABLE"])
}
