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

// policyExplainScopeLayer writes an allow overlay carrying `tools` and returns the layer
// naming it under `scope`.
func policyExplainScopeLayer(t *testing.T, dir, scope string, tools ...string) guardAllowOverlayLayer {
	t.Helper()
	path := filepath.Join(dir, scope+".allow.json")
	if err := saveGuardAllowOverlay(path, guardAllowOverlay{Allow: tools}); err != nil {
		t.Fatal(err)
	}
	return guardAllowOverlayLayer{Name: scope, Path: path}
}

// TestGuardAllowScopeExplainShowsResolvedScopePerEntry is the #5180 witness for the
// "policy explain shows the scope of each overlay entry" half of the done condition.
//
// Three things are pinned at once, because a scope column that got any of them wrong
// would be worse than no column:
//
//   - EVERY overlay entry carries a scope= cell naming the layer it came from;
//   - an entry named by SEVERAL scopes renders ONCE, attributed to the NARROWEST one
//     (guard_allow_scope.go's precedence) — not once per layer, which would show the
//     same widening three times and leave the operator guessing which one governs;
//   - the attribution agrees with guardAllowWinningScope, the overlay's own provenance
//     seam, so the report and the enforcement path cannot disagree.
func TestGuardAllowScopeExplainShowsResolvedScopePerEntry(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	dir := t.TempDir()
	// Broadest-first, exactly as guardAllowOverlayPaths + the session layer order them.
	// zz_everywhere_tool is named by all three scopes; the rest by one each.
	layers := []guardAllowOverlayLayer{
		policyExplainScopeLayer(t, dir, "repo", "zz_repo_tool", "zz_everywhere_tool"),
		policyExplainScopeLayer(t, dir, "user", "zz_user_only_tool", "zz_everywhere_tool"),
		policyExplainScopeLayer(t, dir, guardAllowScopeSession, "zz_session_tool", "zz_everywhere_tool"),
	}
	got := policyExplainRun(t, layers, filepath.Join(dir, "deny.json"))
	idx := policyExplainHeaderIndex(t, got)
	from, to := idx["== GATED-WIDEN"], idx["== SELF-AMENDABLE"]

	for _, tc := range []struct{ tool, scope, provenance string }{
		{"zz_repo_tool", "repo", "repo-overlay"},
		{"zz_user_only_tool", "user", "user-overlay"},
		{"zz_session_tool", guardAllowScopeSession, "session-overlay"},
		// Present in repo, user AND session — the session scope is the narrowest, so it
		// owns the entry and the two broader layers must not claim it.
		{"zz_everywhere_tool", guardAllowScopeSession, "session-overlay"},
	} {
		row := "allow.tools[" + tc.tool + "]"
		line := policyExplainRowLine(t, got, row, from, to)
		if !strings.Contains(line, "scope="+tc.scope) {
			t.Fatalf("%s row lacks scope=%s: %q", row, tc.scope, line)
		}
		if !strings.Contains(line, "provenance="+tc.provenance) {
			t.Fatalf("%s row lacks provenance=%s: %q", row, tc.provenance, line)
		}
		if n := strings.Count(got, row); n != 1 {
			t.Fatalf("%s rendered %d times, want exactly one resolved-scope row", row, n)
		}
	}
}

// TestGuardAllowScopeExplainAgreesWithWinningScope ties the report's attribution to the
// overlay's own resolver: for the same layer set, policyExplainResolvedAllowEntries must
// pick the scope guardAllowWinningScope would. Drift between them is how an operator ends
// up reading a scope the enforcement path does not actually honor.
func TestGuardAllowScopeExplainAgreesWithWinningScope(t *testing.T) {
	scopeTestRepo(t, "explain-agreement")

	if err := saveGuardAllowOverlay(guardAllowOverlayPath(), guardAllowOverlay{Allow: []string{"zz_shared_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(guardAllowSessionOverlayPath(), guardAllowOverlay{Allow: []string{"zz_shared_tool"}}); err != nil {
		t.Fatal(err)
	}

	layers := guardPolicyExplainAllowLayers()
	byLayer := make([]guardAllowOverlay, len(layers))
	for i, l := range layers {
		ov, err := loadGuardAllowOverlay(l.Path)
		if err != nil {
			t.Fatal(err)
		}
		byLayer[i] = ov
	}
	entries := policyExplainResolvedAllowEntries(layers, byLayer, false)

	want, err := guardAllowWinningScope("zz_shared_tool", false)
	if err != nil {
		t.Fatal(err)
	}
	if want != guardAllowScopeSession {
		t.Fatalf("guardAllowWinningScope = %q, want the session scope to own the shared entry", want)
	}
	found := 0
	for _, e := range entries {
		if e.Name != "zz_shared_tool" {
			continue
		}
		found++
		if e.Scope != want {
			t.Fatalf("explain attributed zz_shared_tool to %q, guardAllowWinningScope says %q", e.Scope, want)
		}
	}
	if found != 1 {
		t.Fatalf("zz_shared_tool resolved to %d entries, want exactly 1", found)
	}
}

// TestGuardAllowScopeExplainRendersPrecedenceLegend: the per-row scope= cell is only
// readable against the precedence order, so the GATED-WIDEN group must print the scope
// table — every scope, with the durability answer (does a widening recorded here survive
// the next launch?) that guardAllowScopeDurabilityNote owns.
func TestGuardAllowScopeExplainRendersPrecedenceLegend(t *testing.T) {
	t.Setenv(guardDenyOverlayEnv, "")
	got := policyExplainRun(t, nil, filepath.Join(t.TempDir(), "deny.json"))
	idx := policyExplainHeaderIndex(t, got)
	legend := got[idx["== GATED-WIDEN"]:idx["== SELF-AMENDABLE"]]

	for _, scope := range policyExplainScopeOrder {
		if !strings.Contains(legend, "scope="+scope) {
			t.Fatalf("GATED-WIDEN legend omits scope %q:\n%s", scope, legend)
		}
		if note := guardAllowScopeDurabilityNote(scope); note != "" && !strings.Contains(legend, note) {
			t.Fatalf("GATED-WIDEN legend omits the %s durability note %q:\n%s", scope, note, legend)
		}
	}
	// The legend belongs to the widening channel alone — the frozen floor has no scope.
	if frozen := got[idx["== FROZEN"]:idx["== RATCHET"]]; strings.Contains(frozen, "scope precedence") {
		t.Fatalf("scope precedence legend leaked into the FROZEN group:\n%s", frozen)
	}
}
