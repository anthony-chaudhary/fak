package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guardSelfTightenIsolate points every guard overlay at a fresh temp dir so a floor
// assembly in the test binary reads no repo-local or per-user file, and returns the
// self-tighten overlay path.
func guardSelfTightenIsolate(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(guardAllowOverlayEnv, filepath.Join(dir, "allow.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	selfPath := filepath.Join(dir, "self-tighten.json")
	t.Setenv(guardSelfTightenOverlayEnv, selfPath)
	return selfPath
}

// TestGuardSelfTightenOverlayReachesTheFloorAtLaunch is the ARMING witness for #5411.
// It FAILS at HEAD (where admitSelfTightenOverlay had no caller and no seam, so a
// self-authored tighten overlay could not reach the floor at all) and PASSES with the
// gate wired into loadGuardCapabilityFloor. It deliberately drives the real launch
// entry point rather than the helpers, because "the gate is reachable" is exactly the
// claim the dead-code note said was unwitnessed.
func TestGuardSelfTightenOverlayReachesTheFloorAtLaunch(t *testing.T) {
	selfPath := guardSelfTightenIsolate(t)
	if err := saveGuardSelfTightenOverlay(selfPath, guardSelfTightenOverlay{
		Deny:       []string{"agent_self_denied_tool"},
		BlockHosts: []string{"tighten.invalid"},
	}); err != nil {
		t.Fatal(err)
	}

	rt, source, _, _ := loadGuardCapabilityFloor("")

	if got := rt.Adjudicator.Deny["agent_self_denied_tool"]; got != abi.ReasonPolicyBlock {
		t.Fatalf("self-tighten deny did not reach the launch floor: Deny[agent_self_denied_tool] = %v, want POLICY_BLOCK", got)
	}
	if !guardSelfTightenHas(rt.Adjudicator.EgressBlockHosts, "tighten.invalid") {
		t.Fatalf("self-tighten block host did not reach the launch floor: %v", rt.Adjudicator.EgressBlockHosts)
	}
	// The class must be journaled on the provenance surface — a floor that tightened
	// without saying so is the unauditable outcome the epic's invariants forbid.
	if !strings.Contains(source, "agent self-tighten overlay") || !strings.Contains(source, policy.AmendmentTighten) {
		t.Fatalf("floor source omitted the self-tighten provenance/class: %q", source)
	}
}

// TestGuardSelfTightenAbsentOverlayLeavesFloorUnchanged is the default-behaviour
// witness: with no overlay on disk — the state of every existing install — the floor
// and its provenance are exactly what they were before the gate was armed.
func TestGuardSelfTightenAbsentOverlayLeavesFloorUnchanged(t *testing.T) {
	guardSelfTightenIsolate(t) // path points at a file that is never created
	rt, source, digest, _ := loadGuardCapabilityFloor("")

	base, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	if len(rt.Adjudicator.Deny) != len(base.Adjudicator.Deny) {
		t.Fatalf("absent overlay changed the deny floor: %d entries, want %d", len(rt.Adjudicator.Deny), len(base.Adjudicator.Deny))
	}
	if len(rt.Adjudicator.EgressBlockHosts) != len(base.Adjudicator.EgressBlockHosts) {
		t.Fatalf("absent overlay changed the egress block hosts: %v", rt.Adjudicator.EgressBlockHosts)
	}
	if strings.Contains(source, "self-tighten") {
		t.Fatalf("absent overlay must leave the banner silent, got %q", source)
	}
	if digest == "" {
		t.Fatal("launch digest went empty")
	}
}

// TestGuardSelfTightenNoOpOverlayIsAdmittedSilently: an overlay that proposes only
// what the floor already refuses is AmendmentNone — admitted, adds nothing, and says
// nothing on the banner.
func TestGuardSelfTightenNoOpOverlayIsAdmittedSilently(t *testing.T) {
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{
		Deny: map[string]abi.ReasonCode{"already": abi.ReasonPolicyBlock},
	}}
	admit, class, reason, added := guardApplySelfTightenOverlay(&rt, guardSelfTightenOverlay{Deny: []string{"already"}})
	if !admit || class != policy.AmendmentNone || added != 0 {
		t.Fatalf("no-op overlay = admit %v class %q added %d (%s), want admit/none/0", admit, class, added, reason)
	}
	if note := guardSelfTightenFloorNote("p", admit, class, reason, added); note != "" {
		t.Fatalf("no-op overlay must not annotate the banner, got %q", note)
	}
}

// TestGuardSelfTightenRefusesForgedWideningInSchema is barrier 1 (the red-team case
// the design spike's invariants call for): a hand-forged overlay that tries to grant
// an allow cannot even be DECODED, so it is refused wholesale rather than partially
// applied.
func TestGuardSelfTightenRefusesForgedWideningInSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "self-tighten.json")
	forged := `{"version":"` + guardSelfTightenOverlayVersion + `","deny":["x"],"allow":["Bash"]}`
	if err := os.WriteFile(path, []byte(forged), 0o600); err != nil {
		t.Fatal(err)
	}
	ov, err := loadGuardSelfTightenOverlay(path)
	if err == nil {
		t.Fatalf("a forged allow field must fail loud, got overlay %+v", ov)
	}
	if !ov.empty() {
		t.Fatalf("a refused overlay must carry nothing applicable, got %+v", ov)
	}
	// And the launch treats that error as a refusal that leaves the base floor intact,
	// never as a partial apply of the tighten half it could parse.
	t.Setenv(guardAllowOverlayEnv, filepath.Join(filepath.Dir(path), "allow.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(filepath.Dir(path), "deny.json"))
	t.Setenv(guardSelfTightenOverlayEnv, path)
	rt, source, _, _ := loadGuardCapabilityFloor("")
	if _, denied := rt.Adjudicator.Deny["x"]; denied {
		t.Fatal("a refused overlay was partially applied: its deny half reached the floor")
	}
	if !strings.Contains(source, "REFUSED") {
		t.Fatalf("a refused overlay must be loud in the provenance, got %q", source)
	}
}

// TestGuardSelfTightenApplyRefusesWideningProposal is barrier 2: even a proposal the
// schema could never have produced is still classified before it is installed, and a
// widening one leaves the runtime UNTOUCHED. This is what makes the gate load-bearing
// rather than merely reachable — if the union builder were ever changed to emit a
// widening, this path refuses it.
func TestGuardSelfTightenApplyRefusesWideningProposal(t *testing.T) {
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{
		Allow: map[string]bool{"read_file": true},
		Deny:  map[string]abi.ReasonCode{"Bash": abi.ReasonPolicyBlock},
	}}
	widened := adjudicator.Policy{
		Allow: map[string]bool{"read_file": true, "Bash": true},
		Deny:  map[string]abi.ReasonCode{},
	}
	admit, class, reason := guardAdmitSelfTightenProposal(&rt, widened)
	if admit {
		t.Fatalf("a widening proposal was admitted (%s)", reason)
	}
	if class != policy.AmendmentWiden {
		t.Fatalf("class = %q, want %q", class, policy.AmendmentWiden)
	}
	if rt.Adjudicator.Allow["Bash"] {
		t.Fatal("a REFUSED proposal was installed anyway: Bash became allowed")
	}
	if _, stillDenied := rt.Adjudicator.Deny["Bash"]; !stillDenied {
		t.Fatal("a REFUSED proposal was installed anyway: the Bash deny was dropped")
	}
}

// TestGuardSelfTightenProposalDoesNotMutateTheCurrentFloor: the union must be
// copy-on-write. If it wrote through shared backing storage the diff would compare the
// proposal against itself and the gate would classify every widening as a no-op.
func TestGuardSelfTightenProposalDoesNotMutateTheCurrentFloor(t *testing.T) {
	// Spare capacity is the hazard: append onto this slice would otherwise land in the
	// same backing array the "current" policy still points at.
	globs := make([]string, 1, 8)
	globs[0] = ".git/"
	cur := adjudicator.Policy{
		Deny:            map[string]abi.ReasonCode{"a": abi.ReasonPolicyBlock},
		SelfModifyGlobs: globs,
	}
	next, added := guardSelfTightenProposal(cur, guardSelfTightenOverlay{
		Deny:            []string{"b"},
		SelfModifyGlobs: []string{".ssh/"},
	})
	if added != 2 {
		t.Fatalf("added = %d, want 2", added)
	}
	if len(cur.Deny) != 1 {
		t.Fatalf("current deny map was mutated: %v", cur.Deny)
	}
	if len(cur.SelfModifyGlobs) != 1 || cur.SelfModifyGlobs[0] != ".git/" {
		t.Fatalf("current self-modify globs were mutated: %v", cur.SelfModifyGlobs)
	}
	if cap(globs) > 1 && len(globs) < cap(globs) && globs[:cap(globs)][1] == ".ssh/" {
		t.Fatal("the union wrote through the current floor's spare capacity")
	}
	if !guardSelfTightenHas(next.SelfModifyGlobs, ".ssh/") || !guardSelfTightenHas(next.SelfModifyGlobs, ".git/") {
		t.Fatalf("proposal lost or dropped a glob: %v", next.SelfModifyGlobs)
	}
	// And the gate agrees this is the ratchet direction.
	if admit, class, reason := admitSelfTightenOverlay(cur, next); !admit || class != policy.AmendmentTighten {
		t.Fatalf("union classified %q admit=%v (%s), want tighten/admit", class, admit, reason)
	}
}

// TestGuardSelfTightenOverlayPathIsAgentWritable pins the deviation from the design
// spike's suggested location. The shipped floor self-modify-protects `.fak/guard/`, so
// an overlay under it could never be written by the wrapped agent — the property that
// makes it SELF-authored. The default path must therefore sit outside every shipped
// glob, and `.fak/guard/` must still be protected.
func TestGuardSelfTightenOverlayPathIsAgentWritable(t *testing.T) {
	t.Setenv(guardSelfTightenOverlayEnv, "")
	base, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.ToSlash(guardSelfTightenOverlayPath())
	if !strings.HasSuffix(path, ".fak/agent/self-tighten.json") {
		t.Fatalf("default overlay path = %q, want it under .fak/agent/", path)
	}
	for _, glob := range base.Adjudicator.SelfModifyGlobs {
		if g := filepath.ToSlash(glob); g != "" && strings.Contains(path, g) {
			t.Fatalf("the self-tighten overlay path %q is self-modify-protected by %q — the agent could never write it", path, glob)
		}
	}
	if !guardSelfTightenHas(base.Adjudicator.SelfModifyGlobs, ".fak/guard/") {
		t.Fatal(".fak/guard/ must stay self-modify-protected; the operator overlays live there")
	}
}

// guardSelfTightenHas is a local membership helper. Named for this file rather than
// reusing a neighbouring test's containsString so a peer editing that file cannot
// break this witness.
func guardSelfTightenHas(haystack []string, needle string) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}
