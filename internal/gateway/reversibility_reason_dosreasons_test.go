package gateway

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestReversibilityGateReasonIsDeclaredNotUnclassified is the closed-vocabulary
// conformance witness for #2776: the reversibility rung
// (internal/adjudicator reversibilityGateVerdict, By "monitor/reversibility")
// refuses with Kind=VerdictRequireWitness and sets NO closed-vocabulary Reason, so
// the in-band render falls back to the verdict KIND (reasonOrKind returns v.Reason
// if set, else v.Kind) — the token "REQUIRE_WITNESS". That token was the exact
// reason-drift the closed vocabulary exists to kill: `dos_check_reason
// REQUIRE_WITNESS` returned known=false, category=UNCLASSIFIED, so an agent that
// checked the reason it was refused with was told the reason was invalid — a
// verdict KIND leaking into the reason slot, undermining the kernel's own contract.
//
// The fix (working spine) declared a real OPERATOR_GATE reason for the gate in the
// workspace dos.toml [reasons] table, so the emitted token now resolves known=true
// (an acknowledged pause, not a hard deny). This test pins that end to end: it
// projects the SAME abi.Verdict the adjudicator produces through the real gateway
// render pipeline (renderVerdict -> reasonOrKind), then asserts the emitted reason
// token is DECLARED in dos.toml (known, refusal=true) rather than UNCLASSIFIED.
//
// It fails loudly (a) on the pre-#2776 UNCLASSIFIED state — remove the
// [reasons.REQUIRE_WITNESS] table from dos.toml and this reds — and (b) if a future
// edit changes the emitted reason to a token that is not declared (e.g. setting the
// verdict Reason to an undeclared abi code, which would render an UNCLASSIFIED
// name). This is the "no reason the adjudicator/gateway emits is UNCLASSIFIED"
// assertion the issue's Witness requires.
func TestReversibilityGateReasonIsDeclaredNotUnclassified(t *testing.T) {
	// The exact verdict internal/adjudicator's reversibilityGateVerdict produces:
	// RequireWitness, By "monitor/reversibility", the preview envelope as the bounded
	// claim, and NO closed-vocabulary Reason (Reason defaults to abi.ReasonNone) — so
	// the render MUST fall back to the KIND, reproducing the token the agent sees.
	claim := `{"class":"outward-facing","preview":"outward-facing command: git push origin main","confirm_token":"fak-0011223344556677"}`
	w := renderVerdict(abi.Verdict{
		Kind:    abi.VerdictRequireWitness,
		By:      "monitor/reversibility",
		Payload: abi.WitnessPayload{Claim: claim},
	}, nil)

	// The KIND-fallback token the agent actually reads in-band.
	token := reasonOrKind(w)
	if token == "" {
		t.Fatal("reversibility gate emitted an empty reason token")
	}
	// Witness the exact defect this issue names: the leaked verdict KIND is the token,
	// and it must now be a DECLARED reason rather than UNCLASSIFIED drift.
	if token != "REQUIRE_WITNESS" {
		t.Fatalf("reversibility gate reason token = %q, want the KIND-fallback %q "+
			"(if the adjudicator now sets an explicit closed-vocabulary Reason, update "+
			"this witness to the new token — it must still be declared below)", token, "REQUIRE_WITNESS")
	}

	// The token the agent sees must be declared in the workspace dos.toml [reasons]
	// table — known, refusal=true — so `dos_check_reason <token>` resolves it instead
	// of returning category=UNCLASSIFIED (the pre-#2776 defect).
	content := readRepoDosTomlForReversibility(t)
	header := "[reasons." + token + "]"
	if !strings.Contains(content, header) {
		t.Fatalf("reversibility gate emits reason %q but dos.toml has no %s table — "+
			"dos_check_reason %s would return known=false (UNCLASSIFIED drift, the #2776 defect)",
			token, header, token)
	}
	block := reversibilityDosReasonBlock(content, header)
	if !reversibilityDosReasonFieldTrue(block, "refusal") {
		t.Fatalf("reversibility reason %q is declared but not marked refusal = true — "+
			"dos_check_reason would resolve it as non-refusable", token)
	}
}

// reversibilityDosReasonBlock returns the text of the [reasons.<TOKEN>] table named
// by header: from the header line up to (but excluding) the next top-level
// [section] or EOF, so a field assertion scopes to a single reason's table rather
// than matching a sibling's by accident. (Same shape as the toon/assumecheck
// dos-reason conformance helpers, copied here because those are package-private.)
func reversibilityDosReasonBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// reversibilityDosReasonFieldTrue reports whether block contains a `field = true`
// line, tolerant of the aligned whitespace the dos.toml [reasons] tables use
// (e.g. "refusal  = true").
func reversibilityDosReasonFieldTrue(block, field string) bool {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") && strings.TrimSpace(rest[1:]) == "true" {
			return true
		}
	}
	return false
}

// readRepoDosTomlForReversibility reads the repo-root dos.toml located relative to
// this test's own source path (internal/gateway), so the lookup is independent of
// the test's working directory.
func readRepoDosTomlForReversibility(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}
