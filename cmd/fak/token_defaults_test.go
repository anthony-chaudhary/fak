package main

// token_defaults_test.go — the regression sentinel for fak's OUT-OF-THE-BOX token-saving
// default stack. The token-defaults scorecard (`fak token-defaults-scorecard`) scores
// which stacking savers are on by default; this test is the lock its `default_on_locked`
// KPI requires, so a peer who silently flips a saver back to off — unwires the tool floor,
// drops the vDSO default, or zeroes the compaction budget — reds CI here, not in
// production. Each assertion reads the REAL entrypoint declaration (cmd/fak/serve.go,
// cmd/fak/guard.go) or the gateway Default* constant, so it pins the binary's actual
// default, never a copy that could drift from it.
//
// The off-by-default levers (elision, ctxview) are pinned too — not to keep them off, but
// so each stays wired to its single documented default (the gateway const / the literal
// 0), making a future flip-on one deliberate edit that this test moves with, never a silent
// per-entrypoint drift. The byte-faithful provider-cache passthrough is locked separately by
// internal/gateway's TestAnthropicMessagesPassthroughStreamsLiveAndAdjudicates (upstream
// body byte-identical to the inbound → the client's prompt-cache prefix survives).

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// readEntrypoint reads a cmd/fak source file from the package directory (go test runs with
// CWD = the package dir), so the assertions bind the real flag declarations.
func readEntrypoint(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(b)
}

// TestTokenDefault_CompactHistoryDefaultsOn pins history compaction ON by default: the
// gateway const is a non-zero (default-on) resident-token budget, and both front doors wire
// the flag to it, so a sprawling conversation is compacted with no operator configuration.
func TestTokenDefault_CompactHistoryDefaultsOn(t *testing.T) {
	if gateway.DefaultCompactHistoryBudget <= 0 {
		t.Fatalf("DefaultCompactHistoryBudget must be default-on (>0), got %d", gateway.DefaultCompactHistoryBudget)
	}
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`) {
			t.Errorf("%s must wire --compact-history-budget to gateway.DefaultCompactHistoryBudget (default-on)", f)
		}
	}
}

// TestTokenDefault_VdsoDefaultsOn pins the vDSO dedup fast path ON by default: serve.go's
// flag defaults true and guard.go enables it structurally in the gateway Config.
func TestTokenDefault_VdsoDefaultsOn(t *testing.T) {
	if !strings.Contains(readEntrypoint(t, "serve.go"), `fs.Bool("vdso", true`) {
		t.Errorf("serve.go must default --vdso to true (lossless dedup on by default)")
	}
	if !regexp.MustCompile(`VDSO:\s+true`).MatchString(readEntrypoint(t, "guard.go")) {
		t.Errorf("guard.go must set gateway Config VDSO: true (dedup on by default)")
	}
}

// TestTokenDefault_ToolFloorDefaultsOn pins tool-floor pruning ON by default: both front
// doors set the gateway Config ToolFloorDenies predicate, which prunes provably-unreachable
// tool definitions from the Anthropic passthrough (fewer tool-definition tokens, fail-safe).
func TestTokenDefault_ToolFloorDefaultsOn(t *testing.T) {
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), "ToolFloorDenies:") {
			t.Errorf("%s must set gateway Config ToolFloorDenies (tool-floor pruning on by default)", f)
		}
	}
}

// TestTokenDefault_ElideShipsDarkAtDocumentedDefault pins the oversized-result elision lever
// to its documented default (the gateway const). It ships dark (0) until a savings/fidelity
// witness supports a default-on threshold; wiring the flag to the const keeps the on/off
// decision a single deliberate edit to DefaultElideResultBytes, never a per-entrypoint drift.
func TestTokenDefault_ElideShipsDarkAtDocumentedDefault(t *testing.T) {
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`) {
			t.Errorf("%s must wire --elide-result-bytes to gateway.DefaultElideResultBytes (single documented default)", f)
		}
	}
}

// TestTokenDefault_CtxViewShipsDarkAtZero pins the ctxplan view lever OFF by default (0): it
// rewrites in-flight turn history, so it stays opt-in until a watched-live witness (its bench
// witness — 13.3x fewer resident, 100% exact recall — is tracked by the token-defaults scorecard).
func TestTokenDefault_CtxViewShipsDarkAtZero(t *testing.T) {
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Int("ctx-view-budget", 0`) {
			t.Errorf("%s must default --ctx-view-budget to 0 (off; rewrites in-flight history, gated until a live witness)", f)
		}
	}
}
