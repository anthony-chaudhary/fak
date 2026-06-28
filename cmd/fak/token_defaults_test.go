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
// The remaining off-by-default lever (ctxview) is pinned too — not to keep it off, but so it
// stays wired to its literal 0 default, making a future flip-on one deliberate edit that this
// test moves with, never a silent per-entrypoint drift. (Elision was flipped on by default once
// adversarial verification + a synthetic dogfood + a real-corpus prevalence scan supported it.)
// The byte-faithful provider-cache passthrough is locked separately by
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

// TestTokenDefault_ElideDefaultsOn pins oversized-result elision ON by default: the gateway
// const is a non-zero (default-on) byte threshold, and both front doors wire the flag to it, so
// an old oversized tool_result is shrunk to head+tail with no operator configuration. Wiring the
// flag to the const keeps the on/off decision a single edit to DefaultElideResultBytes.
func TestTokenDefault_ElideDefaultsOn(t *testing.T) {
	if gateway.DefaultElideResultBytes <= 0 {
		t.Fatalf("DefaultElideResultBytes must be default-on (>0), got %d", gateway.DefaultElideResultBytes)
	}
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`) {
			t.Errorf("%s must wire --elide-result-bytes to gateway.DefaultElideResultBytes (default-on threshold)", f)
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

// TestTokenDefaultsSnapshotFresh pins the committed scorecard doc to the binary's REAL
// source-derived defaults: it regenerates the markdown the same way `--markdown` does and asserts
// it byte-equals docs/serving/token-defaults-scorecard.md. This closes the regression hole that
// let the doc drift unnoticed — when oversized-result elision was flipped on by default, the
// committed snapshot kept advertising it OFF (4/6) to cost-conscious operators, because nothing
// bound the doc to the source. A future flip the doc would misreport now reds here, with the exact
// regenerate command. EOL is normalized so a CRLF checkout (autocrlf=true) is not a false failure.
func TestTokenDefaultsSnapshotFresh(t *testing.T) {
	const root = "../.."
	raw, err := os.ReadFile(root + "/docs/serving/token-defaults-scorecard.md")
	if err != nil {
		t.Fatalf("read committed snapshot: %v", err)
	}
	want := strings.ReplaceAll(string(raw), "\r\n", "\n")
	got := renderTokenDefaultsMarkdown(collectTokenDefaultsScorecard(root)["corpus"].(map[string]any))
	if want != got {
		t.Errorf("docs/serving/token-defaults-scorecard.md is STALE vs the source-derived defaults — regenerate it:\n"+
			"  go run ./cmd/fak token-defaults-scorecard --markdown > docs/serving/token-defaults-scorecard.md\n"+
			"committed snapshot len=%d, regenerated len=%d", len(want), len(got))
	}
}

// TestTokenDefaultsLeversDerivedFromSource guards the anti-gaming rule for THIS scorecard: each
// lever's on/off must be DERIVED from the entrypoint source, never a hardcoded roster claim. It
// asserts the elision flip is reflected (elideresult ON), the lone opt-in lever is off-and-gated
// (ctxview), and the headline counters match (5 of 6 stacked), so the scorecard cannot report a
// default that contradicts the binary.
func TestTokenDefaultsLeversDerivedFromSource(t *testing.T) {
	c := collectTokenDefaultsScorecard("../..")["corpus"].(map[string]any)
	if got := c["stacked_on"].(int); got != 5 {
		t.Errorf("stacked_on derived = %d, want 5 (5/6 safe savers on by default)", got)
	}
	if got := c["levers_total"].(int); got != 6 {
		t.Errorf("levers_total = %d, want 6", got)
	}
	on := map[string]bool{}
	gated := map[string]bool{}
	for _, raw := range c["lever_status"].([]map[string]any) {
		on[raw["key"].(string)] = raw["on"].(bool)
		gated[raw["key"].(string)] = raw["gated"].(bool)
	}
	if !on["elideresult"] {
		t.Errorf("elideresult must derive ON from source (the default-on flip), got OFF")
	}
	if on["ctxview"] {
		t.Errorf("ctxview must derive OFF (opt-in, gated), got ON")
	}
	if !gated["ctxview"] {
		t.Errorf("the off-by-default ctxview lever must carry a documented gate")
	}
}
