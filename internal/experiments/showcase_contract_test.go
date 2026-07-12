package experiments

// Contract test for the governed code-review showcase (issue #3291, epic #3256).
//
// The showcase lives at experiments/agent-runtime-showcase/ as a runnable app
// (showcase.py) plus a captured witness (EXAMPLE-OUTPUT.md) and a machine-readable
// contract (showcase-contract.json). Those three artifacts are authored by hand and
// nothing gated them together: an edit to the app or the transcript that drifts from
// the declared contract would go unnoticed. The showcase's own contract even names
// this as its invalidating assumption ("the preset verdicts are stable ... a
// policy-grammar change would drift the captured witness").
//
// This test is that gate. It reads the sibling artifacts from disk (no import edge —
// they are not part of this Go package) and asserts the contract, the human-declared
// summary, and the machine-parsed audit tail all agree on the load-bearing facts:
// a 7-step trajectory, 4 ALLOW / 3 DENY across three distinct governance paths, the
// cost cap that halts spend, and the least-agency preset that is a real file. If any
// of those drift, the experiments lane gate (go test ./internal/experiments) reds.
//
// It deliberately does NOT re-run showcase.py or invoke the fak kernel — it guards
// the captured proof's internal consistency, which is the part that silently rots.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// repoRoot walks up from the test's working directory (the package dir) until it
// finds go.mod, which marks the module root. The showcase is a repo-root-relative
// sibling, so the test resolves it from there rather than hard-coding "../..".
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repo root (go.mod) not found walking up from test dir")
		}
		dir = parent
	}
}

type showcaseContract struct {
	Issue              int  `json:"issue"`
	Epic               int  `json:"epic"`
	IssueFullyResolved bool `json:"issue_fully_resolved"`
	Witness            struct {
		Result string `json:"result"`
		Preset string `json:"preset"`
	} `json:"witness"`
}

type auditEntry struct {
	Seq          int    `json:"seq"`
	Tool         string `json:"tool"`
	Verdict      string `json:"verdict"`
	Reason       string `json:"reason"`
	By           string `json:"by"`
	CumCostCents int    `json:"cum_cost_cents"`
}

// extractJSONL pulls the lines of the first ```jsonl fenced block out of a markdown
// document (the append-only audit tail in EXAMPLE-OUTPUT.md).
func extractJSONL(md string) []string {
	var out []string
	in := false
	for _, ln := range strings.Split(md, "\n") {
		t := strings.TrimSpace(ln)
		switch {
		case !in && strings.HasPrefix(t, "```jsonl"):
			in = true
		case in && strings.HasPrefix(t, "```"):
			return out
		case in && t != "":
			out = append(out, t)
		}
	}
	return out
}

// countBefore returns the integer N in "N <verdict>" within s, or -1 if absent.
func countBefore(s, verdict string) int {
	m := regexp.MustCompile(`(\d+)\s+` + verdict).FindStringSubmatch(s)
	if m == nil {
		return -1
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

func TestShowcaseContractMatchesWitness(t *testing.T) {
	root := repoRoot(t)
	base := filepath.Join(root, "experiments", "agent-runtime-showcase")

	// --- the declared contract ---
	cb, err := os.ReadFile(filepath.Join(base, "showcase-contract.json"))
	if err != nil {
		t.Fatalf("read contract: %v", err)
	}
	var c showcaseContract
	if err := json.Unmarshal(cb, &c); err != nil {
		t.Fatalf("parse contract: %v", err)
	}
	if c.Issue != 3291 {
		t.Errorf("contract issue = %d, want 3291", c.Issue)
	}
	if c.Epic != 3256 {
		t.Errorf("contract epic = %d, want 3256", c.Epic)
	}
	// Honesty boundary: this experiments/ dogfood must never self-declare full
	// closure while the app is not promoted to examples/ and #D1/#D2 are unwired.
	if c.IssueFullyResolved {
		t.Error("contract issue_fully_resolved = true; the experiments/ dogfood must stay false until examples/ promotion + #D1/#D2 wiring land")
	}

	// --- the captured witness (audit tail) ---
	mb, err := os.ReadFile(filepath.Join(base, "EXAMPLE-OUTPUT.md"))
	if err != nil {
		t.Fatalf("read witness: %v", err)
	}
	jsonl := extractJSONL(string(mb))
	if len(jsonl) == 0 {
		t.Fatal("no ```jsonl audit tail found in EXAMPLE-OUTPUT.md")
	}
	var entries []auditEntry
	for i, ln := range jsonl {
		var e auditEntry
		if err := json.Unmarshal([]byte(ln), &e); err != nil {
			t.Fatalf("parse audit line %d: %v\n%s", i+1, err, ln)
		}
		entries = append(entries, e)
	}

	// A1: the captured task is a 7-step trajectory.
	if len(entries) != 7 {
		t.Errorf("audit tail has %d steps, want 7", len(entries))
	}

	var allow, deny, finalCum, maxSeq int
	denyPaths := map[string]string{} // reason -> deciding component
	for _, e := range entries {
		switch e.Verdict {
		case "ALLOW":
			allow++
		case "DENY":
			deny++
			denyPaths[e.Reason] = e.By
		default:
			t.Errorf("seq %d: unexpected verdict %q", e.Seq, e.Verdict)
		}
		if e.Seq >= maxSeq {
			maxSeq, finalCum = e.Seq, e.CumCostCents
		}
	}
	if allow != 4 {
		t.Errorf("ALLOW count = %d, want 4", allow)
	}
	if deny != 3 {
		t.Errorf("DENY count = %d, want 3", deny)
	}

	// A3 + A4: the three distinct governance refusal paths must all fire, each
	// decided by the expected component.
	wantPaths := map[string]string{
		"POLICY_BLOCK": "gitgate", // dangerous git action blocked by policy
		"DEFAULT_DENY": "monitor", // unlisted tool denied by least-agency default
		"COST_CAP":     "budget",  // spend ceiling halts the run
	}
	for reason, by := range wantPaths {
		got, ok := denyPaths[reason]
		if !ok {
			t.Errorf("witness missing DENY path %s", reason)
			continue
		}
		if got != by {
			t.Errorf("DENY %s decided by %q, want %q", reason, got, by)
		}
	}

	// A4: the cost cap halted the run — cumulative spend stayed at 47c because the
	// blocked final step's cost was never added.
	if finalCum != 47 {
		t.Errorf("final cum_cost_cents = %d, want 47 (cap must halt, not spend)", finalCum)
	}

	// Cross-artifact: the human-declared witness.result summary must agree with the
	// machine-parsed audit tail. If the app drifts and only one side is updated,
	// this reds.
	res := c.Witness.Result
	if got := countBefore(res, "ALLOW"); got != allow {
		t.Errorf("contract result says %d ALLOW, audit tail has %d", got, allow)
	}
	if got := countBefore(res, "DENY"); got != deny {
		t.Errorf("contract result says %d DENY, audit tail has %d", got, deny)
	}
	for reason := range wantPaths {
		if !strings.Contains(res, reason) {
			t.Errorf("contract result missing governance path %s", reason)
		}
	}
	if !strings.Contains(res, "cap 50c") {
		t.Errorf("contract result missing 'cap 50c'; got %q", res)
	}

	// A2: the least-agency floor is a real preset file, named identically by the
	// contract and referenced by the runnable app.
	const preset = "examples/presets/coding-agent-safe.json"
	if c.Witness.Preset != preset {
		t.Errorf("contract preset = %q, want %q", c.Witness.Preset, preset)
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(preset))); err != nil {
		t.Errorf("least-agency preset %s not found on disk: %v", preset, err)
	}
	app, err := os.ReadFile(filepath.Join(base, "showcase.py"))
	if err != nil {
		t.Fatalf("read showcase.py: %v", err)
	}
	if !strings.Contains(string(app), preset) {
		t.Errorf("showcase.py does not reference the least-agency preset %s", preset)
	}
}
