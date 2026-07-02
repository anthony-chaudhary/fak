package main

// resume_admit_test.go — witnesses for the source-governor observability contract
// (#2173): a PRESENT-but-malformed policy refuses loudly (never silently permissive),
// a missing policy stays fail-open, `--explain` reports the whole governor posture in
// one command, and the gate_fail_open warning rows launchers append are visible to the
// explain fold but invisible to launch-pressure accounting.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// TestResumeAdmitMalformedPolicyRefuses: a policy file that exists but does not parse
// must REFUSE (exit 3) with the structured POLICY_MALFORMED reason. Launchers fail open
// on unexpected gate exits, so the old exit-2 usage error silently turned a policy typo
// into a fully permissive host — the exact hole #2173 names.
func TestResumeAdmitMalformedPolicyRefuses(t *testing.T) {
	dir := t.TempDir()
	pol := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(pol, []byte("{this is not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runResumeAdmit(&out, &errb, []string{
		"--policy", pol, "--ledger", filepath.Join(dir, "ledger.jsonl"), "--json"})
	if code != 3 {
		t.Fatalf("malformed policy: exit=%d, want 3 (REFUSE)\nstdout: %s\nstderr: %s",
			code, out.String(), errb.String())
	}
	var doc struct {
		Decision resume.SourceDecision `json:"decision"`
	}
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("decode --json output: %v\n%s", err, out.String())
	}
	if doc.Decision.Admit {
		t.Fatal("malformed policy admitted — a typo silently became permissive")
	}
	if doc.Decision.Reason != resume.ReasonPolicyMalformed {
		t.Fatalf("reason = %q, want %q", doc.Decision.Reason, resume.ReasonPolicyMalformed)
	}
}

// TestResumeAdmitMissingPolicyFailsOpen: a MISSING policy file is the permissive
// default (nobody configured the rail), distinct from the malformed refuse above. The
// gates are explicitly disabled by flags so the verdict is deterministic no matter how
// many live resumes the host census sees.
func TestResumeAdmitMissingPolicyFailsOpen(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runResumeAdmit(&out, &errb, []string{
		"--policy", filepath.Join(dir, "nope.json"),
		"--ledger", filepath.Join(dir, "nope.jsonl"),
		"--max-live", "0", "--max-per-window", "0", "--min-spacing-sec", "0",
		"--quiet"})
	if code != 0 {
		t.Fatalf("missing policy: exit=%d, want 0 (fail-open)\nstderr: %s", code, errb.String())
	}
}

// TestResumeAdmitExplainReportsPosture: one `--explain` call must surface the policy
// path + effective values, the ledger path, and the trailing-24h launched / deferred /
// gate_fail_open counts — the operator posture #2173 asks for, without reading .fak/
// or the scheduler actions by hand.
func TestResumeAdmitExplainReportsPosture(t *testing.T) {
	dir := t.TempDir()
	pol := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(pol, []byte(`{"schema":"fak.resume-source-policy.v1","default":{"max_live_resumes":10}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	led := filepath.Join(dir, "ledger.jsonl")
	recent := time.Now().UTC().Add(-time.Hour).Format("2006-01-02T15:04:05Z")
	rows := `{"ts":"` + recent + `","session":"s1","phase":"launched"}
{"ts":"` + recent + `","session":"s2","phase":"deferred","cause":"source_concurrency_gate"}
{"ts":"` + recent + `","phase":"gate_fail_open","cause":"source_governor_unavailable","reason":"no-fak-binary"}
`
	if err := os.WriteFile(led, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runResumeAdmit(&out, &errb, []string{
		"--policy", pol, "--ledger", led,
		"--max-live", "0", "--max-per-window", "0", "--min-spacing-sec", "0",
		"--explain"})
	if code != 0 {
		t.Fatalf("explain: exit=%d\nstdout: %s\nstderr: %s", code, out.String(), errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"source governor posture",
		pol,
		led,
		"launched=1 deferred=1 gate_fail_open=1",
		"WARNING:",
	} {
		if !bytes.Contains([]byte(got), []byte(want)) {
			t.Errorf("explain output missing %q:\n%s", want, got)
		}
	}
}

// TestResumeAdmitExplainJSON: --explain --json carries the machine-readable posture
// (policy existence/source, ledger stats, executable) alongside the decision.
func TestResumeAdmitExplainJSON(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runResumeAdmit(&out, &errb, []string{
		"--policy", filepath.Join(dir, "nope.json"),
		"--ledger", filepath.Join(dir, "nope.jsonl"),
		"--max-live", "0", "--max-per-window", "0", "--min-spacing-sec", "0",
		"--json", "--explain"})
	if code != 0 {
		t.Fatalf("explain --json: exit=%d\nstderr: %s", code, errb.String())
	}
	var doc struct {
		Explain struct {
			PolicyFileExists *bool  `json:"policy_file_exists"`
			LedgerExists     *bool  `json:"ledger_exists"`
			Executable       string `json:"executable"`
			Recent           struct {
				Launched24h int `json:"launched_24h"`
			} `json:"recent"`
		} `json:"explain"`
	}
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("decode: %v\n%s", err, out.String())
	}
	if doc.Explain.PolicyFileExists == nil || *doc.Explain.PolicyFileExists {
		t.Errorf("policy_file_exists should be present and false for a missing file")
	}
	if doc.Explain.LedgerExists == nil || *doc.Explain.LedgerExists {
		t.Errorf("ledger_exists should be present and false for a missing ledger")
	}
}

// TestScanGovernorLedgerStats: the trailing-24h fold classifies rows by phase, skips
// malformed lines, ignores stale rows for the window counts, and still tracks the most
// recent launch overall.
func TestScanGovernorLedgerStats(t *testing.T) {
	dir := t.TempDir()
	led := filepath.Join(dir, "ledger.jsonl")
	now := time.Now().UTC()
	recent := now.Add(-2 * time.Hour).Format("2006-01-02T15:04:05Z")
	stale := now.Add(-48 * time.Hour).Format("2006-01-02T15:04:05Z")
	rows := `{"ts":"` + recent + `","session":"a","phase":"launched"}
{"ts":"` + recent + `","session":"b"}
{"ts":"` + stale + `","session":"c","phase":"launched"}
{"ts":"` + recent + `","session":"d","phase":"deferred"}
{"ts":"` + recent + `","phase":"gate_fail_open"}
{"ts":"` + recent + `","session":"e","phase":"considered"}
not json at all
`
	if err := os.WriteFile(led, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	st := scanGovernorLedgerStats(led, now)
	if st.Launched24h != 2 { // the phase-less row is a launch too
		t.Errorf("Launched24h = %d, want 2", st.Launched24h)
	}
	if st.Deferred24h != 1 {
		t.Errorf("Deferred24h = %d, want 1", st.Deferred24h)
	}
	if st.FailOpen24h != 1 {
		t.Errorf("FailOpen24h = %d, want 1", st.FailOpen24h)
	}
	if st.LastLaunchUnix == 0 {
		t.Error("LastLaunchUnix not tracked")
	}
	// missing ledger yields zeros, not an error
	if z := scanGovernorLedgerStats(filepath.Join(dir, "nope.jsonl"), now); z != (governorLedgerStats{}) {
		t.Errorf("missing ledger: got %+v, want zeros", z)
	}
}

// TestGateFailOpenIsNotLaunchPressure: the governor-unavailable warning row must never
// count as a launch in the snapshot fold — otherwise the warning about a missing rail
// would itself consume the rate window once the rail returns.
func TestGateFailOpenIsNotLaunchPressure(t *testing.T) {
	if !isNonLaunchPhase("gate_fail_open") {
		t.Fatal("gate_fail_open must be a non-launch phase")
	}
	dir := t.TempDir()
	led := filepath.Join(dir, "ledger.jsonl")
	ts := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	if err := os.WriteFile(led, []byte(`{"ts":"`+ts+`","phase":"gate_fail_open"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	times, last := scanLaunchLedger(led)
	if len(times) != 0 || last != 0 {
		t.Fatalf("gate_fail_open counted as launch pressure: times=%v last=%d", times, last)
	}
}
