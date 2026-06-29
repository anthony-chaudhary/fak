// Package auditpane is the one rollup over the tree's many tools/*_audit.py auditors.
//
// # The gap it closes
//
// The repo carries two dozen standalone auditors under tools/*_audit.py (crash,
// security, history-leak, session, plan, readme-freshness, ...). Each is a self-contained
// argparse main() with its own paired *_test.py. They share NO entrypoint: an operator who
// wants to answer "is anything failing an audit right now?" can only `ls tools/*_audit.py`
// and run each by name. There is no meta-runner. (The journal hash-chain `fak audit
// verify|export` in cmd/fak/audit.go is a DIFFERENT thing — the guard decision journal —
// and is explicitly out of scope here.)
//
// This package is the fold: discover the auditors, run each one bounded (degrade to SKIP on
// timeout, the way tools/fresh_status.py folds plan_audit), and roll every per-auditor
// verdict into one schema/ok/verdict control-pane envelope plus a --check CI gate.
//
// # The common verdict contract
//
// Most auditors do not yet expose a uniform machine verdict, so the fold DEFINES one and
// reads each auditor through it (precedence, highest first):
//
//   - the auditor could not be executed at all (spawn failed) -> ERROR (trips the rollup);
//   - the bounded run exceeded its deadline               -> SKIP  (advisory, never trips);
//   - the auditor printed a JSON envelope with an explicit SKIP-ish verdict
//     (SKIP/PREREQ_MISSING/BLOCKED/HOST_GATED/...) -> SKIP;
//   - the auditor printed a JSON envelope with an "ok" boolean -> PASS if true, else FAIL;
//   - the auditor printed a JSON envelope naming a verdict -> PASS if it is a pass-ish
//     token (OK/PASS/GREEN/CLEAN/...), else FAIL;
//   - otherwise the UNIVERSAL fallback is the process exit code: 0 -> PASS, non-zero -> FAIL.
//
// The runner tries `<auditor> --json` first (most auditors expose it and emit the richest
// envelope, including their own SKIP verdicts) and retries the auditor with no flags when
// argparse rejects --json, so the five auditors without a --json flag still fold through
// their exit code. The honest limitation of this FIRST contract: a host-gated auditor that
// exits non-zero because its prerequisite is absent here — and does not yet emit a SKIP-ish
// verdict — shows FAIL, not SKIP. Teaching such an auditor a SKIP verdict is how it degrades
// cleanly; until then the rollup's reason field carries the exit code + a stderr tail so an
// operator sees why.
//
// # Layering
//
// This is a tool leaf (architest tier 1): it imports nothing internal and is NOT on the
// internal/registrations request-path closure, so its os/exec of the Python auditors is
// off the live decision path — the same off-hot-path shell-out posture as internal/
// cadencereport and internal/gardenbundle. The pure surface (Discover / Classify / Fold /
// CheckGate) is split from the live subprocess runner so the fold is unit-testable without
// running a single auditor.
package auditpane

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// SCHEMA is the control-pane envelope version this rollup emits.
const SCHEMA = "fak-audit-control-pane/1"

// The four folded per-auditor verdicts (the common contract this package defines).
const (
	VerdictPass  = "PASS"  // the auditor ran and reported success
	VerdictFail  = "FAIL"  // the auditor ran and reported a failure (trips the rollup)
	VerdictSkip  = "SKIP"  // degraded: timed out, or the auditor itself reported a skip-ish verdict
	VerdictError = "ERROR" // the auditor could not be executed at all (trips the rollup)
)

// passTokens / skipTokens are the verdict strings the contract maps to PASS / SKIP when an
// auditor names a verdict in its envelope but does not carry an explicit ok boolean. Any
// other named verdict is treated as FAIL. Compared upper-cased.
var passTokens = map[string]bool{
	"OK": true, "PASS": true, "PASSED": true, "GREEN": true,
	"CLEAN": true, "ALL_CLEAR": true, "ALLGREEN": true, "ALL_GREEN": true,
}

var skipTokens = map[string]bool{
	"SKIP": true, "SKIPPED": true, "PREREQ_MISSING": true, "PREREQ": true,
	"BLOCKED": true, "NOT_APPLICABLE": true, "NA": true, "N/A": true,
	"ABSTAIN": true, "PRECREDENTIAL": true, "GATED": true, "HOST_GATED": true,
	"DEFERRED": true, "INCONCLUSIVE": true,
}

// Result is one auditor's folded verdict.
type Result struct {
	Name     string `json:"name"`             // the auditor's file name, e.g. "security_audit.py"
	Verdict  string `json:"verdict"`          // PASS | FAIL | SKIP | ERROR
	OK       bool   `json:"ok"`               // true for PASS/SKIP, false for FAIL/ERROR
	ExitCode int    `json:"exit_code"`        // the process exit code (-1 if it never ran)
	Schema   string `json:"schema,omitempty"` // the auditor's own envelope schema, when it emitted one
	Reason   string `json:"reason"`           // one-line why
}

// Payload is the rollup control-pane envelope (the same schema/ok/verdict shape the rest of
// the fak control-pane family emits, so a loop runner reads it identically).
type Payload struct {
	Schema     string   `json:"schema"`
	OK         bool     `json:"ok"`
	Verdict    string   `json:"verdict"`
	Finding    string   `json:"finding"`
	Reason     string   `json:"reason"`
	NextAction string   `json:"next_action"`
	Workspace  string   `json:"workspace,omitempty"`
	Commit     string   `json:"commit,omitempty"`
	Total      int      `json:"total"`
	Passed     int      `json:"passed"`
	Failed     int      `json:"failed"`
	Skipped    int      `json:"skipped"`
	Errored    int      `json:"errored"`
	Results    []Result `json:"results"`
}

// RunOutcome is the raw result of executing one auditor — the seam between the live
// subprocess runner and the pure Classify, so Classify is unit-testable with no subprocess.
type RunOutcome struct {
	ExitCode int    // process exit code; -1 when the process never produced one
	Stdout   []byte // captured stdout (where an auditor prints its JSON envelope)
	Stderr   string // captured stderr (a tail is surfaced on FAIL)
	TimedOut bool   // the bounded run exceeded its deadline -> SKIP
	SpawnErr string // the auditor could not be executed at all -> ERROR
}

// envelope is the slice of an auditor's JSON output the contract reads.
type envelope struct {
	schema  string
	hasOK   bool
	ok      bool
	verdict string
}

// parseEnvelope extracts {schema, ok, verdict} from an auditor's stdout, tolerating an
// envelope wrapped in surrounding text by falling back to the first '{' .. last '}' span.
// Returns ok=false when there is no JSON object at all (the exit-code fallback then applies).
func parseEnvelope(stdout []byte) (envelope, bool) {
	raw := strings.TrimSpace(string(stdout))
	m, ok := decodeObject(raw)
	if !ok {
		// Tolerant retry: some auditors print a human line before the JSON body.
		if i, j := strings.IndexByte(raw, '{'), strings.LastIndexByte(raw, '}'); i >= 0 && j > i {
			m, ok = decodeObject(raw[i : j+1])
		}
	}
	if !ok {
		return envelope{}, false
	}
	var e envelope
	if s, isStr := m["schema"].(string); isStr {
		e.schema = s
	}
	if v, present := m["ok"]; present {
		if b, isBool := v.(bool); isBool {
			e.hasOK, e.ok = true, b
		}
	}
	if s, isStr := m["verdict"].(string); isStr {
		e.verdict = s
	}
	return e, true
}

// verdict folds an envelope into a (verdict, decided) decision per the common contract. When
// decided is false the envelope carried neither an ok boolean nor a verdict string, so the
// caller falls back to the exit code.
func (e envelope) classify() (string, bool) {
	if e.verdict != "" && skipTokens[strings.ToUpper(strings.TrimSpace(e.verdict))] {
		return VerdictSkip, true
	}
	if e.hasOK {
		if e.ok {
			return VerdictPass, true
		}
		return VerdictFail, true
	}
	if e.verdict != "" {
		if passTokens[strings.ToUpper(strings.TrimSpace(e.verdict))] {
			return VerdictPass, true
		}
		return VerdictFail, true
	}
	return "", false
}

// Classify maps one auditor's raw RunOutcome to a folded Result via the common verdict
// contract. Pure: this is the unit-tested heart of the fold.
func Classify(name string, o RunOutcome) Result {
	switch {
	case o.SpawnErr != "":
		return Result{Name: name, Verdict: VerdictError, OK: false, ExitCode: -1,
			Reason: "could not execute auditor: " + o.SpawnErr}
	case o.TimedOut:
		return Result{Name: name, Verdict: VerdictSkip, OK: true, ExitCode: -1,
			Reason: "timed out under the bounded run — degraded to SKIP"}
	}
	if e, ok := parseEnvelope(o.Stdout); ok {
		if v, decided := e.classify(); decided {
			return Result{
				Name: name, Verdict: v, OK: v == VerdictPass || v == VerdictSkip,
				ExitCode: o.ExitCode, Schema: e.schema, Reason: envelopeReason(e, v, o.ExitCode),
			}
		}
	}
	// Universal fallback: the process exit code.
	if o.ExitCode == 0 {
		return Result{Name: name, Verdict: VerdictPass, OK: true, ExitCode: 0,
			Reason: "exit 0 (no JSON verdict; exit-code contract)"}
	}
	return Result{Name: name, Verdict: VerdictFail, OK: false, ExitCode: o.ExitCode,
		Reason: exitFailReason(o)}
}

func envelopeReason(e envelope, verdict string, exit int) string {
	parts := []string{"verdict " + verdict}
	if e.verdict != "" {
		parts = append(parts, "auditor verdict "+e.verdict)
	}
	parts = append(parts, fmt.Sprintf("exit %d", exit))
	if e.schema != "" {
		parts = append(parts, "schema "+e.schema)
	}
	return strings.Join(parts, "; ")
}

func exitFailReason(o RunOutcome) string {
	reason := fmt.Sprintf("exit %d (no JSON verdict; exit-code contract)", o.ExitCode)
	if tail := lastLine(o.Stderr); tail != "" {
		reason += ": " + truncate(tail, 200)
	}
	return reason
}

// Fold rolls per-auditor Results into one control-pane Payload. Pure.
//
// Verdict ladder: any FAIL or ERROR trips the rollup to ACTION (a real audit failure, or an
// auditor that could not run, is something an operator must look at). SKIPs are advisory and
// never trip. Zero discovered auditors is itself ACTION (the discovery root is almost
// certainly wrong).
func Fold(results []Result, workspace, commit string) Payload {
	var passed, failed, skipped, errored int
	var failing []string
	for _, r := range results {
		switch r.Verdict {
		case VerdictPass:
			passed++
		case VerdictFail:
			failed++
			failing = append(failing, r.Name)
		case VerdictSkip:
			skipped++
		case VerdictError:
			errored++
			failing = append(failing, r.Name+" (error)")
		}
	}
	total := len(results)
	bad := failed + errored

	p := Payload{
		Schema: SCHEMA, Workspace: workspace, Commit: commit,
		Total: total, Passed: passed, Failed: failed, Skipped: skipped, Errored: errored,
		Results: results,
	}
	counts := fmt.Sprintf("%d auditors: %d pass, %d fail, %d skip, %d error",
		total, passed, failed, skipped, errored)

	switch {
	case total == 0:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "no_auditors"
		p.Reason = "discovered 0 tools/*_audit.py auditors — the workspace root is likely wrong"
		p.NextAction = "run from the repo root (where tools/ lives), or pass --workspace <repo>"
	case bad > 0:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "audits_failing"
		p.Reason = counts + "; failing: " + strings.Join(failing, ", ")
		p.NextAction = "re-run each failing auditor by name for detail (python tools/<name>); " +
			"a host-gated auditor that needs an absent prerequisite shows FAIL until it emits a SKIP verdict"
	default:
		p.OK, p.Verdict, p.Finding = true, "OK", "all_clear"
		p.Reason = counts
		if skipped > 0 {
			p.NextAction = "rollup is green; no auditor is failing (skips are advisory)"
		} else {
			p.NextAction = "rollup is green; every auditor passed"
		}
	}
	return p
}

// CheckGate is the pure --check decision over a folded Payload: exit code + message.
//
//	0  OK        — no auditor is failing or errored (green even with advisory skips)
//	1  ACTION    — at least one auditor FAILED or could not run
//	2  no work   — discovery found zero auditors (wrong root)
func CheckGate(p Payload) (int, string) {
	if p.Total == 0 {
		return 2, "AUDIT ROLLUP EMPTY: " + p.Reason
	}
	if !p.OK {
		return 1, "AUDIT ROLLUP FAIL: " + p.Reason
	}
	return 0, "AUDIT ROLLUP OK: " + p.Reason
}

// --- discovery --------------------------------------------------------------

// Discover lists the tools/*_audit.py auditors under root, excluding the paired *_test.py
// files (the glob already excludes them, but the explicit filter documents the contract and
// guards a stray *_audit_test.py). Sorted for deterministic output.
func Discover(root string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(root, "tools", "*_audit.py"))
	if err != nil {
		return nil, err
	}
	var out []string
	for _, m := range matches {
		if strings.HasSuffix(filepath.Base(m), "_test.py") {
			continue
		}
		out = append(out, m)
	}
	sort.Strings(out)
	return out, nil
}

// --- live runner (off the request-path closure; not unit-tested) ------------

// Options configures a live Collect.
type Options struct {
	Python      string        // the python interpreter to run auditors with (default "python")
	Timeout     time.Duration // per-auditor bound (default 30s)
	Concurrency int           // bounded parallel auditors (default 8)
}

func (o Options) withDefaults() Options {
	if o.Python == "" {
		o.Python = "python"
	}
	if o.Timeout <= 0 {
		o.Timeout = 30 * time.Second
	}
	if o.Concurrency <= 0 {
		o.Concurrency = 8
	}
	return o
}

// Collect discovers the auditors under root, runs each one bounded (degrade-to-SKIP on
// timeout) with bounded concurrency, and returns the folded Results in stable name order.
func Collect(root string, opts Options) ([]Result, error) {
	opts = opts.withDefaults()
	auditors, err := Discover(root)
	if err != nil {
		return nil, err
	}
	results := make([]Result, len(auditors))
	sem := make(chan struct{}, opts.Concurrency)
	var wg sync.WaitGroup
	for i, script := range auditors {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, path string) {
			defer wg.Done()
			defer func() { <-sem }()
			o := runAuditor(root, path, opts.Python, opts.Timeout)
			results[idx] = Classify(filepath.Base(path), o)
		}(i, script)
	}
	wg.Wait()
	return results, nil
}

// runAuditor runs one auditor bounded by timeout. It tries `--json` first and retries with no
// flags when argparse rejects the flag, so a non-JSON auditor still folds through its exit
// code. The cwd is root so an auditor's repo-relative reads resolve.
func runAuditor(root, script, python string, timeout time.Duration) RunOutcome {
	o := execOnce(root, python, []string{script, "--json"}, timeout)
	if o.SpawnErr == "" && !o.TimedOut && o.ExitCode == 2 && rejectedFlag(o.Stderr) {
		o = execOnce(root, python, []string{script}, timeout)
	}
	return o
}

// rejectedFlag reports whether an argparse exit-2 stderr names an unrecognized --json flag.
func rejectedFlag(stderr string) bool {
	s := strings.ToLower(stderr)
	return strings.Contains(s, "unrecognized arguments") ||
		strings.Contains(s, "no such option") ||
		(strings.Contains(s, "error:") && strings.Contains(s, "--json"))
}

func execOnce(root, python string, args []string, timeout time.Duration) RunOutcome {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Dir = root
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	out := RunOutcome{Stdout: []byte(stdout.String()), Stderr: stderr.String(), ExitCode: -1}
	if ctx.Err() == context.DeadlineExceeded {
		out.TimedOut = true
		return out
	}
	if err == nil {
		out.ExitCode = 0
		return out
	}
	if ee, ok := err.(*exec.ExitError); ok {
		out.ExitCode = ee.ExitCode()
		return out
	}
	// The process never started (interpreter missing, permission, ...).
	out.SpawnErr = err.Error()
	return out
}

// HeadCommit returns the short HEAD sha for stamping the envelope, or "" on any failure.
// Execs the compiled git binary off the hot path.
func HeadCommit(root string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--short", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// --- small stdlib-only helpers ---------------------------------------------

func decodeObject(raw string) (map[string]any, bool) {
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil || m == nil {
		return nil, false
	}
	return m, true
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(lines[i]); t != "" {
			return t
		}
	}
	return ""
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
