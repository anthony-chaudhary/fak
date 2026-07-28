package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/closureaudit"
)

// acceptanceFixtureBody is the issue body every fixture test resolves. Its
// acceptance section names one Go symbol, one refusal token, and one production
// path; the surrounding prose names things that must NEVER be extracted.
const acceptanceFixtureBody = "## Problem\n" +
	"Spill is decided nowhere. `NeverInProse` is only mentioned here.\n" +
	"\n" +
	"## Acceptance\n" +
	"- `ResolveHostSpill` decides the spill target\n" +
	"- the refusal token `OVERLAY_WOULD_GATE` is declared\n" +
	"- it is reachable from `cmd/fak/dispatch_tick_preflight.go`\n"

// newAcceptanceFixtureRepo builds a throwaway git repo whose origin/main carries a
// code-complete-but-UNWIRED primitive: ResolveHostSpill is declared, table-tested,
// and referenced by architest — and by nothing on a production path. That is the
// exact shape #5435 says a symbol-presence check alone would mis-report as SHIPPED.
// Returns the repo root and the path of the issue body (kept OUTSIDE the repo so it
// can never satisfy its own probes).
func newAcceptanceFixtureRepo(t *testing.T) (repo, bodyPath string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	base := t.TempDir()
	repo = filepath.Join(base, "repo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(rel, content string) {
		t.Helper()
		p := filepath.Join(repo, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "fixture")
	git("config", "commit.gpgsign", "false")

	write("internal/hostplacement/spill.go",
		"package hostplacement\n\n// ResolveHostSpill decides the spill target.\nfunc ResolveHostSpill(host string) string { return host }\n")
	write("internal/hostplacement/spill_test.go",
		"package hostplacement\n\nimport \"testing\"\n\nfunc TestResolveHostSpill(t *testing.T) { _ = ResolveHostSpill(\"a\") }\n")
	write("internal/architest/architest_test.go",
		"package architest\n\n// layering pin mentions ResolveHostSpill; proves nothing about wiring\n")
	write("dos.toml", "[reasons.OVERLAY_WOULD_GATE]\nmessage = \"the overlay would gate\"\n")
	// The named seam EXISTS but does not call the primitive: code complete, wired
	// to nothing.
	write("cmd/fak/dispatch_tick_preflight.go",
		"package main\n\nfunc dispatchProbeWorkerCount() int { return 1 }\n")
	git("add", "-A")
	git("commit", "-q", "-m", "seed: primitive landed, no production caller")
	git("update-ref", "refs/remotes/origin/main", "HEAD")

	bodyPath = filepath.Join(base, "issue.md")
	if err := os.WriteFile(bodyPath, []byte(acceptanceFixtureBody), 0o644); err != nil {
		t.Fatal(err)
	}
	return repo, bodyPath
}

// acceptanceFixtureWire lands a production caller on origin/main, turning the
// unwired primitive into a wired one.
func acceptanceFixtureWire(t *testing.T, repo string) {
	t.Helper()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	p := filepath.Join(repo, "cmd", "fak", "dispatch_tick_preflight.go")
	body := "package main\n\nimport \"example.invalid/internal/hostplacement\"\n\n" +
		"func dispatchProbeWorkerCount() int { _ = hostplacement.ResolveHostSpill(\"a\"); return 1 }\n"
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "-A")
	git("commit", "-q", "-m", "wire the primitive into the preflight seam")
	git("update-ref", "refs/remotes/origin/main", "HEAD")
}

// TestDispatchAcceptanceResolveThroughRouter drives the REAL argv entry point —
// runDispatch("acceptance-resolve", ...) — against a real git ref, and is the
// non-vacuity witness for the router wiring in dispatch_order.go: delete the
// `case "acceptance-resolve"` arm and this test fails with exit 2 / "unknown
// subcommand" instead of a verdict.
//
// It also pins the discriminator the ticket is about: with the acceptance symbols
// ALL present on trunk but reaching zero production callers the verdict is PARTIAL
// with the remaining work named "wire it into <seam>" — not SHIPPED.
func TestDispatchAcceptanceResolveThroughRouter(t *testing.T) {
	repo, bodyPath := newAcceptanceFixtureRepo(t)

	run := func() (acceptanceResolveOut, string) {
		t.Helper()
		var stdout, stderr bytes.Buffer
		code := runDispatch(&stdout, &stderr, []string{
			"acceptance-resolve", "--workspace", repo, "--body-file", bodyPath,
			"--ref", "origin/main", "--json",
		})
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
		}
		var got acceptanceResolveOut
		if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
			t.Fatalf("json: %v\n%s", err, stdout.String())
		}
		return got, stdout.String()
	}

	got, raw := run()
	if got.Schema != acceptanceResolveSchema || got.Ref != "origin/main" || got.Resolution == nil {
		t.Fatalf("envelope: %+v\n%s", got, raw)
	}
	res := got.Resolution
	if res.Verdict != closureaudit.AcceptancePartial {
		t.Fatalf("verdict=%q want PARTIAL (present but caller-less): %+v", res.Verdict, res)
	}
	if !strings.Contains(res.Remaining, "wire it into cmd/fak/dispatch_tick_preflight.go") {
		t.Fatalf("remaining must name the seam, got %q", res.Remaining)
	}
	if len(res.Callers) != 1 || !res.Callers[0].Unwired || res.Callers[0].Production != 0 {
		t.Fatalf("caller-count: %+v", res.Callers)
	}
	if res.Callers[0].DefFile != "internal/hostplacement/spill.go" {
		t.Fatalf("declaration site: %+v", res.Callers[0])
	}
	// The problem prose outside the acceptance region is never mined.
	for _, n := range res.Acceptance.Needles {
		if n.Text == "NeverInProse" {
			t.Fatalf("prose outside the acceptance region was extracted: %+v", res.Acceptance.Needles)
		}
	}

	// Land the wiring on origin/main; the same argv now reports SHIPPED.
	acceptanceFixtureWire(t, repo)
	got, raw = run()
	if got.Resolution.Verdict != closureaudit.AcceptanceShipped {
		t.Fatalf("after wiring verdict=%q want SHIPPED: %s", got.Resolution.Verdict, raw)
	}
}

// TestDispatchAcceptanceResolveDeclinesRatherThanGuessing pins the design-honesty
// requirement: a body whose acceptance cannot be extracted resolves UNKNOWN, and the
// reason says so — never a confident SHIPPED.
func TestDispatchAcceptanceResolveDeclinesRatherThanGuessing(t *testing.T) {
	repo, _ := newAcceptanceFixtureRepo(t)
	body := filepath.Join(t.TempDir(), "vague.md")
	if err := os.WriteFile(body, []byte("## Problem\nIt is broken. `ResolveHostSpill` exists already.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatch(&stdout, &stderr, []string{
		"acceptance-resolve", "--workspace", repo, "--body-file", body, "--ref", "origin/main",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, closureaudit.AcceptanceUnknown) ||
		!strings.Contains(out, closureaudit.ReasonNoAcceptanceSection) {
		t.Fatalf("want UNKNOWN(NO_ACCEPTANCE_SECTION), got:\n%s", out)
	}
	if strings.Contains(out, closureaudit.AcceptanceShipped) {
		t.Fatalf("an unextractable body must never read as SHIPPED:\n%s", out)
	}
}

// TestDispatchAcceptanceResolveSymbolCallerCount drives the caller-count half alone
// through the router — the self-application check a worker runs against their own
// new symbol before claiming they are done.
func TestDispatchAcceptanceResolveSymbolCallerCount(t *testing.T) {
	repo, _ := newAcceptanceFixtureRepo(t)
	var stdout, stderr bytes.Buffer
	if code := runDispatch(&stdout, &stderr, []string{
		"acceptance-resolve", "--workspace", repo, "--symbol", "ResolveHostSpill", "--ref", "origin/main",
	}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "UNWIRED") || !strings.Contains(stdout.String(), "production=0") {
		t.Fatalf("caller-count render: %s", stdout.String())
	}
	acceptanceFixtureWire(t, repo)
	stdout.Reset()
	if code := runDispatch(&stdout, &stderr, []string{
		"acceptance-resolve", "--workspace", repo, "--symbol", "ResolveHostSpill", "--ref", "origin/main",
	}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "WIRED") || !strings.Contains(stdout.String(), "production=1") {
		t.Fatalf("after wiring: %s", stdout.String())
	}
}

// TestDispatchAcceptanceResolveUsage pins the mutually-exclusive selector.
func TestDispatchAcceptanceResolveUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDispatch(&stdout, &stderr, []string{"acceptance-resolve"}); code != 2 {
		t.Fatalf("no selector: exit=%d", code)
	}
	stderr.Reset()
	if code := runDispatch(&stdout, &stderr, []string{"acceptance-resolve", "--issue", "1", "--symbol", "X"}); code != 2 {
		t.Fatalf("two selectors: exit=%d", code)
	}
}

// TestDispatchClosureAuditResolveAcceptance is the non-vacuity witness for the
// BACKLOG-VIEW wiring: `fak dispatch closure-audit --resolve-acceptance` must fold
// the resolver over its still-open issues. Delete the resolveAcceptanceForReport
// call in runDispatchClosureAudit and this test fails (acceptance_counts absent, no
// STALE_OPEN line).
//
// The fixture reproduces the ticket's headline defect: issue #5319 is OPEN with NO
// commit whose subject mentions it, so the subject-grep grader buckets it OPEN — yet
// every symbol its acceptance names is present and wired on trunk. Acceptance
// resolution catches it; the subject grep never can.
func TestDispatchClosureAuditResolveAcceptance(t *testing.T) {
	repo, _ := newAcceptanceFixtureRepo(t)
	acceptanceFixtureWire(t, repo) // the work is fully landed AND wired on trunk

	withClosureAuditSeams(t,
		[]closureaudit.Issue{{Number: 5319, State: "OPEN", Title: "spill target is decided nowhere"}},
		[]closureaudit.Commit{{SHA: "abcdef12", Subject: "feat(hostplacement): decide the spill target (#5171, epic #5170)"}},
		map[string]closureaudit.Audit{"abcdef12": {Verdict: "OK", Witness: "diff-witnessed"}},
	)
	prev := acceptanceIssueBody
	t.Cleanup(func() { acceptanceIssueBody = prev })
	acceptanceIssueBody = func(_ string, n int) (string, error) {
		if n != 5319 {
			t.Fatalf("unexpected issue fetch: %d", n)
		}
		return acceptanceFixtureBody, nil
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchClosureAudit(&stdout, &stderr, []string{
		"--workspace", repo, "--resolve-acceptance", "--ref", "origin/main", "--json",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var rep closureaudit.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}
	if rep.AcceptanceCounts[closureaudit.AcceptanceShipped] != 1 {
		t.Fatalf("acceptance_counts=%v want one SHIPPED (the stale open a subject grep misses)", rep.AcceptanceCounts)
	}
	if len(rep.Issues) != 1 || rep.Issues[0].Acceptance == nil {
		t.Fatalf("resolution not attached to the graded issue: %+v", rep.Issues)
	}
	if got := rep.Issues[0].Bucket; got != closureaudit.Open {
		t.Fatalf("subject-grep bucket=%q want OPEN — the defect being corrected", got)
	}
	stale := closureaudit.StaleOpens(rep)
	if len(stale) != 1 || stale[0] != 5319 {
		t.Fatalf("stale opens=%v want [5319]", stale)
	}

	// The human card names it too, so an operator reading the backlog view sees it.
	stdout.Reset()
	if code := runDispatchClosureAudit(&stdout, &stderr, []string{
		"--workspace", repo, "--resolve-acceptance", "--ref", "origin/main",
	}); code != 0 {
		t.Fatalf("human exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "STALE_OPEN") || !strings.Contains(stdout.String(), "#5319") {
		t.Fatalf("human card missing the stale-open line:\n%s", stdout.String())
	}

	// Without the flag the card is unchanged: no acceptance block, no gh reads.
	stdout.Reset()
	if code := runDispatchClosureAudit(&stdout, &stderr, []string{"--workspace", repo, "--json"}); code != 0 {
		t.Fatalf("plain exit=%d stderr=%s", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "acceptance_counts") {
		t.Fatalf("acceptance resolution must be opt-in:\n%s", stdout.String())
	}
}

// TestDispatchClosureAuditAcceptanceLimitStaysVisible pins the cap's honesty. The
// per-issue probe budget is real -- each issue costs a gh read plus git probes -- so a
// cap has to exist. What it must NOT do is drop the tail silently: an issue nobody
// looked at would then render identically to one that was probed and found still open,
// which is this verb's own defect (#5435) reproduced one level up by the tool built to
// remove it.
//
// So the two halves are asserted together: the capped issue is still present with
// UNKNOWN(NOT_RESOLVED), AND it cost no issue-body read. Either alone is the wrong fix
// -- keeping it visible by probing it anyway would discard the budget, and honouring the
// budget by dropping it would discard the honesty.
func TestDispatchClosureAuditAcceptanceLimitStaysVisible(t *testing.T) {
	repo, _ := newAcceptanceFixtureRepo(t)
	acceptanceFixtureWire(t, repo)

	withClosureAuditSeams(t,
		[]closureaudit.Issue{
			{Number: 5319, State: "OPEN", Title: "older, falls past the cap"},
			{Number: 5320, State: "OPEN", Title: "newer, fits under the cap"},
		},
		[]closureaudit.Commit{{SHA: "abcdef12", Subject: "feat(hostplacement): decide the spill target (#5171)"}},
		map[string]closureaudit.Audit{"abcdef12": {Verdict: "OK", Witness: "diff-witnessed"}},
	)
	prev := acceptanceIssueBody
	t.Cleanup(func() { acceptanceIssueBody = prev })
	var fetched []int
	acceptanceIssueBody = func(_ string, n int) (string, error) {
		fetched = append(fetched, n)
		return acceptanceFixtureBody, nil
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchClosureAudit(&stdout, &stderr, []string{
		"--workspace", repo, "--resolve-acceptance", "--ref", "origin/main",
		"--acceptance-limit", "1", "--json",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var rep closureaudit.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, stdout.String())
	}

	// Newest-first: 5320 is probed, 5319 is the capped tail.
	if len(fetched) != 1 || fetched[0] != 5320 {
		t.Fatalf("cap must spend exactly one probe, on the newest issue; fetched=%v", fetched)
	}

	byNum := map[int]*closureaudit.Resolution{}
	for _, g := range rep.Issues {
		byNum[g.Number] = g.Acceptance
	}
	capped, ok := byNum[5319]
	if !ok || capped == nil {
		t.Fatalf("the capped issue vanished from the report -- a silent cap is the defect this verb exists to remove; got %+v", byNum)
	}
	if capped.Verdict != closureaudit.AcceptanceUnknown {
		t.Fatalf("capped issue verdict = %q, want %q -- an unprobed issue must never carry a confident verdict",
			capped.Verdict, closureaudit.AcceptanceUnknown)
	}
	if !strings.Contains(capped.Reason, closureaudit.ReasonNotResolved) {
		t.Fatalf("capped reason must name %s so a reader can tell unprobed from probed-and-open; got %q",
			closureaudit.ReasonNotResolved, capped.Reason)
	}
	if capped.Remaining == "" {
		t.Fatalf("capped issue must say how to get it probed; Remaining was empty")
	}
	if rep.AcceptanceCounts[closureaudit.AcceptanceUnknown] < 1 {
		t.Fatalf("the cap must show up in the counts, got %+v", rep.AcceptanceCounts)
	}
}
