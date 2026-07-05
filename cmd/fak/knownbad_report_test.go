package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// writeReportLeases writes a JSONL lease fixture (the blastradius.Lease shape) the
// report verb folds the affected count over, so the test never touches the real dos
// lease ledger.
func writeReportLeases(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "leases.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write leases fixture: %v", err)
	}
	return path
}

// recordKnownBad records one signature over the given ledger and returns its signature id
// (parsed from the --json row) so the test can claim/reference it.
func recordKnownBad(t *testing.T, ledger string, now int64, tree, reason, by string, ttl string) string {
	t.Helper()
	args := []string{"record", "--tree", tree, "--reason", reason, "--by", by, "--ledger", ledger, "--json"}
	if ttl != "" {
		args = append(args, "--ttl", ttl)
	}
	var out, errb bytes.Buffer
	if rc := runKnownBad(&out, &errb, args, now); rc != 0 {
		t.Fatalf("record rc=%d stderr=%q", rc, errb.String())
	}
	var rec knownbad.Record
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("record --json not valid JSON: %v (%q)", err, out.String())
	}
	return rec.Signature
}

// An empty ledger folds to a quiet all-clear card and never pages (dry-run, so nothing
// posts). This is the honest "no shared blockers" a scheduled render must emit.
func TestKnownBadReportEmptyIsClear(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	var out, errb bytes.Buffer
	rc := runKnownBad(&out, &errb, []string{"report", "--ledger", ledger, "--dry-run"}, 1_700_000_000)
	if rc != 0 {
		t.Fatalf("report rc=%d stderr=%q", rc, errb.String())
	}
	if !strings.Contains(out.String(), "no shared blockers") {
		t.Fatalf("empty ledger should render the all-clear card:\n%s", out.String())
	}
	if strings.Contains(out.String(), "<!here>") {
		t.Fatalf("an all-clear render MUST NOT page:\n%s", out.String())
	}
}

// A live signature WITH an elected fixer whose broken tree intersects a live lease folds
// to a muted status card carrying the blast frame — 1 affected, 1 fixing, 0 parked — and
// does not page.
func TestKnownBadReportClaimedRendersStatusCard(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const now = int64(1_700_000_000)
	// Seed an open row and its superseding claim row directly — report reads the LEDGER,
	// not the lease store, so this exercises the fixer-claimed render without a real git
	// lease acquire (the exactly-one election itself is covered by the W5 claim tests).
	rec := knownbad.NewRecord("build", []string{"internal/foo/**"}, "foo broke", "agent-1", "", now-60, 0)
	if err := appendKnownBadRow(ledger, rec); err != nil {
		t.Fatalf("seed open row: %v", err)
	}
	if err := appendKnownBadRow(ledger, rec.WithClaim("fixer-7", now-30)); err != nil {
		t.Fatalf("seed claim row: %v", err)
	}

	leases := writeReportLeases(t,
		`{"lane":"lane-a","tree_globs":["internal/foo/bar.go"]}`, // intersects the broken tree
		`{"lane":"lane-b","tree_globs":["internal/other/**"]}`,   // disjoint
	)

	var out, errb bytes.Buffer
	rc := runKnownBad(&out, &errb, []string{
		"report", "--ledger", ledger, "--leases", leases, "--dry-run",
	}, now)
	if rc != 0 {
		t.Fatalf("report rc=%d stderr=%q", rc, errb.String())
	}
	got := out.String()
	if strings.Contains(got, "<!here>") {
		t.Fatalf("a fixer-claimed status card MUST NOT page:\n%s", got)
	}
	// Exactly one lease intersects -> 1 affected, 1 fixing (@fixer-7), 0 parked.
	for _, want := range []string{"1 affected", "1 fixing (@fixer-7)", "0 parked", "witness: pending"} {
		if !strings.Contains(got, want) {
			t.Fatalf("status card missing %q:\n%s", want, got)
		}
	}
}

// An UNCLAIMED signature older than --operator-after with an affected lease escalates to
// an operator page; --json emits the folded signature set (and never posts).
func TestKnownBadReportUnclaimedOverdueSurfacesToOperator(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const now = int64(1_700_000_000)
	// discovered 20 minutes ago, no claim -> overdue past the 15m default window.
	recordKnownBad(t, ledger, now-20*60, "internal/foo/**", "build", "agent-1", "")

	leases := writeReportLeases(t, `{"lane":"lane-a","tree_globs":["internal/foo/x.go"]}`)

	var out, errb bytes.Buffer
	rc := runKnownBad(&out, &errb, []string{
		"report", "--ledger", ledger, "--leases", leases, "--repo-url", "https://github.com/o/r", "--dry-run",
	}, now)
	if rc != 0 {
		t.Fatalf("report rc=%d stderr=%q", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "<!here>") {
		t.Fatalf("an unclaimed overdue signature must page:\n%s", got)
	}
	if !strings.Contains(got, "NO FIXER") {
		t.Fatalf("the orphan signature should be marked NO FIXER:\n%s", got)
	}

	// --json emits the folded set with the affected count joined from the leases.
	var jOut, jErr bytes.Buffer
	if rc := runKnownBad(&jOut, &jErr, []string{
		"report", "--ledger", ledger, "--leases", leases, "--json",
	}, now); rc != 0 {
		t.Fatalf("report --json rc=%d stderr=%q", rc, jErr.String())
	}
	var res struct {
		LiveCount  int `json:"live_count"`
		Signatures []struct {
			Affected       int    `json:"Affected"`
			Fixer          string `json:"Fixer"`
			NoFixerOverdue bool   `json:"NoFixerOverdue"`
		} `json:"signatures"`
	}
	if err := json.Unmarshal(jOut.Bytes(), &res); err != nil {
		t.Fatalf("report --json not valid JSON: %v (%q)", err, jOut.String())
	}
	if res.LiveCount != 1 || len(res.Signatures) != 1 {
		t.Fatalf("report --json live_count/signatures = %d/%d, want 1/1", res.LiveCount, len(res.Signatures))
	}
	s := res.Signatures[0]
	if s.Affected != 1 || s.Fixer != "" || !s.NoFixerOverdue {
		t.Fatalf("folded signature = %+v, want affected:1 no-fixer overdue", s)
	}
}

// --operator-after is overridable: with a generous window the same fresh unclaimed
// signature stays a muted status card rather than paging.
func TestKnownBadReportOperatorAfterOverride(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const now = int64(1_700_000_000)
	recordKnownBad(t, ledger, now-60, "internal/foo/**", "build", "agent-1", "")
	leases := writeReportLeases(t, `{"lane":"lane-a","tree_globs":["internal/foo/x.go"]}`)

	var out, errb bytes.Buffer
	rc := runKnownBad(&out, &errb, []string{
		"report", "--ledger", ledger, "--leases", leases, "--operator-after", "1h", "--dry-run",
	}, now)
	if rc != 0 {
		t.Fatalf("report rc=%d stderr=%q", rc, errb.String())
	}
	if strings.Contains(out.String(), "<!here>") {
		t.Fatalf("with a 1h window a 1-minute-old signature must not page:\n%s", out.String())
	}
}
