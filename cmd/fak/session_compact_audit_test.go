package main

// session_compact_audit_test.go — the wired `fak session compact-audit` verb end to end
// over the internal/session fixtures (#4763). The classification logic is proven in
// internal/session; this pins the CLI seam: flag parsing, --json/--scrub, exit codes.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func fixtureCorpusRoot() string {
	return filepath.Join("..", "..", "internal", "session", "testdata", "compactaudit")
}

func TestSessionCompactAuditHumanReport(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot()})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"compaction health", "resident context", "append-only", "verdicts:"} {
		if !strings.Contains(got, want) {
			t.Errorf("report missing %q\n%s", want, got)
		}
	}
}

func TestSessionCompactAuditJSONScrubbed(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--json", "--scrub"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	var res struct {
		Root      string `json:"root"`
		Aggregate struct {
			Sessions int `json:"sessions"`
			Fires    int `json:"fires"`
		} `json:"aggregate"`
		Sessions []struct {
			Path string `json:"path"`
			Cwd  string `json:"cwd"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if res.Aggregate.Sessions != 6 {
		t.Errorf("sessions = %d, want 6", res.Aggregate.Sessions)
	}
	if res.Root != "" {
		t.Errorf("scrubbed root = %q, want empty", res.Root)
	}
	for _, s := range res.Sessions {
		if s.Path != "" || s.Cwd != "" {
			t.Errorf("scrubbed session kept path/cwd: %q/%q", s.Path, s.Cwd)
		}
	}
	if strings.Contains(out.String(), "MUST_NOT_LEAK") {
		t.Error("json leaked a prompt body")
	}
}

// The committed image-wedge fixture is the oversized-single-item wedge rollout: compaction
// fires twice and resident context never comes off the late-fire ceiling. WEDGED_AT_CEILING is
// classified in internal/session, but an operator only ever meets it through this verb, so pin
// the whole seam — corpus walk, per-session row, aggregate roll-up, and the default human
// report — not just the in-package scan (#5168).
func TestSessionCompactAuditSurfacesWedgedAtCeiling(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--json"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	var res struct {
		Aggregate struct {
			VerdictCounts map[string]int `json:"verdict_counts"`
			AnomalyCounts map[string]int `json:"anomaly_counts"`
		} `json:"aggregate"`
		Sessions []struct {
			Path      string   `json:"path"`
			Verdict   string   `json:"verdict"`
			FireCount int      `json:"fire_count"`
			Anomalies []string `json:"anomalies"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}

	found := false
	for _, s := range res.Sessions {
		if filepath.Base(s.Path) != "image-wedge.jsonl" {
			continue
		}
		found = true
		if s.Verdict != session.VerdictWedgedAtCeiling {
			t.Errorf("verdict = %q, want %q", s.Verdict, session.VerdictWedgedAtCeiling)
		}
		if s.FireCount != 2 {
			t.Errorf("fire count = %d, want 2 — the wedge is repeated fire WITHOUT relief, not absent fire", s.FireCount)
		}
		if !slices.Contains(s.Anomalies, session.AnomalyWedgedAtCeiling) {
			t.Errorf("anomalies = %v, want %s among them", s.Anomalies, session.AnomalyWedgedAtCeiling)
		}
	}
	if !found {
		t.Fatalf("corpus walk never reached image-wedge.jsonl; scanned %d sessions", len(res.Sessions))
	}

	// Exactly one fixture is wedged: a roll-up that counted every repeated-fire session as
	// wedged would satisfy the per-session check above while being useless to an operator.
	if got := res.Aggregate.VerdictCounts[session.VerdictWedgedAtCeiling]; got != 1 {
		t.Errorf("aggregate verdict_counts[%s] = %d, want 1", session.VerdictWedgedAtCeiling, got)
	}
	if got := res.Aggregate.AnomalyCounts[session.AnomalyWedgedAtCeiling]; got != 1 {
		t.Errorf("aggregate anomaly_counts[%s] = %d, want 1", session.AnomalyWedgedAtCeiling, got)
	}
	// The wedge fixture's image body must not ride out on the un-scrubbed JSON either.
	if strings.Contains(out.String(), "MUST_NOT_LEAK") {
		t.Error("json leaked a prompt or image body")
	}

	// --json is the script path; the bare verb is what an operator actually runs.
	var hout, herrb bytes.Buffer
	if rc := runSession(&hout, &herrb, []string{"compact-audit", "--root", fixtureCorpusRoot()}); rc != 0 {
		t.Fatalf("human rc = %d, stderr = %s", rc, herrb.String())
	}
	if !strings.Contains(hout.String(), session.VerdictWedgedAtCeiling) {
		t.Errorf("human report never names the wedge verdict:\n%s", hout.String())
	}
}

func TestSessionCompactAuditAggregateOnly(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--json", "--aggregate-only"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	if strings.Contains(out.String(), `"sessions": [`) {
		t.Errorf("--aggregate-only still emitted per-session rows:\n%s", out.String())
	}
}

// TestSessionCompactAuditGuardedOnly pins the CLI half of the provenance filter (#5254).
// The cohort logic is proven in internal/session; what can only break here is the wiring —
// that --guarded-only and --guard-witness-dir reach CompactAuditOptions at all, that the
// provenance block rides out on the --scrub'd JSON an operator checks in, and that a sweep
// with no ledger EXITS NONZERO instead of printing a clean-looking zero report.
func TestSessionCompactAuditGuardedOnly(t *testing.T) {
	// One of the six corpus fixtures, named by its real session_meta id.
	const healthy = "019ed887-0603-7e90-84fc-2ff941120848"
	ledger := t.TempDir()
	witness, err := json.Marshal(map[string]any{
		"schema":     session.GuardWitnessSchema,
		"session_id": healthy,
		"guarded_at": "2026-08-04T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("marshal witness: %v", err)
	}
	if err := os.WriteFile(filepath.Join(ledger, healthy+".json"), witness, 0o600); err != nil {
		t.Fatalf("write witness: %v", err)
	}

	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{
		"compact-audit", "--root", fixtureCorpusRoot(),
		"--guarded-only", "--guard-witness-dir", ledger, "--json", "--scrub",
	})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	var res struct {
		Provenance struct {
			GuardedOnly    bool `json:"guarded_only"`
			LedgerSessions int  `json:"ledger_sessions"`
			Guarded        int  `json:"guarded_sessions"`
			Unguarded      int  `json:"unguarded_sessions"`
		} `json:"provenance"`
		Aggregate struct {
			Sessions int `json:"sessions"`
		} `json:"aggregate"`
		Sessions []struct {
			SessionID string `json:"session_id"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if res.Aggregate.Sessions != 1 || len(res.Sessions) != 1 || res.Sessions[0].SessionID != healthy {
		t.Fatalf("cohort = %d sessions %+v, want only the ledger-present %s",
			res.Aggregate.Sessions, res.Sessions, healthy)
	}
	// --scrub drops paths so the JSON is checkinable; it must NOT drop the one field that
	// says which population these numbers describe.
	if !res.Provenance.GuardedOnly || res.Provenance.LedgerSessions != 1 {
		t.Errorf("provenance = %+v, want guarded_only with the ledger's own size", res.Provenance)
	}
	if res.Provenance.Guarded != 1 || res.Provenance.Unguarded != 5 {
		t.Errorf("cohort split = %d guarded / %d unguarded, want 1/5 of the 6 fixtures",
			res.Provenance.Guarded, res.Provenance.Unguarded)
	}

	// No ledger: an empty guarded cohort renders exactly like "the anomaly class is gone",
	// so the verb must refuse rather than emit that document.
	var eout, eerr bytes.Buffer
	rc = runSession(&eout, &eerr, []string{
		"compact-audit", "--root", fixtureCorpusRoot(),
		"--guarded-only", "--guard-witness-dir", filepath.Join(t.TempDir(), "no-such-ledger"), "--json",
	})
	if rc == 0 {
		t.Fatalf("guarded-only with no ledger exited 0:\n%s", eout.String())
	}
	if eout.Len() != 0 {
		t.Errorf("refused sweep still printed a report:\n%s", eout.String())
	}
	if !strings.Contains(eerr.String(), "guard witness ledger") {
		t.Errorf("refusal does not name the missing ledger: %s", eerr.String())
	}
}

func TestSessionCompactAuditMissingRoot(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", filepath.Join(t.TempDir(), "does-not-exist")})
	if rc != 1 {
		t.Errorf("rc = %d, want 1 for a missing corpus root; stderr = %s", rc, errb.String())
	}
}

func TestSessionCompactAuditBadSince(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--since", "last-tuesday"})
	if rc != 2 {
		t.Errorf("rc = %d, want 2 for a malformed --since", rc)
	}
}

func TestSessionCompactAuditTopByPeakResident(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--top", "2", "--top-by", "peak-resident"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "top 2 sessions by peak-resident tokens") {
		t.Fatalf("missing token trajectory ranking:\n%s", got)
	}
	if strings.Contains(got, "sessions by fires") {
		t.Fatalf("fire ranking leaked into token trajectory view:\n%s", got)
	}
	if !strings.Contains(got, "final RESIDENT") || !strings.Contains(got, "cumulative input") {
		t.Fatalf("token dimensions missing:\n%s", got)
	}
}

func TestSessionCompactAuditRejectsUnknownTopBy(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"compact-audit", "--root", fixtureCorpusRoot(), "--top-by", "bytes"})
	if rc != 2 {
		t.Fatalf("rc=%d want 2; stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "want fires, peak-resident, or cumulative-input") {
		t.Fatalf("stderr=%s", errb.String())
	}
}
