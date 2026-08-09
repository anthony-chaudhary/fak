package session

// compactaudit_provenance_test.go — the acceptance bar for the guarded-only sweep (#5254).
//
// The filter's whole job is to make a gateway-side before/after FALSIFIABLE. The
// 2026-07-19 decision note measured that only 120 of 2,448 corpus sessions (4.9%) ever
// crossed fak's wire, so an unscoped re-run moves for reasons unrelated to the dedup
// port. These tests pin the two properties that make the scoped number trustworthy:
// the cohort is actually restricted to ledger-present sessions, and a sweep with no
// ledger to stand on REFUSES instead of reporting an empty corpus as a clean zero.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeGuardLedger materialises a witness directory holding exactly the given session
// ids, in the on-disk shape `fak guard` writes (cmd/fak/sessions_codex_loop.go).
func writeGuardLedger(t *testing.T, ids ...string) string {
	t.Helper()
	dir := t.TempDir()
	for i, id := range ids {
		raw, err := json.Marshal(map[string]any{
			"schema":     GuardWitnessSchema,
			"session_id": id,
			"guarded_at": "2026-08-04T00:00:00Z",
		})
		if err != nil {
			t.Fatalf("marshal witness %d: %v", i, err)
		}
		if err := os.WriteFile(filepath.Join(dir, id+".json"), raw, 0o600); err != nil {
			t.Fatalf("write witness %d: %v", i, err)
		}
	}
	return dir
}

func corpusRoot() string { return filepath.Join("testdata", "compactaudit") }

// TestLoadGuardWitnessIDsReadsTheLedgerAndSkipsJunk covers the live-directory posture: a
// half-written or foreign file must not sink a sweep, but it must not be counted either.
func TestLoadGuardWitnessIDsReadsTheLedgerAndSkipsJunk(t *testing.T) {
	dir := writeGuardLedger(t, "sess-a", "sess-b")

	// Three shapes that are all "not a guard witness" and must each be ignored rather
	// than fataled or miscounted.
	if err := os.WriteFile(filepath.Join(dir, "half-written.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write junk: %v", err)
	}
	other, err := json.Marshal(map[string]any{"schema": "fak.something.else.v1", "session_id": "sess-foreign"})
	if err != nil {
		t.Fatalf("marshal foreign: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "foreign.json"), other, 0o600); err != nil {
		t.Fatalf("write foreign: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignored"), 0o600); err != nil {
		t.Fatalf("write non-json: %v", err)
	}

	ids, err := LoadGuardWitnessIDs(dir)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("ids = %v, want exactly the 2 well-formed witnesses", ids)
	}
	for _, want := range []string{"sess-a", "sess-b"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("ledger dropped %q: %v", want, ids)
		}
	}
	if _, ok := ids["sess-foreign"]; ok {
		t.Errorf("a foreign schema was accepted as a guard witness: %v", ids)
	}
}

// TestLoadGuardWitnessIDsRefusesAnEmptyLedger is the fail-closed line. Returning an empty
// set here would make a guarded-only sweep scan nothing and report a zero aggregate —
// which renders identically to "the anomaly class is gone".
func TestLoadGuardWitnessIDsRefusesAnEmptyLedger(t *testing.T) {
	empty := t.TempDir()
	junkOnly := t.TempDir()
	if err := os.WriteFile(filepath.Join(junkOnly, "x.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write junk: %v", err)
	}

	for name, dir := range map[string]string{
		"no directory": "",
		"absent":       filepath.Join(empty, "no-such-dir"),
		"empty":        empty,
		"only junk":    junkOnly,
		"only blank":   "   ",
	} {
		t.Run(name, func(t *testing.T) {
			ids, err := LoadGuardWitnessIDs(dir)
			if err == nil {
				t.Fatalf("empty ledger returned %d ids and no error — a zero-row sweep would read as a pass", len(ids))
			}
			if ids != nil {
				t.Errorf("refusal still handed back a set: %v", ids)
			}
		})
	}
}

// TestAuditCompactCorpusGuardedOnlyKeepsTheRoutedCohort is the DoD line: the sweep
// measures the fak-routed subset, and SAYS which subset it measured.
func TestAuditCompactCorpusGuardedOnlyKeepsTheRoutedCohort(t *testing.T) {
	// Two of the six corpus fixtures, named by their real session_meta ids.
	const healthy = "019ed887-0603-7e90-84fc-2ff941120848"
	const dupe = "019ed900-0000-7000-8000-00000000dupe"
	dir := writeGuardLedger(t, healthy, dupe, "sess-not-in-this-corpus")

	all, err := AuditCompactCorpus(CompactAuditOptions{Root: corpusRoot()})
	if err != nil {
		t.Fatalf("unscoped audit: %v", err)
	}
	if all.Aggregate.Sessions != 6 {
		t.Fatalf("unscoped sessions = %d, want the 6 fixtures", all.Aggregate.Sessions)
	}
	// An unscoped sweep must not claim provenance it does not have.
	if all.Provenance.GuardedOnly {
		t.Errorf("unscoped sweep reported guarded_only=true")
	}
	if all.Provenance.Guarded != 0 || all.Provenance.Unguarded != 0 || all.Provenance.LedgerSessions != 0 {
		t.Errorf("unscoped sweep carried a cohort split: %+v", all.Provenance)
	}

	scoped, err := AuditCompactCorpus(CompactAuditOptions{
		Root:            corpusRoot(),
		GuardedOnly:     true,
		GuardWitnessDir: dir,
	})
	if err != nil {
		t.Fatalf("guarded-only audit: %v", err)
	}
	if scoped.Aggregate.Sessions != 2 {
		t.Fatalf("guarded sessions = %d, want the 2 ledger-present fixtures", scoped.Aggregate.Sessions)
	}
	for _, s := range scoped.Sessions {
		if s.SessionID != healthy && s.SessionID != dupe {
			t.Errorf("guarded-only sweep kept an unrouted session %q", s.SessionID)
		}
	}
	if !scoped.Provenance.GuardedOnly {
		t.Errorf("scoped sweep did not record guarded_only")
	}
	// The ledger holds an id this corpus does not contain: the count is the LEDGER's
	// size, not the intersection, so a reader can see the join was partial.
	if scoped.Provenance.LedgerSessions != 3 {
		t.Errorf("ledger_sessions = %d, want 3 (the ledger's own size)", scoped.Provenance.LedgerSessions)
	}
	if scoped.Provenance.Guarded != 2 || scoped.Provenance.Unguarded != 4 {
		t.Errorf("cohort split = %d guarded / %d unguarded, want 2/4",
			scoped.Provenance.Guarded, scoped.Provenance.Unguarded)
	}
}

// TestAuditCompactCorpusGuardedOnlyRefusesRatherThanReportZero is the green-looking-lie
// guard at the sweep level: with no ledger, the verb must error out BEFORE the walk
// rather than hand back a clean-looking empty aggregate.
func TestAuditCompactCorpusGuardedOnlyRefusesRatherThanReportZero(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{
		Root:            corpusRoot(),
		GuardedOnly:     true,
		GuardWitnessDir: filepath.Join(t.TempDir(), "no-such-ledger"),
	})
	if err == nil {
		t.Fatalf("guarded-only sweep with no ledger returned %d sessions and no error", res.Aggregate.Sessions)
	}
	if res.Aggregate.Sessions != 0 {
		t.Errorf("refused sweep still reported %d sessions", res.Aggregate.Sessions)
	}
}

// TestScrubCompactResultKeepsProvenance guards the checked-in-witness case. --scrub drops
// paths so the JSON can land in the repo; if it also dropped provenance, a guarded
// aggregate and a whole-corpus one would be byte-shaped the same and the reader could no
// longer tell which population the numbers describe.
func TestScrubCompactResultKeepsProvenance(t *testing.T) {
	in := CompactAuditResult{
		Root:       `C:\Users\someone\.codex\sessions`,
		Provenance: CompactProvenance{GuardedOnly: true, LedgerSessions: 154, Guarded: 120, Unguarded: 2328},
		Sessions:   []CompactSessionReport{scanFixture(t, "healthy-repeated-fire.jsonl")},
	}
	out := ScrubCompactResult(in)

	if out.Provenance != in.Provenance {
		t.Errorf("scrub altered provenance: %+v, want %+v", out.Provenance, in.Provenance)
	}
	if out.Root != "" {
		t.Errorf("scrub kept the corpus root %q", out.Root)
	}
	for _, s := range out.Sessions {
		if s.Path != "" || s.Cwd != "" {
			t.Errorf("scrub kept a private path: path=%q cwd=%q", s.Path, s.Cwd)
		}
	}

	// The provenance block must also survive the JSON round trip an operator checks in.
	raw, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal scrubbed: %v", err)
	}
	var back CompactAuditResult
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal scrubbed: %v", err)
	}
	if back.Provenance != in.Provenance {
		t.Errorf("provenance did not survive the JSON round trip: %+v", back.Provenance)
	}
}
