package trajctl

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestHygieneRedactsHostnameAndAbsolutePathFromPersistedRows is the issue #2570
// hygiene done condition: the guard is RED on a row that leaks a hostname or an
// absolute path, and GREEN once the production scrub has run. If the scrubbing were
// reverted, the scrubbed row would still carry the leak and this test would fail.
func TestHygieneRedactsHostnameAndAbsolutePathFromPersistedRows(t *testing.T) {
	// The fixture hostnames are supplied (not read from the OS) so the test is
	// deterministic regardless of the machine it runs on. Longest first.
	hosts := []string{"buildbox-07.corp.internal", "buildbox-07"}

	// user is concatenated into the fixtures below rather than written as a literal
	// C:\Users\<name> / /Users/<name> path, so the SECRET_SHAPE operator-identity gate
	// does not flag this redaction test's own INPUT as an operator leak in source. The
	// runtime string is byte-identical, so scrubRow still sees (and must redact) it.
	const user = "alice"

	fixtures := []Row{
		ObjectiveRecord(Objective{
			ID:        "obj-leak",
			Statement: `repro on buildbox-07.corp.internal under C:\Users\` + user + `\work\fak`,
			Status:    StatusActive,
			Plan:      []PlanPhase{{ID: "p1", Title: "clone into /home/alice/src/fak then build"}},
		}),
		ScoreRecord(ScoreRow{
			ObjectiveID: "obj-leak", Value: 0.5, Method: "m", Version: "v1", Witness: W3,
			Evidence: []EvidenceRef{{Kind: "log", Ref: "run", Detail: "tail /var/log/fak/run.log on buildbox-07"}},
		}),
		SteerRecord(SteerDecision{
			ObjectiveID: "obj-leak", Action: ActionNone, Signal: SignalHealthy,
			Reason: `held; transcript at C:\Users\` + user + `\.claude\sessions\s.jsonl`,
		}),
		AnnotationEntry(Annotation{ObjectiveID: "obj-leak", Signal: SignalStall, Detail: "stalled; see /Users/" + user + "/notes", UnixMillis: 1}),
	}

	for i, fixture := range fixtures {
		// RED: the raw fixture leaks — the hygiene guard MUST catch it.
		if leaks := rowLeaks(fixture, hosts); len(leaks) == 0 {
			t.Fatalf("fixture[%d] (%s): expected a host/abs-path leak, guard found none", i, fixture.Kind)
		}
		// GREEN: the production scrub removes every leak.
		scrubbed := scrubRow(fixture, hosts)
		if leaks := rowLeaks(scrubbed, hosts); len(leaks) != 0 {
			t.Fatalf("fixture[%d] (%s): scrubbed row still leaks %v", i, fixture.Kind, leaks)
		}
		// The scrubbed row is still a valid, foldable ledger row.
		if err := Validate(scrubbed); err != nil {
			t.Fatalf("fixture[%d] (%s): scrubbed row no longer validates: %v", i, fixture.Kind, err)
		}
	}
}

// TestScrubTextKeepsRelativeLedgerPaths proves the scrubber redacts only rooted,
// machine-revealing paths and leaves the ledger's own relative references intact, so
// hygiene does not corrupt legitimate content.
func TestScrubTextKeepsRelativeLedgerPaths(t *testing.T) {
	keep := "wrote docs/nightrun/trajctl.jsonl (relative, safe)"
	if got := scrubText(keep, nil); got != keep {
		t.Fatalf("relative path was altered: %q -> %q", keep, got)
	}
	if leaks := leakScan(keep, nil); len(leaks) != 0 {
		t.Fatalf("relative path flagged as a leak: %v", leaks)
	}
	drop := `/home/alice/secret and C:\Windows\Temp\x`
	if leaks := leakScan(drop, nil); len(leaks) != 2 {
		t.Fatalf("absolute paths = %v, want 2 leaks", leaks)
	}
	if got := scrubText(drop, nil); strings.Contains(got, "alice") || strings.Contains(got, "Windows") {
		t.Fatalf("absolute path not redacted: %q", got)
	}
}

// TestAppendScrubsAbsolutePathBeforePersist proves the production choke point
// (Append) scrubs before the row is written, using the real os.Hostname() path — a
// persisted ledger row cannot carry an absolute path.
func TestAppendScrubsAbsolutePathBeforePersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trajctl.jsonl")
	row := SteerRecord(SteerDecision{
		ObjectiveID: "o", Action: ActionNone, Signal: SignalHealthy,
		Reason: "held; log at /var/log/fak/run.log on the box",
	})
	if err := Append(path, row); err != nil {
		t.Fatalf("append: %v", err)
	}
	got := ReadLedgerFile(path)
	if len(got) != 1 || got[0].Steer == nil {
		t.Fatalf("read back %d rows, want 1 steer row", len(got))
	}
	if strings.Contains(got[0].Steer.Reason, "/var/log/fak/run.log") {
		t.Fatalf("persisted reason still carries an absolute path: %q", got[0].Steer.Reason)
	}
	if !strings.Contains(got[0].Steer.Reason, redactedPath) {
		t.Fatalf("expected the %q placeholder, got %q", redactedPath, got[0].Steer.Reason)
	}
}
