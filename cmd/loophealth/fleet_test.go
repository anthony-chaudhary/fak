package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fixedNow is the injected wall-clock for the fleet fold tests so every derived
// state is deterministic (no real time.Now()).
var fixedNow = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

// writeLedger writes JSONL rows under root at rel, creating parent dirs.
func writeLedger(t *testing.T, root, rel string, rows []map[string]any) {
	t.Helper()
	path := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
	var buf bytes.Buffer
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func nanos(at time.Time) int64 { return at.UnixNano() }

func rowFor(rep fleetReport, kind string) (fleetRow, bool) {
	for _, l := range rep.Loops {
		if l.Kind == kind {
			return l, true
		}
	}
	return fleetRow{}, false
}

// TestFoldFleetStatesAndDark is the acceptance check: across ≥3 ledgers the fold
// renders one row per loop with a health state, a loop whose last tick is older
// than its cadence renders `dark`, and a missing ledger is surfaced as `unknown`
// (skipped, not fatal).
func TestFoldFleetStatesAndDark(t *testing.T) {
	root := t.TempDir()

	// loopmgr: three fires an hour apart, the last 5h before now → observed
	// cadence 1h, age 5h > cadence → DARK.
	writeLedger(t, root, filepath.Join(".fak", "loops.jsonl"), []map[string]any{
		{"kind": "fire", "ts_unix_nano": nanos(fixedNow.Add(-7 * time.Hour))},
		{"kind": "fire", "ts_unix_nano": nanos(fixedNow.Add(-6 * time.Hour))},
		{"kind": "end", "status": "claimed-done", "ts_unix_nano": nanos(fixedNow.Add(-5 * time.Hour))},
	})

	// dispatch: two ticks 5m apart, last 1m before now → observed cadence 5m,
	// age 1m ≤ 0.5·cadence → LIVE.
	writeLedger(t, root, filepath.Join(".dispatch-runs", "progress.jsonl"), []map[string]any{
		{"utc": fixedNow.Add(-6 * time.Minute).Format(time.RFC3339), "ok": true, "closed_now": 1},
		{"utc": fixedNow.Add(-1 * time.Minute).Format(time.RFC3339), "ok": true, "closed_now": 0},
	})

	// dojo: two ticks 24h apart, last 18h before now → observed cadence 24h, age
	// 18h is past 0.5·24h=12h but ≤ 24h → STALE.
	writeLedger(t, root, filepath.Join("docs", "dojo", "history.jsonl"), []map[string]any{
		{"generated_at": fixedNow.Add(-42 * time.Hour).Format(time.RFC3339), "verdict": "OK", "calibrated": 3},
		{"generated_at": fixedNow.Add(-18 * time.Hour).Format(time.RFC3339), "verdict": "OK", "calibrated": 4},
	})

	// guard-audit.jsonl is intentionally absent → UNKNOWN, surfaced not fatal.

	rep := foldFleet(root, fleetAdapters(), fixedNow)

	if got := len(rep.Loops); got != len(fleetAdapters()) {
		t.Fatalf("rows = %d, want one per adapter (%d)", got, len(fleetAdapters()))
	}

	// Acceptance: ≥3 ledgers render with a (non-unknown) health state.
	withState := 0
	for _, l := range rep.Loops {
		if l.State != stateUnknown {
			withState++
		}
	}
	if withState < 3 {
		t.Fatalf("only %d loop(s) rendered a known health state, want ≥3", withState)
	}

	cases := map[string]string{
		"loopmgr":  stateDark,
		"dispatch": stateLive,
		"dojo":     stateStale,
		"guardrsi": stateUnknown,
	}
	for kind, want := range cases {
		row, ok := rowFor(rep, kind)
		if !ok {
			t.Fatalf("no row for %q", kind)
		}
		if row.State != want {
			t.Errorf("%s state = %q, want %q (age=%.0fs cadObs=%.0fs)", kind, row.State, want, row.AgeSeconds, row.CadenceObserved)
		}
	}

	if rep.DarkCount != 1 {
		t.Fatalf("dark count = %d, want 1", rep.DarkCount)
	}

	// The dark loop must carry a real last-tick and run count (not silence).
	dark, _ := rowFor(rep, "loopmgr")
	if dark.LastTickUnixNano == 0 || dark.Runs != 3 {
		t.Errorf("dark loopmgr row last=%d runs=%d, want last>0 runs=3", dark.LastTickUnixNano, dark.Runs)
	}

	// A surfaced unknown carries a note explaining the absence (not silence).
	un, _ := rowFor(rep, "guardrsi")
	if un.Note == "" {
		t.Errorf("absent guardrsi ledger rendered unknown with no surfaced note")
	}
}

// TestHealthStateBoundaries pins the live/stale/dark ladder, including the exact
// "older than its cadence → dark" rule the acceptance requires.
func TestHealthStateBoundaries(t *testing.T) {
	cad := time.Hour
	cases := []struct {
		age   time.Duration
		state string
	}{
		{0, stateLive},
		{29 * time.Minute, stateLive},  // ≤ 0.5·cadence
		{31 * time.Minute, stateStale}, // > 0.5·cadence, ≤ cadence
		{cad, stateStale},              // exactly at cadence is not yet dark
		{cad + time.Second, stateDark}, // older than its cadence → dark
		{10 * cad, stateDark},
	}
	for _, c := range cases {
		if got := healthState(c.age, cad); got != c.state {
			t.Errorf("healthState(age=%s, cad=%s) = %q, want %q", c.age, cad, got, c.state)
		}
	}
	if got := healthState(time.Hour, 0); got != stateUnknown {
		t.Errorf("healthState with zero cadence = %q, want unknown", got)
	}
}

// TestFleetMainDarkExit3 proves the scheduler gate: a dark loop yields exit 3,
// and the --json view round-trips to a fleetReport carrying the dark count.
func TestFleetMainDarkExit3(t *testing.T) {
	root := t.TempDir()
	writeLedger(t, root, filepath.Join(".fak", "loops.jsonl"), []map[string]any{
		{"kind": "fire", "ts_unix_nano": nanos(fixedNow.Add(-50 * time.Hour))},
		{"kind": "fire", "ts_unix_nano": nanos(fixedNow.Add(-49 * time.Hour))},
	})

	var out, errBuf bytes.Buffer
	code := fleetMain(&out, &errBuf, root, true, fixedNow)
	if code != 3 {
		t.Fatalf("exit code = %d, want 3 (a dark loop present)", code)
	}
	var rep fleetReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("--json output not valid fleetReport: %v", err)
	}
	if rep.Schema != FleetSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, FleetSchema)
	}
	if rep.DarkCount < 1 {
		t.Errorf("dark count = %d, want ≥1", rep.DarkCount)
	}
}

// TestFleetMainNoDarkExit0 proves the inverse: with no dark loop the gate is
// clean (exit 0) and the human render names the dark tally.
func TestFleetMainNoDarkExit0(t *testing.T) {
	root := t.TempDir()
	writeLedger(t, root, filepath.Join(".dispatch-runs", "progress.jsonl"), []map[string]any{
		{"utc": fixedNow.Add(-10 * time.Minute).Format(time.RFC3339), "ok": true},
		{"utc": fixedNow.Add(-1 * time.Minute).Format(time.RFC3339), "ok": true},
	})

	var out, errBuf bytes.Buffer
	code := fleetMain(&out, &errBuf, root, false, fixedNow)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (no dark loop)", code)
	}
	if !bytes.Contains(out.Bytes(), []byte("0 dark loop(s)")) {
		t.Errorf("human render did not surface the dark tally:\n%s", out.String())
	}
}

// TestFoldFleetReadOnly proves the fold appends nothing: the bytes of a ledger
// are identical before and after a fold (no event, no control verb).
func TestFoldFleetReadOnly(t *testing.T) {
	root := t.TempDir()
	rel := filepath.Join(".fak", "loops.jsonl")
	writeLedger(t, root, rel, []map[string]any{
		{"kind": "fire", "ts_unix_nano": nanos(fixedNow.Add(-time.Hour))},
	})
	before, err := os.ReadFile(filepath.Join(root, rel))
	if err != nil {
		t.Fatalf("read before: %v", err)
	}
	_ = foldFleet(root, fleetAdapters(), fixedNow)
	after, err := os.ReadFile(filepath.Join(root, rel))
	if err != nil {
		t.Fatalf("read after: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Errorf("fold mutated the ledger; read-only violated")
	}
}
