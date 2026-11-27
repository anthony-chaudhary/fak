package ops

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestPauseStateDurable is the witness for #14476:
//  1. Tripping a pause persists fak-ops-load-pause/1 with paused=true, a closed
//     reason_class, and a non-empty reactivate_when.
//  2. A fresh Engine over the same root reads back the same paused state.
//  3. Resuming flips paused=false on disk with the transition preserved.
//  4. Out-of-vocabulary reason_class values are refused.
//  5. A corrupt/truncated pause file fails safe as "not paused".
//  6. Skipped ticks while paused do not rewrite the pause file on every tick.
//
// fak-test:runtime fast est=1s lane=default
func TestPauseStateDurable(t *testing.T) {
	repo := t.TempDir()
	e1 := newHermeticTestEngine(t, repo)

	baseTime := time.Date(2026, 11, 27, 9, 0, 0, 0, time.UTC)
	clock, advance := fixedClock(baseTime)
	e1.SetClock(clock)
	e1.SetSelfCostFunc(func() (time.Duration, bool) { return 10 * time.Millisecond, true })

	cond := ReactivateCondition{
		PressureRatioBelow: 0.5,
		ConsecutiveSamples: 3,
		MaxPause:           "30m",
	}

	// (4) Out-of-vocabulary reason_class is refused before any state changes.
	if err := e1.Pause(PauseReasonClass("BOGUS_REASON"), "nope", cond); err == nil {
		t.Fatalf("expected out-of-vocabulary reason_class to be refused")
	}

	// Empty reactivate_when is also refused when entering a pause.
	if err := e1.Pause(PauseReasonHostPressure, "missing cond", ReactivateCondition{}); err == nil {
		t.Fatalf("expected empty reactivate_when to be refused")
	}

	// (1) Trip a pause and assert the on-disk record.
	if err := e1.Pause(PauseReasonHostPressure, "ram pressure 92%", cond); err != nil {
		t.Fatalf("Pause: %v", err)
	}

	pausePath := DefaultPauseStatePath(repo)
	if filepath.Dir(pausePath) != filepath.Dir(DefaultLedgerPath(repo)) {
		t.Fatalf("pause path %q must be a sibling of ops ledger %q", pausePath, DefaultLedgerPath(repo))
	}

	raw, err := os.ReadFile(pausePath)
	if err != nil {
		t.Fatalf("read on-disk pause record: %v", err)
	}
	var disk PauseState
	if err := json.Unmarshal(raw, &disk); err != nil {
		t.Fatalf("unmarshal pause record: %v", err)
	}
	if disk.Schema != PauseSchemaV1 {
		t.Errorf("schema = %q, want %q", disk.Schema, PauseSchemaV1)
	}
	if !disk.Paused {
		t.Errorf("paused = false, want true")
	}
	if disk.ReasonClass != PauseReasonHostPressure {
		t.Errorf("reason_class = %q, want %q", disk.ReasonClass, PauseReasonHostPressure)
	}
	if !ValidPauseReasonClass(disk.ReasonClass) {
		t.Errorf("reason_class %q not in closed vocabulary", disk.ReasonClass)
	}
	if disk.ReactivateWhen.IsEmpty() {
		t.Errorf("expected non-empty reactivate_when")
	}
	if !disk.PausedAt.Equal(baseTime) {
		t.Errorf("paused_at = %v, want %v", disk.PausedAt, baseTime)
	}

	// Verify UNKNOWN_HOST_STATE is distinct from HOST_PRESSURE and valid.
	if PauseReasonUnknownHostState == PauseReasonHostPressure {
		t.Fatalf("UNKNOWN_HOST_STATE must be distinct from HOST_PRESSURE")
	}
	if !ValidPauseReasonClass(PauseReasonUnknownHostState) {
		t.Fatalf("UNKNOWN_HOST_STATE must be in closed vocabulary")
	}

	// (6) Skipped ticks while paused must not rewrite the pause file.
	infoBefore, err := os.Stat(pausePath)
	if err != nil {
		t.Fatalf("stat pause file: %v", err)
	}
	advance(1 * time.Minute)
	if err := e1.Tick(context.Background(), false); err != nil {
		t.Fatalf("Tick while paused: %v", err)
	}
	infoAfter, err := os.Stat(pausePath)
	if err != nil {
		t.Fatalf("stat pause file after tick: %v", err)
	}
	if !infoAfter.ModTime().Equal(infoBefore.ModTime()) {
		t.Errorf("pause file was rewritten on skipped tick: before=%v after=%v", infoBefore.ModTime(), infoAfter.ModTime())
	}
	if got := e1.PauseState().PausedTicks; got != 1 {
		t.Errorf("paused_ticks = %d, want 1", got)
	}

	// (2) Construct a fresh Engine over the same root and assert it reads back the paused state.
	e2 := newHermeticTestEngine(t, repo)
	if !e2.Paused() {
		t.Fatalf("fresh Engine did not restore paused=true from disk")
	}
	got2 := e2.PauseState()
	if got2.ReasonClass != PauseReasonHostPressure || got2.ReasonDetail != "ram pressure 92%" {
		t.Errorf("fresh Engine pause state mismatch: %+v", got2)
	}

	// (3) Resume on the fresh Engine and assert the record flips to paused=false with transition preserved.
	if err := e2.Resume(); err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if e2.Paused() {
		t.Fatalf("expected Engine.Paused() == false after Resume")
	}
	rawResumed, err := os.ReadFile(pausePath)
	if err != nil {
		t.Fatalf("read resumed pause record: %v", err)
	}
	var resumed PauseState
	if err := json.Unmarshal(rawResumed, &resumed); err != nil {
		t.Fatalf("unmarshal resumed pause record: %v", err)
	}
	if resumed.Paused {
		t.Errorf("on-disk paused = true after Resume, want false")
	}
	if resumed.ReasonClass != PauseReasonHostPressure {
		t.Errorf("resumed record lost transition reason_class: got %q", resumed.ReasonClass)
	}
	if !resumed.PausedAt.Equal(baseTime) {
		t.Errorf("resumed record lost paused_at: got %v, want %v", resumed.PausedAt, baseTime)
	}

	// (5) Corrupt/truncated file fails safe as "not paused", never "paused forever".
	if err := os.WriteFile(pausePath, []byte(`{"schema":"fak-ops-load-pause/1","paused":true,"reason_cla`), 0o644); err != nil {
		t.Fatalf("write corrupt pause file: %v", err)
	}
	e3 := newHermeticTestEngine(t, repo)
	if e3.Paused() {
		t.Fatalf("corrupt/truncated pause file must fail safe as not paused")
	}
}
