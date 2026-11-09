package benchsweep

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// mixedOutcomes is the shared fixture: two plain accepts, one rescued short
// run, one hard-excluded crash, and one non-whitelisted status.
func mixedOutcomes() []Outcome {
	return []Outcome{
		oc("m1", "ok", 1, 60*time.Second),        // accept
		oc("m2", "ok", 1, time.Second),           // rescue (below 5s floor)
		oc("m3", "crashed", 1, 60*time.Second),   // reject: hard-excluded
		oc("m4", "timeout", 1, 60*time.Second),   // reject: not whitelisted
		oc("m5", "degraded", 2, 120*time.Second), // accept
	}
}

func TestSummaryWriter(t *testing.T) {
	t.Run("ledger_accounts_every_outcome_with_per_status_counts", func(t *testing.T) {
		ch := NewChain(testCfg, nil)
		l, err := BuildLedger(Sweep(ch, mixedOutcomes()))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if l.Schema != SummarySchema {
			t.Fatalf("schema = %q, want %q", l.Schema, SummarySchema)
		}
		if l.Total != 5 || l.Accepted != 2 || l.Rescued != 1 || l.Rejected != 2 {
			t.Fatalf("verdict counts wrong: %+v", l)
		}
		wantStatus := map[string]int{"ok": 2, "crashed": 1, "timeout": 1, "degraded": 1}
		if !reflect.DeepEqual(l.ByStatus, wantStatus) {
			t.Fatalf("by_status = %v, want %v", l.ByStatus, wantStatus)
		}
		if len(l.Rescues) != 1 || l.Rescues[0].ComboID != "m2" || l.Rescues[0].Check != "duration-rescue" {
			t.Fatalf("rescue rows wrong: %+v", l.Rescues)
		}
		if len(l.Rejections) != 2 {
			t.Fatalf("rejection rows = %d, want 2: %+v", len(l.Rejections), l.Rejections)
		}
		byID := map[string]Rejection{}
		for _, j := range l.Rejections {
			byID[j.ComboID] = j
		}
		if j := byID["m3"]; j.Reason != ReasonHardExcludedStatus || j.Check != "hard-exclude-statuses" || j.Status != "crashed" {
			t.Fatalf("m3 rejection row wrong: %+v", j)
		}
		if j := byID["m4"]; j.Reason != ReasonStatusNotWhitelisted || j.Check != "status-whitelist" || j.Status != "timeout" {
			t.Fatalf("m4 rejection row wrong: %+v", j)
		}
		if err := l.Validate(); err != nil {
			t.Fatalf("built ledger must validate: %v", err)
		}
	})

	t.Run("zero_eligible_input_yields_valid_empty_ledger", func(t *testing.T) {
		for _, pairs := range [][]Paired{nil, {}} {
			l, err := BuildLedger(pairs)
			if err != nil {
				t.Fatalf("empty input must not be an error: %v", err)
			}
			if l.Total != 0 || l.Accepted != 0 || l.Rescued != 0 || l.Rejected != 0 {
				t.Fatalf("empty ledger must carry zero counts, got %+v", l)
			}
			if l.ByStatus == nil || len(l.ByStatus) != 0 {
				t.Fatalf("empty ledger must carry an empty by_status map, got %v", l.ByStatus)
			}
			if err := l.Validate(); err != nil {
				t.Fatalf("empty ledger must validate: %v", err)
			}
		}
	})

	t.Run("all_rejected_sweep_still_yields_a_complete_ledger", func(t *testing.T) {
		ch := NewChain(testCfg, nil)
		outcomes := []Outcome{
			oc("r1", "crashed", 1, 60*time.Second),
			oc("r2", "unknown_status", 1, 60*time.Second),
		}
		l, err := BuildLedger(Sweep(ch, outcomes))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if l.Total != 2 || l.Rejected != 2 || l.Accepted != 0 || l.Rescued != 0 {
			t.Fatalf("counts wrong: %+v", l)
		}
		if !reflect.DeepEqual(l.ByStatus, map[string]int{"crashed": 1, "unknown_status": 1}) {
			t.Fatalf("by_status wrong: %v", l.ByStatus)
		}
		if err := l.Validate(); err != nil {
			t.Fatalf("all-rejected ledger must validate: %v", err)
		}
	})

	t.Run("build_refuses_verdict_kind_outside_the_closed_vocabulary", func(t *testing.T) {
		pairs := []Paired{{Outcome: oc("z1", "ok", 1, 60*time.Second), Verdict: Verdict{Kind: "garbage"}}}
		_, err := BuildLedger(pairs)
		var le *LedgerError
		if !errors.As(err, &le) || le.Reason != ReasonLedgerInconsistent {
			t.Fatalf("want typed LEDGER_INCONSISTENT refusal, got %v", err)
		}
	})

	t.Run("write_is_single_call_and_idempotent", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "ledger.json")
		w := NewWriter(path)
		if w.Path() != path {
			t.Fatalf("Path() = %q, want %q", w.Path(), path)
		}
		ch := NewChain(testCfg, nil)
		l, err := BuildLedger(Sweep(ch, mixedOutcomes()))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("ledger must not exist before the first write, stat err = %v", err)
		}
		if err := w.Write(l); err != nil {
			t.Fatalf("first write failed: %v", err)
		}
		afterFirst, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ledger missing after write: %v", err)
		}
		if err := w.Write(l); err != nil {
			t.Fatalf("second write of the same ledger must succeed: %v", err)
		}
		afterSecond, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ledger missing after re-write: %v", err)
		}
		if string(afterFirst) != string(afterSecond) {
			t.Fatalf("idempotent write changed the bytes:\nfirst:\n%s\nsecond:\n%s", afterFirst, afterSecond)
		}
		var back Ledger
		if err := json.Unmarshal(afterSecond, &back); err != nil {
			t.Fatalf("written ledger does not parse: %v", err)
		}
		if !reflect.DeepEqual(back, l) {
			t.Fatalf("round-trip mismatch:\ngot  %+v\nwant %+v", back, l)
		}
	})

	t.Run("write_replaces_a_stale_ledger_wholesale", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "ledger.json")
		w := NewWriter(path)
		ch := NewChain(testCfg, nil)
		old, err := BuildLedger(Sweep(ch, mixedOutcomes()[:2]))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := w.Write(old); err != nil {
			t.Fatalf("stale write failed: %v", err)
		}
		newLedger, err := BuildLedger(Sweep(ch, mixedOutcomes()))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := w.Write(newLedger); err != nil {
			t.Fatalf("replacement write failed: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ledger missing: %v", err)
		}
		var back Ledger
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("written ledger does not parse: %v", err)
		}
		if back.Total != newLedger.Total || back.Accepted != newLedger.Accepted {
			t.Fatalf("file still carries the stale ledger: %+v", back)
		}
	})

	t.Run("write_refuses_inconsistent_ledgers_typed", func(t *testing.T) {
		cases := []struct {
			name   string
			mutate func(*Ledger)
		}{
			{"wrong_schema", func(l *Ledger) { l.Schema = "fak.benchsweep.summary/0" }},
			{"verdict_counts_do_not_sum", func(l *Ledger) { l.Accepted++ }},
			{"per_status_does_not_sum", func(l *Ledger) { l.ByStatus["ok"]++ }},
			{"rejection_row_without_token", func(l *Ledger) { l.Rejections[0].Reason = "" }},
			{"rescue_row_unaccounted_status", func(l *Ledger) { l.Rescues[0].Status = "ghost" }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				dir := t.TempDir()
				w := NewWriter(filepath.Join(dir, "ledger.json"))
				l, err := BuildLedger(Sweep(NewChain(testCfg, nil), mixedOutcomes()))
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				tc.mutate(&l)
				err = w.Write(l)
				var le *LedgerError
				if !errors.As(err, &le) || le.Reason != ReasonLedgerInconsistent {
					t.Fatalf("want typed LEDGER_INCONSISTENT refusal, got %v", err)
				}
				if _, statErr := os.Stat(w.Path()); !os.IsNotExist(statErr) {
					t.Fatalf("refused ledger must not be written, stat err = %v", statErr)
				}
			})
		}
	})

	t.Run("deferred_write_after_interrupt_leaves_a_complete_ledger", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "ledger.json")
		w := NewWriter(path)
		ch := NewChain(testCfg, nil)
		outcomes := mixedOutcomes()

		interrupted := func() {
			defer func() { _ = recover() }()
			defer func() {
				// finally: the always-written guarantee. Even though the sweep
				// panics below, the deferred write emits the complete ledger
				// over every outcome recorded so far.
				l, err := BuildLedger(Sweep(ch, outcomes))
				if err == nil {
					if werr := w.Write(l); werr != nil {
						t.Errorf("deferred write failed: %v", werr)
					}
				}
			}()
			panic("simulated sweep interrupt")
		}
		interrupted()

		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("interrupted sweep left no ledger: %v", err)
		}
		var back Ledger
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("written ledger does not parse: %v", err)
		}
		if back.Total != len(outcomes) || back.Accepted != 2 || back.Rescued != 1 || back.Rejected != 2 {
			t.Fatalf("interrupted ledger incomplete: %+v", back)
		}
		if !reflect.DeepEqual(back.ByStatus, map[string]int{"ok": 2, "crashed": 1, "timeout": 1, "degraded": 1}) {
			t.Fatalf("interrupted ledger lost per-status counts: %v", back.ByStatus)
		}
	})
}
