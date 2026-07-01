package auditusage

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

func TestFold_AllAbsent(t *testing.T) {
	now := time.Unix(1700000000, 0)
	rep := Fold(Input{Now: now})

	if len(rep.Sinks) != 6 {
		t.Fatalf("want 6 sinks, got %d: %+v", len(rep.Sinks), rep.Sinks)
	}
	for _, s := range rep.Sinks {
		if s.Present {
			t.Errorf("sink %s: want absent, got present", s.Kind)
		}
		if s.Chain != ChainAbsent {
			t.Errorf("sink %s: want chain=absent, got %s", s.Kind, s.Chain)
		}
		if s.RowCount != 0 {
			t.Errorf("sink %s: want row_count=0, got %d", s.Kind, s.RowCount)
		}
	}
	if len(rep.Findings) != 0 {
		t.Errorf("want no findings for an all-absent fold, got %+v", rep.Findings)
	}
	if rep.Guard.Total != 0 || rep.Usage.Total != 0 || rep.Loop.Loops != 0 || rep.Dispatch.Workers != 0 {
		t.Errorf("want every rollup zeroed, got %+v", rep)
	}
}

func TestFold_GuardRollup_WithSince(t *testing.T) {
	old := time.Unix(1000, 0)
	recent := time.Unix(5000, 0)
	since := time.Unix(4000, 0)
	rows := []journal.Row{
		{TSUnixNano: old.UnixNano(), Kind: "DECIDE", Verdict: "ALLOW"},
		{TSUnixNano: recent.UnixNano(), Kind: "DECIDE", Verdict: "ALLOW"},
		{TSUnixNano: recent.UnixNano(), Kind: "DENY", Verdict: "DENY"},
	}
	rep := Fold(Input{
		Now:   time.Unix(9000, 0),
		Since: since,
		DecisionJournal: DecisionJournalInput{
			Path: "/x/guard-audit.jsonl", Present: true, Rows: rows,
		},
	})

	if rep.Guard.Basis != "witnessed" {
		t.Errorf("guard basis: want witnessed, got %q", rep.Guard.Basis)
	}
	if rep.Guard.Total != 2 {
		t.Fatalf("want 2 rows within --since, got %d", rep.Guard.Total)
	}
	if rep.Guard.ByVerdict["ALLOW"] != 1 || rep.Guard.ByVerdict["DENY"] != 1 {
		t.Errorf("by_verdict mismatch: %+v", rep.Guard.ByVerdict)
	}
	if rep.Guard.ByKind["DECIDE"] != 1 || rep.Guard.ByKind["DENY"] != 1 {
		t.Errorf("by_kind mismatch: %+v", rep.Guard.ByKind)
	}

	for _, s := range rep.Sinks {
		if s.Kind == SinkDecisionJournal {
			if s.Chain != ChainVerified {
				t.Errorf("decision journal: want verified (no VerifyErr set), got %s", s.Chain)
			}
			if s.RowCount != 3 {
				t.Errorf("decision journal row_count: want 3 (raw rows, not since-filtered), got %d", s.RowCount)
			}
		}
	}
}

func TestFold_UsageRollup_VerbWeeksAndErrors(t *testing.T) {
	t1 := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC) // 2026-W25
	t2 := time.Date(2026, 6, 16, 12, 0, 0, 0, time.UTC) // same week
	t3 := time.Date(2026, 6, 23, 12, 0, 0, 0, time.UTC) // next week
	rows := []usagelog.Row{
		{TSUnixNano: t1.UnixNano(), Verb: "guard", ExitCode: 0},
		{TSUnixNano: t2.UnixNano(), Verb: "guard", ExitCode: 1},
		{TSUnixNano: t3.UnixNano(), Verb: "audit", ExitCode: 0},
	}
	rep := Fold(Input{
		Now: time.Now(),
		UsageLog: UsageLogInput{
			Path: "/x/usage.jsonl", Present: true, Rows: rows,
		},
	})

	if rep.Usage.Basis != "observed" {
		t.Errorf("usage basis: want observed, got %q", rep.Usage.Basis)
	}
	if rep.Usage.Total != 3 || rep.Usage.Errors != 1 {
		t.Fatalf("usage rollup mismatch: %+v", rep.Usage)
	}
	if len(rep.InvocationsByVerbWeek) != 2 {
		t.Fatalf("want 2 (week,verb) buckets, got %d: %+v", len(rep.InvocationsByVerbWeek), rep.InvocationsByVerbWeek)
	}
	first := rep.InvocationsByVerbWeek[0]
	if first.Verb != "guard" || first.Total != 2 || first.Errors != 1 {
		t.Errorf("first bucket mismatch: %+v", first)
	}
	second := rep.InvocationsByVerbWeek[1]
	if second.Verb != "audit" || second.Total != 1 || second.Errors != 0 {
		t.Errorf("second bucket mismatch: %+v", second)
	}
}

func TestFold_ChainBroken_SurfacesFindingButKeepsRecoveredRows(t *testing.T) {
	rep := Fold(Input{
		Now: time.Now(),
		DecisionJournal: DecisionJournalInput{
			Path: "/x/guard-audit.jsonl", Present: true,
			Rows:      []journal.Row{{Kind: "DECIDE", Verdict: "ALLOW"}},
			VerifyErr: errors.New("journal: tampered row at seq 2: hash mismatch"),
		},
		UsageLog: UsageLogInput{
			Path: "/x/usage.jsonl", Present: true,
			Rows:      []usagelog.Row{{Verb: "guard", ExitCode: 0}},
			VerifyErr: errors.New("usagelog: broken chain at seq 3"),
		},
		LoopLedger: LoopLedgerInput{
			Path: "/x/loops.jsonl", Present: true,
			Events:    []loopmgr.Event{{LoopID: "l1", Kind: loopmgr.EventFire}},
			Integrity: loopmgr.Integrity{Broken: true, Reason: "loopmgr: forked seq at line 4", Recovered: 1},
		},
	})

	if len(rep.Findings) != 3 {
		t.Fatalf("want 3 CHAIN_BROKEN findings, got %d: %+v", len(rep.Findings), rep.Findings)
	}
	wantSinks := []SinkKind{SinkDecisionJournal, SinkLoopLedger, SinkUsageLog}
	for i, f := range rep.Findings {
		if f.Kind != "CHAIN_BROKEN" {
			t.Errorf("finding %d: want kind CHAIN_BROKEN, got %s", i, f.Kind)
		}
		if f.Sink != wantSinks[i] {
			t.Errorf("finding %d: want sink %s, got %s (findings must sort by sink)", i, wantSinks[i], f.Sink)
		}
		if f.Reason == "" {
			t.Errorf("finding %d: want a non-empty reason", i)
		}
	}

	// A broken chain never silently drops the rows a tolerant read DID recover.
	if rep.Guard.Total != 1 {
		t.Errorf("guard: want the 1 recovered row still folded in, got total=%d", rep.Guard.Total)
	}
	if rep.Usage.Total != 1 {
		t.Errorf("usage: want the 1 recovered row still folded in, got total=%d", rep.Usage.Total)
	}
	if rep.Loop.Fires != 1 {
		t.Errorf("loop: want the 1 recovered event still folded in, got fires=%d", rep.Loop.Fires)
	}

	for _, s := range rep.Sinks {
		switch s.Kind {
		case SinkDecisionJournal, SinkUsageLog, SinkLoopLedger:
			if s.Chain != ChainBroken {
				t.Errorf("sink %s: want chain=broken, got %s", s.Kind, s.Chain)
			}
			if s.BrokenReason == "" {
				t.Errorf("sink %s: want a broken_reason", s.Kind)
			}
		}
	}
}

func TestFold_LoopRollup(t *testing.T) {
	events := []loopmgr.Event{
		{LoopID: "l1", Kind: loopmgr.EventFire},
		{LoopID: "l1", Kind: loopmgr.EventAdmit},
		{LoopID: "l1", Kind: loopmgr.EventStart},
		{LoopID: "l1", Kind: loopmgr.EventEnd, Status: loopmgr.StatusWitnessedDone},
		{LoopID: "l2", Kind: loopmgr.EventFire},
		{LoopID: "l2", Kind: loopmgr.EventFire, Status: loopmgr.StatusRefused},
	}
	rep := Fold(Input{
		Now: time.Now(),
		LoopLedger: LoopLedgerInput{
			Path: "/x/loops.jsonl", Present: true,
			Events:    events,
			Integrity: loopmgr.Integrity{Recovered: len(events)},
		},
	})
	if rep.Loop.Basis != "witnessed" {
		t.Errorf("loop basis: want witnessed, got %q", rep.Loop.Basis)
	}
	if rep.Loop.Loops != 2 {
		t.Errorf("want 2 distinct loops, got %d", rep.Loop.Loops)
	}
	if rep.Loop.Fires != 3 {
		t.Errorf("want 3 total fires across both loops, got %d", rep.Loop.Fires)
	}
	for _, s := range rep.Sinks {
		if s.Kind == SinkLoopLedger && s.Chain != ChainVerified {
			t.Errorf("loop ledger: want verified, got %s", s.Chain)
		}
	}
}

func TestFold_DispatchRollup(t *testing.T) {
	workers := []dispatchaudit.Worker{
		{Log: "resolve-1-a.log", Issue: "1", Lane: "cmd", HeaderBackend: dispatchaudit.BackendClaude, CommitSHA: "deadbeef"},
		{Log: "resolve-2-b.log", Issue: "2", Lane: "cmd", HeaderBackend: dispatchaudit.BackendClaude, BannerOnly: true},
	}
	rep := Fold(Input{
		Now: time.Now(),
		DispatchRuns: DispatchRunsInput{
			RunsDir: "/x/.dispatch-runs", Present: true, Workers: workers,
		},
	})
	if rep.Dispatch.Basis != "observed" {
		t.Errorf("dispatch basis: want observed, got %q", rep.Dispatch.Basis)
	}
	if rep.Dispatch.Workers != 2 {
		t.Errorf("want 2 workers, got %d", rep.Dispatch.Workers)
	}
	if len(rep.Dispatch.ByBackend) != 1 || rep.Dispatch.ByBackend[0].Shipped != 1 {
		t.Errorf("want 1 backend rollup with 1 shipped worker, got %+v", rep.Dispatch.ByBackend)
	}
}

func TestFold_GatewayAndCache_SinceFilterAndTrend(t *testing.T) {
	since := time.Unix(2000, 0)
	gwRows := []gatewayusageledger.Row{
		{Schema: gatewayusageledger.Schema, UnixMillis: 1000 * 1000, SessionType: "guard"},
		{Schema: gatewayusageledger.Schema, UnixMillis: 3000 * 1000, SessionType: "guard"},
		{Schema: gatewayusageledger.Schema, UnixMillis: 4000 * 1000, SessionType: "guard"},
	}
	cvRows := []cachevalueledger.Row{
		{Schema: cachevalueledger.Schema, UnixMillis: 1000 * 1000},
		{Schema: cachevalueledger.Schema, UnixMillis: 3000 * 1000},
	}
	rep := Fold(Input{
		Now:          time.Unix(9000, 0),
		Since:        since,
		GatewayUsage: GatewayUsageInput{Path: "/x/gateway-usage.jsonl", Present: true, Rows: gwRows},
		CacheValue:   CacheValueInput{Path: "/x/cache-value.jsonl", Present: true, Rows: cvRows},
	})
	if rep.Gateway.Basis != "observed" || rep.Cache.Basis != "observed" {
		t.Fatalf("want both observed, got gateway=%q cache=%q", rep.Gateway.Basis, rep.Cache.Basis)
	}
	if rep.Gateway.Sessions != 2 {
		t.Errorf("gateway: want 2 sessions within --since, got %d", rep.Gateway.Sessions)
	}
	if rep.Gateway.Trend == nil {
		t.Fatalf("want a trend over >=2 rows")
	}
	if rep.Cache.Sessions != 1 {
		t.Errorf("cache: want 1 session within --since, got %d", rep.Cache.Sessions)
	}
}

func TestFold_SinksAlwaysSorted(t *testing.T) {
	rep := Fold(Input{Now: time.Now()})
	for i := 1; i < len(rep.Sinks); i++ {
		if rep.Sinks[i-1].Kind >= rep.Sinks[i].Kind {
			t.Fatalf("sinks not sorted at index %d: %s >= %s", i, rep.Sinks[i-1].Kind, rep.Sinks[i].Kind)
		}
	}
}
