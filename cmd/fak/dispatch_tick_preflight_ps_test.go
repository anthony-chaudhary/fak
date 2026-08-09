package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// #5537. dispatchScanProcessesPOSIX used to run its OWN `ps -eo pid=,nlwp=,rss=,comm=` and
// parse the table inline. `nlwp` is a procps-ng extension; BSD `ps` has no thread-count
// keyword at all and rejects the whole invocation rather than dropping the column, so the
// probe returned an error on every macOS tick and the procguard dimension was skipped.
//
// The fix routes the census through internal/procguard, which picks the argv per GOOS and
// leaves a column the dialect cannot supply NIL (psNoColumn). What is left here is the
// projection onto dispatchtick.ProcInfo, and the property that projection must hold is the
// one these tests pin: an unmeasured dimension stays unmeasured.
//
// Why that property and not "the census works on darwin": the second is only observable on a
// BSD-dialect host, and none is available to this suite. The per-GOOS argv itself is pinned
// where it lives, by internal/procguard's collect_posix_test.go, and the structural rule
// that no fork of it may reappear is pinned by internal/architest/ps_dialect_test.go.

// TestDispatchProcInfoFromProcguardKeepsAbsentDimensionsNil is the darwin shape: procguard
// hands back rows whose Threads is nil because BSD `ps` cannot answer it. A nil must survive
// as nil. Defaulting it to 0 would state that the process has no threads — a measurement
// nobody took — and EvaluateProcGuard would then read every macOS process as permanently
// under the thread ceiling, silently disabling the dimension this guard exists for.
func TestDispatchProcInfoFromProcguardKeepsAbsentDimensionsNil(t *testing.T) {
	rows := []procguard.Proc{
		// The darwin census: pid, rss and comm are real, thread count is unavailable.
		{PID: 4021, Name: "llama-cli", Threads: nil, Handles: nil, WSMB: procguard.IntPtr(812)},
		// A row where even the resident set did not parse.
		{PID: 4022, Name: "zombie", Threads: nil, Handles: nil, WSMB: nil},
	}

	got := dispatchProcInfoFromProcguard(rows)
	if len(got) != 2 {
		t.Fatalf("rows = %d, want 2 — every census row must be projected", len(got))
	}
	if got[0].PID != 4021 || got[0].Name != "llama-cli" {
		t.Fatalf("row 0 = %+v, want pid 4021 named llama-cli", got[0])
	}
	if got[0].Threads != nil {
		t.Fatalf("Threads = %d, want nil: BSD `ps` has no thread-count keyword, so a thread count here is a fabricated measurement", *got[0].Threads)
	}
	if got[0].Handles != nil {
		t.Fatalf("Handles = %d, want nil: POSIX `ps` reports no handle count at all", *got[0].Handles)
	}
	if got[0].WorkingSetMB == nil || *got[0].WorkingSetMB != 812 {
		t.Fatalf("WorkingSetMB = %v, want 812 — a dimension the dialect DOES answer must survive", got[0].WorkingSetMB)
	}
	if got[1].Threads != nil || got[1].WorkingSetMB != nil || got[1].Handles != nil {
		t.Fatalf("row 1 = %+v, want every dimension nil", got[1])
	}
}

// TestDispatchProcInfoFromProcguardCarriesLinuxDimensions is the other direction, and it is
// what keeps the test above from being satisfiable by a projection that simply drops every
// pointer. On a procps host the thread count is the whole point — the incident this guard
// was built for was one process at ~129,427 threads — so a measured value must arrive intact
// and must still be over the ceiling when the classifier looks at it.
func TestDispatchProcInfoFromProcguardCarriesLinuxDimensions(t *testing.T) {
	rows := []procguard.Proc{
		{PID: 7, Name: "runaway", Threads: procguard.IntPtr(129427), WSMB: procguard.IntPtr(64)},
	}

	got := dispatchProcInfoFromProcguard(rows)
	if len(got) != 1 {
		t.Fatalf("rows = %d, want 1", len(got))
	}
	if got[0].Threads == nil {
		t.Fatal("Threads = nil, want 129427 — a measured dimension must not be dropped")
	}
	if *got[0].Threads != 129427 {
		t.Fatalf("Threads = %d, want 129427", *got[0].Threads)
	}

	th := dispatchtick.DefaultProcGuardThresholds()
	if th.MaxThreads <= 0 {
		t.Fatalf("MaxThreads = %d, want a positive default ceiling", th.MaxThreads)
	}
	if *got[0].Threads <= th.MaxThreads {
		t.Fatalf("129427 threads must exceed the default ceiling %d, else this test asserts nothing", th.MaxThreads)
	}
}

// TestDispatchProcInfoFromProcguardEmptyCensus pins the empty case as EMPTY, not nil-panicky:
// zero rows project to zero rows. dispatchScanProcessesPOSIX reports "there is no census" via
// its error return instead, which is the distinction #5385 restored — an empty set and a
// failed scan must not be the same value.
func TestDispatchProcInfoFromProcguardEmptyCensus(t *testing.T) {
	if got := dispatchProcInfoFromProcguard(nil); len(got) != 0 {
		t.Fatalf("rows = %d, want 0", len(got))
	}
}
