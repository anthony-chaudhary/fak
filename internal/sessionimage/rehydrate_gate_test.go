package sessionimage

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rehydrate"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// recordingGate builds a rehydrate.Gate of the five canonical rungs, each recording that it
// ran (into *ran) and clearing — unless its reason is in refuse, in which case it refuses.
// It stands in for the four real rungs (#1182-#1186) that compose into this boundary later;
// the wire-in's job is only to feed the dormancy band in and surface the verdict out.
func recordingGate(ran *[]rehydrate.Reason, refuse map[rehydrate.Reason]bool) *rehydrate.Gate {
	mk := func(r rehydrate.Reason) rehydrate.Rung {
		return rehydrate.NewRung(r, func(context.Context) rehydrate.Verdict {
			*ran = append(*ran, r)
			if refuse[r] {
				return rehydrate.Refuse(r, "wire-in test refusal")
			}
			return rehydrate.Clear()
		})
	}
	return rehydrate.NewGate(
		mk(rehydrate.ColdCache), mk(rehydrate.StaleCred), mk(rehydrate.StaleRecall),
		mk(rehydrate.StaleLease), mk(rehydrate.StalePlan),
	)
}

// TestRehydrateGateWarmRunsNoRungs: an image resumed within minutes of its dump runs zero
// rehydration rungs and is admitted — the warm band is read from the image's own
// UpdatedUnix vs the injected resume clock, so a short gap is the verbatim resume path.
func TestRehydrateGateWarmRunsNoRungs(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const dumpedAt = 1_700_000_000
	in := Input{SessionID: "sess-warm", Drive: session.DefaultState("sess-warm"), Now: dumpedAt}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	var ran []rehydrate.Reason
	res, err := img.Rehydrate(ctx, RehydrateOptions{
		Table: session.NewTable(),
		Gate:  recordingGate(&ran, nil),
		Now:   dumpedAt + 100, // 100s after the dump => Warm
	})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}
	if !res.Gated {
		t.Fatal("a gate was supplied but Resumed.Gated is false")
	}
	if !res.Admitted() {
		t.Fatalf("warm resume not admitted: %+v", res.Admission)
	}
	if len(res.Admission.Ran) != 0 || len(ran) != 0 {
		t.Fatalf("warm resume ran rungs: %v", res.Admission.RanReasons())
	}
}

// TestRehydrateGateFrozenRunsAllRungs: an image dormant ~10 days runs all five rungs and,
// when each clears, admits; when one refuses, the resumed handle is NOT admitted (so the
// caller withholds the first post-wake action) even though the drive is faithfully restored.
func TestRehydrateGateFrozenRunsAllRungs(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const dumpedAt = 1_700_000_000
	const tenDays = 10 * 24 * 3600 // within [24h, 30d) => Frozen
	in := Input{SessionID: "sess-frozen", Drive: session.DefaultState("sess-frozen"), Now: dumpedAt}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}

	// All rungs clear -> admitted, all five ran.
	var ran []rehydrate.Reason
	res, err := img.Rehydrate(ctx, RehydrateOptions{
		Table: session.NewTable(),
		Gate:  recordingGate(&ran, nil),
		Now:   dumpedAt + tenDays,
	})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}
	if !res.Admitted() {
		t.Fatalf("frozen all-clear not admitted: %+v", res.Admission)
	}
	if len(res.Admission.Ran) != 5 {
		t.Fatalf("frozen resume ran %d rungs, want 5: %v", len(res.Admission.Ran), res.Admission.RanReasons())
	}

	// STALE_LEASE refuses -> not admitted, RefusedBy names it; the drive is still restored.
	tbl := session.NewTable()
	var ran2 []rehydrate.Reason
	res2, err := img.Rehydrate(ctx, RehydrateOptions{
		Table: tbl,
		Gate:  recordingGate(&ran2, map[rehydrate.Reason]bool{rehydrate.StaleLease: true}),
		Now:   dumpedAt + tenDays,
	})
	if err != nil {
		t.Fatalf("Rehydrate(refuse): %v", err)
	}
	if res2.Admitted() {
		t.Fatalf("admitted despite a STALE_LEASE refusal: %+v", res2.Admission)
	}
	if res2.Admission.RefusedBy != rehydrate.StaleLease {
		t.Fatalf("RefusedBy = %q, want STALE_LEASE", res2.Admission.RefusedBy)
	}
	if tbl.Get("sess-frozen").Run != session.Running {
		t.Fatal("drive not restored on a gated (refused) resume — restore must not depend on admission")
	}
}

// TestRehydrateNoGateIsUnconditional: with no Gate, a long-dormant image resumes admitted
// and ungated — the wire-in is strictly additive (today's behavior is preserved).
func TestRehydrateNoGateIsUnconditional(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const dumpedAt = 1_700_000_000
	in := Input{SessionID: "sess-nogate", Drive: session.DefaultState("sess-nogate"), Now: dumpedAt}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	res, err := img.Rehydrate(ctx, RehydrateOptions{Table: session.NewTable(), Now: dumpedAt + 100*24*3600})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}
	if res.Gated || !res.Admitted() {
		t.Fatalf("no-gate resume should be ungated+admitted: Gated=%v Admitted=%v", res.Gated, res.Admitted())
	}
}
