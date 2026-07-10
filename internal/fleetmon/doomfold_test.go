package fleetmon

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/doomloop"
)

// doomStream builds a worker's trailing verified-progress history from parallel
// effort/progress counters (Alive throughout — the common live-worker case).
func doomStream(effort, progress []int64) []doomloop.Sample {
	if len(effort) != len(progress) {
		panic("effort/progress length mismatch")
	}
	out := make([]doomloop.Sample, len(effort))
	for i := range effort {
		out[i] = doomloop.Sample{UnixMillis: int64(i) * 60_000, Effort: effort[i], Progress: progress[i], Alive: true}
	}
	return out
}

// burningFlat builds an n-sample burning-flat history: effort climbs every
// sample, verified progress never moves — the doom-loop signature at any length.
func burningFlat(n int) []doomloop.Sample {
	out := make([]doomloop.Sample, n)
	for i := 0; i < n; i++ {
		out[i] = doomloop.Sample{UnixMillis: int64(i) * 60_000, Effort: int64(i+1) * 10, Progress: 7, Alive: true}
	}
	return out
}

func hasSubstr(reasons []string, sub string) bool {
	for _, r := range reasons {
		if strings.Contains(r, sub) {
			return true
		}
	}
	return false
}

// healthyEvidence is a worker the activity-only liveness read calls healthy:
// PID alive, transcript fresh (10s old) and growing (+15 lines). This is the
// exact shape #4148 is about — active, but the fold has yet to consult progress.
func healthyEvidence(now time.Time) WorkerEvidence {
	prev := iptr(40)
	return WorkerEvidence{
		Session: "w", HasPID: true, PID: 10, PIDAlive: true, PIDLivenessKnown: true, PrevLines: prev,
		Transcript: TranscriptSignal{Exists: true, Lines: 55, HasTimestamp: true, LastTimestamp: now.Add(-10 * time.Second)},
	}
}

// TestDoomLoopWorkerNoLongerReadsHealthy is the #4148 witness: a worker whose
// transcript is fresh and growing (activity-only "healthy") but whose VERIFIED
// progress has been flat while effort burns for >= TripWindows windows must NOT
// be called healthy. It flips to attention with a doom-loop reason.
func TestDoomLoopWorkerNoLongerReadsHealthy(t *testing.T) {
	now := time.Now()
	base := healthyEvidence(now)

	// Control: no progress evidence => activity-only read => healthy (fail-open).
	if got := Classify(base, now, DefaultThresholds()); got.Class != ClassHealthy {
		t.Fatalf("control: activity-only worker must read healthy, got %s (reasons=%v)", got.Class, got.Reasons)
	}

	// Fold: same worker, burning effort with flat verified progress.
	doom := base
	doom.DoomSamples = doomStream(
		[]int64{10, 20, 30, 40, 50, 60},
		[]int64{4, 4, 4, 4, 4, 4},
	)
	got := Classify(doom, now, DefaultThresholds())
	if got.Class == ClassHealthy {
		t.Fatal("a burning-flat doom loop must NOT read healthy even though its transcript is actively growing")
	}
	if got.Class != ClassAttention {
		t.Fatalf("a burning-flat doom loop should route to attention, got %s", got.Class)
	}
	if !hasSubstr(got.Reasons, "doom loop") {
		t.Fatalf("attention reason should name the doom loop, got %v", got.Reasons)
	}
}

// TestGenuinelyAdvancingWorkerStaysHealthy: a worker burning effort AND landing
// real verified progress every window is healthy — spend with landings is not a
// doom loop. And the byte-for-byte healthy classification is UNCHANGED versus the
// no-samples baseline: the fold only ever demotes a confirmed doom loop, it never
// alters a genuinely-advancing worker's row.
func TestGenuinelyAdvancingWorkerStaysHealthy(t *testing.T) {
	now := time.Now()
	base := healthyEvidence(now)

	withProgress := base
	withProgress.DoomSamples = doomStream(
		[]int64{10, 20, 30, 40, 50, 60},
		[]int64{1, 2, 3, 4, 5, 6},
	)
	got := Classify(withProgress, now, DefaultThresholds())
	if got.Class != ClassHealthy {
		t.Fatalf("a worker landing real verified progress stays healthy, got %s (reasons=%v)", got.Class, got.Reasons)
	}

	baseline := Classify(base, now, DefaultThresholds())
	if !reflect.DeepEqual(got, baseline) {
		t.Fatalf("the healthy path changed under the fold:\n with-progress: %+v\n baseline:      %+v", got, baseline)
	}
}

// TestDoomFoldCostBounded is the "meter the meter" assertion: the per-worker fold
// reads a FIXED trailing window (doomWindowCap = EscalateWindows+1), so its cost
// is O(EscalateWindows) — independent of how long the worker's real sample history
// is — and the bound loses no decision (the bounded tail yields the same DOOM_LOOP
// verdict the full-history classify would).
func TestDoomFoldCostBounded(t *testing.T) {
	cfg := doomloop.DefaultConfig()
	windowCap := doomWindowCap(cfg)
	if windowCap != cfg.EscalateWindows+1 {
		t.Fatalf("window cap = %d, want EscalateWindows+1 = %d", windowCap, cfg.EscalateWindows+1)
	}

	for _, n := range []int{8, 5_000, 50_000} {
		hist := burningFlat(n)
		tail := boundedDoomTail(hist, windowCap)
		if len(tail) != windowCap {
			t.Fatalf("history %d: bounded tail = %d samples, want the fixed cap %d (fold cost must not grow with history)", n, len(tail), windowCap)
		}
		boundedConfirmed, _ := doomLoopConfirmed(hist, cfg)
		fullConfirmed := doomloop.Classify(hist, cfg).Verdict == doomloop.VerdictDoomLoop
		if !boundedConfirmed || !fullConfirmed {
			t.Fatalf("history %d: bounded=%v full=%v, want both true (the bound must lose no decision)", n, boundedConfirmed, fullConfirmed)
		}
	}

	// A history shorter than the cap passes through unchanged — no false padding.
	if got := boundedDoomTail(burningFlat(4), windowCap); len(got) != 4 {
		t.Fatalf("short history should pass through unchanged, got %d samples", len(got))
	}
	// Absent progress evidence never manufactures a doom verdict (fail-open).
	if confirmed, _ := doomLoopConfirmed(nil, cfg); confirmed {
		t.Fatal("empty samples must never confirm a doom loop")
	}
}
