package witness

// Synthetic-producer proof for the settled-artifact receipt rung (#5646).
//
// Every case below drives a REAL file on disk through a REAL producer timeline,
// but the timeline is stepped by the watcher's own sleep seam rather than by wall
// clock: each inter-sample gap runs the next producer step. So "the producer
// appended between observation 1 and observation 2" is an exact, reproducible
// fact, and the suite costs no sleeping.

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// harness is a synthetic asynchronous producer plus a deterministic clock. steps[i]
// runs during the gap between sample i and sample i+1; a nil step means the
// producer did nothing in that gap (the quiescent case).
type harness struct {
	steps []func()
	n     int
	clock time.Time
}

func newHarness(steps ...func()) *harness {
	return &harness{steps: steps, clock: time.Unix(1_700_000_000, 0).UTC()}
}

func (h *harness) watcher() SettleWatcher {
	return SettleWatcher{
		Now: func() time.Time { return h.clock },
		Sleep: func(ctx context.Context, d time.Duration) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if h.n < len(h.steps) && h.steps[h.n] != nil {
				h.steps[h.n]()
			}
			h.n++
			h.clock = h.clock.Add(d)
			return nil
		},
	}
}

func settleWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func settleAppend(t *testing.T, path, body string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(body); err != nil {
		t.Fatal(err)
	}
}

// spec builds a 3-sample spec. The interval is only reporting data here: the
// harness sleep never actually waits.
func settleSpecFor(path string) SettleSpec {
	return SettleSpec{Path: path, IntervalNanos: int64(10 * time.Millisecond), Samples: 3}
}

// itoa / itoa64 format a sample's integer identity fields the same way Render's
// %d verbs do, so the output-equivalence test asserts against the rendered text
// rather than against a second, drifting restatement of the format.
func itoa(v int) string     { return strconv.Itoa(v) }
func itoa64(v int64) string { return strconv.FormatInt(v, 10) }

func wantState(t *testing.T, rec SettleReceipt, state SettleState, reason string) {
	t.Helper()
	if rec.State != state || rec.Reason != reason {
		t.Fatalf("state=%s reason=%s, want %s/%s\n%s", rec.State, rec.Reason, state, reason, rec.Render())
	}
}

// --- the six producer cases the issue names ------------------------------------

// GRADUAL APPEND: the canonical worker-log case. The file exists at every sample —
// existence alone would say "ready" — but it is still being written.
func TestSettleGradualAppendIsStillGrowing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "worker.log")
	settleWrite(t, p, "line 1\n")
	h := newHarness(
		func() { settleAppend(t, p, "line 2\n") },
		func() { settleAppend(t, p, "line 3\n") },
	)
	rec := h.watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleStillGrowing, "identity_changed_during_window")
	if len(rec.Samples) != 3 {
		t.Fatalf("want 3 samples, got %d", len(rec.Samples))
	}
	// The receipt carries the growth as evidence, not just a verdict.
	if !(rec.Samples[0].Size < rec.Samples[1].Size && rec.Samples[1].Size < rec.Samples[2].Size) {
		t.Fatalf("samples must record the growth: %+v", rec.Samples)
	}
}

// The SAME artifact, once the producer stops, settles — and the settled receipt is
// the thing that changed, not the file's existence.
func TestSettleQuiescentFileIsSettled(t *testing.T) {
	p := filepath.Join(t.TempDir(), "worker.log")
	settleWrite(t, p, "line 1\nline 2\nline 3\n")
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleSettled, "quiescent")
	if rec.Basis != SettleBasisDigest {
		t.Fatalf("basis = %q, want the digest basis by default", rec.Basis)
	}
}

// TRUNCATE-AND-REWRITE, in its most adversarial form: the rewrite lands the SAME
// byte count and the mtime is forced back to the original value, so size and mtime
// are pinned identical across every sample. Only the content digest can see it.
// This is why the digest is on by default.
func TestSettleTruncateAndRewriteCannotBeMisclassifiedAsSettled(t *testing.T) {
	p := filepath.Join(t.TempDir(), "result.json")
	pinned := time.Unix(1_600_000_000, 0)
	settleWrite(t, p, "AAAA")
	if err := os.Chtimes(p, pinned, pinned); err != nil {
		t.Fatal(err)
	}
	h := newHarness(func() {
		settleWrite(t, p, "BBBB") // truncate + rewrite, identical length
		if err := os.Chtimes(p, pinned, pinned); err != nil {
			t.Fatal(err)
		}
	}, nil)
	rec := h.watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleStillGrowing, "identity_changed_during_window")
	if rec.Samples[0].Size != rec.Samples[1].Size || rec.Samples[0].ModUnixNano != rec.Samples[1].ModUnixNano {
		t.Fatal("fixture broken: size and mtime were meant to be pinned identical")
	}
	if rec.Samples[0].Digest == rec.Samples[1].Digest {
		t.Fatal("digest failed to distinguish a same-size rewrite")
	}
}

// COMPLETED MARKER: quiescence plus the producer's own declaration.
func TestSettleCompletedMarkerIsSettled(t *testing.T) {
	dir := t.TempDir()
	p, marker := filepath.Join(dir, "trace.jsonl"), filepath.Join(dir, "trace.done")
	settleWrite(t, p, "{}\n")
	settleWrite(t, marker, "run-7\n")
	spec := settleSpecFor(p)
	spec.Marker, spec.RunID = marker, "run-7"
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleSettled, "marker_and_quiescent")
	if rec.Samples[2].MarkerState != SettleMarkerReady {
		t.Fatalf("marker state = %q, want ready", rec.Samples[2].MarkerState)
	}
}

// A marker that lands while the artifact is STILL moving does not launder the
// growth: the artifact's own identity must also be quiet.
func TestSettleMarkerDuringGrowthIsStillGrowing(t *testing.T) {
	dir := t.TempDir()
	p, marker := filepath.Join(dir, "trace.jsonl"), filepath.Join(dir, "trace.done")
	settleWrite(t, p, "{}\n")
	spec := settleSpecFor(p)
	spec.Marker = marker
	h := newHarness(func() {
		settleAppend(t, p, "{}\n")
		settleWrite(t, marker, "done")
	}, nil)
	rec := h.watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleStillGrowing, "marker_present_but_artifact_changed")
}

// PRODUCER FAILURE: the producer wrote a prefix and died. The file is perfectly
// quiet — indistinguishable from "finished" by quiescence alone — but the declared
// completion marker never appeared, so the receipt says stalled, not settled.
func TestSettleProducerDiedMidWriteIsStalled(t *testing.T) {
	dir := t.TempDir()
	p, marker := filepath.Join(dir, "eval.json"), filepath.Join(dir, "eval.done")
	settleWrite(t, p, `{"partial":`)
	spec := settleSpecFor(p)
	spec.Marker = marker
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleStalled, "quiet_but_no_completion_marker")
	if rec.Samples[0].MarkerState != SettleMarkerAbsent {
		t.Fatalf("marker state = %q, want absent", rec.Samples[0].MarkerState)
	}
}

// STALE FILE, binding 1 — the run-start floor. Last run's result is quiet, present,
// and byte-stable. Quiescence alone would call it settled; the floor refuses it.
func TestSettleStaleFileRefusedByRunFloor(t *testing.T) {
	p := filepath.Join(t.TempDir(), "result.json")
	settleWrite(t, p, "last run's answer\n")
	old := time.Unix(1_500_000_000, 0)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	spec := settleSpecFor(p)
	spec.NotBeforeUnixNano = time.Unix(1_600_000_000, 0).UnixNano()
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleMissing, "predates_run_floor")

	// The same observation with no floor and no marker settles — proving the
	// refusal came from the run binding, not from some incidental instability.
	unbound := newHarness(nil, nil).watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, unbound, SettleSettled, "quiescent")
}

// STALE FILE, binding 2 — run-bound marker identity. Both the artifact AND a
// completion marker are present and quiet, but the marker names the PREVIOUS run.
func TestSettleStaleFileRefusedByMarkerRunMismatch(t *testing.T) {
	dir := t.TempDir()
	p, marker := filepath.Join(dir, "result.json"), filepath.Join(dir, "result.done")
	settleWrite(t, p, "last run's answer\n")
	settleWrite(t, marker, "run-6\n")
	spec := settleSpecFor(p)
	spec.Marker, spec.RunID = marker, "run-7"
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleMissing, "marker_bound_to_other_run")

	// The new producer's marker for run-7 flips the same artifact to settled.
	settleWrite(t, marker, "run-7\n")
	fresh := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, fresh, SettleSettled, "marker_and_quiescent")
}

// A run id with nothing to bind it to cannot prove identity, so it abstains rather
// than silently degrading to bare quiescence.
func TestSettleRunIDWithoutMarkerIsUnverifiable(t *testing.T) {
	p := filepath.Join(t.TempDir(), "result.json")
	settleWrite(t, p, "x\n")
	spec := settleSpecFor(p)
	spec.RunID = "run-7"
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleUnverifiable, "run_id_without_marker")
}

// MISSING FILE: never appeared at all.
func TestSettleMissingFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "never.json")
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleMissing, "absent_throughout")
}

// An artifact that APPEARS mid-window is still growing, not settled: the consumer
// caught the producer in the act of creating it.
func TestSettleAppearsMidWindowIsStillGrowing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "late.json")
	rec := newHarness(func() { settleWrite(t, p, "{}\n") }, nil).
		watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleStillGrowing, "identity_changed_during_window")
}

// An artifact that is deleted mid-window (a producer cleaning up after a failure)
// is missing, not settled on its last-seen bytes.
func TestSettleVanishedDuringWindowIsMissing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "gone.json")
	settleWrite(t, p, "{}\n")
	rec := newHarness(func() { os.Remove(p) }, nil).
		watcher().Observe(context.Background(), settleSpecFor(p))
	wantState(t, rec, SettleMissing, "vanished_during_window")
}

// --- the helper's own contract --------------------------------------------------

// CANCELLABLE: a cancelled context returns promptly with an unverifiable receipt —
// never a settled one, and never a completed window.
func TestSettleCancelledIsUnverifiable(t *testing.T) {
	p := filepath.Join(t.TempDir(), "worker.log")
	settleWrite(t, p, "x\n")

	// Cancelled before the first observation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pre := newHarness(nil, nil).watcher().Observe(ctx, settleSpecFor(p))
	wantState(t, pre, SettleUnverifiable, "canceled")
	if len(pre.Samples) != 0 {
		t.Fatalf("a pre-cancelled observation took %d samples, want 0", len(pre.Samples))
	}

	// Cancelled mid-window: the wait returns immediately instead of sleeping out
	// the rest of the interval.
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	w := SettleWatcher{Sleep: func(c context.Context, _ time.Duration) error {
		cancel2()
		return c.Err()
	}}
	mid := w.Observe(ctx2, settleSpecFor(p))
	wantState(t, mid, SettleUnverifiable, "canceled")
}

// READ-ONLY: the rung must be safe to point at a live producer's output. Nothing in
// the observed tree may change — not content, not size, not mtime — and no new
// entry (lock, temp file, marker) may appear.
func TestSettleIsReadOnly(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "bundle")
	settleWrite(t, filepath.Join(root, "a.txt"), "alpha\n")
	settleWrite(t, filepath.Join(root, "nested", "b.txt"), "beta\n")

	before := treeSnapshot(t, dir)
	spec := settleSpecFor(root)
	spec.Marker = filepath.Join(dir, "bundle.done") // declared but absent
	rec := newHarness(nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleStalled, "quiet_but_no_completion_marker")

	if after := treeSnapshot(t, dir); after != before {
		t.Fatalf("the settle rung mutated the tree it observed:\nbefore:\n%s\nafter:\n%s", before, after)
	}
}

// treeSnapshot renders every entry's path, size, mtime and content — the full state
// a read-only observer must leave untouched.
func treeSnapshot(t *testing.T, root string) string {
	t.Helper()
	var lines []string
	err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		if fi.IsDir() {
			lines = append(lines, "dir "+filepath.ToSlash(rel))
			return nil
		}
		body, rerr := os.ReadFile(p)
		if rerr != nil {
			return rerr
		}
		lines = append(lines, strings.Join([]string{
			"file " + filepath.ToSlash(rel),
			time.Duration(fi.Size()).String(),
			fi.ModTime().UTC().Format(time.RFC3339Nano),
			string(body),
		}, "\x00"))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(lines)
	return strings.Join(lines, "\n")
}

// A directory artifact (the diagnostic-bundle case) is observed as one whole: a
// member appearing anywhere in the tree is growth.
func TestSettleDirectoryBundleGrowthAndQuiescence(t *testing.T) {
	root := filepath.Join(t.TempDir(), "bundle")
	settleWrite(t, filepath.Join(root, "a.txt"), "alpha\n")

	growing := newHarness(func() { settleWrite(t, filepath.Join(root, "nested", "b.txt"), "beta\n") }, nil).
		watcher().Observe(context.Background(), settleSpecFor(root))
	wantState(t, growing, SettleStillGrowing, "identity_changed_during_window")

	quiet := newHarness(nil, nil).watcher().Observe(context.Background(), settleSpecFor(root))
	wantState(t, quiet, SettleSettled, "quiescent")
	if quiet.Samples[0].Files != 2 || quiet.Samples[0].Size == 0 {
		t.Fatalf("tree sample must carry member count and total bytes: %+v", quiet.Samples[0])
	}
}

// A single look can never witness quiescence, so the sample count floors at two and
// the receipt reports the basis it actually used.
func TestSettleSampleCountIsFlooredAndReported(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.log")
	settleWrite(t, p, "x\n")
	spec := SettleSpec{Path: p, Samples: 1, IntervalNanos: 1, NoDigest: true}
	rec := newHarness(nil).watcher().Observe(context.Background(), spec)
	if rec.WantSamples != SettleMinSamples || len(rec.Samples) != SettleMinSamples {
		t.Fatalf("want %d samples, got want=%d actual=%d", SettleMinSamples, rec.WantSamples, len(rec.Samples))
	}
	if rec.Basis != SettleBasisStat {
		t.Fatalf("basis = %q, want %q when digesting is off", rec.Basis, SettleBasisStat)
	}
	if rec.Samples[0].Digest != "" {
		t.Fatal("NoDigest must not record a digest")
	}
}

// A window that would park the kernel is refused up front rather than served.
func TestSettleOverlongWindowIsUnverifiable(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.log")
	settleWrite(t, p, "x\n")
	spec := SettleSpec{Path: p, IntervalNanos: int64(SettleMaxWindow), Samples: 4}
	rec := newHarness(nil, nil, nil).watcher().Observe(context.Background(), spec)
	wantState(t, rec, SettleUnverifiable, "window_exceeds_max")
}

func TestSettleEmptyPathIsUnverifiable(t *testing.T) {
	rec := Settle(context.Background(), SettleSpec{})
	wantState(t, rec, SettleUnverifiable, "empty_path")
}

// --- output equivalence ---------------------------------------------------------

// HUMAN AND JSON PRESERVE THE SAME STATE AND SAMPLES. Both shapes come from one
// struct: JSON round-trips it exactly, and every sample field the JSON carries is
// present verbatim in the rendered text.
func TestSettleHumanAndJSONAgree(t *testing.T) {
	dir := t.TempDir()
	p, marker := filepath.Join(dir, "trace.jsonl"), filepath.Join(dir, "trace.done")
	settleWrite(t, p, "{}\n")
	spec := settleSpecFor(p)
	spec.Marker, spec.RunID = marker, "run-9"
	h := newHarness(func() { settleAppend(t, p, "{}\n") }, func() { settleWrite(t, marker, "run-9") })
	rec := h.watcher().Observe(context.Background(), spec)

	raw, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var back SettleReceipt
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if back.State != rec.State || back.Reason != rec.Reason || len(back.Samples) != len(rec.Samples) {
		t.Fatalf("JSON round-trip lost the verdict: %+v vs %+v", back, rec)
	}
	for i := range rec.Samples {
		if back.Samples[i] != rec.Samples[i] {
			t.Fatalf("JSON round-trip lost sample %d: %+v vs %+v", i, back.Samples[i], rec.Samples[i])
		}
	}

	text := rec.Render()
	if !strings.Contains(text, "state:    "+string(rec.State)) {
		t.Fatalf("human output does not carry the state:\n%s", text)
	}
	if !strings.Contains(text, "reason:   "+rec.Reason) || !strings.Contains(text, "basis:    "+rec.Basis) {
		t.Fatalf("human output does not carry reason/basis:\n%s", text)
	}
	if !strings.Contains(text, rec.RunID) || !strings.Contains(text, rec.Marker) {
		t.Fatalf("human output does not carry the run binding:\n%s", text)
	}
	// Every sample the JSON carries is rendered, with its identity fields intact.
	for _, s := range rec.Samples {
		for _, want := range []string{
			"[" + itoa(s.Index) + "]",
			"size=" + itoa64(s.Size),
			"mod=" + itoa64(s.ModUnixNano),
			"digest=" + s.Digest,
			"marker=" + s.MarkerState,
		} {
			if !strings.Contains(text, want) {
				t.Fatalf("human output is missing %q for sample %d:\n%s", want, s.Index, text)
			}
		}
	}
}

// --- the claim rung -------------------------------------------------------------

// The kernel-facing grammar: settled => CONFIRMED, a growing artifact => REFUTED,
// an incoherent spec => ABSTAIN (never a false confirm).
func TestSettleClaimRungOutcomes(t *testing.T) {
	dir := t.TempDir()
	settleWrite(t, filepath.Join(dir, "done.log"), "complete\n")

	// The resolver must not need git for this rung: a runner that fails the test
	// proves the settled rung neither caches nor shells out.
	r := NewWithRunner(func(context.Context, string, ...string) (string, int, error) {
		t.Fatal("the settled rung must not invoke git")
		return "", 0, nil
	}, dir)

	fast := `,"interval_nanos":1000,"samples":2}`
	cases := []struct {
		name  string
		claim string
		want  abi.WitnessOutcome
	}{
		// A relative path resolves against the resolver's dir, so the claim means
		// the same thing wherever the kernel runs.
		{"settled", `settled:{"path":"done.log"` + fast, abi.WitnessConfirmed},
		{"missing", `settled:{"path":"never.log"` + fast, abi.WitnessRefuted},
		{"malformed", `settled:not-json`, abi.WitnessAbstain},
		{"unverifiable", `settled:{"path":""` + fast, abi.WitnessAbstain},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := r.Resolve(context.Background(), nil, tc.claim); got != tc.want {
				t.Fatalf("Resolve(%s) = %v, want %v", tc.claim, got, tc.want)
			}
		})
	}
}

// A settled verdict is a live filesystem observation of something that is BY
// DEFINITION still moving; memoizing it would be the exact staleness the rung
// exists to refuse.
func TestSettleClaimIsNeverCached(t *testing.T) {
	r := NewWithRunner(func(context.Context, string, ...string) (string, int, error) {
		t.Fatal("a non-cacheable claim must not resolve cache anchors")
		return "", 0, nil
	}, t.TempDir())
	if _, ok := r.cacheKey(context.Background(), "settled", `{"path":"x"}`); ok {
		t.Fatal("settled: must never be cacheable")
	}
}
