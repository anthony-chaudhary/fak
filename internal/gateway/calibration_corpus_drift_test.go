package gateway

// calibration_corpus_drift_test.go — the executable guard that closes the open loop behind
// two HAND-calibrated constants in gateway.go: DefaultAssumedSessionTurns and
// headHorizonHeavyResidentFloor. Both were sized from the observed distribution in
// docs/nightrun/gateway-usage.jsonl (the session-length p90, and the per-turn cached-prompt
// p10), but the calibration lived only in a prose comment — nothing FAILED if the corpus
// drifted out from under the number, or if an editor changed the const without re-checking
// the corpus. These tests recompute the same percentiles the comment cites and assert each
// const still sits in its documented band, so a real drift (either direction) is caught in
// CI instead of silently degrading the head-anchored burst gate's economics.
//
// WHICH CORPUS (#5406). The guard reads gatewayusageledger.PublishedLedgerRel — the TRACKED
// docs/nightrun snapshot — never DefaultLedgerRel, the gitignored per-box .fak runtime
// ledger. That distinction is the whole point of this file's second revision:
//
//   - reading the live ledger made the verdict UNREPRODUCIBLE. It is empty in CI and on a
//     fresh clone, so the guard skipped ("thin corpus: only 0 nonzero-length sessions") on
//     every tree CI ever blessed, and red only on the boxes running the headless fleet. The
//     person who saw the red could not show it to anyone else.
//   - the two populations disagree, and the constants belong to the tracked one. Measured
//     2026-07-28 with the interpolation below, exit rows only: the tracked corpus carries
//     n=803 sessions with cached_turns (p50 26, p75 39, p90 54, p95 73) and per-turn
//     cached-prompt p10 51986 / p50 64963 — DefaultAssumedSessionTurns=50 and
//     headHorizonHeavyResidentFloor=60000 both sit comfortably INSIDE. The live ledger on a
//     fleet box carries n=2436 sessions with p50 6, p75 6, p95 18: night-dispatch traffic,
//     where each headless resolver is a short mostly-automated session. Recalibrating 50→~10
//     against that would size the INTERACTIVE burst gate's economics on fleet automation.
//     The guard was not detecting that the world moved; it was detecting that the SAMPLE
//     changed composition, which is not what its failure message claimed.
//
// ONE LENGTH BASIS, NAMED (#5406 scope 2). The session-length distribution is computed from
// cached_turns ALONE — the proxy the constant was actually calibrated on. The earlier
// revision preferred ObservedTurns and fell back to cached_turns per row, which silently
// mixed two bases inside one distribution (on the tracked corpus that blend reads p50=25,
// a value neither basis produces). ObservedTurns is still computed and LOGGED beside it as
// an unasserted cross-check, because the two bases disagree on the same corpus (p50 35 vs
// 26) and whether that is a population shift or a change in what ObservedTurns counts is
// NOT settled — see #5406. Logging keeps that divergence visible without gating the trunk
// on an unresolved question.
//
// Posture: the tests SKIP (never fail) when the corpus is absent or thin, but a permanent
// skip can no longer be mistaken for a pass — TestCalibrationDriftCorpusIsPresentAndThick
// FAILS loudly if the tracked corpus ever goes missing or thin, so the guard cannot quietly
// go vacuous. The bands are deliberately WIDE (a drift GUARD, not an exact-match assertion)
// and now carry an explicit driftBandMarginFrac tolerance: they catch a gross shift (someone
// setting the const to 5 or 500, or the real distribution moving a whole quartile), not the
// sub-percent wobble of an appended corpus nudging a band edge past the constant.

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// minCorpusForDrift is the nonzero-sample floor below which the distribution is too thin to
// assert a percentile band on — the tests skip rather than fail (the cachevalueledger
// thin-corpus posture). 100 is comfortably below the ~800 real sessions the tracked corpus
// carries today while still refusing to judge a near-empty file.
const minCorpusForDrift = 100

// driftBandMarginFrac is how far OUTSIDE a band edge the constant must land before the guard
// fails, as a fraction of that edge. It exists because a band edge with no hysteresis flips
// red and green as rows are appended: #5406 caught headHorizonHeavyResidentFloor=60000
// failing a p10 of 60385 — a 0.6% miss, which is exactly the "normal wobble of an appended
// corpus" this file's header promises NOT to fail on. 5% is an order of magnitude above that
// wobble and an order of magnitude below the gross shift the guard is for (a const set to 5
// or 500, or the distribution moving a whole quartile), so it buys hysteresis without buying
// blindness. It is deliberately NOT a config knob — a drift guard whose tolerance can be
// widened from outside is a guard that gets widened instead of investigated.
const driftBandMarginFrac = 0.05

// resolveCorpusFrom walks up from startDir to the module root (the dir holding go.mod) and
// returns the path to the TRACKED gateway-usage corpus, or ok=false if neither the root nor
// the corpus is found. Split out from findCorpus so the resolution itself is testable against
// a synthetic tree — the #5406 regression (resolving the gitignored per-box ledger instead of
// the tracked one) is invisible to any test that can only observe the resolution through the
// real repo, where BOTH files exist.
func resolveCorpusFrom(startDir string) (string, bool) {
	dir := startDir
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			p := filepath.Join(dir, filepath.FromSlash(gatewayusageledger.PublishedLedgerRel))
			if _, err := os.Stat(p); err == nil {
				return p, true
			}
			return "", false
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// findCorpus resolves the tracked corpus from the test's working directory, keeping the test
// independent of the caller's cwd.
func findCorpus(t *testing.T) (string, bool) {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	return resolveCorpusFrom(dir)
}

// percentile returns the linearly-interpolated p-th percentile (0..100) of xs, which need
// not be pre-sorted (it sorts a copy). Matches the numpy-style interpolation the calibration
// comment's percentiles were computed with, so the band edges line up with the documented
// numbers.
func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	k := float64(len(s)-1) * p / 100
	lo := int(k)
	hi := lo + 1
	if hi >= len(s) {
		return s[len(s)-1]
	}
	return s[lo] + (s[hi]-s[lo])*(k-float64(lo))
}

// withinBand reports whether got sits inside [lo, hi] after each edge is relaxed outward by
// marginFrac of its own magnitude. Returns the relaxed edges too, so a failure message can
// state the tolerance it actually applied rather than the raw percentiles (an operator
// reading "outside [39, 73]" for a value of 38.9 would reasonably think the guard is broken).
func withinBand(got, lo, hi, marginFrac float64) (ok bool, loTol, hiTol float64) {
	loTol = lo - math.Abs(lo)*marginFrac
	hiTol = hi + math.Abs(hi)*marginFrac
	return got >= loTol && got <= hiTol, loTol, hiTol
}

// cachedTurnLengths returns the per-session length distribution on the DECLARED basis:
// cached_turns, the proxy the calibration in gateway.go was computed over ("n=1893 real
// guard/serve exits, `cached_turns`"). Every exit row carrying a positive cached_turns
// contributes exactly one sample — no fallback, no second signal folded in. A zero-turn exit
// is a serve process that served nothing, not a length sample.
func cachedTurnLengths(rows []gatewayusageledger.Row) []float64 {
	var out []float64
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		if n := r.Counters.CachedTurns; n > 0 {
			out = append(out, float64(n))
		}
	}
	return out
}

// observedTurnLengths returns the same distribution on the ObservedTurns basis — the honest
// served-turn count recorded once a session runs a provenance-stamped build. It is REPORTED,
// never asserted: it is a different (and on the tracked corpus, materially different)
// population from the one the constant was sized on, and #5406 leaves open whether that gap
// is a population shift or a change in what the field counts.
func observedTurnLengths(rows []gatewayusageledger.Row) []float64 {
	var out []float64
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		if n := r.Counters.ObservedTurns; n > 0 {
			out = append(out, float64(n))
		}
	}
	return out
}

// perTurnCachedPrompt returns the per-turn cached-prompt volume (cached_prompt_tokens divided
// by cached_turns) for every exit row that carries both — the working-set signal
// headHorizonHeavyResidentFloor was sized against.
func perTurnCachedPrompt(rows []gatewayusageledger.Row) []float64 {
	var out []float64
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		c := r.Counters
		if c.CachedTurns > 0 && c.CachedPromptTokens > 0 {
			out = append(out, float64(c.CachedPromptTokens)/float64(c.CachedTurns))
		}
	}
	return out
}

// TestDefaultAssumedSessionTurnsTracksCorpusP90 asserts the assumed session-length prior
// still sits in the upper tail of the tracked corpus's cached_turns session-length
// distribution it was calibrated to (~p90). Below p75 it would over-shed short sessions
// (early bursts justified by turns that never arrive); above p95 it would assume a repaying
// tail that almost never materializes. The [p75, p95] band is the documented span, relaxed by
// driftBandMarginFrac at each edge.
func TestDefaultAssumedSessionTurnsTracksCorpusP90(t *testing.T) {
	path, ok := findCorpus(t)
	if !ok {
		t.Skip("tracked gateway-usage corpus (" + gatewayusageledger.PublishedLedgerRel + ") not present — drift guard is corpus-dependent")
	}
	rows := gatewayusageledger.ReadLedgerFile(path)
	lengths := cachedTurnLengths(rows)
	if len(lengths) < minCorpusForDrift {
		t.Skipf("thin corpus: only %d sessions with cached_turns (<%d) in %s — not enough to assert a band",
			len(lengths), minCorpusForDrift, gatewayusageledger.PublishedLedgerRel)
	}
	p75, p90, p95 := percentile(lengths, 75), percentile(lengths, 90), percentile(lengths, 95)
	got := float64(DefaultAssumedSessionTurns)
	inBand, loTol, hiTol := withinBand(got, p75, p95, driftBandMarginFrac)
	// The population is logged as the RESOLVED path, never as the constant the resolver was
	// meant to use: printing the intended name would have let #5406 hide in plain sight, since
	// a guard reading the wrong corpus still reports the right label.
	t.Logf("population=%s basis=cached_turns n=%d p75=%.1f p90=%.1f p95=%.1f tolerated=[%.1f, %.1f] const=%d",
		filepath.ToSlash(path), len(lengths), p75, p90, p95, loTol, hiTol, DefaultAssumedSessionTurns)
	// Cross-check only (see file header): the observed-turn basis on the same corpus.
	if obs := observedTurnLengths(rows); len(obs) > 0 {
		t.Logf("cross-check (NOT asserted) basis=observed_turns n=%d p75=%.1f p90=%.1f p95=%.1f",
			len(obs), percentile(obs, 75), percentile(obs, 90), percentile(obs, 95))
	}
	if !inBand {
		t.Fatalf("DefaultAssumedSessionTurns=%d has drifted outside the calibrated [p75=%.0f, p95=%.0f] session-length band "+
			"(tolerated [%.0f, %.0f] at a %.0f%% edge margin; corpus p90=%.0f over n=%d sessions on the cached_turns basis in %s). "+
			"Recompute the distribution over THAT corpus (not the gitignored %s) and either recalibrate the const in "+
			"internal/gateway/gateway.go or update the calibration comment to the new corpus shape.",
			DefaultAssumedSessionTurns, p75, p95, loTol, hiTol, driftBandMarginFrac*100, p90, len(lengths),
			gatewayusageledger.PublishedLedgerRel, gatewayusageledger.DefaultLedgerRel)
	}
}

// TestHeavyResidentFloorTracksCorpusCachedPromptFloor asserts the heavy/thin split of the
// volume-aware horizon still sits at the real per-turn working-set floor it was calibrated
// to. A trace qualifies as "heavy" (keeps a repaying-turn floor) once its peak resident
// window crosses headHorizonHeavyResidentFloor; that number was sized to the per-turn
// cached-prompt p10 (~51k, with resident adding this turn's input on top). We assert it lands
// in [p10, p50] of the tracked corpus's per-turn cached-prompt volume, relaxed by
// driftBandMarginFrac: below p10 a short chat would wrongly qualify as heavy; above p50 a
// genuine working session would be denied the floor.
func TestHeavyResidentFloorTracksCorpusCachedPromptFloor(t *testing.T) {
	path, ok := findCorpus(t)
	if !ok {
		t.Skip("tracked gateway-usage corpus (" + gatewayusageledger.PublishedLedgerRel + ") not present — drift guard is corpus-dependent")
	}
	perTurn := perTurnCachedPrompt(gatewayusageledger.ReadLedgerFile(path))
	if len(perTurn) < minCorpusForDrift {
		t.Skipf("thin corpus: only %d sessions with per-turn cached-prompt volume (<%d) in %s",
			len(perTurn), minCorpusForDrift, gatewayusageledger.PublishedLedgerRel)
	}
	p10, p50 := percentile(perTurn, 10), percentile(perTurn, 50)
	got := float64(headHorizonHeavyResidentFloor)
	inBand, loTol, hiTol := withinBand(got, p10, p50, driftBandMarginFrac)
	t.Logf("population=%s n=%d p10=%.0f p50=%.0f tolerated=[%.0f, %.0f] const=%d",
		filepath.ToSlash(path), len(perTurn), p10, p50, loTol, hiTol, headHorizonHeavyResidentFloor)
	if !inBand {
		t.Fatalf("headHorizonHeavyResidentFloor=%d has drifted outside the calibrated [p10=%.0f, p50=%.0f] per-turn "+
			"cached-prompt band (tolerated [%.0f, %.0f] at a %.0f%% edge margin; n=%d sessions in %s). Recompute the "+
			"distribution over THAT corpus (not the gitignored %s) and either recalibrate the const in "+
			"internal/gateway/gateway.go or update the calibration comment to the new corpus shape.",
			headHorizonHeavyResidentFloor, p10, p50, loTol, hiTol, driftBandMarginFrac*100, len(perTurn),
			gatewayusageledger.PublishedLedgerRel, gatewayusageledger.DefaultLedgerRel)
	}
}

// TestCalibrationDriftCorpusResolvesTheTrackedMirror is the #5406 regression witness, and the
// only test here that can see the bug: in the real repo BOTH corpora exist, so a resolver
// pointed at the wrong one still returns a path and still produces numbers — it just produces
// numbers nobody else can reproduce. Against a synthetic module root the two cases separate.
func TestCalibrationDriftCorpusResolvesTheTrackedMirror(t *testing.T) {
	plant := func(root, rel string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", p, err)
		}
		if err := os.WriteFile(p, []byte("{\"schema\":\""+gatewayusageledger.Schema+"\",\"kind\":\"exit\"}\n"), 0o644); err != nil {
			t.Fatalf("write %s: %v", p, err)
		}
	}
	newRoot := func() string {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module synthetic\n"), 0o644); err != nil {
			t.Fatalf("write go.mod: %v", err)
		}
		return root
	}

	t.Run("live ledger alone does not satisfy the guard", func(t *testing.T) {
		root := newRoot()
		plant(root, gatewayusageledger.DefaultLedgerRel)
		if got, ok := resolveCorpusFrom(root); ok {
			t.Fatalf("resolved %q from a tree carrying ONLY the gitignored live ledger — the guard would judge the "+
				"constants against per-box state that CI and a fresh clone never see (#5406)", got)
		}
	})

	t.Run("tracked mirror alone satisfies the guard", func(t *testing.T) {
		root := newRoot()
		plant(root, gatewayusageledger.PublishedLedgerRel)
		got, ok := resolveCorpusFrom(root)
		if !ok {
			t.Fatal("did not resolve the tracked corpus in a tree that carries it — the guard would skip on every " +
				"clean checkout, i.e. exactly where it is supposed to be able to red")
		}
		if !strings.HasSuffix(filepath.ToSlash(got), gatewayusageledger.PublishedLedgerRel) {
			t.Fatalf("resolved %q, want a path ending in %q", filepath.ToSlash(got), gatewayusageledger.PublishedLedgerRel)
		}
	})

	t.Run("both present resolves the tracked mirror", func(t *testing.T) {
		root := newRoot()
		plant(root, gatewayusageledger.DefaultLedgerRel)
		plant(root, gatewayusageledger.PublishedLedgerRel)
		got, ok := resolveCorpusFrom(root)
		if !ok {
			t.Fatal("did not resolve a corpus in a tree carrying both")
		}
		if !strings.HasSuffix(filepath.ToSlash(got), gatewayusageledger.PublishedLedgerRel) {
			t.Fatalf("resolved %q — a fleet box carries BOTH corpora, and preferring the live one is what made this "+
				"guard's verdict per-box and unshowable (#5406); want a path ending in %q",
				filepath.ToSlash(got), gatewayusageledger.PublishedLedgerRel)
		}
	})

	t.Run("no module root resolves nothing", func(t *testing.T) {
		if got, ok := resolveCorpusFrom(t.TempDir()); ok {
			t.Fatalf("resolved %q outside any module root", got)
		}
	})
}

// TestDriftBandMarginTolerance pins the hysteresis contract of the band check WITHOUT a
// corpus, so both halves of it are witnessable in CI on a clean checkout: a sub-percent edge
// crossing (the #5406 case — 60000 against a p10 of 60385) must NOT fail, and a gross shift
// (a const set to 5 or 500 against the real band) must still fail. A guard that reds on
// normal append wobble teaches people to ignore it; a guard that tolerates a quartile-scale
// move is not a guard.
func TestDriftBandMarginTolerance(t *testing.T) {
	// The real per-turn cached-prompt band and the real session-length band, as measured over
	// the tracked corpus on 2026-07-28.
	const volLo, volHi = 51986.0, 64963.0
	const lenLo, lenHi = 39.0, 73.0
	cases := []struct {
		name     string
		got      float64
		lo, hi   float64
		wantPass bool
	}{
		{"production floor sits inside the volume band", 60000, volLo, volHi, true},
		{"the 0.6% edge crossing that reported #5406 is tolerated", 60000, 60385, 69462, true},
		{"a 0.6% crossing of the upper edge is tolerated too", 65353, volLo, volHi, true},
		{"a 20% shortfall below the volume floor still reds", 41589, volLo, volHi, false},
		{"production turn prior sits inside the length band", 50, lenLo, lenHi, true},
		{"a const of 5 still reds", 5, lenLo, lenHi, false},
		{"a const of 500 still reds", 500, lenLo, lenHi, false},
		{"a whole-quartile downward move still reds", 20, lenLo, lenHi, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ok, loTol, hiTol := withinBand(tc.got, tc.lo, tc.hi, driftBandMarginFrac)
			if ok != tc.wantPass {
				t.Fatalf("withinBand(%.0f, [%.0f, %.0f], %.2f) = %v (tolerated [%.1f, %.1f]), want %v",
					tc.got, tc.lo, tc.hi, driftBandMarginFrac, ok, loTol, hiTol, tc.wantPass)
			}
		})
	}
	// The tolerance must be a real margin, not an unbounded one: a value the margin admits
	// must still be genuinely near the edge.
	if ok, _, _ := withinBand(volLo*0.5, volLo, volHi, driftBandMarginFrac); ok {
		t.Fatal("the edge margin swallowed a 50% shortfall — the band has stopped guarding anything")
	}
}

// TestCalibrationDriftCorpusIsPresentAndThick is the anti-vacuity backstop. The two band tests
// above SKIP on an absent or thin corpus, which is the right posture for them but the wrong
// posture for the repo: a guard that skips forever is indistinguishable, in a CI summary, from
// a guard that passes. Now that the corpus is TRACKED, its absence or thinning is a repo
// defect rather than an environment fact, so this test FAILS on it and the skip can never be
// silent.
func TestCalibrationDriftCorpusIsPresentAndThick(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Skipf("cannot resolve working directory: %v", err)
	}
	root := dir
	for {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(root)
		if parent == root {
			t.Skip("no module root above the test's working directory — cannot judge corpus presence")
		}
		root = parent
	}
	path := filepath.Join(root, filepath.FromSlash(gatewayusageledger.PublishedLedgerRel))
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("tracked calibration corpus %s is missing from the module root %s (%v). The two drift guards in this "+
			"file SKIP without it, so removing it silently disarms them — restore the tracked snapshot, or retire the "+
			"guards deliberately rather than by deletion.", gatewayusageledger.PublishedLedgerRel, root, err)
	}
	rows := gatewayusageledger.ReadLedgerFile(path)
	lengths, perTurn := cachedTurnLengths(rows), perTurnCachedPrompt(rows)
	t.Logf("tracked corpus %s: %d rows, %d cached_turns length samples, %d per-turn volume samples",
		filepath.ToSlash(path), len(rows), len(lengths), len(perTurn))
	if len(lengths) < minCorpusForDrift || len(perTurn) < minCorpusForDrift {
		t.Fatalf("tracked calibration corpus %s has thinned below the %d-sample floor (%d cached_turns length samples, "+
			"%d per-turn volume samples): both drift guards in this file are now SKIPPING, which reads as green. Refresh "+
			"the publication snapshot or lower the floor deliberately.",
			gatewayusageledger.PublishedLedgerRel, minCorpusForDrift, len(lengths), len(perTurn))
	}
}
