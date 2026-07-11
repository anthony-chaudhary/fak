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
// Posture: the tests SKIP (never fail) when the corpus is absent or thin — the corpus is a
// living artifact, not a fixture, and a fresh checkout / CI box may not carry it. The bands
// are deliberately WIDE (a drift GUARD, not an exact-match assertion): they catch a gross
// shift (someone setting the const to 5 or 500, or the real distribution moving a whole
// quartile), not the normal wobble of an appended corpus.

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// minCorpusForDrift is the nonzero-sample floor below which the distribution is too thin to
// assert a percentile band on — the tests skip rather than fail (the cachevalueledger
// thin-corpus posture). 100 is comfortably below the ~500 real sessions the corpus carries
// today while still refusing to judge a near-empty file.
const minCorpusForDrift = 100

// findCorpus walks up from the test's working directory to the module root (the dir holding
// go.mod) and returns the path to the gateway-usage ledger, or ok=false if neither the root
// nor the ledger is found. Keeps the test independent of the caller's cwd.
func findCorpus(t *testing.T) (string, bool) {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			p := filepath.Join(dir, filepath.FromSlash(gatewayusageledger.DefaultLedgerRel))
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

// sessionLengths returns the per-session length signal from the exit rows. It PREFERS the
// real ObservedTurns count (the honest served-turn signal recorded once a session runs the
// provenance-stamped build) and falls back to CachedTurns — the proxy the original
// calibration used because Submits is 0 on the guard proxy path — for the historical rows
// that predate ObservedTurns. Only sessions with a positive length count as real sessions
// (a zero-turn exit is a serve process that served nothing, not a length sample).
func sessionLengths(rows []gatewayusageledger.Row) []float64 {
	var out []float64
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		n := r.Counters.ObservedTurns
		if n == 0 {
			n = r.Counters.CachedTurns
		}
		if n > 0 {
			out = append(out, float64(n))
		}
	}
	return out
}

// TestDefaultAssumedSessionTurnsTracksCorpusP90 asserts the assumed session-length prior
// still sits in the upper tail of the REAL session-length distribution it was calibrated to
// (~p90). Below p75 it would over-shed short sessions (early bursts justified by turns that
// never arrive); above p95 it would assume a repaying tail that almost never materializes.
// The [p75, p95] band is the documented span (comment in gateway.go: median 7, p75 33, p90
// 52, p95 70 — 50 sits just under p90).
func TestDefaultAssumedSessionTurnsTracksCorpusP90(t *testing.T) {
	path, ok := findCorpus(t)
	if !ok {
		t.Skip("gateway-usage corpus not present — drift guard is corpus-dependent")
	}
	lengths := sessionLengths(gatewayusageledger.ReadLedgerFile(path))
	if len(lengths) < minCorpusForDrift {
		t.Skipf("thin corpus: only %d nonzero-length sessions (<%d) — not enough to assert a band", len(lengths), minCorpusForDrift)
	}
	p75, p90, p95 := percentile(lengths, 75), percentile(lengths, 90), percentile(lengths, 95)
	got := float64(DefaultAssumedSessionTurns)
	if got < p75 || got > p95 {
		t.Fatalf("DefaultAssumedSessionTurns=%d has drifted outside the calibrated [p75=%.0f, p95=%.0f] session-length band "+
			"(corpus p90=%.0f over n=%d real sessions). Recompute the distribution and either recalibrate the const in "+
			"internal/gateway/gateway.go or update the calibration comment to the new corpus shape.",
			DefaultAssumedSessionTurns, p75, p95, p90, len(lengths))
	}
}

// TestHeavyResidentFloorTracksCorpusCachedPromptFloor asserts the heavy/thin split of the
// volume-aware horizon still sits at the real per-turn working-set floor it was calibrated
// to. A trace qualifies as "heavy" (keeps a repaying-turn floor) once its peak resident
// window crosses headHorizonHeavyResidentFloor; that number was sized to the per-turn
// cached-prompt p10 (~51k, with resident adding this turn's input on top). We assert it lands
// in [p10, p50] of the observed per-turn cached-prompt volume: below p10 a short chat would
// wrongly qualify as heavy; above p50 a genuine working session would be denied the floor.
func TestHeavyResidentFloorTracksCorpusCachedPromptFloor(t *testing.T) {
	path, ok := findCorpus(t)
	if !ok {
		t.Skip("gateway-usage corpus not present — drift guard is corpus-dependent")
	}
	rows := gatewayusageledger.ReadLedgerFile(path)
	var perTurn []float64
	for _, r := range rows {
		if r.Kind != "exit" {
			continue
		}
		c := r.Counters
		if c.CachedTurns > 0 && c.CachedPromptTokens > 0 {
			perTurn = append(perTurn, float64(c.CachedPromptTokens)/float64(c.CachedTurns))
		}
	}
	if len(perTurn) < minCorpusForDrift {
		t.Skipf("thin corpus: only %d sessions with per-turn cached-prompt volume (<%d)", len(perTurn), minCorpusForDrift)
	}
	p10, p50 := percentile(perTurn, 10), percentile(perTurn, 50)
	got := float64(headHorizonHeavyResidentFloor)
	if got < p10 || got > p50 {
		t.Fatalf("headHorizonHeavyResidentFloor=%d has drifted outside the calibrated [p10=%.0f, p50=%.0f] per-turn "+
			"cached-prompt band (n=%d sessions). Recompute the distribution and either recalibrate the const in "+
			"internal/gateway/gateway.go or update the calibration comment to the new corpus shape.",
			headHorizonHeavyResidentFloor, p10, p50, len(perTurn))
	}
}
