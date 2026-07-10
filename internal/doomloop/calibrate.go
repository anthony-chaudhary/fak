package doomloop

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// calibrate.go is the offline threshold-calibration fold (#4149): it reads the
// accountability ledger the doomloop shell appends (one Decision per worker per
// tick) and folds it into an evidence-backed proposal for the two dials the live
// classifier reads — TripWindows (when a burning-flat streak trips a NUDGE) and
// EscalateWindows (when a persistent doom loop climbs to an operator ESCALATE).
//
// The signal is drawn from what actually happened to real episodes, never from a
// hand-tuned guess:
//
//   - EscalateWindows is proposed from the RECOVERY-STREAK distribution of NUDGED
//     episodes that recovered (the burning-flat streak they reached before the
//     pattern broke). If 90% of nudged workers that recover have done so by streak
//     Q, a worker still burning-flat past Q is very unlikely to recover — escalate
//     it. Proposal: p90(recovery streak) + 1.
//   - TripWindows is proposed from the peak-streak distribution of NON-NUDGED
//     burning-flat stalls that SELF-recovered (a big commit legitimately lands a
//     window or two into a stall). Setting the trip just above where natural
//     recovery is common avoids nudging a worker that would have recovered anyway.
//     Proposal: p90(self-recovery peak) + 1.
//
// Honest limitation, stated plainly: a ledger recorded under the CURRENT
// thresholds censors the evidence for loosening. A nudged episode can only be
// observed to recover at a streak below EscalateWindows (past it, it escalates
// instead), and a natural self-recovery can only be observed below TripWindows
// (past it, a nudge fires). So this fold recommends TIGHTENING (act sooner) or
// holding — it never proposes a value above the current dial. Loosening requires a
// ledger gathered with the dial already relaxed. The proposals name this in their
// evidence so the reader is not misled into thinking "hold" means "well-centered."
//
// Pure and deterministic: it reads only the decisions and the config, carries no
// clock, and fails closed — below a minimum nudge-episode count it returns an
// INSUFFICIENT verdict and emits NO proposal, so a thin ledger can never move a
// dial on noise.

// Decision mirrors one accountability-ledger row the doomloop shell appends
// (cmd/fak/doomloop.go dlDecision). It is defined here so the calibration fold
// parses the ledger without importing the impure shell and stays compile-testable.
// Only the fields the fold reads are typed; unknown fields are ignored.
type Decision struct {
	UnixMillis    int64  `json:"unix_millis"`
	Session       string `json:"session"`
	Verdict       string `json:"verdict"`
	Correction    string `json:"correction"`
	Reason        string `json:"reason"`
	Streak        int    `json:"burning_flat_streak"`
	EffortDelta   int64  `json:"effort_delta"`
	ProgressDelta int64  `json:"progress_delta"`
	Samples       int    `json:"samples"`
}

// ParseDecisions reads a decisions.jsonl stream (one Decision per line) from r.
// Blank/whitespace-only lines are skipped; a malformed line is a hard error (the
// ledger is machine-written, so a bad row is a bug to surface, not silently drop).
func ParseDecisions(r io.Reader) ([]Decision, error) {
	var out []Decision
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var d Decision
		if err := json.Unmarshal([]byte(raw), &d); err != nil {
			return nil, fmt.Errorf("decisions line %d: %w", line, err)
		}
		out = append(out, d)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read decisions: %w", err)
	}
	return out, nil
}

// CalibrateConfig tunes the calibration fold.
type CalibrateConfig struct {
	// Base is the classifier config whose TripWindows/EscalateWindows the proposal
	// is measured against and would change. The zero value folds DefaultConfig.
	Base Config
	// MinNudgeEpisodes is the INSUFFICIENT floor: below this many observed nudge
	// episodes the fold refuses to propose (fail closed). Must be >= 1.
	MinNudgeEpisodes int
}

// DefaultCalibrateConfig is the tuned starting point: calibrate against the
// default classifier dials and require at least 3 nudge episodes to propose.
func DefaultCalibrateConfig() CalibrateConfig {
	return CalibrateConfig{Base: DefaultConfig(), MinNudgeEpisodes: 3}
}

func (c CalibrateConfig) withDefaults() CalibrateConfig {
	c.Base = c.Base.withDefaults()
	if c.MinNudgeEpisodes < 1 {
		c.MinNudgeEpisodes = DefaultCalibrateConfig().MinNudgeEpisodes
	}
	return c
}

// CalibrationVerdict is the closed outcome band of a calibration run.
type CalibrationVerdict string

const (
	// CalibrationOK: enough nudge episodes were observed; the proposals are backed.
	CalibrationOK CalibrationVerdict = "OK"
	// CalibrationInsufficient: below the nudge-episode floor; NO proposal is emitted.
	CalibrationInsufficient CalibrationVerdict = "INSUFFICIENT"
)

// Distribution is a nearest-rank summary of an integer sample (windows/streaks).
type Distribution struct {
	Count int `json:"count"`
	Min   int `json:"min"`
	P50   int `json:"p50"`
	P90   int `json:"p90"`
	Max   int `json:"max"`
}

// ThresholdProposal is one dial's evidence-backed recommendation.
type ThresholdProposal struct {
	Name     string `json:"name"`     // "TripWindows" | "EscalateWindows"
	Current  int    `json:"current"`  // the dial's current value (from Base)
	Proposed int    `json:"proposed"` // the recommended value
	Delta    int    `json:"delta"`    // proposed - current; its magnitude ranks worst-first
	Evidence string `json:"evidence"` // the one-line justification (with the honest caveat)
}

// CalibrationReport is the folded result. Proposals are ordered worst-first (the
// dial furthest from its evidence-backed value leads).
type CalibrationReport struct {
	Schema          string              `json:"schema"`
	Verdict         CalibrationVerdict  `json:"verdict"`
	Reason          string              `json:"reason"`
	Workers         int                 `json:"workers"`
	NudgeEpisodes   int                 `json:"nudge_episodes"`
	Recovered       int                 `json:"recovered"`
	Escalated       int                 `json:"escalated"`
	Ongoing         int                 `json:"ongoing"`
	SelfRecovered   int                 `json:"self_recovered"`
	RecoveryRate    float64             `json:"recovery_rate"`
	RecoveryStreak  Distribution        `json:"recovery_streak"`
	SelfRecoverPeak Distribution        `json:"self_recovery_peak_streak"`
	Proposals       []ThresholdProposal `json:"proposals"`
}

// CalibrationSchema is the report's schema tag.
const CalibrationSchema = "fak.doomloop.calibration.v1"

// episode is one reconstructed burning-flat run for a worker.
type episode struct {
	peakStreak int
	nudged     bool
	escalated  bool
	recovered  bool // the streak reset before the stream ended
}

// Calibrate folds the decision ledger into a threshold-calibration report. Pure
// and deterministic. Below the nudge-episode floor it returns INSUFFICIENT with
// no proposals (fail closed).
func Calibrate(decisions []Decision, cfg CalibrateConfig) CalibrationReport {
	cfg = cfg.withDefaults()
	rep := CalibrationReport{Schema: CalibrationSchema}

	// Group by worker, oldest-first within each worker, sessions in stable order.
	byWorker := map[string][]Decision{}
	var order []string
	for _, d := range decisions {
		if _, ok := byWorker[d.Session]; !ok {
			order = append(order, d.Session)
		}
		byWorker[d.Session] = append(byWorker[d.Session], d)
	}
	sort.Strings(order)
	rep.Workers = len(order)

	var recoveryStreaks, selfRecoverPeaks []int
	for _, sess := range order {
		ds := byWorker[sess]
		sort.SliceStable(ds, func(i, j int) bool { return ds[i].UnixMillis < ds[j].UnixMillis })
		for _, ep := range reconstructEpisodes(ds) {
			switch {
			case ep.nudged && ep.escalated:
				rep.Escalated++
			case ep.nudged && ep.recovered:
				rep.Recovered++
				recoveryStreaks = append(recoveryStreaks, ep.peakStreak)
			case ep.nudged:
				rep.Ongoing++ // nudged but neither escalated nor reset before the stream ended
			case ep.recovered:
				rep.SelfRecovered++
				selfRecoverPeaks = append(selfRecoverPeaks, ep.peakStreak)
			}
		}
	}

	rep.NudgeEpisodes = rep.Recovered + rep.Escalated + rep.Ongoing
	if resolved := rep.Recovered + rep.Escalated; resolved > 0 {
		rep.RecoveryRate = float64(rep.Recovered) / float64(resolved)
	}
	rep.RecoveryStreak = distOf(recoveryStreaks)
	rep.SelfRecoverPeak = distOf(selfRecoverPeaks)

	if rep.NudgeEpisodes < cfg.MinNudgeEpisodes {
		rep.Verdict = CalibrationInsufficient
		rep.Reason = fmt.Sprintf("%d nudge episode(s) < min %d: too little evidence to move a dial (fail closed)", rep.NudgeEpisodes, cfg.MinNudgeEpisodes)
		return rep
	}

	rep.Verdict = CalibrationOK
	rep.Reason = fmt.Sprintf("%d nudge episode(s): %d recovered, %d escalated, %d ongoing (recovery rate %.0f%%)",
		rep.NudgeEpisodes, rep.Recovered, rep.Escalated, rep.Ongoing, rep.RecoveryRate*100)
	rep.Proposals = worstFirst([]ThresholdProposal{
		proposeEscalate(cfg.Base, recoveryStreaks, rep.Escalated),
		proposeTrip(cfg.Base, selfRecoverPeaks),
	})
	return rep
}

// CalibrateReader is the ledger-reading convenience: parse then fold.
func CalibrateReader(r io.Reader, cfg CalibrateConfig) (CalibrationReport, error) {
	ds, err := ParseDecisions(r)
	if err != nil {
		return CalibrationReport{}, err
	}
	return Calibrate(ds, cfg), nil
}

// reconstructEpisodes walks one worker's ordered decisions and emits a closed
// episode per burning-flat run. A run is a maximal block of Streak>0 decisions; a
// Streak==0 decision ends it (recovered=true); the stream ending mid-run leaves it
// unrecovered (ongoing/censored). nudged/escalated are read from the recorded
// correction the shell actually issued, so recovery rate reflects reality.
func reconstructEpisodes(ds []Decision) []episode {
	var out []episode
	var cur *episode
	for _, d := range ds {
		if d.Streak > 0 {
			if cur == nil {
				cur = &episode{}
			}
			if d.Streak > cur.peakStreak {
				cur.peakStreak = d.Streak
			}
			switch strings.ToUpper(strings.TrimSpace(d.Correction)) {
			case string(CorrectNudge):
				cur.nudged = true
			case string(CorrectEscalate):
				cur.nudged = true
				cur.escalated = true
			}
			continue
		}
		// Streak == 0: any run in progress ended here by a reset (recovery).
		if cur != nil {
			cur.recovered = true
			out = append(out, *cur)
			cur = nil
		}
	}
	if cur != nil { // stream ended mid-run: unrecovered (censored)
		out = append(out, *cur)
	}
	return out
}

// proposeEscalate recommends EscalateWindows from the recovery-streak distribution
// of nudged episodes that recovered. See calibrate.go's header for the censoring
// caveat carried in the evidence string.
func proposeEscalate(base Config, recoveryStreaks []int, escalated int) ThresholdProposal {
	p := ThresholdProposal{Name: "EscalateWindows", Current: base.EscalateWindows, Proposed: base.EscalateWindows}
	if len(recoveryStreaks) == 0 {
		if escalated > 0 {
			// Every nudge escalated; none recovered. The soft nudge is not working at
			// this dial — escalate as soon as the loop is confirmed.
			p.Proposed = clampInt(base.TripWindows+1, base.TripWindows+1, base.EscalateWindows)
			p.Delta = p.Proposed - p.Current
			p.Evidence = fmt.Sprintf("0/%d nudge episodes recovered before escalation: the soft nudge is not recovering workers at EscalateWindows=%d — escalate immediately after the trip", escalated, base.EscalateWindows)
			return p
		}
		p.Evidence = "no recovered nudge episodes to measure recovery latency: EscalateWindows held (loosening needs a ledger gathered at a higher dial)"
		return p
	}
	d := distOf(recoveryStreaks)
	proposed := clampInt(d.P90+1, base.TripWindows+1, base.EscalateWindows)
	p.Proposed = proposed
	p.Delta = proposed - p.Current
	caveat := "cannot exceed the current dial (recoveries past it are censored to escalations)"
	p.Evidence = fmt.Sprintf("nudged-recovery streak p50=%d p90=%d over %d episodes: 90%% recover by streak %d, so escalate at p90+1=%d — %s",
		d.P50, d.P90, d.Count, d.P90, d.P90+1, caveat)
	return p
}

// proposeTrip recommends TripWindows from the peak-streak distribution of
// non-nudged burning-flat stalls that self-recovered.
func proposeTrip(base Config, selfRecoverPeaks []int) ThresholdProposal {
	p := ThresholdProposal{Name: "TripWindows", Current: base.TripWindows, Proposed: base.TripWindows}
	if len(selfRecoverPeaks) == 0 {
		p.Evidence = "no non-nudged self-recovery evidence: TripWindows held (a natural recovery past the trip is censored to a nudge)"
		return p
	}
	d := distOf(selfRecoverPeaks)
	proposed := clampInt(d.P90+1, 2, base.TripWindows)
	p.Proposed = proposed
	p.Delta = proposed - p.Current
	caveat := "cannot exceed the current dial (natural recoveries past it are censored to nudges)"
	p.Evidence = fmt.Sprintf("self-recovery peak streak p50=%d p90=%d over %d natural stalls: trip just above the bulk at p90+1=%d — %s",
		d.P50, d.P90, d.Count, d.P90+1, caveat)
	return p
}

// worstFirst orders proposals by the magnitude of their delta (the dial furthest
// from its evidence-backed value leads), tie-broken by name for determinism.
func worstFirst(ps []ThresholdProposal) []ThresholdProposal {
	sort.SliceStable(ps, func(i, j int) bool {
		ai, aj := abs(ps[i].Delta), abs(ps[j].Delta)
		if ai != aj {
			return ai > aj
		}
		return ps[i].Name < ps[j].Name
	})
	return ps
}

func distOf(xs []int) Distribution {
	if len(xs) == 0 {
		return Distribution{}
	}
	s := append([]int(nil), xs...)
	sort.Ints(s)
	return Distribution{
		Count: len(s),
		Min:   s[0],
		P50:   pctl(s, 50),
		P90:   pctl(s, 90),
		Max:   s[len(s)-1],
	}
}

// pctl is the nearest-rank percentile of an ascending-sorted slice (p in 0..100).
func pctl(sorted []int, p int) int {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[n-1]
	}
	rank := int(math.Ceil(float64(p) / 100 * float64(n)))
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return sorted[rank-1]
}

func clampInt(v, lo, hi int) int {
	if lo > hi {
		lo = hi
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
