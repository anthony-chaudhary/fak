package cvregress

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// Baseline is the PINNED per-session cache-efficiency reference cvregress flags against.
//
// This pinning is the load-bearing distinction from
// cachevaluereport.foldSessionOutliers, whose hit-rate baseline is the window's OWN
// session-median. A self-relative median floats WITH a uniform fleet-wide drift and is
// therefore blind to it: if every session degrades from 80% to 30% together, the median
// degrades too and nothing flags. A pinned floor/ceiling still trips when the whole fleet
// slides in the same direction — which is exactly the slow-rot regression a trust-but-verify
// loop (epic #3569) has to catch. The two checks are complementary, not redundant: the median
// catches a lone outlier inside a healthy fleet; the pin catches the fleet moving as one.
type Baseline struct {
	// HitRatePctFloor: a scored session whose realized hit-rate (100*reused/prompt) is below
	// this pinned floor is flagged. Fleet-wide, not median-relative.
	HitRatePctFloor float64 `json:"hit_rate_pct_floor"`
	// WriteAmpCeiling: a scored session whose write-amplification ((prompt-reused)/reused, the
	// WITNESSED-ledger analog of cache_creation/cache_read) exceeds this pinned ceiling is
	// flagged — the prefix churned enough to re-prefill more than it reused. 0 when a session
	// reused nothing (the hit-rate floor owns the zero-reuse case), mirroring
	// cachevaluereport's write-amp convention.
	WriteAmpCeiling float64 `json:"write_amp_ceiling"`
	// MinPromptTokens: sessions below this prompt-token size are dropped as noise before
	// scoring, so a handful of tiny turns cannot manufacture a phantom regression.
	MinPromptTokens uint64 `json:"min_prompt_tokens"`
}

// DefaultBaseline pins conservative references calibrated to keep a currently-healthy
// docs/nightrun/cache-value.jsonl corpus GREEN (its multi-turn sessions realize 44.9%–80.8%
// hit and write-amp 0.24–1.23) while still tripping on a real fleet-wide slide below the pin.
// A consumer wanting a tighter ratchet passes its own Baseline; nothing here is a schema.
func DefaultBaseline() Baseline {
	return Baseline{
		HitRatePctFloor: 40.0,
		WriteAmpCeiling: 1.5,
		MinPromptTokens: 300,
	}
}

// Session is one MULTI-TURN ledger row scored on its own per-session cache efficiency. Only
// turns >= 2 are scored: a single-turn cold run has no previous turn to reuse from, so folding
// it in would manufacture a false hit-rate regression (the same posture cachevalueledger's own
// ScoreLedger/FoldTrendGate take).
type Session struct {
	Date         string  `json:"date"`
	SessionType  string  `json:"session_type,omitempty"`
	Provider     string  `json:"provider,omitempty"`
	Context      string  `json:"context,omitempty"`
	UnixMillis   int64   `json:"unix_millis,omitempty"`
	Turns        uint64  `json:"turns"`
	PromptTokens uint64  `json:"prompt_tokens"`
	ReusedTokens uint64  `json:"reused_tokens"`
	HitRatePct   float64 `json:"hit_rate_pct"` // 100 * reused/prompt
	WriteAmp     float64 `json:"write_amp"`    // (prompt-reused)/reused; 0 when reused==0
	Regressed    bool    `json:"regressed"`
	Reason       string  `json:"reason,omitempty"`
}

// Report is the verdict of the pinned-baseline per-session regression fold. OK is CI-green:
// true for OK and INSUFFICIENT, false ONLY for REGRESSED — a thin corpus is never a failure,
// matching the fall-open posture of the sibling ledger gates.
type Report struct {
	Baseline    Baseline  `json:"baseline"`
	Verdict     string    `json:"verdict"` // OK | REGRESSED | INSUFFICIENT
	OK          bool      `json:"ok"`
	Scored      int       `json:"scored"`  // multi-turn sessions above the min-prompt size
	Skipped     int       `json:"skipped"` // dropped: single-turn or below the min-prompt size
	Regressions []Session `json:"regressions,omitempty"`
	Finding     string    `json:"finding"`
}

// scoreRow computes the per-session hit-rate and write-amp for one multi-turn ledger row.
func scoreRow(r cachevalueledger.Row) Session {
	s := Session{
		Date:         r.Date,
		SessionType:  r.SessionType,
		Provider:     r.Provider,
		Context:      r.Context,
		UnixMillis:   r.UnixMillis,
		Turns:        r.Turns,
		PromptTokens: r.PromptTokens,
		ReusedTokens: r.ReusedTokens,
	}
	if r.PromptTokens > 0 {
		s.HitRatePct = 100 * float64(r.ReusedTokens) / float64(r.PromptTokens)
	}
	// created = freshly-prefilled tokens (the WITNESSED analog of cache_creation); reused is a
	// subset of prompt so the subtraction never underflows, but guard defensively anyway.
	if r.ReusedTokens > 0 && r.PromptTokens > r.ReusedTokens {
		s.WriteAmp = float64(r.PromptTokens-r.ReusedTokens) / float64(r.ReusedTokens)
	}
	return s
}

// Fold runs the pinned-baseline per-session regression fold over a slice of cache-value ledger
// rows. It is PURE and deterministic — no clock, no I/O. Single-turn rows and rows below the
// baseline's MinPromptTokens are dropped; every remaining session is scored and flagged against
// the PINNED floor/ceiling. Regressions are returned worst-first (lowest hit-rate, then highest
// write-amp, then oldest). An all-dropped corpus falls open INSUFFICIENT (OK stays true).
func Fold(rows []cachevalueledger.Row, base Baseline) Report {
	rep := Report{Baseline: base, Verdict: "INSUFFICIENT", OK: true}
	for _, r := range rows {
		if r.Turns < 2 {
			rep.Skipped++
			continue
		}
		if r.PromptTokens < base.MinPromptTokens {
			rep.Skipped++
			continue
		}
		rep.Scored++
		s := scoreRow(r)
		var reasons []string
		if s.HitRatePct < base.HitRatePctFloor {
			reasons = append(reasons, fmt.Sprintf("hit-rate %.1f%% below pinned floor %.1f%%",
				s.HitRatePct, base.HitRatePctFloor))
		}
		if s.WriteAmp > base.WriteAmpCeiling {
			reasons = append(reasons, fmt.Sprintf("write-amp %.2f exceeds pinned ceiling %.2f (prefix re-written faster than reused)",
				s.WriteAmp, base.WriteAmpCeiling))
		}
		if len(reasons) > 0 {
			s.Regressed = true
			s.Reason = strings.Join(reasons, "; ")
			rep.Regressions = append(rep.Regressions, s)
		}
	}
	sort.SliceStable(rep.Regressions, func(i, j int) bool {
		a, b := rep.Regressions[i], rep.Regressions[j]
		if a.HitRatePct != b.HitRatePct {
			return a.HitRatePct < b.HitRatePct
		}
		if a.WriteAmp != b.WriteAmp {
			return a.WriteAmp > b.WriteAmp
		}
		if a.UnixMillis != b.UnixMillis {
			return a.UnixMillis < b.UnixMillis
		}
		return a.Date < b.Date
	})

	switch {
	case rep.Scored == 0:
		rep.Finding = fmt.Sprintf("INSUFFICIENT — no multi-turn session above %d prompt tokens to score (%d row(s) dropped)",
			base.MinPromptTokens, rep.Skipped)
	case len(rep.Regressions) > 0:
		rep.Verdict = "REGRESSED"
		rep.OK = false
		rep.Finding = fmt.Sprintf("REGRESSED — %d of %d scored session(s) fell below the pinned baseline (hit>=%.1f%%, write-amp<=%.2f)",
			len(rep.Regressions), rep.Scored, base.HitRatePctFloor, base.WriteAmpCeiling)
	default:
		rep.Verdict = "OK"
		rep.Finding = fmt.Sprintf("OK — all %d scored session(s) held the pinned baseline (hit>=%.1f%%, write-amp<=%.2f)",
			rep.Scored, base.HitRatePctFloor, base.WriteAmpCeiling)
	}
	return rep
}

// ScoreLedgerFile reads the cache-value ledger at path and runs the pinned-baseline fold. It is
// the file-reading convenience over Fold; a missing/unreadable file yields an empty INSUFFICIENT
// report (fall-open), never an error, mirroring cachevalueledger.ReadLedgerFile.
func ScoreLedgerFile(path string, base Baseline) Report {
	return Fold(cachevalueledger.ReadLedgerFile(path), base)
}
