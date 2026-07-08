package cachevaluereport

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// ShapeSchema versions the session-shape roll-up envelope so a downstream reader
// can pin it independently of the by-week trend Schema.
const ShapeSchema = "fak-cache-value-shapes/1"

// The session-shape fold answers a DIFFERENT question than Fold. Fold buckets by
// week × session_type — a TREND over time and an attribution to the front door a
// session came through. This fold buckets by the SHAPE of each session — how long
// it ran (length band) crossed with how much KV-prefix reuse it actually realized
// (outcome band). It asks "what KINDS of sessions do we run, and which shapes earn
// reuse?" rather than "is reuse trending up this week?".
//
// The two axes are deliberately orthogonal to the week/type view so a reader can
// see, e.g., that long warm sessions are rare but carry most of the realized reuse
// while single-turn cold runs dominate the row count — a fact the time trend hides.
//
// #1066 honesty fence: outcome bands are cut on the WITNESSED realized reuse ratio
// (reused/prompt), never the forbidden vs-naive 1/(1-reuse) re-prefill multiple.
// The self-labels (PublishableValueFamily, VsNaiveMultipleExcluded) ride along on
// the report exactly as Fold's do.

// Length-band thresholds (in turns). A single-turn cold run has no previous turn to
// reuse from, so it is its own band; short vs long splits the multi-turn population
// at MinLongTurns so the "does a long trajectory earn its warm KV?" question has a
// clean population to read.
const (
	// MinShortTurns is the smallest turn count that counts as multi-turn (>= this is
	// not "single"). It mirrors the turns >= 2 multi-turn floor used everywhere else.
	MinShortTurns = 2
	// MinLongTurns is the turn count at or above which a multi-turn session is "long".
	// Below it (2..MinLongTurns-1) a multi-turn session is "short".
	MinLongTurns = 5
)

// Outcome-band thresholds on the realized reuse ratio (reused/prompt). A multi-turn
// session below coldOutcomeMax realized effectively no reuse (cold); at or above
// warmOutcomeMin it ran warm; in between it is partial. Single-turn sessions have no
// prior turn to reuse from and are recorded under the "n/a" outcome, never cold —
// folding a structurally-impossible reuse into "cold" would slander the shape.
const (
	coldOutcomeMax = 0.10
	warmOutcomeMin = 0.50
)

// LengthBand is a session's length classification.
type LengthBand string

const (
	LengthSingle LengthBand = "single" // exactly 1 turn
	LengthShort  LengthBand = "short"  // 2..MinLongTurns-1 turns
	LengthLong   LengthBand = "long"   // >= MinLongTurns turns
)

// OutcomeBand is a session's realized-reuse classification.
type OutcomeBand string

const (
	OutcomeNA      OutcomeBand = "n/a"     // single-turn: no prior turn to reuse from
	OutcomeCold    OutcomeBand = "cold"    // multi-turn, realized reuse < coldOutcomeMax
	OutcomePartial OutcomeBand = "partial" // multi-turn, coldOutcomeMax..warmOutcomeMin
	OutcomeWarm    OutcomeBand = "warm"    // multi-turn, realized reuse >= warmOutcomeMin
)

// lengthBand classifies a row by its turn count.
func lengthBand(turns uint64) LengthBand {
	switch {
	case turns >= MinLongTurns:
		return LengthLong
	case turns >= MinShortTurns:
		return LengthShort
	default:
		return LengthSingle
	}
}

// outcomeBand classifies a row by its realized reuse. Single-turn rows are n/a; a
// multi-turn row is cut on its own recorded reuse_ratio.
func outcomeBand(turns uint64, reuseRatio float64) OutcomeBand {
	if turns < MinShortTurns {
		return OutcomeNA
	}
	switch {
	case reuseRatio >= warmOutcomeMin:
		return OutcomeWarm
	case reuseRatio >= coldOutcomeMax:
		return OutcomePartial
	default:
		return OutcomeCold
	}
}

// ShapeHealth is a per-cluster reading of whether a shape is EARNING its reuse or is a
// failure mode. It is a pure function of the (length × outcome) bands — the whole point
// of clustering by shape is to surface the expensive failure class, a long session that
// ran turn after turn and never warmed, which the neutral cluster table buries.
type ShapeHealth string

const (
	// HealthEarning: the shape is fine — either warm (any length) or the structurally
	// reuse-free single/n-a cluster, which is not a failure, just a cold single-turn run.
	HealthEarning ShapeHealth = "earning"
	// HealthWeak: short multi-turn sessions that ran cold/partial — cheap and low-stakes;
	// a 2–4 turn session earning little reuse wastes little.
	HealthWeak ShapeHealth = "weak"
	// HealthUnderwarmed: long sessions stuck at partial reuse — a near-miss worth a look.
	HealthUnderwarmed ShapeHealth = "underwarmed"
	// HealthWasteful: long sessions that ran cold — the expensive failure mode. They paid
	// the full prompt-token cost every turn and realized effectively no KV-prefix reuse.
	HealthWasteful ShapeHealth = "wasteful"
)

// classifyHealth maps a (length × outcome) pair to its health reading. Warm is always
// earning; the single/n-a cluster is earning (structurally reuse-free, not a failure);
// the cold/partial classes split on length so the expensive long failure stands out.
func classifyHealth(l LengthBand, o OutcomeBand) ShapeHealth {
	if o == OutcomeWarm || o == OutcomeNA {
		return HealthEarning
	}
	// o is cold or partial here.
	if l == LengthLong {
		if o == OutcomeCold {
			return HealthWasteful
		}
		return HealthUnderwarmed
	}
	// short cold/partial (single-turn never reaches here — it is always n/a).
	return HealthWeak
}

// ShapeCluster is one (length × outcome) cell of the shape roll-up. RealizedReuseRatio
// is the cluster's aggregate reused/prompt over its rows (0 for the single/n/a cluster,
// where reuse is structurally impossible). ShareOfSessions and ShareOfReusedTokens let a
// reader see the "few long warm sessions carry most of the reuse" story at a glance.
type ShapeCluster struct {
	Length  LengthBand  `json:"length"`
	Outcome OutcomeBand `json:"outcome"`
	Health  ShapeHealth `json:"health"`

	Sessions uint64 `json:"sessions"`
	Turns    uint64 `json:"turns"`

	PromptTokens uint64 `json:"prompt_tokens"`
	ReusedTokens uint64 `json:"reused_tokens"`

	RealizedReuseRatio  float64 `json:"realized_reuse_ratio"`
	ShareOfSessions     float64 `json:"share_of_sessions"`      // this cluster's sessions / all sessions
	ShareOfReusedTokens float64 `json:"share_of_reused_tokens"` // this cluster's reused tokens / all reused tokens

	// BySessionType attributes the cluster's sessions back to the front door they
	// came through, so a shape ("long warm") can still be traced to guard/serve/run.
	BySessionType map[string]uint64 `json:"by_session_type"`
}

// ShapeReport is the rolled-up session-shape envelope. Clusters are ordered by a
// stable (length, outcome) key so the render and JSON are deterministic.
type ShapeReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"`

	TotalSessions      uint64 `json:"total_sessions"`
	MultiTurnSessions  uint64 `json:"multi_turn_sessions"`
	SingleTurnSessions uint64 `json:"single_turn_sessions"`
	TotalTurns         uint64 `json:"total_turns"`
	TotalReusedTokens  uint64 `json:"total_reused_tokens"`

	// WastefulSessions counts sessions in HealthWasteful clusters (long × cold) — the
	// expensive failure mode. WastefulSessionShare is that count over all sessions.
	WastefulSessions     uint64  `json:"wasteful_sessions"`
	WastefulSessionShare float64 `json:"wasteful_session_share"`

	Clusters []ShapeCluster `json:"clusters"`

	// #1066 fence self-labels — a downstream reader can never mistake the realized
	// reuse for the forbidden vs-naive multiple.
	PublishableValueFamily  string `json:"publishable_value_family"`
	VsNaiveMultipleExcluded bool   `json:"vs_naive_multiple_excluded"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

// shapeKey orders clusters length-major (single, short, long) then outcome-major
// (n/a, cold, partial, warm), so the table reads from "no reuse possible" to
// "warm" within each length band.
func shapeKey(l LengthBand, o OutcomeBand) string {
	lengthRank := map[LengthBand]int{LengthSingle: 0, LengthShort: 1, LengthLong: 2}
	outcomeRank := map[OutcomeBand]int{OutcomeNA: 0, OutcomeCold: 1, OutcomePartial: 2, OutcomeWarm: 3}
	return fmt.Sprintf("%d\x00%d", lengthRank[l], outcomeRank[o])
}

// FoldShapes rolls a slice of ledger rows up into a session-shape ShapeReport. It is
// PURE and deterministic: the only time input is `now`, used solely to stamp
// GeneratedAt. Rows with zero turns (no session activity) are skipped, the same way
// Fold and ScoreLedger skip them.
func FoldShapes(rows []cachevalueledger.Row, now time.Time) ShapeReport {
	r := ShapeReport{
		Schema:                  ShapeSchema,
		GeneratedAt:             now.UTC().Format(time.RFC3339),
		PublishableValueFamily:  PublishableValueFamily,
		VsNaiveMultipleExcluded: true,
		Verdict:                 "INSUFFICIENT",
		OK:                      true,
	}

	type agg struct {
		c ShapeCluster
	}
	byShape := map[string]*agg{}

	for _, row := range rows {
		if row.Turns == 0 {
			continue
		}
		l := lengthBand(row.Turns)
		o := outcomeBand(row.Turns, row.ReuseRatio)
		key := shapeKey(l, o)
		a := byShape[key]
		if a == nil {
			a = &agg{c: ShapeCluster{Length: l, Outcome: o, BySessionType: map[string]uint64{}}}
			byShape[key] = a
		}
		c := &a.c
		c.Sessions++
		c.Turns += row.Turns
		c.PromptTokens += row.PromptTokens
		c.ReusedTokens += row.ReusedTokens
		st := row.SessionType
		if st == "" {
			st = "unknown"
		}
		c.BySessionType[st]++

		r.TotalSessions++
		r.TotalTurns += row.Turns
		r.TotalReusedTokens += row.ReusedTokens
		if row.Turns >= MinShortTurns {
			r.MultiTurnSessions++
		} else {
			r.SingleTurnSessions++
		}
	}

	keys := make([]string, 0, len(byShape))
	for k := range byShape {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		c := byShape[k].c
		c.Health = classifyHealth(c.Length, c.Outcome)
		if c.PromptTokens > 0 {
			c.RealizedReuseRatio = float64(c.ReusedTokens) / float64(c.PromptTokens)
		}
		if r.TotalSessions > 0 {
			c.ShareOfSessions = float64(c.Sessions) / float64(r.TotalSessions)
		}
		if r.TotalReusedTokens > 0 {
			c.ShareOfReusedTokens = float64(c.ReusedTokens) / float64(r.TotalReusedTokens)
		}
		if c.Health == HealthWasteful {
			r.WastefulSessions += c.Sessions
		}
		r.Clusters = append(r.Clusters, c)
	}
	if r.TotalSessions > 0 {
		r.WastefulSessionShare = float64(r.WastefulSessions) / float64(r.TotalSessions)
	}

	r.fillShapeVerdict()
	return r
}

// fillShapeVerdict sets the report-contract fields. This is a REPORT, not a gate: OK
// stays true; Verdict is INSUFFICIENT only when no multi-turn session exists to give a
// shape any realized reuse (mirroring Fold's fall-open posture on a thin corpus).
func (r *ShapeReport) fillShapeVerdict() {
	if r.MultiTurnSessions == 0 {
		r.Verdict = "INSUFFICIENT"
		r.Finding = fmt.Sprintf("%d session(s), all single-turn; no multi-turn shape to cluster reuse on yet", r.TotalSessions)
		r.Reason = "realized KV-prefix reuse needs sessions with >= 2 turns; single-turn cold runs have no prior turn to reuse from"
		r.NextAction = "accumulate multi-turn guard/serve sessions into docs/nightrun/cache-value.jsonl, then re-fold"
		return
	}
	warm := r.warmMultiTurnSessions()
	r.Verdict = "MEASURED"
	r.Finding = fmt.Sprintf("%d shape cluster(s) over %d session(s); %d multi-turn, of which %d ran warm (reuse >= %.2f), %d ran wasteful (long × cold)",
		len(r.Clusters), r.TotalSessions, r.MultiTurnSessions, warm, warmOutcomeMin, r.WastefulSessions)
	r.Reason = "WITNESSED in-kernel KV-prefix reuse, clustered by session length × realized-reuse outcome; " + PublishableValueFamily
	if r.WastefulSessions > 0 {
		r.NextAction = fmt.Sprintf("investigate the %d long session(s) that ran cold (never warmed despite >= %d turns) — the expensive failure shape; check for cache-busting prefix churn",
			r.WastefulSessions, MinLongTurns)
	} else {
		r.NextAction = "read which shapes carry share_of_reused_tokens; long warm sessions should dominate reuse despite being rare"
	}
}

// warmMultiTurnSessions counts sessions in warm-outcome clusters.
func (r *ShapeReport) warmMultiTurnSessions() uint64 {
	var n uint64
	for _, c := range r.Clusters {
		if c.Outcome == OutcomeWarm {
			n += c.Sessions
		}
	}
	return n
}

// RenderShapes produces a compact, deterministic terminal table of the shape clusters.
func RenderShapes(r ShapeReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-value session shapes (Track 1, WITNESSED kernel reuse) — %s\n", r.Verdict)
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	fmt.Fprintf(&sb, "  fence: %s\n", PublishableValueFamily)
	if len(r.Clusters) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-7s  %-8s  %-11s  %8s  %7s  %6s  %8s  %8s  %s\n",
		"length", "outcome", "health", "sessions", "turns", "reuse", "sess%", "reuse-tok%", "by session_type")
	for _, c := range r.Clusters {
		fmt.Fprintf(&sb, "  %-7s  %-8s  %-11s  %8d  %7d  %5.1f%%  %6.1f%%  %8.1f%%  %s\n",
			c.Length, c.Outcome, c.Health, c.Sessions, c.Turns, 100*c.RealizedReuseRatio,
			100*c.ShareOfSessions, 100*c.ShareOfReusedTokens, renderShapeSessionTypes(c.BySessionType))
	}
	return sb.String()
}

// renderShapeSessionTypes formats a cluster's session_type attribution deterministically.
func renderShapeSessionTypes(m map[string]uint64) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, m[k]))
	}
	return strings.Join(parts, ", ")
}
