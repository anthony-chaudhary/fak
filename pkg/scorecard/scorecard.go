// Package scorecard is the IMPORTABLE shared kernel behind fak's scorecard family.
//
// The tree carries ~22 tools/*_scorecard.py scorecards, each a standalone module that
// copy-pastes the same skeleton: an A-F grade table, a per-KPI {score, defects, soft}
// shape, a fold into a "control-pane" JSON envelope, a Jekyll-front-matter markdown
// renderer, and a --json/--markdown/--compare CLI. There is no shared library on the
// Python side -- only a duck-typed JSON contract that tools/scorecard_control_pane.py
// reads via find_int(corpus.<name>_debt) / find_grade(corpus.grade).
//
// The first Go ports (internal/guardrsi, internal/dogfoodscore) re-copied that skeleton
// and have already DRIFTED on the one thing that must not drift -- the grade table
// (guardrsi 90/80/70/60, vcachescore 90/75/60/40, the Python conflation card 95/85/75/60).
// This package builds the skeleton ONCE so the next scorecards port onto it instead of
// re-deriving the fold/grade/markdown machinery, and the grade tables live in exactly one
// place (grade.go) as named functions a card selects.
//
// The kernel now emits corpus.value (a continuous quality ratio, 1.0 == the old
// 100/100) as the primary numeric signal. The legacy corpus.score remains during the
// migration so Python-era consumers can still read the old control-pane contract.
//
// Import this package under pkg/ (like pkg/abi) so an out-of-tree tool can build a
// scorecard against the same fold; the per-card KPI logic stays in internal/<name>score.
package scorecard

// KPI is one scored dimension. Score is the legacy 0-100 input because the Python
// family still emits that shape; Value is the continuous quality ratio derived from it
// by Fold (1.0 == the old 100/100). New callers should reason about Value and use
// Score only when bridging old payloads.
//
// Defects are the HARD debt of this KPI -- each entry is one concrete, re-derivable thing
// to fix, and debt is the count of them across all KPIs. Soft entries are advisory nudges
// that NEVER count as debt and never gate ok (the deliberate anti-gaming rule: the cheap
// way to move a soft signal is prose spam, so a soft signal must not be able to red a gate).
type KPI struct {
	Key     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   float64  `json:"score"`
	Value   float64  `json:"value"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// Payload is the control-pane envelope every scorecard emits. The shape is the Python
// run() return verbatim (conflation_scorecard.py:245-258) so the control-pane fold and any
// downstream consumer read a Go card and a Python card identically.
//
// Corpus is map[string]any so a card adds its own <name>_debt plus bespoke keys while the
// kernel writes value/grade/the debt count plus a legacy score; keeping it a map (rather
// than a struct) is what lets one fold serve every card without knowing its private corpus
// keys.
type Payload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace,omitempty"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPI          `json:"kpis"`
}

// Messages carries the per-card prose + extra corpus keys + the grade table the fold needs
// but cannot derive. A card supplies these so it never re-implements the fold itself.
type Messages struct {
	// Finding/NextAction are used when there is debt; FindingClean/NextActionClean when debt==0.
	Finding         string
	FindingClean    string
	NextAction      string
	NextActionClean string
	// ExtraCorpus is merged into corpus alongside the kernel-written value/grade/<debtKey>
	// and the legacy score compatibility fields.
	ExtraCorpus map[string]any
	// Grade selects the A-F table (e.g. GradeStd or GradeStrict). Nil defaults to GradeStd.
	Grade func(float64) string
	// Reason, when set, overrides the default (the joined defect list, or "clean").
	Reason string
}

// Fold turns a slice of scored KPIs into the control-pane Payload.
//
//   - composite = the weighted mean of kpi.Score (mean when weights is nil), retained
//     as legacy 0-100 input compatibility.
//   - value = composite / 100, the continuous value new consumers should trend.
//   - debt = Σ len(kpi.Defects) -- the headline integer, written into corpus[debtKey].
//   - grade = msgs.Grade(composite), rounded; ok = debt==0; verdict = OK | ACTION.
//
// weights maps a KPI Group (or Key -- Group is tried first, then Key) to a relative weight;
// a KPI with no entry weighs 1. This covers both the equal-weight cards (weights nil) and
// the GROUP_WEIGHTS cards without forcing a second fold.
func Fold(schema string, kpis []KPI, debtKey string, weights map[string]float64, msgs Messages) Payload {
	composite := weightedMean(kpis, weights)
	value := ValueFromScore(composite)
	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}
	gradeFn := msgs.Grade
	if gradeFn == nil {
		gradeFn = GradeStd
	}
	grade := gradeFn(composite)
	ok := debt == 0

	verdict := "ACTION"
	finding := msgs.Finding
	next := msgs.NextAction
	if ok {
		verdict = "OK"
		finding = msgs.FindingClean
		next = msgs.NextActionClean
	}

	reason := msgs.Reason
	if reason == "" {
		reason = defectReason(kpis)
	}

	corpus := map[string]any{
		"value":              Round3(value),
		"value_unit":         "quality_ratio",
		"score":              Round1(composite),
		"legacy_score":       Round1(composite),
		"legacy_score_scale": 100,
		"grade":              grade,
		debtKey:              debt,
	}
	for k, v := range msgs.ExtraCorpus {
		corpus[k] = v
	}
	if _, explicitValue := msgs.ExtraCorpus["value"]; !explicitValue {
		if score, ok := anyFloat(corpus["score"]); ok {
			StampLegacyScore(corpus, score)
		}
	}

	return Payload{
		Schema:     schema,
		OK:         ok,
		Verdict:    verdict,
		Finding:    finding,
		Reason:     reason,
		NextAction: next,
		Corpus:     corpus,
		KPIs:       normalizeKPIs(kpis),
	}
}

// weightedMean averages kpi.Score by Group/Key weight (mean when weights is nil/empty).
func weightedMean(kpis []KPI, weights map[string]float64) float64 {
	if len(kpis) == 0 {
		return 0
	}
	var sum, total float64
	for _, k := range kpis {
		w := 1.0
		if len(weights) > 0 {
			if wv, ok := weights[k.Group]; ok {
				w = wv
			} else if wv, ok := weights[k.Key]; ok {
				w = wv
			}
		}
		sum += w * k.Score
		total += w
	}
	if total == 0 {
		return 0
	}
	return sum / total
}

// defectReason joins every defect across the KPIs, or "clean" when there are none --
// matching the Python `"; ".join(...) or "clean"` (conflation_scorecard.py:250).
func defectReason(kpis []KPI) string {
	var ds []string
	for _, k := range kpis {
		ds = append(ds, k.Defects...)
	}
	if len(ds) == 0 {
		return "clean"
	}
	return joinSemicolon(ds)
}

// normalizeKPIs guarantees Defects/Soft marshal as [] not null, matching the Python lists.
func normalizeKPIs(kpis []KPI) []KPI {
	out := make([]KPI, len(kpis))
	for i, k := range kpis {
		k.Value = Round3(ValueFromScore(k.Score))
		if k.Defects == nil {
			k.Defects = []string{}
		}
		if k.Soft == nil {
			k.Soft = []string{}
		}
		out[i] = k
	}
	return out
}
