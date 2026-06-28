package model

import (
	"fmt"
	"math"
	"sort"
)

// DefaultSpanRewardCorrelationThreshold is the pre-registered bar used when a
// caller does not provide one. A report at or above this Spearman rho, with
// confound-normalized reward beating raw attention, is CORRELATE; otherwise the
// shadow result is REFUTE.
const DefaultSpanRewardCorrelationThreshold = 0.5

// SpanRewardVerdict is the closed verdict for the shadow reward experiment.
type SpanRewardVerdict string

const (
	SpanRewardCorrelate    SpanRewardVerdict = "CORRELATE"
	SpanRewardRefute       SpanRewardVerdict = "REFUTE"
	SpanRewardInsufficient SpanRewardVerdict = "INSUFFICIENT"
)

// SpanRewardSegment names one token span in a recorded model context.
type SpanRewardSegment struct {
	ID string `json:"id"`
	// From and Len are token offsets inside RecordedSpanRewardSession.ContextIDs.
	From int `json:"from"`
	Len  int `json:"len"`
	// ExpectedAttentionMass is an optional confound baseline. When it is zero, the
	// scorer uses the per-row uniform-visible baseline implied by the attention row.
	ExpectedAttentionMass float64 `json:"expected_attention_mass,omitempty"`
	// Age is the turn lag for the recency discount: reward is multiplied by gamma^Age.
	Age int `json:"age,omitempty"`
}

// RecordedSpanRewardSession is the shadow-only input for issue #866. ContextIDs is
// the resident context whose spans may be evicted; ProbeIDs are the future turn
// tokens that consume that context and whose final logits are measured.
type RecordedSpanRewardSession struct {
	ContextIDs   []int               `json:"context_ids"`
	ProbeIDs     []int               `json:"probe_ids"`
	Spans        []SpanRewardSegment `json:"spans"`
	SuccessGate  float64             `json:"success_gate"`
	RecencyGamma float64             `json:"recency_gamma,omitempty"`
}

// SpanRewardOptions controls how much leave-one-out work the shadow scorer runs.
type SpanRewardOptions struct {
	// TopBottomK runs exact-eviction LOO only for the top K and bottom K spans by
	// normalized reward. <=0 means all spans.
	TopBottomK int `json:"top_bottom_k,omitempty"`
	// CorrelationThreshold is the pre-registered Spearman bar. <=0 uses
	// DefaultSpanRewardCorrelationThreshold.
	CorrelationThreshold float64 `json:"correlation_threshold,omitempty"`
}

// SpanRewardRow is one per-span shadow measurement.
type SpanRewardRow struct {
	ID   string `json:"id"`
	From int    `json:"from"`
	Len  int    `json:"len"`

	RawAttentionMass       float64 `json:"raw_attention_mass"`
	ExpectedAttentionMass  float64 `json:"expected_attention_mass"`
	NormalizedReward       float64 `json:"normalized_reward"`
	SuccessGate            float64 `json:"success_gate"`
	RecencyDiscount        float64 `json:"recency_discount"`
	Rank                   int     `json:"rank"`
	LeaveOneOutChecked     bool    `json:"leave_one_out_checked"`
	LeaveOneOutDelta       float64 `json:"leave_one_out_delta,omitempty"`
	EvictRemoved           int     `json:"evict_removed,omitempty"`
	EvictRepositionMaxDiff float64 `json:"evict_reposition_max_diff,omitempty"`
	// EvictVsRecomputeMaxDiff is diagnostic: zero means the resident-row eviction
	// matches a full prompt recompute with this span omitted for the measured probe.
	// Nonzero does not affect the shadow KV-eviction counterfactual; it records the
	// known boundary that deleting already-consumed earlier context is not a suffix
	// recompute.
	EvictVsRecomputeMaxDiff float64 `json:"evict_vs_recompute_max_diff,omitempty"`
}

// SpanRewardReport is the issue #866 shadow result. It is measurement only: no
// forecast, planner, cache, or eviction decision is changed by producing it.
type SpanRewardReport struct {
	Rows []SpanRewardRow `json:"rows"`

	CheckedRows int `json:"checked_rows"`
	// ExactRecomputeRows counts checked rows whose evicted probe logits also match
	// a full prompt recompute with that span omitted.
	ExactRecomputeRows int `json:"exact_recompute_rows"`

	SpearmanRewardDelta  float64 `json:"spearman_reward_delta"`
	SpearmanDefined      bool    `json:"spearman_defined"`
	SpearmanRawDelta     float64 `json:"spearman_raw_delta"`
	SpearmanRawDefined   bool    `json:"spearman_raw_defined"`
	CorrelationThreshold float64 `json:"correlation_threshold"`

	ConfoundNormalizationImproved bool              `json:"confound_normalization_improved"`
	Verdict                       SpanRewardVerdict `json:"verdict"`
}

// ShadowSpanRewardLeaveOneOut computes the span-attention reward and validates it
// against an exact KV-eviction leave-one-out counterfactual. It observes attention
// mass on ProbeIDs, computes the success-gated / recency-discounted / confound-
// normalized reward, then evicts the top/bottom-K spans from a cloned context cache
// and measures the actual final-logit delta.
func (m *Model) ShadowSpanRewardLeaveOneOut(rec RecordedSpanRewardSession, opts SpanRewardOptions) (SpanRewardReport, error) {
	if m == nil {
		return SpanRewardReport{}, fmt.Errorf("model: nil model")
	}
	if err := validateSpanRewardSession(m.Cfg, rec); err != nil {
		return SpanRewardReport{}, err
	}
	threshold := opts.CorrelationThreshold
	if threshold <= 0 {
		threshold = DefaultSpanRewardCorrelationThreshold
	}
	if threshold > 1 || math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		return SpanRewardReport{}, fmt.Errorf("model: correlation threshold %.4f outside (0,1]", threshold)
	}
	gamma := rec.RecencyGamma
	if gamma == 0 {
		gamma = 1
	}
	gate := clamp01(rec.SuccessGate)

	raw, baseline, err := m.observeSpanRewardMass(rec)
	if err != nil {
		return SpanRewardReport{}, err
	}

	rows := make([]SpanRewardRow, len(rec.Spans))
	for i, sp := range rec.Spans {
		exp := sp.ExpectedAttentionMass
		if exp == 0 {
			exp = baseline[sp.ID]
		}
		discount := math.Pow(gamma, float64(sp.Age))
		if sp.Age < 0 || discount < 0 || math.IsNaN(discount) || math.IsInf(discount, 0) {
			discount = 0
		}
		reward := (raw[sp.ID] - exp) * gate * discount
		if reward < 0 {
			reward = 0
		}
		rows[i] = SpanRewardRow{
			ID:                    sp.ID,
			From:                  sp.From,
			Len:                   sp.Len,
			RawAttentionMass:      raw[sp.ID],
			ExpectedAttentionMass: exp,
			NormalizedReward:      reward,
			SuccessGate:           gate,
			RecencyDiscount:       discount,
		}
	}
	assignSpanRewardRanks(rows)

	context := m.NewSession()
	context.PrefillNoLogits(rec.ContextIDs)
	base := m.SessionFromPrefix(context.Cache)
	baseLogits := base.Prefill(rec.ProbeIDs)

	for _, idx := range selectSpanRewardChecks(rows, opts.TopBottomK) {
		r := &rows[idx]
		loo := m.SessionFromPrefix(context.Cache)
		removed, err := loo.Cache.TryEvict(r.From, r.Len)
		if err != nil {
			return SpanRewardReport{}, fmt.Errorf("span %q evict: %w", r.ID, err)
		}
		if removed != r.Len {
			return SpanRewardReport{}, fmt.Errorf("span %q evict removed %d tokens, want %d", r.ID, removed, r.Len)
		}
		evictedLogits := loo.Prefill(rec.ProbeIDs)
		never := m.NewSession()
		never.PrefillNoLogits(removeTokenRange(rec.ContextIDs, r.From, r.Len))
		neverLogits := never.Prefill(rec.ProbeIDs)

		r.LeaveOneOutChecked = true
		r.EvictRemoved = removed
		r.EvictRepositionMaxDiff = loo.Cache.MaxRepositionResidual()
		r.LeaveOneOutDelta = maxAbsDiffF32(baseLogits, evictedLogits)
		r.EvictVsRecomputeMaxDiff = maxAbsDiffF32(evictedLogits, neverLogits)
	}

	return summarizeSpanRewardReport(rows, threshold), nil
}

func validateSpanRewardSession(cfg Config, rec RecordedSpanRewardSession) error {
	if len(rec.ProbeIDs) == 0 {
		return fmt.Errorf("model: span reward session needs at least one probe token")
	}
	if len(rec.Spans) == 0 {
		return fmt.Errorf("model: span reward session has no spans")
	}
	if rec.SuccessGate < 0 || rec.SuccessGate > 1 || math.IsNaN(rec.SuccessGate) || math.IsInf(rec.SuccessGate, 0) {
		return fmt.Errorf("model: success gate %.4f outside [0,1]", rec.SuccessGate)
	}
	if rec.RecencyGamma < 0 || rec.RecencyGamma > 1 || math.IsNaN(rec.RecencyGamma) || math.IsInf(rec.RecencyGamma, 0) {
		return fmt.Errorf("model: recency gamma %.4f outside [0,1]", rec.RecencyGamma)
	}
	for _, id := range rec.ContextIDs {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("model: context token id %d outside vocab [0,%d)", id, cfg.VocabSize)
		}
	}
	for _, id := range rec.ProbeIDs {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("model: probe token id %d outside vocab [0,%d)", id, cfg.VocabSize)
		}
	}
	seen := make(map[string]bool, len(rec.Spans))
	intervals := make([]SpanRewardSegment, len(rec.Spans))
	copy(intervals, rec.Spans)
	for _, sp := range intervals {
		if sp.ID == "" {
			return fmt.Errorf("model: span has empty id")
		}
		if seen[sp.ID] {
			return fmt.Errorf("model: duplicate span id %q", sp.ID)
		}
		seen[sp.ID] = true
		if sp.From < 0 || sp.Len <= 0 || sp.From+sp.Len > len(rec.ContextIDs) {
			return fmt.Errorf("model: span %q range [%d,%d) outside context length %d",
				sp.ID, sp.From, sp.From+sp.Len, len(rec.ContextIDs))
		}
		if sp.Age < 0 {
			return fmt.Errorf("model: span %q has negative age %d", sp.ID, sp.Age)
		}
		if sp.ExpectedAttentionMass < 0 || sp.ExpectedAttentionMass > 1 {
			return fmt.Errorf("model: span %q expected attention mass %.4f outside [0,1]", sp.ID, sp.ExpectedAttentionMass)
		}
	}
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].From != intervals[j].From {
			return intervals[i].From < intervals[j].From
		}
		return intervals[i].ID < intervals[j].ID
	})
	for i := 1; i < len(intervals); i++ {
		prev := intervals[i-1]
		cur := intervals[i]
		if prev.From+prev.Len > cur.From {
			return fmt.Errorf("model: spans %q and %q overlap", prev.ID, cur.ID)
		}
	}
	return nil
}

func (m *Model) observeSpanRewardMass(rec RecordedSpanRewardSession) (map[string]float64, map[string]float64, error) {
	raw := make(map[string]float64, len(rec.Spans))
	baseline := make(map[string]float64, len(rec.Spans))
	ids := append(append([]int(nil), rec.ContextIDs...), rec.ProbeIDs...)
	queryStart := len(rec.ContextIDs)
	rows := 0

	prev := m.attnObs
	m.SetAttnObserver(func(layer, queryPos, head int, keyPositions []int, weights []float32) {
		if queryPos < queryStart {
			return
		}
		rows++
		for _, sp := range rec.Spans {
			var mass float64
			visible := 0
			end := sp.From + sp.Len
			for i, kp := range keyPositions {
				if kp >= sp.From && kp < end {
					mass += float64(weights[i])
					visible++
				}
			}
			raw[sp.ID] += mass
			if len(keyPositions) > 0 {
				baseline[sp.ID] += float64(visible) / float64(len(keyPositions))
			}
		}
	})
	defer func() { m.attnObs = prev }()

	m.Forward(ids)
	if rows == 0 {
		return nil, nil, fmt.Errorf("model: attention observer saw no probe rows")
	}
	for _, sp := range rec.Spans {
		raw[sp.ID] /= float64(rows)
		baseline[sp.ID] /= float64(rows)
	}
	return raw, baseline, nil
}

func summarizeSpanRewardReport(rows []SpanRewardRow, threshold float64) SpanRewardReport {
	report := SpanRewardReport{
		Rows:                 append([]SpanRewardRow(nil), rows...),
		CorrelationThreshold: threshold,
		Verdict:              SpanRewardInsufficient,
	}
	var rewards, raw, deltas []float64
	for _, r := range rows {
		if !r.LeaveOneOutChecked {
			continue
		}
		report.CheckedRows++
		if r.EvictVsRecomputeMaxDiff == 0 {
			report.ExactRecomputeRows++
		}
		rewards = append(rewards, r.NormalizedReward)
		raw = append(raw, r.RawAttentionMass)
		deltas = append(deltas, r.LeaveOneOutDelta)
	}
	report.SpearmanRewardDelta, report.SpearmanDefined = spearman(rewards, deltas)
	report.SpearmanRawDelta, report.SpearmanRawDefined = spearman(raw, deltas)
	report.ConfoundNormalizationImproved = report.SpearmanDefined &&
		(!report.SpearmanRawDefined || report.SpearmanRewardDelta > report.SpearmanRawDelta)
	if report.CheckedRows >= 2 && report.SpearmanDefined {
		if report.SpearmanRewardDelta >= threshold && report.ConfoundNormalizationImproved {
			report.Verdict = SpanRewardCorrelate
		} else {
			report.Verdict = SpanRewardRefute
		}
	}
	return report
}

// spanRewardOrder returns row indices [0,len(rows)) sorted by DESCENDING NormalizedReward,
// ties broken by ASCENDING ID — the shared ranking order of the shadow span-reward rows.
func spanRewardOrder(rows []SpanRewardRow) []int {
	order := iotaInts(len(rows))
	sort.Slice(order, func(i, j int) bool {
		a, b := rows[order[i]], rows[order[j]]
		if a.NormalizedReward != b.NormalizedReward {
			return a.NormalizedReward > b.NormalizedReward
		}
		return a.ID < b.ID
	})
	return order
}

func selectSpanRewardChecks(rows []SpanRewardRow, k int) []int {
	n := len(rows)
	if k <= 0 || 2*k >= n {
		return iotaInts(n)
	}
	order := spanRewardOrder(rows)
	seen := make(map[int]bool, 2*k)
	var out []int
	for _, idx := range append(order[:k:k], order[n-k:]...) {
		if !seen[idx] {
			seen[idx] = true
			out = append(out, idx)
		}
	}
	sort.Ints(out)
	return out
}

func assignSpanRewardRanks(rows []SpanRewardRow) {
	order := spanRewardOrder(rows)
	for i, idx := range order {
		rows[idx].Rank = i + 1
	}
}

func spearman(x, y []float64) (float64, bool) {
	if len(x) != len(y) || len(x) < 2 {
		return 0, false
	}
	rx, okx := averageRanksDesc(x)
	ry, oky := averageRanksDesc(y)
	if !okx || !oky {
		return 0, false
	}
	r, ok := pearson(rx, ry)
	return r, ok
}

func averageRanksDesc(v []float64) ([]float64, bool) {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, false
		}
	}
	idx := iotaInts(len(v))
	sort.Slice(idx, func(i, j int) bool {
		if v[idx[i]] != v[idx[j]] {
			return v[idx[i]] > v[idx[j]]
		}
		return idx[i] < idx[j]
	})
	ranks := make([]float64, len(v))
	for i := 0; i < len(idx); {
		j := i + 1
		for j < len(idx) && v[idx[j]] == v[idx[i]] {
			j++
		}
		avg := (float64(i+1) + float64(j)) / 2
		for k := i; k < j; k++ {
			ranks[idx[k]] = avg
		}
		i = j
	}
	return ranks, true
}

func pearson(x, y []float64) (float64, bool) {
	var mx, my float64
	for i := range x {
		mx += x[i]
		my += y[i]
	}
	mx /= float64(len(x))
	my /= float64(len(y))
	var num, dx, dy float64
	for i := range x {
		xv := x[i] - mx
		yv := y[i] - my
		num += xv * yv
		dx += xv * xv
		dy += yv * yv
	}
	if dx == 0 || dy == 0 {
		return 0, false
	}
	return num / math.Sqrt(dx*dy), true
}

func maxAbsDiffF32(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var mx float64
	for i := 0; i < n; i++ {
		if d := math.Abs(float64(a[i] - b[i])); d > mx {
			mx = d
		}
	}
	return mx
}

func removeTokenRange(ids []int, from, n int) []int {
	out := make([]int, 0, len(ids)-n)
	out = append(out, ids[:from]...)
	out = append(out, ids[from+n:]...)
	return out
}

func clamp01(v float64) float64 {
	if v < 0 || math.IsNaN(v) {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
