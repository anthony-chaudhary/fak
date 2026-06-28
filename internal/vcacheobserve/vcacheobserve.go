package vcacheobserve

import (
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

// Schema is the versioned report contract emitted by `fak vcache observe --json`.
const Schema = "fak.vcache.observe.v1"

// ttl5mMillis is the 5-minute provider TTL window the governor and warmth-belief
// estimator decay against — the Anthropic-5m default the live loop would re-measure
// (Law D2). It is the only clock constant in the leaf; callers inject per-turn millis.
const ttl5mMillis int64 = 5 * 60 * 1000

// Turn is one real provider call's prompt-cache telemetry, tagged with the prefix
// family (a Claude session id == one shared system prefix) it belongs to and the
// wall-clock millis it happened at. The CLI parses these from real Claude transcripts;
// the leaf stays clock-free and pure.
type Turn struct {
	Family        string `json:"family"`
	UnixMillis    int64  `json:"unix_millis"`
	InputTokens   int64  `json:"input_tokens"`
	CacheCreation int64  `json:"cache_creation_input_tokens"`
	CacheRead     int64  `json:"cache_read_input_tokens"`
	Ephemeral1h   int64  `json:"ephemeral_1h_input_tokens"`
	Ephemeral5m   int64  `json:"ephemeral_5m_input_tokens"`
}

// Multipliers carries the provider economics constants (Anthropic defaults: a 0.1x
// cached read, a 1.25x 5-minute write, a 2.0x 1-hour write).
type Multipliers struct {
	Read    float64 `json:"read"`
	Write5m float64 `json:"write_5m"`
	Write1h float64 `json:"write_1h"`
}

// DefaultMultipliers returns the Anthropic prompt-cache multipliers.
func DefaultMultipliers() Multipliers {
	return Multipliers{Read: 0.1, Write5m: vcachegov.WriteMult5Minutes, Write1h: vcachegov.WriteMult1Hour}
}

func (m Multipliers) normalized() Multipliers {
	if m.Read < 0 {
		m.Read = 0
	}
	if m.Read == 0 {
		m.Read = 0.1
	}
	if m.Write5m <= 0 {
		m.Write5m = vcachegov.WriteMult5Minutes
	}
	if m.Write1h <= 0 {
		m.Write1h = vcachegov.WriteMult1Hour
	}
	return m
}

// Provenance labels whether a panel value is OBSERVED (relayed from the provider's
// own counters — fak does not control it) or a DECISION (fak's deterministic verdict
// over those counters). The label is what keeps the report from conflating a provider
// effect with a fak action.
type Provenance string

const (
	Observed Provenance = "OBSERVED"
	Decision Provenance = "DECISION"
)

// Family is one prefix family's observed economics and warmth-belief prediction.
type Family struct {
	Key                 string                          `json:"key"`
	Turns               int                             `json:"turns"`
	ElapsedSeconds      float64                         `json:"elapsed_seconds"`
	InputTokens         int64                           `json:"input_tokens"`
	CacheReadTokens     int64                           `json:"cache_read_tokens"`
	CacheCreationTokens int64                           `json:"cache_creation_tokens"`
	MeanPrefixTokens    float64                         `json:"mean_prefix_tokens"`
	HitRate             float64                         `json:"hit_rate"`
	ArrivalRatePerSec   float64                         `json:"arrival_rate_per_sec"`
	GovernorDecision    vcachegov.GovernorDecision      `json:"governor_decision"`
	Economics           vcachegov.TelemetrySavingsProof `json:"economics"`
	Prediction          vcachecal.PredictionError       `json:"prediction"`
}

// Panel is one vCache sub-concept's observability slice.
type Panel struct {
	Name       string     `json:"name"`
	Pkg        string     `json:"package"`
	Milestone  string     `json:"milestone"`
	Question   string     `json:"question"`
	Value      string     `json:"value"`
	Provenance Provenance `json:"provenance"`
	Verdict    string     `json:"verdict"`
	Witness    string     `json:"witness"`
	Detail     string     `json:"detail"`
}

// Report is the full per-sub-concept observability over one real run.
type Report struct {
	Schema           string                          `json:"schema"`
	Turns            int                             `json:"turns"`
	FamilyCount      int                             `json:"family_count"`
	Families         []Family                        `json:"families"`
	Aggregate        vcachegov.TelemetrySavingsProof `json:"aggregate"`
	HitRate          float64                         `json:"hit_rate"`
	Multiplier       float64                         `json:"multiplier"`
	MeanPrefixTokens float64                         `json:"mean_prefix_tokens"`
	Concentration    vcachecal.Concentration         `json:"concentration"`
	Prediction       vcachecal.PredictionError       `json:"prediction"`
	Recall           vcachechain.RecallProof         `json:"recall"`
	GradeMeasured    string                          `json:"grade_measured"`
	ScoreMeasured    int                             `json:"score_measured"`
	GradeSynthetic   string                          `json:"grade_synthetic"`
	ScoreSynthetic   int                             `json:"score_synthetic"`
	Panels           []Panel                         `json:"panels"`
}

// Observe groups the run's turns by prefix family and runs every shipped vCache
// decision leaf over the real telemetry, returning one panel per sub-concept. It is
// pure and deterministic: same turns in, same report out.
func Observe(turns []Turn, m Multipliers) Report {
	m = m.normalized()
	rep := Report{Schema: Schema, Turns: len(turns)}
	if len(turns) == 0 {
		rep.Concentration = vcachecal.FitConcentration(nil)
		return rep
	}

	families := groupFamilies(turns)
	rep.FamilyCount = len(families)

	var allRows []vcachegov.TelemetryRow
	var ranked []vcachecal.RankedVBlock
	var residentSum float64
	var residentN int
	policy := vcachecal.BeliefPolicy{ProviderTTLMillis: ttl5mMillis, GraceMillis: 30 * 1000}

	for i := range families {
		fam := &families[i]
		rows := make([]vcachegov.TelemetryRow, 0, fam.Turns)
		famResidentSum := 0.0
		for _, t := range famTurns(turns, fam.Key) {
			rows = append(rows, telemetryRow(t))
			resident := float64(t.InputTokens + t.CacheRead + t.CacheCreation)
			famResidentSum += resident
			residentSum += resident
			residentN++
		}
		fam.Economics = vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows: rows, ReadMult: m.Read, Write5mMult: m.Write5m, Write1hMult: m.Write1h,
		})
		if fam.Economics.BaselineTokenEquiv > 0 {
			fam.HitRate = fam.Economics.CacheReadTokens / fam.Economics.BaselineTokenEquiv
		}
		fam.MeanPrefixTokens = famResidentSum / float64(fam.Turns)
		fam.Prediction = predictFamily(famTurns(turns, fam.Key), policy)
		fam.ArrivalRatePerSec, fam.GovernorDecision = classifyFamily(famTurns(turns, fam.Key), m)

		// Fold aggregates.
		rep.Prediction.Total += fam.Prediction.Total
		rep.Prediction.TrueWarm += fam.Prediction.TrueWarm
		rep.Prediction.FalseWarm += fam.Prediction.FalseWarm
		rep.Prediction.TrueCold += fam.Prediction.TrueCold
		rep.Prediction.FalseCold += fam.Prediction.FalseCold
		allRows = append(allRows, rows...)
		ranked = append(ranked, vcachecal.RankedVBlock{
			Key:          fam.Key,
			Frequency:    float64(fam.Turns),
			Size:         fam.MeanPrefixTokens,
			ReuseDensity: float64(fam.CacheReadTokens) / float64(fam.Turns),
		})
	}
	rep.Families = families

	rep.Aggregate = vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
		Rows: allRows, ReadMult: m.Read, Write5mMult: m.Write5m, Write1hMult: m.Write1h,
	})
	if rep.Aggregate.BaselineTokenEquiv > 0 {
		rep.HitRate = rep.Aggregate.CacheReadTokens / rep.Aggregate.BaselineTokenEquiv
	}
	rep.Multiplier = ratio(rep.Aggregate.BaselineTokenEquiv, rep.Aggregate.ActualTokenEquiv)
	if residentN > 0 {
		rep.MeanPrefixTokens = residentSum / float64(residentN)
	}

	// Concentration MEASURED from the account's family distribution (must be sorted
	// descending by ranking weight before fitting — §5.2).
	sort.SliceStable(ranked, func(i, j int) bool {
		wi, wj := ranked[i].Weight(), ranked[j].Weight()
		if wi == wj {
			return ranked[i].Key < ranked[j].Key
		}
		return wi > wj
	})
	rep.Concentration = vcachecal.FitConcentration(ranked)

	// M4 recall cost gate at the account's real mean prefix size, one unit, no fan-out.
	rep.Recall = vcachechain.ProveRecall(vcachechain.ProveRecallInput{
		PrefixTokens: int64(math.Round(rep.MeanPrefixTokens)),
		UnitTokens:   10,
		ReadMult:     m.Read,
		Siblings:     1,
	})

	// Score the run two ways: with the account's MEASURED concentration + prediction,
	// and with the scorecard's SYNTHETIC defaults — the headline contrast.
	measured := vcachescore.Score(vcachescore.Input{
		TelemetryRows:     allRows,
		TelemetryReadMult: m.Read,
		TelemetryWrite5m:  m.Write5m,
		TelemetryWrite1h:  m.Write1h,
		Ranked:            ranked,
		Prediction:        rep.Prediction,
	})
	synthetic := vcachescore.Score(vcachescore.Input{
		TelemetryRows:     allRows,
		TelemetryReadMult: m.Read,
		TelemetryWrite5m:  m.Write5m,
		TelemetryWrite1h:  m.Write1h,
	})
	rep.GradeMeasured, rep.ScoreMeasured = measured.Grade, measured.Score
	rep.GradeSynthetic, rep.ScoreSynthetic = synthetic.Grade, synthetic.Score

	rep.Panels = buildPanels(rep)
	return rep
}

// groupFamilies buckets turns by family preserving first-seen order and records the
// per-family totals and elapsed span.
func groupFamilies(turns []Turn) []Family {
	idx := map[string]int{}
	out := []Family{}
	minT := map[string]int64{}
	maxT := map[string]int64{}
	for _, t := range turns {
		i, ok := idx[t.Family]
		if !ok {
			i = len(out)
			idx[t.Family] = i
			out = append(out, Family{Key: t.Family})
			minT[t.Family] = t.UnixMillis
			maxT[t.Family] = t.UnixMillis
		}
		f := &out[i]
		f.Turns++
		f.InputTokens += t.InputTokens
		f.CacheReadTokens += t.CacheRead
		f.CacheCreationTokens += t.CacheCreation
		if t.UnixMillis < minT[t.Family] {
			minT[t.Family] = t.UnixMillis
		}
		if t.UnixMillis > maxT[t.Family] {
			maxT[t.Family] = t.UnixMillis
		}
	}
	for i := range out {
		span := maxT[out[i].Key] - minT[out[i].Key]
		out[i].ElapsedSeconds = float64(span) / 1000.0
	}
	return out
}

// famTurns returns a family's turns sorted ascending by time so the warmth-belief
// sequencing and arrival rate are real.
func famTurns(turns []Turn, key string) []Turn {
	out := make([]Turn, 0)
	for _, t := range turns {
		if t.Family == key {
			out = append(out, t)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].UnixMillis < out[j].UnixMillis })
	return out
}

// predictFamily runs the SHIPPED warmth-belief estimator across a family's turns,
// reconciling each predicted belief against the real cache_read the provider
// returned. The belief starts cold (zero value → IsWarm false), so turn 1 predicts
// cold; a confirmed read revives it and the TTL clock decays it across idle gaps.
func predictFamily(ft []Turn, policy vcachecal.BeliefPolicy) vcachecal.PredictionError {
	var pe vcachecal.PredictionError
	var b vcachecal.Belief
	started := false
	for _, t := range ft {
		if started {
			b = b.Advance(policy, t.UnixMillis)
		}
		var outcome vcachecal.PredictionOutcome
		b, outcome = b.Observe(policy, t.UnixMillis, t.CacheRead)
		pe.Add(outcome)
		started = true
	}
	return pe
}

// classifyFamily projects the family's OBSERVED arrival rate (turns over elapsed) onto
// the shipped Governor and returns λ and the steady-state verdict (§5.4).
func classifyFamily(ft []Turn, m Multipliers) (float64, vcachegov.GovernorDecision) {
	if len(ft) == 0 {
		return 0, vcachegov.DecisionEvict
	}
	span := ft[len(ft)-1].UnixMillis - ft[0].UnixMillis
	last := ft[len(ft)-1].UnixMillis
	var lambda float64
	if span > 0 {
		lambda = float64(len(ft)) / (float64(span) / 1000.0)
	} else {
		// All turns at one instant: a burst — treat as comfortably hot.
		lambda = float64(len(ft))
	}
	stats := vcachegov.PrefixStats{
		ArrivalRatePerSec: lambda,
		TTLMillis:         ttl5mMillis,
		WriteMult:         m.Write5m,
		LatencyValue:      1,
		RateShadowPrice:   0,
		Secret:            vcachegov.Cacheable,
		LastAccessMillis:  last,
		NowMillis:         last,
	}
	return lambda, vcachegov.Classify(stats)
}

func telemetryRow(t Turn) vcachegov.TelemetryRow {
	return vcachegov.TelemetryRow{
		InputTokens:              float64(t.InputTokens),
		CacheCreationInputTokens: float64(t.CacheCreation),
		CacheReadInputTokens:     float64(t.CacheRead),
		Ephemeral1hInputTokens:   float64(t.Ephemeral1h),
		Ephemeral5mInputTokens:   float64(t.Ephemeral5m),
	}
}

// Rows converts observed turns to the vcachegov.TelemetryRow form vcachescore.Score
// consumes, using the SAME per-turn mapping Observe uses internally — so a caller that
// folds a persisted turn window into `fak vcache score` gets a result that reconciles
// with `fak vcache observe` by construction. Returns nil for an empty input.
func Rows(turns []Turn) []vcachegov.TelemetryRow {
	if len(turns) == 0 {
		return nil
	}
	rows := make([]vcachegov.TelemetryRow, len(turns))
	for i := range turns {
		rows[i] = telemetryRow(turns[i])
	}
	return rows
}

func ratio(num, den float64) float64 {
	if den == 0 {
		if num > 0 {
			return math.Inf(1)
		}
		return 0
	}
	return num / den
}
