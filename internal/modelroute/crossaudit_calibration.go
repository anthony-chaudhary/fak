package modelroute

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const CrossAuditCalibrationSchema = "fak-crossaudit-calibration/v1"

type CalibrationTruth struct {
	ID                 string                 `json:"id"`
	Class              AccidentalFailureClass `json:"class"`
	Corrupt            bool                   `json:"corrupt"`
	BundleDigest       string                 `json:"bundle_digest"`
	WrapperAdversarial bool                   `json:"wrapper_adversarial,omitempty"`
}

type CalibrationObservation struct {
	ID                string            `json:"id"`
	Auditor           AuditIdentity     `json:"auditor"`
	Verdict           CrossAuditVerdict `json:"verdict"`
	BundleDigest      string            `json:"bundle_digest"`
	PolicyDigest      string            `json:"policy_digest"`
	PromptVersion     string            `json:"prompt_version"`
	PromptDigest      string            `json:"prompt_digest"`
	DurationNanos     int64             `json:"duration_nanos"`
	Usage             AuditTokenCost    `json:"usage"`
	EvidenceTruncated bool              `json:"evidence_truncated,omitempty"`
}

type CalibrationConfusion struct {
	TruePositive  int `json:"true_positive"`
	FalsePositive int `json:"false_positive"`
	TrueNegative  int `json:"true_negative"`
	FalseNegative int `json:"false_negative"`
	Abstained     int `json:"abstained"`
	Unavailable   int `json:"unavailable"`
}

type CalibrationMetrics struct {
	Samples           int                  `json:"samples"`
	Confusion         CalibrationConfusion `json:"confusion"`
	Precision         float64              `json:"precision"`
	Recall            float64              `json:"recall"`
	FalsePositiveRate float64              `json:"false_positive_rate"`
	AbstentionRate    float64              `json:"abstention_rate"`
	UnavailableRate   float64              `json:"unavailable_rate"`
	LatencyP50Nanos   int64                `json:"latency_p50_nanos"`
	LatencyP95Nanos   int64                `json:"latency_p95_nanos"`
	InputTokens       int64                `json:"input_tokens"`
	OutputTokens      int64                `json:"output_tokens"`
	CostMicrosUSD     int64                `json:"cost_micros_usd"`
	EvidenceTruncated int                  `json:"evidence_truncated"`
	RecallWilsonLow95 float64              `json:"recall_wilson_low_95"`
	FPRWilsonHigh95   float64              `json:"fpr_wilson_high_95"`
}

type CalibrationArm struct {
	Auditor       AuditIdentity                                 `json:"auditor"`
	Metrics       CalibrationMetrics                            `json:"metrics"`
	ByClass       map[AccidentalFailureClass]CalibrationMetrics `json:"by_class"`
	WrapperRecall *float64                                      `json:"wrapper_recall,omitempty"`
	CleanRecall   *float64                                      `json:"clean_wrapper_recall,omitempty"`
	Status        string                                        `json:"status"`
	NotYetReason  string                                        `json:"not_yet_reason,omitempty"`
}

type CalibrationDisagreement struct {
	AuditorA  string  `json:"auditor_a"`
	AuditorB  string  `json:"auditor_b"`
	Compared  int     `json:"compared"`
	Different int     `json:"different"`
	Rate      float64 `json:"rate"`
}

type CrossAuditCalibrationReport struct {
	Schema        string                    `json:"schema"`
	CorpusDigest  string                    `json:"corpus_digest"`
	PolicyDigest  string                    `json:"policy_digest"`
	PromptVersion string                    `json:"prompt_version"`
	PromptDigest  string                    `json:"prompt_digest"`
	Arms          []CalibrationArm          `json:"arms"`
	Disagreements []CalibrationDisagreement `json:"disagreements"`
}

func BuildCrossAuditCalibrationReport(truth []CalibrationTruth, observations []CalibrationObservation) (CrossAuditCalibrationReport, error) {
	if len(truth) == 0 {
		return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: calibration truth is empty")
	}
	truthByID := map[string]CalibrationTruth{}
	var digestParts []string
	for _, row := range truth {
		if row.ID == "" || row.BundleDigest == "" || truthByID[row.ID].ID != "" {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: invalid/duplicate calibration truth %q", row.ID)
		}
		truthByID[row.ID] = row
		digestParts = append(digestParts, row.ID+":"+row.BundleDigest+fmt.Sprintf(":%t", row.Corrupt))
	}
	sort.Strings(digestParts)
	report := CrossAuditCalibrationReport{Schema: CrossAuditCalibrationSchema, CorpusDigest: IssueAuditContentDigest(strings.Join(digestParts, "\n"))}
	byAuditor := map[string][]CalibrationObservation{}
	seen := map[string]bool{}
	for _, obs := range observations {
		row, ok := truthByID[obs.ID]
		if !ok {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: observation %q has no truth row", obs.ID)
		}
		if obs.BundleDigest != row.BundleDigest {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: observation %q bundle provenance mismatch", obs.ID)
		}
		if obs.Auditor.Provider == "" || obs.Auditor.Family == "" || obs.Auditor.Model == "" {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: observation %q auditor provenance incomplete", obs.ID)
		}
		if obs.PolicyDigest == "" || obs.PromptVersion == "" || obs.PromptDigest == "" {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: observation %q policy/prompt provenance incomplete", obs.ID)
		}
		key := calibrationAuditorKey(obs.Auditor)
		uniq := key + ":" + obs.ID
		if seen[uniq] {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: duplicate observation %s", uniq)
		}
		seen[uniq] = true
		byAuditor[key] = append(byAuditor[key], obs)
		if report.PromptVersion == "" {
			report.PromptVersion = obs.PromptVersion
			report.PromptDigest = IssueAuditContentDigest(CrossAuditSystemPrompt)
		}
		if report.PromptVersion != obs.PromptVersion || obs.PromptDigest == "" {
			return CrossAuditCalibrationReport{}, fmt.Errorf("modelroute: calibration prompt provenance is not frozen")
		}
		// Policy digests legitimately differ when each arm's authoritative roster
		// binds a different auditor identity. Preserve the per-row digest and stamp
		// the report with the sorted set rather than pretending one roster applied.
		if report.PolicyDigest == "" {
			report.PolicyDigest = obs.PolicyDigest
		} else if !strings.Contains(report.PolicyDigest, obs.PolicyDigest) {
			parts := strings.Split(report.PolicyDigest, ",")
			parts = append(parts, obs.PolicyDigest)
			sort.Strings(parts)
			report.PolicyDigest = strings.Join(parts, ",")
		}
	}
	keys := make([]string, 0, len(byAuditor))
	for k := range byAuditor {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, key := range keys {
		obs := byAuditor[key]
		arm := CalibrationArm{Auditor: obs[0].Auditor, Metrics: scoreCalibration(truthByID, obs), ByClass: map[AccidentalFailureClass]CalibrationMetrics{}, Status: "calibrated"}
		classObs := map[AccidentalFailureClass][]CalibrationObservation{}
		for _, o := range obs {
			classObs[truthByID[o.ID].Class] = append(classObs[truthByID[o.ID].Class], o)
		}
		for c, rows := range classObs {
			arm.ByClass[c] = scoreCalibration(truthByID, rows)
		}
		report.Arms = append(report.Arms, arm)
	}
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			report.Disagreements = append(report.Disagreements, scoreDisagreement(keys[i], byAuditor[keys[i]], keys[j], byAuditor[keys[j]]))
		}
	}
	return report, nil
}

func calibrationAuditorKey(a AuditIdentity) string {
	return strings.Join([]string{a.Provider, a.Family, a.Model, a.WeightsRevision, a.ReasoningPosture, a.Driver}, "/")
}

func scoreCalibration(truth map[string]CalibrationTruth, obs []CalibrationObservation) CalibrationMetrics {
	m := CalibrationMetrics{Samples: len(obs)}
	lat := make([]int64, 0, len(obs))
	for _, o := range obs {
		t := truth[o.ID]
		switch o.Verdict {
		case CrossAuditRefute:
			if t.Corrupt {
				m.Confusion.TruePositive++
			} else {
				m.Confusion.FalsePositive++
			}
		case CrossAuditPass:
			if t.Corrupt {
				m.Confusion.FalseNegative++
			} else {
				m.Confusion.TrueNegative++
			}
		case CrossAuditInconclusive:
			m.Confusion.Abstained++
		case CrossAuditUnavailable:
			m.Confusion.Unavailable++
		}
		if o.DurationNanos > 0 {
			lat = append(lat, o.DurationNanos)
		}
		u := o.Usage.Normalize()
		m.InputTokens += u.InputTokens
		m.OutputTokens += u.OutputTokens
		m.CostMicrosUSD += u.CostMicrosUSD
		if o.EvidenceTruncated {
			m.EvidenceTruncated++
		}
	}
	m.Precision = ratio(m.Confusion.TruePositive, m.Confusion.TruePositive+m.Confusion.FalsePositive)
	m.Recall = ratio(m.Confusion.TruePositive, m.Confusion.TruePositive+m.Confusion.FalseNegative)
	m.FalsePositiveRate = ratio(m.Confusion.FalsePositive, m.Confusion.FalsePositive+m.Confusion.TrueNegative)
	m.AbstentionRate = ratio(m.Confusion.Abstained, m.Samples)
	m.UnavailableRate = ratio(m.Confusion.Unavailable, m.Samples)
	m.RecallWilsonLow95 = wilson(m.Confusion.TruePositive, m.Confusion.TruePositive+m.Confusion.FalseNegative, false)
	m.FPRWilsonHigh95 = wilson(m.Confusion.FalsePositive, m.Confusion.FalsePositive+m.Confusion.TrueNegative, true)
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	m.LatencyP50Nanos = percentile(lat, .50)
	m.LatencyP95Nanos = percentile(lat, .95)
	return m
}
func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}
func percentile(v []int64, p float64) int64 {
	if len(v) == 0 {
		return 0
	}
	i := int(math.Ceil(float64(len(v))*p)) - 1
	if i < 0 {
		i = 0
	}
	return v[i]
}
func wilson(success, total int, upper bool) float64 {
	if total == 0 {
		return 0
	}
	z := 1.959963984540054
	n := float64(total)
	ph := float64(success) / n
	center := (ph + z*z/(2*n)) / (1 + z*z/n)
	margin := z * math.Sqrt((ph*(1-ph)+z*z/(4*n))/n) / (1 + z*z/n)
	if upper {
		return math.Min(1, center+margin)
	}
	return math.Max(0, center-margin)
}
func scoreDisagreement(ka string, a []CalibrationObservation, kb string, b []CalibrationObservation) CalibrationDisagreement {
	bm := map[string]CrossAuditVerdict{}
	for _, o := range b {
		bm[o.ID] = o.Verdict
	}
	d := CalibrationDisagreement{AuditorA: ka, AuditorB: kb}
	for _, o := range a {
		if v, ok := bm[o.ID]; ok {
			d.Compared++
			if v != o.Verdict {
				d.Different++
			}
		}
	}
	d.Rate = ratio(d.Different, d.Compared)
	return d
}
