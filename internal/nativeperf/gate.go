package nativeperf

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const (
	GateRequestSchema  = "fak-native-performance-gate-request/v1"
	GateVerdictSchema  = "fak-native-performance-gate-verdict/v1"
	BisectPacketSchema = "fak-native-performance-bisect-packet/v1"
	GatePass           = "pass"
	GateInvestigate    = "investigate"
	GateRegression     = "regression"
)

// QualityMetric binds quality evidence to the same end-to-end receipt.
type QualityMetric struct {
	Name           string  `json:"name"`
	Score          float64 `json:"score"`
	HigherIsBetter bool    `json:"higher_is_better"`
}

// ModuleRevision is a derived module@rev identity at the measured commit.
type ModuleRevision struct {
	Module   string `json:"module"`
	Revision string `json:"revision"`
}

// GatePolicy is valid only for one exact envelope and accepted revision.
type GatePolicy struct {
	Schema                 string  `json:"schema"`
	EnvelopeID             string  `json:"envelope_id"`
	ChangedLeverID         string  `json:"changed_lever_id"`
	AcceptedRevision       string  `json:"accepted_revision"`
	MinimumRepetitions     int     `json:"minimum_repetitions"`
	MaximumNoisePercent    float64 `json:"maximum_noise_percent"`
	InvestigateDropPercent float64 `json:"investigate_drop_percent"`
	RegressionDropPercent  float64 `json:"regression_drop_percent"`
	MinimumThroughput      float64 `json:"minimum_throughput_tokens_per_second"`
	MaximumPeakBytes       uint64  `json:"maximum_peak_bytes"`
	QualityMetric          string  `json:"quality_metric"`
	MinimumQualityScore    float64 `json:"minimum_quality_score"`
	QualityHigherIsBetter  bool    `json:"quality_higher_is_better"`
	RequiredEngine         string  `json:"required_engine"`
	RequiredForwardPath    string  `json:"required_forward_path"`
	RequireAmbientEvidence bool    `json:"require_ambient_evidence,omitempty"`
}

// GateRequest compares one candidate with the envelope's last accepted witness.
type GateRequest struct {
	Schema       string            `json:"schema"`
	Policy       GatePolicy        `json:"policy"`
	LastAccepted ExperimentReceipt `json:"last_accepted"`
	Candidate    ExperimentReceipt `json:"candidate"`
}
type GateCheck struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}
type SuspectModule struct {
	Module            string `json:"module"`
	AcceptedRevision  string `json:"accepted_revision"`
	CandidateRevision string `json:"candidate_revision"`
}
type BisectPacket struct {
	Schema          string          `json:"schema"`
	EnvelopeID      string          `json:"envelope_id"`
	GoodRevision    string          `json:"good_revision"`
	BadRevision     string          `json:"bad_revision"`
	SuspectModules  []SuspectModule `json:"suspect_modules"`
	GuardedCommands []string        `json:"guarded_commands"`
	StopCondition   string          `json:"stop_condition"`
}
type GateVerdict struct {
	Schema                  string          `json:"schema"`
	EnvelopeID              string          `json:"envelope_id"`
	Classification          string          `json:"classification"`
	AcceptedRevision        string          `json:"accepted_revision"`
	CandidateRevision       string          `json:"candidate_revision"`
	AcceptedMeanTokensPerS  float64         `json:"accepted_mean_tokens_per_second"`
	CandidateMeanTokensPerS float64         `json:"candidate_mean_tokens_per_second"`
	ThroughputDeltaPercent  float64         `json:"throughput_delta_percent"`
	AcceptedNoisePercent    float64         `json:"accepted_noise_percent"`
	CandidateNoisePercent   float64         `json:"candidate_noise_percent"`
	Checks                  []GateCheck     `json:"checks"`
	SuspectModules          []SuspectModule `json:"suspect_modules,omitempty"`
	Bisect                  *BisectPacket   `json:"bisect,omitempty"`
}

// Gate returns a typed verdict and, for non-pass results, a guarded bisect packet.
func Gate(r GateRequest) (GateVerdict, error) {
	if r.Schema != GateRequestSchema {
		return GateVerdict{}, fmt.Errorf("gate request schema must be %q", GateRequestSchema)
	}
	p := r.Policy
	if err := validateGatePolicy(p); err != nil {
		return GateVerdict{}, err
	}
	g := ActiveGraph()
	a, c := r.LastAccepted, r.Candidate
	if err := ValidateReceipt(g, a); err != nil {
		return GateVerdict{}, fmt.Errorf("last accepted: %w", err)
	}
	// Identity and fallback are gate dimensions, so normalize only those fields
	// for structural validation and classify their observed values below.
	candidateEvidence := c
	candidateEvidence.Execution.Engine = "fak-native"
	candidateEvidence.Execution.ForwardPath = a.Execution.ForwardPath
	candidateEvidence.Execution.FallbackCount = 0
	if err := ValidateReceipt(g, candidateEvidence); err != nil {
		return GateVerdict{}, fmt.Errorf("candidate: %w", err)
	}
	if a.Revision != p.AcceptedRevision {
		return GateVerdict{}, fmt.Errorf("policy is bound to accepted revision %q, got %q", p.AcceptedRevision, a.Revision)
	}
	if a.Role != RoleBaseline || c.Role != RoleCandidate {
		return GateVerdict{}, fmt.Errorf("gate requires last_accepted baseline and candidate roles")
	}
	if a.EnvelopeID != p.EnvelopeID || c.EnvelopeID != p.EnvelopeID {
		return GateVerdict{}, fmt.Errorf("policy and receipts must use exact envelope %q", p.EnvelopeID)
	}
	if a.ChangedLeverID != p.ChangedLeverID || c.ChangedLeverID != p.ChangedLeverID {
		return GateVerdict{}, fmt.Errorf("policy and receipts must use exact lever %q", p.ChangedLeverID)
	}
	if a.ArtifactSHA256 != c.ArtifactSHA256 || a.Machine != c.Machine || a.Controls != c.Controls || a.ChangedLeverID != c.ChangedLeverID || !sameStrings(a.UnchangedControls, c.UnchangedControls) {
		return GateVerdict{}, fmt.Errorf("receipts are incomparable: an envelope control axis drifted")
	}
	if len(a.Repetitions) < p.MinimumRepetitions || len(c.Repetitions) < p.MinimumRepetitions {
		return GateVerdict{}, fmt.Errorf("policy requires at least %d repetitions", p.MinimumRepetitions)
	}
	if p.RequireAmbientEvidence {
		for _, pair := range []struct {
			name    string
			receipt ExperimentReceipt
		}{{"last accepted", a}, {"candidate", c}} {
			if len(pair.receipt.AmbientEvidence) != len(pair.receipt.Repetitions) {
				return GateVerdict{}, fmt.Errorf("%s ambient evidence must align 1:1 with repetitions", pair.name)
			}
			for i, evidence := range pair.receipt.AmbientEvidence {
				if err := ValidateAmbientEvidence(evidence); err != nil {
					return GateVerdict{}, fmt.Errorf("%s ambient evidence %d: %w", pair.name, i, err)
				}
				if evidence.Verdict != AmbientClean {
					return GateVerdict{Schema: GateVerdictSchema, Classification: GateInvestigate, EnvelopeID: c.EnvelopeID, AcceptedRevision: a.Revision, CandidateRevision: c.Revision, Checks: []GateCheck{{Name: "ambient_evidence", Status: "fail", Detail: fmt.Sprintf("%s repetition %d verdict=%s", pair.name, i, evidence.Verdict)}}}, nil
				}
			}
		}
	}
	am, cm := meanTPS(a.Repetitions), meanTPS(c.Repetitions)
	an, cn := noisePercent(a.Repetitions), noisePercent(c.Repetitions)
	drop := (am - cm) / am * 100
	checks := []GateCheck{gateCheck("engine", c.Execution.Engine == p.RequiredEngine, fmt.Sprintf("got %s; require %s", c.Execution.Engine, p.RequiredEngine)), gateCheck("forward_path", c.Execution.ForwardPath == p.RequiredForwardPath, fmt.Sprintf("got %s; require %s", c.Execution.ForwardPath, p.RequiredForwardPath)), gateCheck("fallback", c.Execution.FallbackCount == 0, fmt.Sprintf("count=%d", c.Execution.FallbackCount)), gateCheck("memory", c.Memory.PeakBytes <= p.MaximumPeakBytes, fmt.Sprintf("peak=%d ceiling=%d", c.Memory.PeakBytes, p.MaximumPeakBytes)), gateCheck("quality", qualityPass(c.Quality, p), fmt.Sprintf("%s=%g floor=%g", c.Quality.Name, c.Quality.Score, p.MinimumQualityScore)), gateCheck("throughput_floor", cm >= p.MinimumThroughput, fmt.Sprintf("mean=%g floor=%g", cm, p.MinimumThroughput))}
	class, hard := GatePass, false
	for _, x := range checks {
		if x.Status == "fail" {
			hard = true
		}
	}
	if hard || drop >= p.RegressionDropPercent {
		class = GateRegression
	} else if an > p.MaximumNoisePercent || cn > p.MaximumNoisePercent || drop >= p.InvestigateDropPercent {
		class = GateInvestigate
	}
	suspects := changedModules(a.ModuleVersions, c.ModuleVersions)
	v := GateVerdict{GateVerdictSchema, p.EnvelopeID, class, a.Revision, c.Revision, am, cm, -drop, an, cn, checks, nil, nil}
	if class != GatePass {
		v.SuspectModules = suspects
		v.Bisect = &BisectPacket{BisectPacketSchema, p.EnvelopeID, a.Revision, c.Revision, suspects, []string{fmt.Sprintf("dos arbitrate --lane native-performance --paths %s", suspectPaths(suspects)), "fak native-performance --gate gate-request.json"}, "first revision classified regression under the exact envelope, quality, identity, memory, and throughput policy"}
	}
	return v, nil
}
func validateGatePolicy(p GatePolicy) error {
	_, envelope, err := findLeverEnvelope(ActiveGraph(), p.ChangedLeverID)
	if err != nil || envelope.ID != p.EnvelopeID {
		return fmt.Errorf("gate policy lever must belong to its exact envelope")
	}
	if p.Schema != "fak-native-performance-gate-policy/v1" || p.EnvelopeID == "" || p.ChangedLeverID == "" || p.AcceptedRevision == "" || p.MinimumRepetitions < 2 || p.MaximumNoisePercent < 0 || p.InvestigateDropPercent < 0 || p.RegressionDropPercent <= p.InvestigateDropPercent || p.MinimumThroughput <= 0 || p.MaximumPeakBytes == 0 || p.QualityMetric == "" || p.RequiredEngine != "fak-native" || p.RequiredForwardPath == "" {
		return fmt.Errorf("invalid envelope-scoped gate policy")
	}
	return nil
}
func gateCheck(n string, ok bool, d string) GateCheck {
	s := "pass"
	if !ok {
		s = "fail"
	}
	return GateCheck{n, s, d}
}
func qualityPass(q QualityMetric, p GatePolicy) bool {
	if q.Name != p.QualityMetric || q.HigherIsBetter != p.QualityHigherIsBetter || math.IsNaN(q.Score) || math.IsInf(q.Score, 0) {
		return false
	}
	if p.QualityHigherIsBetter {
		return q.Score >= p.MinimumQualityScore
	}
	return q.Score <= p.MinimumQualityScore
}
func noisePercent(rs []Repetition) float64 {
	m := meanTPS(rs)
	if m == 0 {
		return math.Inf(1)
	}
	var s float64
	for _, r := range rs {
		d := r.TokensPerSecond - m
		s += d * d
	}
	return math.Sqrt(s/float64(len(rs))) / m * 100
}
func changedModules(a, b []ModuleRevision) []SuspectModule {
	before := map[string]string{}
	for _, m := range a {
		before[m.Module] = m.Revision
	}
	var out []SuspectModule
	for _, m := range b {
		if old, ok := before[m.Module]; ok && old != m.Revision {
			out = append(out, SuspectModule{m.Module, old, m.Revision})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Module < out[j].Module })
	return out
}
func suspectPaths(s []SuspectModule) string {
	if len(s) == 0 {
		return "internal/nativeperf"
	}
	p := make([]string, len(s))
	for i, m := range s {
		p[i] = m.Module
	}
	return strings.Join(p, ",")
}
