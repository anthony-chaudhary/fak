// Package qwen4exp contains bounded contracts for Qwen4-Exp native execution.
//
// The MTP seam in this file is deliberately independent of the base model
// implementation. A zero-value gate and proposal path are inert. A caller can arm
// the one-layer, three-future-token proposal path only with a matched receipt that
// proves exact fak-native output equivalence and a net end-to-end gain.
package qwen4exp

import (
	"fmt"
	"math"
	"strings"
)

const (
	// NativeEngine is the only execution identity admitted by the MTP gate.
	NativeEngine = "fak-native"
	// MTPNgramSize is the fixed number of future tokens predicted by the Qwen4-Exp MTP head.
	MTPNgramSize = 3
	// Architecture is the exact checkpoint architecture this contract admits.
	Architecture = "qwen4_exp_text"
	// MTPParameterSharing identifies the parameter-shared future-token heads.
	MTPParameterSharing = "shared"
	// MTPFirstFutureAttention is the attention mode for the first future token.
	MTPFirstFutureAttention = "autoregressive"
	// MTPLaterFutureAttention is the attention mode for the remaining future tokens.
	MTPLaterFutureAttention = "bidirectional"

	DecisionEnable = "enable"
	DecisionReject = "reject"
)

// ExecutionIdentity binds a measured arm and a live proposer to one native path.
// FallbackCount must remain zero: a reference runtime is not fak-native evidence.
type ExecutionIdentity struct {
	Engine        string
	ForwardPath   string
	FallbackCount int
}

// Controls are the axes which must be identical for the ordinary and MTP arms.
// PromptDigest and QualityDigest bind the exact prompt corpus and quality witness
// without requiring the gate to retain their potentially large artifacts.
type Controls struct {
	ModelRevision string
	PromptDigest  string
	QualityDigest string
	Batch         int
	ContextTokens int
	MaxNewTokens  int
	Sampling      string
}

// MemoryMeasurement is the end-to-end peak and resident memory observed for one arm.
type MemoryMeasurement struct {
	PeakBytes     uint64
	ResidentBytes uint64
}

// Arm is one matched ordinary-decode or MTP-decode measurement. OutputTokens are
// retained as token IDs so equivalence is exact rather than inferred from text.
type Arm struct {
	Controls            Controls
	Execution           ExecutionIdentity
	MTPEnabled          bool
	OutputTokens        []int
	EndToEndNanoseconds uint64
	Memory              MemoryMeasurement
}

// MeasuredDuration distinguishes an observed zero cost from an omitted category.
// Durations in Overheads are disjoint and their sum must be included in the
// candidate arm's end-to-end duration.
type MeasuredDuration struct {
	Measured    bool
	Nanoseconds uint64
}

// Overheads accounts for every cost introduced by the optional proposal path.
type Overheads struct {
	Proposal        MeasuredDuration
	Verification    MeasuredDuration
	Rejection       MeasuredDuration
	Synchronization MeasuredDuration
	Recovery        MeasuredDuration
}

// MTPConfig is the exact Qwen4-Exp predictor shape and attention contract. It
// distinguishes the checkpoint's parameter-shared hybrid head from prompt lookup
// and other unrelated three-token drafters.
type MTPConfig struct {
	Architecture          string
	Hybrid                bool
	Layers                int
	NgramSize             int
	ParameterSharing      string
	FirstFutureAttention  string
	LaterFutureAttention  string
	LaterFutureTokenCount int
}

// RequiredMTPConfig returns the sole predictor configuration admitted by this gate.
func RequiredMTPConfig() MTPConfig {
	return MTPConfig{
		Architecture:          Architecture,
		Hybrid:                true,
		Layers:                1,
		NgramSize:             MTPNgramSize,
		ParameterSharing:      MTPParameterSharing,
		FirstFutureAttention:  MTPFirstFutureAttention,
		LaterFutureAttention:  MTPLaterFutureAttention,
		LaterFutureTokenCount: MTPNgramSize - 1,
	}
}

// MTPEvidence describes the measured optional path. AcceptanceLengths contains
// the accepted prefix length (0..3) for every fixed three-token proposal round.
type MTPEvidence struct {
	Config                    MTPConfig
	ProposedTokens            int
	AcceptedTokens            int
	RejectedTokens            int
	AcceptanceLengths         []int
	Overheads                 Overheads
	EndToEndIncludesOverheads bool
	MemoryIncludesMTP         bool
}

// Receipt is a matched MTP-off/MTP-on comparison.
type Receipt struct {
	Baseline  Arm
	Candidate Arm
	MTP       MTPEvidence
}

// GatePolicy declares the smallest improvement and memory envelope worth arming.
// Zero gain thresholds still require a strict end-to-end latency and throughput win.
type GatePolicy struct {
	MinimumLatencyGainPercent      float64
	MinimumThroughputGainPercent   float64
	MaximumAdditionalPeakBytes     uint64
	MaximumAdditionalResidentBytes uint64
}

// MTPGate is explicit opt-in. Its zero value never enables proposals.
type MTPGate struct {
	Enabled bool
	Policy  GatePolicy
}

// AcceptanceDistribution is derived from every proposal round. Counts[i] is the
// number of rounds that accepted exactly i of the three proposed tokens.
type AcceptanceDistribution struct {
	Counts [MTPNgramSize + 1]uint64
	Rounds uint64
	Mean   float64
}

// MemoryDelta reports candidate minus baseline bytes; positive values are costs.
type MemoryDelta struct {
	PeakBytes     int64
	ResidentBytes int64
}

// Verdict reports the complete net-true fold. An admission can be obtained only
// from an enabling verdict and cannot be constructed by another package.
type Verdict struct {
	Decision                      string
	Enabled                       bool
	Reasons                       []string
	BaselineTokensPerSecond       float64
	CandidateTokensPerSecond      float64
	LatencyGainPercent            float64
	ThroughputGainPercent         float64
	NetLatencyGainNanoseconds     int64
	IntroducedOverheadNanoseconds uint64
	MemoryDelta                   MemoryDelta
	Acceptance                    AcceptanceDistribution
	admission                     Admission
}

// Admission is the opaque capability produced by a passing MTP gate.
type Admission struct {
	enabled bool
	binding ProposerBinding
}

// Enabled reports whether the admission can arm a proposal path.
func (a Admission) Enabled() bool { return a.enabled }

// Admission returns the opaque proposal capability only for an enabling verdict.
func (v Verdict) Admission() (Admission, bool) {
	return v.admission, v.Enabled && v.admission.enabled
}

// Evaluate validates a matched receipt and admits the optional MTP path only when
// every correctness, identity, accounting, memory, and performance check passes.
func (g MTPGate) Evaluate(r Receipt) Verdict {
	v := Verdict{Decision: DecisionReject}
	fail := func(reason string) { v.Reasons = append(v.Reasons, reason) }

	if !g.Enabled {
		fail("MTP gate is disabled")
	}
	if !validPercent(g.Policy.MinimumLatencyGainPercent) || !validPercent(g.Policy.MinimumThroughputGainPercent) {
		fail("gain thresholds must be finite and non-negative")
	}
	if err := validateControls(r.Baseline.Controls); err != nil {
		fail("baseline controls: " + err.Error())
	}
	if err := validateControls(r.Candidate.Controls); err != nil {
		fail("candidate controls: " + err.Error())
	}
	if r.Baseline.Controls != r.Candidate.Controls {
		fail("baseline and candidate controls are not identical")
	}
	if !validNativeIdentity(r.Baseline.Execution) {
		fail("baseline execution must be fak-native with zero fallback and a named forward path")
	}
	if !validNativeIdentity(r.Candidate.Execution) {
		fail("candidate execution must be fak-native with zero fallback and a named forward path")
	}
	if r.Baseline.MTPEnabled || !r.Candidate.MTPEnabled {
		fail("comparison must measure MTP off in the baseline and on in the candidate")
	}
	if err := validateArm(r.Baseline); err != nil {
		fail("baseline arm: " + err.Error())
	}
	if err := validateArm(r.Candidate); err != nil {
		fail("candidate arm: " + err.Error())
	}
	if !equalTokens(r.Baseline.OutputTokens, r.Candidate.OutputTokens) {
		fail("candidate output tokens differ from ordinary fak-native decode")
	}

	if r.MTP.Config != RequiredMTPConfig() {
		fail("candidate MTP configuration does not match the one-layer parameter-shared hybrid ngram-3 contract")
	}
	if !r.MTP.EndToEndIncludesOverheads {
		fail("candidate end-to-end duration does not attest inclusion of every overhead category")
	}
	if !r.MTP.MemoryIncludesMTP {
		fail("candidate memory measurement does not attest inclusion of the MTP path")
	}

	accepted := 0
	if len(r.MTP.AcceptanceLengths) == 0 {
		fail("acceptance distribution has no proposal rounds")
	}
	for i, n := range r.MTP.AcceptanceLengths {
		if n < 0 || n > MTPNgramSize {
			fail(fmt.Sprintf("acceptance length %d at round %d is outside 0..3", n, i))
			continue
		}
		v.Acceptance.Counts[n]++
		v.Acceptance.Rounds++
		accepted += n
	}
	if v.Acceptance.Rounds > 0 {
		v.Acceptance.Mean = float64(accepted) / float64(v.Acceptance.Rounds)
	}
	wantProposed := len(r.MTP.AcceptanceLengths) * MTPNgramSize
	if r.MTP.ProposedTokens != wantProposed || r.MTP.ProposedTokens <= 0 {
		fail(fmt.Sprintf("proposed tokens = %d, want %d from fixed ngram-3 rounds", r.MTP.ProposedTokens, wantProposed))
	}
	if r.MTP.AcceptedTokens != accepted {
		fail(fmt.Sprintf("accepted tokens = %d, want %d from acceptance distribution", r.MTP.AcceptedTokens, accepted))
	}
	if r.MTP.RejectedTokens != r.MTP.ProposedTokens-r.MTP.AcceptedTokens || r.MTP.RejectedTokens < 0 {
		fail("rejected token count does not reconcile proposed minus accepted tokens")
	}

	overhead, overheadReasons := validateOverheads(r.MTP.Overheads, r.MTP.RejectedTokens)
	v.IntroducedOverheadNanoseconds = overhead
	for _, reason := range overheadReasons {
		fail(reason)
	}
	if overhead > r.Candidate.EndToEndNanoseconds {
		fail("introduced overhead exceeds candidate end-to-end duration")
	}

	if memoryFitsInt64(r.Baseline.Memory) && memoryFitsInt64(r.Candidate.Memory) {
		v.MemoryDelta = MemoryDelta{
			PeakBytes:     int64(r.Candidate.Memory.PeakBytes) - int64(r.Baseline.Memory.PeakBytes),
			ResidentBytes: int64(r.Candidate.Memory.ResidentBytes) - int64(r.Baseline.Memory.ResidentBytes),
		}
		if additionalBytes(r.Baseline.Memory.PeakBytes, r.Candidate.Memory.PeakBytes) > g.Policy.MaximumAdditionalPeakBytes {
			fail("candidate peak-memory delta exceeds the declared envelope")
		}
		if additionalBytes(r.Baseline.Memory.ResidentBytes, r.Candidate.Memory.ResidentBytes) > g.Policy.MaximumAdditionalResidentBytes {
			fail("candidate resident-memory delta exceeds the declared envelope")
		}
	} else {
		fail("memory measurement exceeds the signed delta reporting range")
	}

	if validTiming(r.Baseline) && validTiming(r.Candidate) {
		baselineNS := float64(r.Baseline.EndToEndNanoseconds)
		candidateNS := float64(r.Candidate.EndToEndNanoseconds)
		v.BaselineTokensPerSecond = float64(len(r.Baseline.OutputTokens)) * 1e9 / baselineNS
		v.CandidateTokensPerSecond = float64(len(r.Candidate.OutputTokens)) * 1e9 / candidateNS
		v.LatencyGainPercent = (baselineNS - candidateNS) / baselineNS * 100
		v.ThroughputGainPercent = (v.CandidateTokensPerSecond - v.BaselineTokensPerSecond) / v.BaselineTokensPerSecond * 100
		v.NetLatencyGainNanoseconds = int64(r.Baseline.EndToEndNanoseconds) - int64(r.Candidate.EndToEndNanoseconds)
		if r.Candidate.EndToEndNanoseconds >= r.Baseline.EndToEndNanoseconds || v.CandidateTokensPerSecond <= v.BaselineTokensPerSecond {
			fail("candidate has no strict net end-to-end latency and throughput gain")
		}
		if v.LatencyGainPercent < g.Policy.MinimumLatencyGainPercent {
			fail("candidate latency gain is below policy")
		}
		if v.ThroughputGainPercent < g.Policy.MinimumThroughputGainPercent {
			fail("candidate throughput gain is below policy")
		}
	}

	if len(v.Reasons) == 0 {
		v.Decision = DecisionEnable
		v.Enabled = true
		v.admission = Admission{enabled: true, binding: ProposerBinding{
			Execution:     r.Candidate.Execution,
			ModelRevision: r.Candidate.Controls.ModelRevision,
			Config:        r.MTP.Config,
		}}
	}
	return v
}

// Ngram3Proposal is the fixed three-future-token output of the optional MTP head.
type Ngram3Proposal struct {
	Tokens [MTPNgramSize]int
}

// ProposerBinding joins the native execution path to the exact measured model
// revision and hybrid MTP configuration. All fields are comparable so admission
// and live-call checks are exact.
type ProposerBinding struct {
	Execution     ExecutionIdentity
	ModelRevision string
	Config        MTPConfig
}

// MTPProposer is the narrow, package-sealed seam a future Qwen4-Exp native head
// implements. The unexported marker prevents an unrelated prompt-lookup drafter
// in another package from satisfying the interface by relabeling its identity.
// The seam contains no base-decode API and therefore cannot become a correctness
// dependency of ordinary autoregressive decoding.
type MTPProposer interface {
	isQwen4ExpHybridMTP()
	Binding() ProposerBinding
	ProposeHybridNgram3(history []int) (Ngram3Proposal, error)
}

// ProposalPath is inert until constructed from a passing gate admission.
type ProposalPath struct {
	admission Admission
	proposer  MTPProposer
}

// NewProposalPath binds a passing receipt to the exact live native proposer identity.
func NewProposalPath(admission Admission, proposer MTPProposer) (ProposalPath, error) {
	if !admission.enabled {
		return ProposalPath{}, fmt.Errorf("qwen4exp: MTP admission is not enabled")
	}
	if proposer == nil {
		return ProposalPath{}, fmt.Errorf("qwen4exp: nil MTP proposer")
	}
	if binding := proposer.Binding(); binding != admission.binding || !validProposerBinding(binding) {
		return ProposalPath{}, fmt.Errorf("qwen4exp: proposer binding does not match the admitted model revision and hybrid fak-native path")
	}
	return ProposalPath{admission: admission, proposer: proposer}, nil
}

// Enabled reports whether the optional proposal path is armed.
func (p ProposalPath) Enabled() bool {
	return p.admission.enabled && p.proposer != nil
}

// Propose returns ok=false for the default-off path. An armed path rechecks its
// native identity on every call and gives the proposer a history copy so it cannot
// mutate correctness-critical base-decode state.
func (p ProposalPath) Propose(history []int) (proposal Ngram3Proposal, ok bool, err error) {
	if !p.Enabled() {
		return Ngram3Proposal{}, false, nil
	}
	if len(history) == 0 {
		return Ngram3Proposal{}, false, fmt.Errorf("qwen4exp: MTP proposal requires committed history")
	}
	if binding := p.proposer.Binding(); binding != p.admission.binding || !validProposerBinding(binding) {
		return Ngram3Proposal{}, false, fmt.Errorf("qwen4exp: admitted proposer binding changed")
	}
	proposal, err = p.proposer.ProposeHybridNgram3(append([]int(nil), history...))
	if err != nil {
		return Ngram3Proposal{}, false, err
	}
	for i, token := range proposal.Tokens {
		if token < 0 {
			return Ngram3Proposal{}, false, fmt.Errorf("qwen4exp: proposal token %d at position %d is negative", token, i)
		}
	}
	return proposal, true, nil
}

func validateControls(c Controls) error {
	if strings.TrimSpace(c.ModelRevision) == "" || strings.TrimSpace(c.PromptDigest) == "" || strings.TrimSpace(c.QualityDigest) == "" || strings.TrimSpace(c.Sampling) == "" {
		return fmt.Errorf("model, prompt, quality, and sampling identities are required")
	}
	if c.Batch <= 0 || c.ContextTokens <= 0 || c.MaxNewTokens <= 0 {
		return fmt.Errorf("batch, context tokens, and max new tokens must be positive")
	}
	return nil
}

func validateArm(a Arm) error {
	if a.EndToEndNanoseconds == 0 || a.EndToEndNanoseconds > math.MaxInt64 {
		return fmt.Errorf("end-to-end duration must be positive and reportable")
	}
	if a.Memory.PeakBytes == 0 || a.Memory.ResidentBytes == 0 {
		return fmt.Errorf("peak and resident memory must be measured")
	}
	if len(a.OutputTokens) == 0 || len(a.OutputTokens) > a.Controls.MaxNewTokens {
		return fmt.Errorf("output token count must be within the decode budget")
	}
	for i, token := range a.OutputTokens {
		if token < 0 {
			return fmt.Errorf("output token %d at position %d is negative", token, i)
		}
	}
	return nil
}

func validateOverheads(o Overheads, rejectedTokens int) (uint64, []string) {
	phases := []struct {
		name       string
		cost       MeasuredDuration
		mustBeBusy bool
	}{
		{"proposal", o.Proposal, true},
		{"verification", o.Verification, true},
		{"rejection", o.Rejection, rejectedTokens > 0},
		{"synchronization", o.Synchronization, true},
		{"recovery", o.Recovery, rejectedTokens > 0},
	}
	var total uint64
	var reasons []string
	for _, phase := range phases {
		if !phase.cost.Measured {
			reasons = append(reasons, phase.name+" overhead was not measured")
		}
		if phase.mustBeBusy && phase.cost.Nanoseconds == 0 {
			reasons = append(reasons, phase.name+" overhead must be positive for the observed work")
		}
		if math.MaxUint64-total < phase.cost.Nanoseconds {
			reasons = append(reasons, "overhead total overflows uint64")
			continue
		}
		total += phase.cost.Nanoseconds
	}
	return total, reasons
}

func validNativeIdentity(identity ExecutionIdentity) bool {
	return identity.Engine == NativeEngine && strings.TrimSpace(identity.ForwardPath) != "" && identity.FallbackCount == 0
}

func validProposerBinding(binding ProposerBinding) bool {
	return validNativeIdentity(binding.Execution) && strings.TrimSpace(binding.ModelRevision) != "" && binding.Config == RequiredMTPConfig()
}

func validPercent(v float64) bool {
	return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func validTiming(a Arm) bool {
	return a.EndToEndNanoseconds > 0 && a.EndToEndNanoseconds <= math.MaxInt64 && len(a.OutputTokens) > 0
}

func memoryFitsInt64(m MemoryMeasurement) bool {
	return m.PeakBytes <= math.MaxInt64 && m.ResidentBytes <= math.MaxInt64
}

func additionalBytes(baseline, candidate uint64) uint64 {
	if candidate <= baseline {
		return 0
	}
	return candidate - baseline
}

func equalTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
