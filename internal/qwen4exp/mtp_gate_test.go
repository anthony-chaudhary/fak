package qwen4exp

import (
	"errors"
	"math"
	"strings"
	"testing"
)

const millisecond = uint64(1_000_000)

func passingGateAndReceipt() (MTPGate, Receipt) {
	controls := Controls{
		ModelRevision: "qwen4-exp-test-revision",
		PromptDigest:  "prompt-corpus-sha256",
		QualityDigest: "exact-token-oracle-sha256",
		Batch:         1,
		ContextTokens: 4096,
		MaxNewTokens:  6,
		Sampling:      "greedy",
	}
	baselineIdentity := ExecutionIdentity{Engine: NativeEngine, ForwardPath: "qwen4exp/autoregressive"}
	candidateIdentity := ExecutionIdentity{Engine: NativeEngine, ForwardPath: "qwen4exp/mtp-ngram3"}
	output := []int{11, 12, 13, 14, 15, 16}
	gate := MTPGate{
		Enabled: true,
		Policy: GatePolicy{
			MinimumLatencyGainPercent:      10,
			MinimumThroughputGainPercent:   10,
			MaximumAdditionalPeakBytes:     200,
			MaximumAdditionalResidentBytes: 100,
		},
	}
	receipt := Receipt{
		Baseline: Arm{
			Controls: controls, Execution: baselineIdentity, OutputTokens: append([]int(nil), output...),
			EndToEndNanoseconds: 100 * millisecond,
			Memory:              MemoryMeasurement{PeakBytes: 1_000, ResidentBytes: 900},
		},
		Candidate: Arm{
			Controls: controls, Execution: candidateIdentity, MTPEnabled: true, OutputTokens: append([]int(nil), output...),
			EndToEndNanoseconds: 80 * millisecond,
			Memory:              MemoryMeasurement{PeakBytes: 1_120, ResidentBytes: 850},
		},
		MTP: MTPEvidence{
			Config:         RequiredMTPConfig(),
			ProposedTokens: 9, AcceptedTokens: 5, RejectedTokens: 4,
			AcceptanceLengths: []int{3, 2, 0},
			Overheads: Overheads{
				Proposal:        MeasuredDuration{Measured: true, Nanoseconds: 2 * millisecond},
				Verification:    MeasuredDuration{Measured: true, Nanoseconds: 5 * millisecond},
				Rejection:       MeasuredDuration{Measured: true, Nanoseconds: 1 * millisecond},
				Synchronization: MeasuredDuration{Measured: true, Nanoseconds: 1 * millisecond},
				Recovery:        MeasuredDuration{Measured: true, Nanoseconds: 1 * millisecond},
			},
			EndToEndIncludesOverheads: true,
			MemoryIncludesMTP:         true,
		},
	}
	return gate, receipt
}

func TestMTPGateDefaultOff(t *testing.T) {
	_, receipt := passingGateAndReceipt()
	verdict := (MTPGate{}).Evaluate(receipt)
	if verdict.Enabled || verdict.Decision != DecisionReject || !hasReason(verdict, "gate is disabled") {
		t.Fatalf("zero-value gate verdict = %+v, want disabled rejection", verdict)
	}
	if _, ok := verdict.Admission(); ok {
		t.Fatal("disabled gate exposed an admission")
	}
	var path ProposalPath
	if proposal, ok, err := path.Propose([]int{1}); err != nil || ok || proposal != (Ngram3Proposal{}) {
		t.Fatalf("zero-value path = %+v ok=%v err=%v, want inert", proposal, ok, err)
	}
}

func TestMTPGateEnablesEquivalentNetTrueNativePath(t *testing.T) {
	gate, receipt := passingGateAndReceipt()
	verdict := gate.Evaluate(receipt)
	if !verdict.Enabled || verdict.Decision != DecisionEnable || len(verdict.Reasons) != 0 {
		t.Fatalf("verdict = %+v, want enable", verdict)
	}
	if verdict.NetLatencyGainNanoseconds != int64(20*millisecond) || verdict.IntroducedOverheadNanoseconds != 10*millisecond {
		t.Fatalf("latency fold = net %d overhead %d", verdict.NetLatencyGainNanoseconds, verdict.IntroducedOverheadNanoseconds)
	}
	if verdict.MemoryDelta != (MemoryDelta{PeakBytes: 120, ResidentBytes: -50}) {
		t.Fatalf("memory delta = %+v", verdict.MemoryDelta)
	}
	wantCounts := [MTPNgramSize + 1]uint64{1, 0, 1, 1}
	if verdict.Acceptance.Counts != wantCounts || verdict.Acceptance.Rounds != 3 || math.Abs(verdict.Acceptance.Mean-5.0/3.0) > 1e-12 {
		t.Fatalf("acceptance = %+v", verdict.Acceptance)
	}
	if math.Abs(verdict.LatencyGainPercent-20) > 1e-12 || math.Abs(verdict.ThroughputGainPercent-25) > 1e-12 {
		t.Fatalf("gains = latency %.3f throughput %.3f", verdict.LatencyGainPercent, verdict.ThroughputGainPercent)
	}

	admission, ok := verdict.Admission()
	if !ok || !admission.Enabled() {
		t.Fatal("passing gate did not expose an admission")
	}
	head := &fakeProposer{binding: admittedBinding(receipt), proposal: Ngram3Proposal{Tokens: [3]int{21, 22, 23}}, mutateHistory: true}
	path, err := NewProposalPath(admission, head)
	if err != nil {
		t.Fatal(err)
	}
	history := []int{7, 8, 9}
	proposal, proposed, err := path.Propose(history)
	if err != nil || !proposed || proposal.Tokens != [3]int{21, 22, 23} {
		t.Fatalf("proposal = %+v ok=%v err=%v", proposal, proposed, err)
	}
	if history[0] != 7 {
		t.Fatalf("proposer mutated base history: %v", history)
	}
}

func TestMTPGateRejectsIdentityEquivalenceAndMatchedControlFailures(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Receipt)
		want string
	}{
		{"baseline external", func(r *Receipt) { r.Baseline.Execution.Engine = "vllm" }, "baseline execution must be fak-native"},
		{"candidate external", func(r *Receipt) { r.Candidate.Execution.Engine = "llama.cpp" }, "candidate execution must be fak-native"},
		{"fallback", func(r *Receipt) { r.Candidate.Execution.FallbackCount = 1 }, "candidate execution must be fak-native"},
		{"output drift", func(r *Receipt) { r.Candidate.OutputTokens[3]++ }, "output tokens differ"},
		{"prompt drift", func(r *Receipt) { r.Candidate.Controls.PromptDigest = "other" }, "controls are not identical"},
		{"quality drift", func(r *Receipt) { r.Candidate.Controls.QualityDigest = "other" }, "controls are not identical"},
		{"base accidentally MTP", func(r *Receipt) { r.Baseline.MTPEnabled = true }, "MTP off in the baseline"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gate, receipt := passingGateAndReceipt()
			tt.edit(&receipt)
			verdict := gate.Evaluate(receipt)
			if verdict.Enabled || !hasReason(verdict, tt.want) {
				t.Fatalf("verdict = %+v, want rejection containing %q", verdict, tt.want)
			}
			if _, ok := verdict.Admission(); ok {
				t.Fatal("rejected evidence exposed an admission")
			}
		})
	}
}

func TestMTPGateRequiresCompleteReconciledOverheadAndAcceptance(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Receipt)
		want string
	}{
		{"wrong shape", func(r *Receipt) { r.MTP.Config.NgramSize = 4 }, "configuration does not match"},
		{"prompt lookup semantics", func(r *Receipt) { r.MTP.Config.LaterFutureAttention = "history-match" }, "configuration does not match"},
		{"unshared heads", func(r *Receipt) { r.MTP.Config.ParameterSharing = "independent" }, "configuration does not match"},
		{"no rounds", func(r *Receipt) {
			r.MTP.AcceptanceLengths = nil
			r.MTP.ProposedTokens = 0
			r.MTP.AcceptedTokens = 0
			r.MTP.RejectedTokens = 0
		}, "no proposal rounds"},
		{"invalid acceptance", func(r *Receipt) { r.MTP.AcceptanceLengths[0] = 4 }, "outside 0..3"},
		{"proposed mismatch", func(r *Receipt) { r.MTP.ProposedTokens-- }, "proposed tokens"},
		{"accepted mismatch", func(r *Receipt) { r.MTP.AcceptedTokens-- }, "accepted tokens"},
		{"rejected mismatch", func(r *Receipt) { r.MTP.RejectedTokens-- }, "rejected token count"},
		{"proposal omitted", func(r *Receipt) { r.MTP.Overheads.Proposal.Measured = false }, "proposal overhead was not measured"},
		{"verification zero", func(r *Receipt) { r.MTP.Overheads.Verification.Nanoseconds = 0 }, "verification overhead must be positive"},
		{"rejection omitted", func(r *Receipt) { r.MTP.Overheads.Rejection.Measured = false }, "rejection overhead was not measured"},
		{"sync omitted", func(r *Receipt) { r.MTP.Overheads.Synchronization.Measured = false }, "synchronization overhead was not measured"},
		{"recovery omitted", func(r *Receipt) { r.MTP.Overheads.Recovery.Measured = false }, "recovery overhead was not measured"},
		{"overheads excluded", func(r *Receipt) { r.MTP.EndToEndIncludesOverheads = false }, "does not attest inclusion"},
		{"memory excludes MTP", func(r *Receipt) { r.MTP.MemoryIncludesMTP = false }, "does not attest inclusion of the MTP path"},
		{"overheads exceed end-to-end", func(r *Receipt) { r.MTP.Overheads.Proposal.Nanoseconds = 100 * millisecond }, "overhead exceeds"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gate, receipt := passingGateAndReceipt()
			tt.edit(&receipt)
			verdict := gate.Evaluate(receipt)
			if verdict.Enabled || !hasReason(verdict, tt.want) {
				t.Fatalf("verdict = %+v, want rejection containing %q", verdict, tt.want)
			}
		})
	}
}

func TestMTPGateRejectsNoNetGainAndMemoryOutsideEnvelope(t *testing.T) {
	tests := []struct {
		name string
		edit func(*MTPGate, *Receipt)
		want string
	}{
		{"equal latency", func(_ *MTPGate, r *Receipt) { r.Candidate.EndToEndNanoseconds = r.Baseline.EndToEndNanoseconds }, "no strict net"},
		{"slower", func(_ *MTPGate, r *Receipt) { r.Candidate.EndToEndNanoseconds = 110 * millisecond }, "no strict net"},
		{"below latency policy", func(g *MTPGate, _ *Receipt) { g.Policy.MinimumLatencyGainPercent = 30 }, "latency gain is below"},
		{"below throughput policy", func(g *MTPGate, _ *Receipt) { g.Policy.MinimumThroughputGainPercent = 30 }, "throughput gain is below"},
		{"peak memory", func(g *MTPGate, _ *Receipt) { g.Policy.MaximumAdditionalPeakBytes = 119 }, "peak-memory delta"},
		{"resident memory", func(g *MTPGate, r *Receipt) {
			r.Candidate.Memory.ResidentBytes = 950
			g.Policy.MaximumAdditionalResidentBytes = 49
		}, "resident-memory delta"},
		{"invalid threshold", func(g *MTPGate, _ *Receipt) { g.Policy.MinimumLatencyGainPercent = math.NaN() }, "gain thresholds"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gate, receipt := passingGateAndReceipt()
			tt.edit(&gate, &receipt)
			verdict := gate.Evaluate(receipt)
			if verdict.Enabled || !hasReason(verdict, tt.want) {
				t.Fatalf("verdict = %+v, want rejection containing %q", verdict, tt.want)
			}
		})
	}
}

func TestProposalPathRefusesIdentityDriftAndProposerErrors(t *testing.T) {
	gate, receipt := passingGateAndReceipt()
	verdict := gate.Evaluate(receipt)
	admission, ok := verdict.Admission()
	if !ok {
		t.Fatalf("gate did not pass: %+v", verdict)
	}

	head := &fakeProposer{binding: admittedBinding(receipt), proposal: Ngram3Proposal{Tokens: [3]int{1, 2, 3}}}
	path, err := NewProposalPath(admission, head)
	if err != nil {
		t.Fatal(err)
	}
	head.binding.Execution.ForwardPath = "changed"
	if _, ok, err := path.Propose([]int{1}); err == nil || ok || !strings.Contains(err.Error(), "binding changed") {
		t.Fatalf("binding drift ok=%v err=%v", ok, err)
	}

	head.binding = admittedBinding(receipt)
	head.err = errors.New("head failure")
	if _, ok, err := path.Propose([]int{1}); !errors.Is(err, head.err) || ok {
		t.Fatalf("proposer error ok=%v err=%v", ok, err)
	}
	if _, _, err := path.Propose(nil); err == nil || !strings.Contains(err.Error(), "requires committed history") {
		t.Fatalf("empty history err=%v", err)
	}

	other := &fakeProposer{binding: admittedBinding(receipt)}
	other.binding.Config.LaterFutureAttention = "history-match"
	if _, err := NewProposalPath(admission, other); err == nil {
		t.Fatal("prompt-lookup-shaped proposer binding was admitted")
	}
}

type fakeProposer struct {
	binding       ProposerBinding
	proposal      Ngram3Proposal
	err           error
	mutateHistory bool
}

func (*fakeProposer) isQwen4ExpHybridMTP() {}

func (f *fakeProposer) Binding() ProposerBinding { return f.binding }

func (f *fakeProposer) ProposeHybridNgram3(history []int) (Ngram3Proposal, error) {
	if f.mutateHistory && len(history) > 0 {
		history[0] = 999
	}
	return f.proposal, f.err
}

func admittedBinding(receipt Receipt) ProposerBinding {
	return ProposerBinding{
		Execution:     receipt.Candidate.Execution,
		ModelRevision: receipt.Candidate.Controls.ModelRevision,
		Config:        receipt.MTP.Config,
	}
}

func hasReason(verdict Verdict, want string) bool {
	for _, reason := range verdict.Reasons {
		if strings.Contains(reason, want) {
			return true
		}
	}
	return false
}
