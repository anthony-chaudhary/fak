package qwen38quantrun

import (
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestBuildAMDScoreboardComparableEmitsRatios(t *testing.T) {
	in := validAMDScoreboardInput()
	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatal(err)
	}
	if !report.Comparable || report.Verdict != "comparable" || report.ReferenceOverCandidate == nil {
		t.Fatalf("report=%+v", report)
	}
	if got, want := report.ReferenceOverCandidate.Decode, 0.8; math.Abs(got-want) > 1e-12 {
		t.Fatalf("decode ratio=%v want %v", got, want)
	}
	if got, want := report.ReferenceOverCandidate.Prefill, 0.8; math.Abs(got-want) > 1e-12 {
		t.Fatalf("prefill ratio=%v want %v", got, want)
	}
}

func TestBuildAMDScoreboardMismatchSuppressesRatios(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Reference.PromptTokenIDs[0]++
	in.Reference.FallbackActive = true
	report := BuildAMDScoreboard(in)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		t.Fatal(err)
	}
	if report.Comparable || report.ReferenceOverCandidate != nil {
		t.Fatalf("unsafe ratio emitted: %+v", report)
	}
	for _, reason := range []string{"prompt-or-tokenization-mismatch", "reference-not-explicit-llamacpp-comparator"} {
		if !slices.Contains(report.Reasons, reason) {
			t.Fatalf("missing %q in %v", reason, report.Reasons)
		}
	}
}

func TestBuildAMDScoreboardRequiresBoundPacketAndTemplateIdentity(t *testing.T) {
	t.Run("missing attestation", func(t *testing.T) {
		in := validAMDScoreboardInput()
		in.Candidate.PromptPacket = nil
		in.Candidate.PromptPacketDigest = ""
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.ReferenceOverCandidate != nil || !slices.Contains(report.Reasons, "candidate-prompt-attestation-incomplete") {
			t.Fatalf("unattested ratio emitted: %+v", report)
		}
	})

	t.Run("outer template identity mismatch", func(t *testing.T) {
		in := validAMDScoreboardInput()
		in.Reference.TemplateDigest = "7777777777777777777777777777777777777777777777777777777777777777"
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.ReferenceOverCandidate != nil || !slices.Contains(report.Reasons, "reference-prompt-packet-identity-mismatch") {
			t.Fatalf("misbound ratio emitted: %+v", report)
		}
	})

	t.Run("independently bound templates differ", func(t *testing.T) {
		in := validAMDScoreboardInput()
		mismatched := *in.Reference.PromptPacket
		mismatched.TemplateDigest = strings.Repeat("d", 64)
		mismatched.PacketDigest = ""
		mismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		in.Reference.TemplateDigest = mismatched.TemplateDigest
		in.Reference.PromptPacketDigest = mismatched.PacketDigest
		in.Reference.PromptPacket = &mismatched
		report := BuildAMDScoreboard(in)
		if report.Comparable || report.ReferenceOverCandidate != nil || !slices.Contains(report.Reasons, "template-digest-mismatch") {
			t.Fatalf("mismatched-template ratio emitted: %+v", report)
		}
	})
}

func TestBuildAMDScoreboardRequiresFakNativeCandidate(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Candidate.Engine = "llama.cpp"
	report := BuildAMDScoreboard(in)
	if report.Comparable || !slices.Contains(report.Reasons, "candidate-not-fak-native-no-fallback") {
		t.Fatalf("report=%+v", report)
	}
}

func TestBuildAMDScoreboardRequiresMemoryAndThreeTrials(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Candidate.PeakVRAMBytes = 0
	in.Candidate.Trials = in.Candidate.Trials[:2]
	report := BuildAMDScoreboard(in)
	if report.Comparable || !slices.Contains(report.Reasons, "candidate-memory-evidence-missing") || !slices.Contains(report.Reasons, "candidate-five-trials-required") {
		t.Fatalf("reasons=%v", report.Reasons)
	}
}

func TestBuildAMDScoreboardFixed128BindsIgnoreEOSAndPhysicalObservation(t *testing.T) {
	in := validAMDScoreboardInput()
	observedTrue, observedFalse := true, false
	bindFixed128 := func(arm *AMDArmReceipt) {
		packet := *arm.PromptPacket
		packet.StopTokens = nil
		packet.StopTokenIDs = nil
		packet.GenerationControls.StopTokens = nil
		packet.GenerationControls.StopTokenIDs = nil
		packet.GenerationControls.MaxOutputTokens = 128
		packet.GenerationControls.IgnoreEOS = true
		packet.PacketDigest = ""
		var err error
		packet, err = FreezePromptPacket(packet)
		if err != nil {
			t.Fatal(err)
		}
		arm.PromptPacket = &packet
		arm.PromptPacketDigest = packet.PacketDigest
		arm.StopTokens = nil
		arm.StopTokenIDs = nil
		arm.DecodeTokens = 128
		arm.IgnoreEOS = true
		fixedLogits := make([]float64, 128)
		for j := range fixedLogits {
			fixedLogits[j] = -1
		}
		for i := range arm.Trials {
			arm.Trials[i].OutputTokenIDs = make([]int, 128)
			arm.Trials[i].Logits = slices.Clone(fixedLogits)
			arm.Trials[i].WarmDecodeSeconds = float64(128) / arm.Trials[i].WarmDecodeTokensPerSecond
			if arm.Trials[i].NativeInferenceReceipt != nil {
				arm.Trials[i].NativeInferenceReceipt.DecodeSeconds = arm.Trials[i].WarmDecodeSeconds
				arm.Trials[i].NativeInferenceReceipt.TokenIDs = make([]int, 128)
				arm.Trials[i].NativeInferenceReceipt.TokenLogprobs = slices.Clone(fixedLogits)
			}
			arm.Trials[i].ObservedIgnoreEOS = &observedTrue
			arm.Trials[i].EOSStopped = &observedFalse
			arm.Trials[i].AcceptedOutputTokens = 128
		}
	}
	bindFixed128(&in.Candidate)
	bindFixed128(&in.Reference)

	withoutIgnoreEOS := *in.Candidate.PromptPacket
	withoutIgnoreEOS.GenerationControls.IgnoreEOS = false
	withoutIgnoreEOS.PacketDigest = ""
	withoutIgnoreEOS, err := FreezePromptPacket(withoutIgnoreEOS)
	if err != nil {
		t.Fatal(err)
	}
	if withoutIgnoreEOS.PacketDigest == in.Candidate.PromptPacketDigest {
		t.Fatal("ignore_eos toggle did not change the frozen packet digest")
	}

	if report := BuildAMDScoreboard(in); !report.Comparable {
		t.Fatalf("fully observed fixed-128 cell rejected: %v", report.Reasons)
	}

	tests := []struct {
		name       string
		wantReason string
		mutate     func(*AMDScoreboardInput)
	}{
		{"requested false", "candidate-fixed-128-ignore-eos-required", func(got *AMDScoreboardInput) {
			for _, arm := range []*AMDArmReceipt{&got.Candidate, &got.Reference} {
				packet := *arm.PromptPacket
				packet.GenerationControls.IgnoreEOS = false
				packet.PacketDigest = ""
				packet, err = FreezePromptPacket(packet)
				if err != nil {
					t.Fatal(err)
				}
				arm.PromptPacket = &packet
				arm.PromptPacketDigest = packet.PacketDigest
				arm.IgnoreEOS = false
			}
		}},
		{"unbound arm request", "candidate-prompt-packet-identity-mismatch", func(got *AMDScoreboardInput) { got.Candidate.IgnoreEOS = false }},
		{"missing ignore-EOS observation", "candidate-fixed-128-ignore-eos-observation-missing", func(got *AMDScoreboardInput) { got.Candidate.Trials[0].ObservedIgnoreEOS = nil }},
		{"false ignore-EOS observation", "candidate-fixed-128-ignore-eos-not-observed", func(got *AMDScoreboardInput) { got.Candidate.Trials[0].ObservedIgnoreEOS = &observedFalse }},
		{"missing EOS-stop observation", "candidate-fixed-128-eos-stopped-observation-missing", func(got *AMDScoreboardInput) { got.Candidate.Trials[0].EOSStopped = nil }},
		{"EOS stopped", "candidate-fixed-128-eos-stopped", func(got *AMDScoreboardInput) { got.Candidate.Trials[0].EOSStopped = &observedTrue }},
		{"short accepted output", "candidate-fixed-128-accepted-output-token-count-mismatch", func(got *AMDScoreboardInput) { got.Candidate.Trials[0].AcceptedOutputTokens = 127 }},
		{"short output IDs", "candidate-fixed-128-output-token-count-mismatch", func(got *AMDScoreboardInput) {
			got.Candidate.Trials[0].OutputTokenIDs = got.Candidate.Trials[0].OutputTokenIDs[:127]
		}},
		{"active stops", "candidate-fixed-128-stop-controls-active", func(got *AMDScoreboardInput) {
			for _, arm := range []*AMDArmReceipt{&got.Candidate, &got.Reference} {
				packet := *arm.PromptPacket
				packet.StopTokens = []string{"stop"}
				packet.GenerationControls.StopTokens = []string{"stop"}
				packet.PacketDigest = ""
				packet, err = FreezePromptPacket(packet)
				if err != nil {
					t.Fatal(err)
				}
				arm.PromptPacket = &packet
				arm.PromptPacketDigest = packet.PacketDigest
				arm.StopTokens = []string{"stop"}
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := in
			got.Candidate.Trials = slices.Clone(in.Candidate.Trials)
			got.Reference.Trials = slices.Clone(in.Reference.Trials)
			tc.mutate(&got)
			report := BuildAMDScoreboard(got)
			if report.Comparable || !slices.Contains(report.Reasons, tc.wantReason) {
				t.Fatalf("fixed-128 refusal = comparable:%t reasons:%v, want %q", report.Comparable, report.Reasons, tc.wantReason)
			}
		})
	}

	receipt := &model.NativeInferenceReceipt{Engine: "inkernel", Planner: "inkernel", Owner: "fak", ForwardPath: "test-vulkan", Backend: "vulkan", TokenIDs: []int{1}, TokenLogprobs: []float64{-1}}
	captured, err := CaptureAMDScoreboardTrial(1, receipt, 1, 1, 1, 1, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if captured.ObservedIgnoreEOS != nil || captured.EOSStopped != nil || captured.AcceptedOutputTokens != 0 {
		t.Fatalf("trial capture synthesized physical EOS/output observation: %+v", captured)
	}
}

func validAMDScoreboardInput() AMDScoreboardInput {
	sha := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	prompt := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	packet, err := FreezePromptPacket(PromptTokenPacket{
		Schema: PromptTokenPacketSchema, PacketID: "amd-scoreboard-test", ArtifactSHA256: sha,
		TokenizerIdentity: "test-tokenizer", TokenizerDigest: strings.Repeat("b", 64), TemplateDigest: strings.Repeat("c", 64),
		PromptTokenIDs: []int{1, 2, 3}, StopTokens: []string{"stop"}, StopTokenIDs: []int{4},
		ContextBudget:      ContextBudget{ContextTokens: 256, ContextBudgetBytes: 1 << 30},
		GenerationControls: GenerationControls{Temperature: 0, TopP: 1, MaxOutputTokens: 4, StopTokens: []string{"stop"}, StopTokenIDs: []int{4}},
	})
	if err != nil {
		panic(err)
	}
	arm := AMDArmReceipt{Name: "fak", Engine: "fak-native", Backend: "vulkan", Runtime: "native", ArtifactSHA256: sha, PromptSHA256: prompt, PromptTokenIDs: []int{1, 2, 3}, ContextTokens: 256, ContextBudgetBytes: 1 << 30, KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUMemoryBudget: 6 << 30, HostSpillPolicy: "bounded", Temperature: 0, PrefillTokens: 17, DecodeTokens: 4, Hardware: "AMD Radeon RX 7600 / driver 26.8.1", SoftwareRevision: "internal/compute@r212+gfc6393fe90", BuildFlags: []string{"vulkan"}, PeakRSSBytes: 20 << 30, PeakVRAMBytes: 6 << 30, ResidentModelBytes: 1 << 30, TokenizerDigest: packet.TokenizerDigest, TemplateDigest: packet.TemplateDigest, PromptPacketDigest: packet.PacketDigest, StopTokens: slices.Clone(packet.StopTokens), StopTokenIDs: slices.Clone(packet.StopTokenIDs), TopP: 1, TopK: packet.GenerationControls.TopK, PromptPacket: &packet}
	arm.PrefillTokens = len(arm.PromptTokenIDs)
	for i := 1; i <= 5; i++ {
		seq := 2*i - 1
		if i%2 == 0 {
			seq++
		}
		arm.Trials = append(arm.Trials, AMDScoreboardTrial{EvidenceKind: "selected-token-logprobs", Repetition: i, Sequence: seq, ColdSetupSeconds: 300, PrefillSeconds: .1, PrefillTokensPerSecond: 30, WarmDecodeSeconds: .1, WarmDecodeTokensPerSecond: 40, OutputTokenIDs: []int{4, 5, 6, 7}, Logits: []float64{-1, -2, -3, -4}, H2DBytes: 1, D2HBytes: 1, D2DBytes: 1, QueueSubmissions: 1, NativeInferenceReceipt: &model.NativeInferenceReceipt{Engine: "inkernel", Planner: "inkernel", Owner: "fak", ForwardPath: "test-vulkan", Backend: "vulkan", PrefillSeconds: .1, DecodeSeconds: .1, TokenIDs: []int{4, 5, 6, 7}, TokenLogprobs: []float64{-1, -2, -3, -4}}})
	}
	ref := arm
	refPacket := packet
	ref.PromptPacket = &refPacket
	ref.PromptTokenIDs = slices.Clone(arm.PromptTokenIDs)
	ref.Name = "llama.cpp"
	ref.Engine = "llama.cpp"
	ref.ComparatorOnly = true
	ref.SoftwareRevision = "llama.cpp@50f068ffffc3e0e4c9c2e4139281c6075224f429"
	ref.BuildFlags = []string{"GGML_VULKAN=ON"}
	ref.PeakVRAMBytes = 6 << 30
	ref.ResidentModelBytes = 6 << 30
	ref.Trials = slices.Clone(arm.Trials)
	for i := range ref.Trials {
		ref.Trials[i].Sequence = 4*i + 3 - arm.Trials[i].Sequence
		ref.Trials[i].NativeInferenceReceipt = nil
		ref.Trials[i].PrefillSeconds = .125
		ref.Trials[i].WarmDecodeSeconds = .125
		ref.Trials[i].PrefillTokensPerSecond = 24
		ref.Trials[i].WarmDecodeTokensPerSecond = 32
		ref.Trials[i].Logits = slices.Clone(ref.Trials[i].Logits)
	}
	return AMDScoreboardInput{Schema: AMDScoreboardInputSchema, Concurrency: 1, LogitTolerance: 1e-3, Candidate: arm, Reference: ref}
}

func TestAMDStatisticalContract(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		candidate, reference   []float64
		absoluteLCB, pairedLCB float64
		absolute, paired       bool
	}{
		{"positive", []float64{19.6, 19.8, 20, 20.2, 20.4}, []float64{16, 16, 16, 16, 16}, 19.698511336187536, 1.231156958511721, true, true},
		{"absolute failure", []float64{15.9, 16.7, 16.8, 16.9, 17}, []float64{14, 14.1, 14, 14.1, 14}, 16.241158562453133, 1.1576629436966597, false, true},
		{"paired failure", []float64{20, 20.1, 19.9, 20.05, 19.95}, []float64{19.8, 20.2, 19.8, 20.2, 19.8}, 19.924627834046884, .9946369617100597, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := validAMDScoreboardInput()
			for i := range tc.candidate {
				setAMDTrialRate(&in.Candidate, i, tc.candidate[i])
				setAMDTrialRate(&in.Reference, i, tc.reference[i])
			}
			r := BuildAMDScoreboard(in)
			if !r.PairedComparable || r.Statistics == nil {
				t.Fatalf("reasons=%v", r.Reasons)
			}
			s := r.Statistics
			if math.Abs(s.CandidateLCB95-tc.absoluteLCB) > 1e-12 || math.Abs(s.PairedRatioLCB95-tc.pairedLCB) > 1e-12 || (s.CandidateLCB95 > frozenStrixBarAMD(1)) != tc.absolute || s.PairedPass != tc.paired {
				t.Fatalf("stats=%+v", s)
			}
			if r.StrixAbsoluteEligible || r.OverallWin || s.AbsolutePass || s.AbsoluteBar != 0 {
				t.Fatal("generic fixture earned Strix credit")
			}
		})
	}
	for n := 0; n < 5; n++ {
		if _, _, _, ok := oneSided95LCBAMDChecked(make([]float64, n)); ok {
			t.Fatalf("accepted n=%d", n)
		}
	}
	_, cv, _, ok := oneSided95LCBAMDChecked([]float64{19, 19, 20, 21, 21})
	if !ok || !cvAdmittedAMD(cv) {
		t.Fatalf("exact 5 percent rejected: %.18g", cv)
	}
	if cvAdmittedAMD(.0500001) {
		t.Fatal("CV above 5 percent accepted")
	}
	for _, v := range [][]float64{{1, 1, 1, 1, math.NaN()}, {1, 1, 1, 1, math.Inf(1)}, {1, 1, 1, 1, math.MaxFloat64}} {
		if _, _, _, ok := oneSided95LCBAMDChecked(v); ok {
			t.Fatalf("invalid vector accepted: %v", v)
		}
	}
}

func setAMDTrialRate(arm *AMDArmReceipt, i int, tps float64) {
	t := &arm.Trials[i]
	t.PrefillSeconds = float64(arm.DecodeTokens) / tps / 2
	t.WarmDecodeSeconds = t.PrefillSeconds
	t.PrefillTokensPerSecond = float64(arm.PrefillTokens) / t.PrefillSeconds
	t.WarmDecodeTokensPerSecond = float64(arm.DecodeTokens) / t.WarmDecodeSeconds
	if t.NativeInferenceReceipt != nil {
		t.NativeInferenceReceipt.PrefillSeconds = t.PrefillSeconds
		t.NativeInferenceReceipt.DecodeSeconds = t.WarmDecodeSeconds
	}
}

func TestAMDScoreboardV2FailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*AMDScoreboardInput)
	}{
		{"four trials", func(in *AMDScoreboardInput) {
			in.Candidate.Trials = in.Candidate.Trials[:4]
			in.Reference.Trials = in.Reference.Trials[:4]
		}},
		{"nonalternating", func(in *AMDScoreboardInput) { in.Candidate.Trials[1].Sequence-- }},
		{"native timing laundering", func(in *AMDScoreboardInput) {
			in.Candidate.Trials[0].PrefillSeconds /= 2
			in.Candidate.Trials[0].PrefillTokensPerSecond *= 2
		}},
		{"missing native receipt", func(in *AMDScoreboardInput) { in.Candidate.Trials[0].NativeInferenceReceipt = nil }},
		{"one byte resource breach", func(in *AMDScoreboardInput) { in.Reference.PeakVRAMBytes = in.Reference.GPUMemoryBudget + 1 }},
		{"overflow", func(in *AMDScoreboardInput) {
			in.Reference.Trials[0].PrefillSeconds = math.MaxFloat64
			in.Reference.Trials[0].WarmDecodeSeconds = math.MaxFloat64
		}},
		{"unstable", func(in *AMDScoreboardInput) { setAMDTrialRate(&in.Candidate, 0, 10) }},
		{"serial c4", func(in *AMDScoreboardInput) { in.Concurrency = 4 }},
		{"serial c8", func(in *AMDScoreboardInput) { in.Concurrency = 8 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := validAMDScoreboardInput()
			tc.mutate(&in)
			r := BuildAMDScoreboard(in)
			if r.Comparable || r.PairedComparable || r.OverallWin || r.Statistics != nil || r.ReferenceOverCandidate != nil {
				t.Fatalf("credit leaked: %+v", r)
			}
			if len(r.RawInput.Candidate.Trials) != len(in.Candidate.Trials) {
				t.Fatal("raw trial lost")
			}
			if err := ValidateAMDScoreboardReport(r); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, mutate := range []func(*AMDScoreboardReport){
		func(r *AMDScoreboardReport) { r.OverallWin = true },
		func(r *AMDScoreboardReport) { r.StrixAbsoluteEligible = true },
		func(r *AMDScoreboardReport) { r.Statistics.CandidateLCB95++ },
		func(r *AMDScoreboardReport) { r.Candidate.GPUMemoryBudget++ },
		func(r *AMDScoreboardReport) { r.RawInput.Candidate.Trials[0].PrefillSeconds /= 2 },
	} {
		r := BuildAMDScoreboard(validAMDScoreboardInput())
		mutate(&r)
		if ValidateAMDScoreboardReport(r) == nil {
			t.Fatal("tampered report accepted")
		}
	}
}

func TestAMDScoreboardCanonicalArithmeticAndRawSnapshot(t *testing.T) {
	in := validAMDScoreboardInput()
	original := BuildAMDScoreboard(in)
	for i := range in.Candidate.Trials {
		// Redundant rates may round slightly, but cannot shift a strict bound.
		in.Candidate.Trials[i].PrefillTokensPerSecond *= 1 + 1e-10
		in.Candidate.Trials[i].WarmDecodeTokensPerSecond *= 1 + 1e-10
	}
	rounded := BuildAMDScoreboard(in)
	if !rounded.Comparable || !reflect.DeepEqual(original.Statistics, rounded.Statistics) || !reflect.DeepEqual(original.ReferenceOverCandidate, rounded.ReferenceOverCandidate) {
		t.Fatal("redundant rates changed canonical statistical credit")
	}
	in.Candidate.Trials[0].OutputTokenIDs[0]++
	in.Candidate.Trials[0].NativeInferenceReceipt.TokenIDs[0]++
	if err := ValidateAMDScoreboardReport(original); err != nil {
		t.Fatalf("caller mutation rewrote report snapshot: %v", err)
	}
	for _, concurrency := range []int{1, 4, 8} {
		bar := frozenStrixBarAMD(concurrency)
		_, _, lcb, ok := oneSided95LCBAMDChecked([]float64{bar, bar, bar, bar, bar})
		if !ok || lcb != bar || lcb > bar {
			t.Fatalf("strict absolute boundary c%d: %g", concurrency, lcb)
		}
	}
	_, _, lcb, ok := oneSided95LCBAMDChecked([]float64{1, 1, 1, 1, 1})
	if !ok || lcb != 1 || lcb > 1 {
		t.Fatalf("strict paired boundary: %g", lcb)
	}
	in = validAMDScoreboardInput()
	in.Candidate.Trials[0].PrefillSeconds /= 2
	in.Candidate.Trials[0].PrefillTokensPerSecond *= 2
	if r := BuildAMDScoreboard(in); !slices.Contains(r.Reasons, "candidate-native-timing-mismatch") {
		t.Fatalf("missing native timing refusal: %v", r.Reasons)
	}
}

func TestBuildAMDScoreboardRequiresTransferAndSubmissionAccounting(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Candidate.Trials[0].QueueSubmissions = 0
	report := BuildAMDScoreboard(in)
	if report.Comparable || !slices.Contains(report.Reasons, "candidate-transfer-or-submission-accounting-missing") {
		t.Fatalf("report=%+v", report)
	}
}

func TestBuildAMDScoreboardBindsPlacementEnvelope(t *testing.T) {
	in := validAMDScoreboardInput()
	in.Reference.GPUMemoryBudget++
	in.Reference.HostSpillPolicy = "unbounded"
	report := BuildAMDScoreboard(in)
	if report.Comparable || !slices.Contains(report.Reasons, "memory-placement-envelope-mismatch") {
		t.Fatalf("report=%+v", report)
	}
}
