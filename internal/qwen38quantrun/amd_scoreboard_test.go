package qwen38quantrun

import (
	"slices"
	"strings"
	"testing"
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
	if got, want := report.ReferenceOverCandidate.Decode, 17.2; got != want {
		t.Fatalf("decode ratio=%v want %v", got, want)
	}
	if got, want := report.ReferenceOverCandidate.Prefill, 12.0; got != want {
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
	if report.Comparable || !slices.Contains(report.Reasons, "candidate-memory-evidence-missing") || !slices.Contains(report.Reasons, "candidate-three-trials-required") {
		t.Fatalf("reasons=%v", report.Reasons)
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
	for i := 1; i <= 3; i++ {
		arm.Trials = append(arm.Trials, AMDScoreboardTrial{Repetition: i, ColdSetupSeconds: 300, PrefillSeconds: 10, PrefillTokensPerSecond: .5, WarmDecodeSeconds: 60, WarmDecodeTokensPerSecond: .065, OutputTokenIDs: []int{4, 5, 6, 7}, Logits: []float64{1, 2}, H2DBytes: 1, D2HBytes: 1, D2DBytes: 1, QueueSubmissions: 1})
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
	ref.PeakVRAMBytes = 7 << 30
	ref.ResidentModelBytes = 6 << 30
	ref.Trials = slices.Clone(arm.Trials)
	for i := range ref.Trials {
		ref.Trials[i].PrefillTokensPerSecond = 6
		ref.Trials[i].WarmDecodeTokensPerSecond = 1.118
		ref.Trials[i].Logits = slices.Clone(ref.Trials[i].Logits)
	}
	return AMDScoreboardInput{Schema: AMDScoreboardInputSchema, LogitTolerance: 1e-3, Candidate: arm, Reference: ref}
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
