package qwen38quantrun

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// strixConcurrentBatchTestSlots builds exactly `c` fully-overlapping slot
// observations for one arm. Every slot is admitted on the shared monotonic
// clock at admission and completes at completion, so max(admission) <
// min(completion) holds by construction. The slots carry the fixed-128
// accepted-token identity the scoreboard's c>1 path requires.
func strixConcurrentBatchTestSlots(c int, admission, completion float64, accepted int) []StrixConcurrentBatchSlotObservation {
	tokens := make([]int, accepted)
	logprobs := make([]float64, accepted)
	for i := range tokens {
		tokens[i] = i
		logprobs[i] = -1
	}
	yes, no := true, false
	slots := make([]StrixConcurrentBatchSlotObservation, 0, c)
	for i := 0; i < c; i++ {
		slots = append(slots, StrixConcurrentBatchSlotObservation{
			RequestID:              "req-" + string(rune('a'+i)),
			AdmissionSec:           admission,
			CompletionSec:          completion,
			AcceptedOutputTokenIDs: slices.Clone(tokens),
			AcceptedTokenLogprobs:  slices.Clone(logprobs),
			ObservedIgnoreEOS:      &yes,
			EOSStopped:             &no,
			PrefillSeconds:         0.5,
			DecodeSeconds:          1.0,
			PrefillTokens:          32,
			H2DBytes:               1,
			D2HBytes:               1,
			D2DBytes:               1,
			QueueSubmissions:       1,
			PeakRSSBytes:           20 << 30,
			PeakVRAMBytes:          6 << 30,
			ResidentModelBytes:     1 << 30,
		})
	}
	return slots
}

// strixConcurrentBatchTestPair builds one canonical c4/c8 pair observation. The
// candidate and reference differ only in identity, matching the AMD scoreboard's
// own fixture contract; both arms share the packet digest, artifact, and the
// exact overlapping interval.
func strixConcurrentBatchTestPair(c int) StrixConcurrentBatchPairObservation {
	packet := strixHeadToHeadTestPacketForBatch()
	sha := packet.ArtifactSHA256
	candidateArm := AMDArmReceipt{
		Name: "fak", Engine: "fak-native", Backend: "vulkan", Runtime: "native",
		ArtifactSHA256: sha, PromptSHA256: strings.Repeat("a", 64), PromptTokenIDs: slices.Clone(packet.PromptTokenIDs),
		ContextTokens: packet.ContextBudget.ContextTokens, ContextBudgetBytes: packet.ContextBudget.ContextBudgetBytes,
		KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUMemoryBudget: 6 << 30,
		HostSpillPolicy: "bounded", Temperature: 0, PrefillTokens: len(packet.PromptTokenIDs), DecodeTokens: 128,
		Hardware: "AMD Radeon 8060S / driver 26.8.1", SoftwareRevision: "internal/compute@r212+gfc6393fe90",
		BuildFlags: []string{"vulkan"}, PeakRSSBytes: 20 << 30, PeakVRAMBytes: 6 << 30, ResidentModelBytes: 1 << 30,
		TokenizerDigest: packet.TokenizerDigest, TemplateDigest: packet.TemplateDigest,
		PromptPacketDigest: packet.PacketDigest, StopTokens: slices.Clone(packet.StopTokens),
		StopTokenIDs: slices.Clone(packet.StopTokenIDs), TopP: 1, TopK: packet.GenerationControls.TopK,
		IgnoreEOS: packet.GenerationControls.IgnoreEOS,
	}
	bound := packet
	candidateArm.PromptPacket = &bound
	candidateArm.Trials = []AMDScoreboardTrial{{ColdSetupSeconds: 300}}

	referenceArm := candidateArm
	referenceArm.Name = "llama.cpp"
	referenceArm.Engine = "llama.cpp"
	referenceArm.ComparatorOnly = true
	referenceArm.SoftwareRevision = "llama.cpp@50f068ffffc3e0e4c9c2e4139281c6075224f429"
	referenceArm.BuildFlags = []string{"GGML_VULKAN=ON"}

	candidateSlots := strixConcurrentBatchTestSlots(c, 100, 102, 128)
	referenceSlots := strixConcurrentBatchTestSlots(c, 100, 102.25, 128)
	for i := range candidateSlots {
		receipt := &model.NativeInferenceReceipt{
			Engine: "inkernel", Planner: "inkernel", Owner: "fak", ForwardPath: "vulkan/qwen35-gdn-ssm-decode-v1", Backend: "vulkan",
			PrefillSeconds: 0.5, DecodeSeconds: 1.0,
			TokenIDs:      slices.Clone(candidateSlots[i].AcceptedOutputTokenIDs),
			TokenLogprobs: slices.Clone(candidateSlots[i].AcceptedTokenLogprobs),
		}
		candidateSlots[i].NativeInferenceReceipt = receipt
	}

	return StrixConcurrentBatchPairObservation{
		Concurrency:    c,
		PacketDigest:   packet.PacketDigest,
		LogitTolerance: 1e-3,
		Candidate:      StrixConcurrentBatchArmObservation{Arm: candidateArm, Concurrency: c, Slots: candidateSlots},
		Reference:      StrixConcurrentBatchArmObservation{Arm: referenceArm, Concurrency: c, Slots: referenceSlots},
	}
}

// strixConcurrentBatchTestPairs builds the canonical 5-pair set the creditable
// producer requires. Each pair is an identical genuinely-overlapping c4/c8
// observation, so the arm identity is stable across the measured pairs.
func strixConcurrentBatchTestPairs(c int) []StrixConcurrentBatchPairObservation {
	pairs := make([]StrixConcurrentBatchPairObservation, 0, StrixComparisonMinimumMeasuredPairs)
	for i := 0; i < StrixComparisonMinimumMeasuredPairs; i++ {
		pairs = append(pairs, strixConcurrentBatchTestPair(c))
	}
	return pairs
}

func strixHeadToHeadTestPacketForBatch() PromptTokenPacket {
	sha := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	packet, err := FreezePromptPacket(PromptTokenPacket{
		Schema: PromptTokenPacketSchema, PacketID: "strix-concurrent-batch-test", ArtifactSHA256: sha,
		TokenizerIdentity: "test-tokenizer", TokenizerDigest: strings.Repeat("b", 64), TemplateDigest: strings.Repeat("c", 64),
		PromptTokenIDs: []int{1, 2, 3}, StopTokens: []string{"stop"}, StopTokenIDs: []int{4},
		ContextBudget:      ContextBudget{ContextTokens: 32768, ContextBudgetBytes: 1 << 30},
		GenerationControls: GenerationControls{Temperature: 0, TopP: 1, MaxOutputTokens: 128, StopTokens: []string{"stop"}, StopTokenIDs: []int{4}},
	})
	if err != nil {
		panic(err)
	}
	return packet
}

// TestStrixConcurrentBatchPairRequiresOverlappingObservedRequests is the issue's
// named witness. It proves that a genuine c4/c8 batch pair is creditable only
// from complete, overlapping per-request evidence, and that every relabeled,
// serial, duplicated, multiplied, shortened, replayed, or arm-drifted
// observation fails closed with a named refusal.
func TestStrixConcurrentBatchPairRequiresOverlappingObservedRequests(t *testing.T) {
	t.Run("c4 genuine overlap is creditable", func(t *testing.T) {
		pairs := strixConcurrentBatchTestPairs(4)
		report, err := ScoreStrixConcurrentBatchPairs(pairs)
		if err != nil {
			t.Fatalf("ScoreStrixConcurrentBatchPairs: %v", err)
		}
		if report.Concurrency != 4 {
			t.Fatalf("concurrency=%d want 4", report.Concurrency)
		}
		if !report.Comparable {
			t.Fatalf("genuine c4 observation not comparable: %v", report.Reasons)
		}
		if report.Candidate.Requests != 4 || report.Candidate.AcceptedToken != 5*512 {
			t.Fatalf("candidate cell tile=%+v want 4 requests and 2560 tokens across 5 pairs", report.Candidate)
		}
		tile, err := BatchTileFromObservation(pairs[0].Candidate, "candidate", pairs[0].PacketDigest)
		if err != nil {
			t.Fatalf("BatchTileFromObservation: %v", err)
		}
		if tile.Requests != 4 || tile.AcceptedToken != 512 || tile.AggregateTPS != 256 {
			t.Fatalf("single c4 pair tile=%+v want 4 requests, 512 tokens, 256 tok/s", tile)
		}
	})

	t.Run("c8 genuine overlap is creditable", func(t *testing.T) {
		report, err := ScoreStrixConcurrentBatchPairs(strixConcurrentBatchTestPairs(8))
		if err != nil {
			t.Fatalf("ScoreStrixConcurrentBatchPairs: %v", err)
		}
		if !report.Comparable {
			t.Fatalf("genuine c8 observation not comparable: %v", report.Reasons)
		}
		if report.Candidate.Requests != 8 || report.Candidate.AcceptedToken != 5*1024 {
			t.Fatalf("candidate cell tile=%+v want 8 requests and 5120 tokens across 5 pairs", report.Candidate)
		}
		if report.AbsoluteBar != 29.42 {
			t.Fatalf("c8 absolute bar=%v want 29.42", report.AbsoluteBar)
		}
	})

	t.Run("c8 aggregate rate is N over T not a multiplied c1", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(8)
		// Each slot accepts 128 tokens; N = 8*128 = 1024 over the shared 2s
		// interval, so the credited tile rate is 512 tok/s, never a caller's
		// multiplied c1. Prove the producer re-derives it from slots.
		tile, err := BatchTileFromObservation(obs.Candidate, "candidate", obs.PacketDigest)
		if err != nil {
			t.Fatalf("BatchTileFromObservation: %v", err)
		}
		if tile.AcceptedToken != 1024 || tile.Requests != 8 {
			t.Fatalf("tile=%+v want 8 requests and 1024 tokens", tile)
		}
		if got := tile.AggregateTPS; got != 512 {
			t.Fatalf("aggregate tps=%v want 512", got)
		}
	})

	for _, c := range []int{4, 8} {
		c := c
		t.Run("serial slots refused", func(t *testing.T) {
			obs := strixConcurrentBatchTestPair(c)
			// Slot 0 completes before slot 1 is admitted: no shared interval.
			for i := range obs.Candidate.Slots {
				obs.Candidate.Slots[i].AdmissionSec = float64(i)*2 + 1
				obs.Candidate.Slots[i].CompletionSec = float64(i)*2 + 2.5
			}
			_, err := ScoreStrixConcurrentBatchPair(obs)
			assertBatchRefusal(t, err, "slot-not-overlapping")
		})
		t.Run("duplicated request id refused", func(t *testing.T) {
			obs := strixConcurrentBatchTestPair(c)
			obs.Reference.Slots[1].RequestID = obs.Reference.Slots[0].RequestID
			_, err := ScoreStrixConcurrentBatchPair(obs)
			assertBatchRefusal(t, err, "duplicate-request-id")
		})

		t.Run("missing slot refused", func(t *testing.T) {
			obs := strixConcurrentBatchTestPair(c)
			obs.Candidate.Slots = obs.Candidate.Slots[:len(obs.Candidate.Slots)-1]
			_, err := ScoreStrixConcurrentBatchPair(obs)
			assertBatchRefusal(t, err, "slot-count-mismatch")
		})

		t.Run("shortened accepted count refused", func(t *testing.T) {
			obs := strixConcurrentBatchTestPair(c)
			obs.Candidate.Slots[0].AcceptedOutputTokenIDs = obs.Candidate.Slots[0].AcceptedOutputTokenIDs[:127]
			_, err := ScoreStrixConcurrentBatchPair(obs)
			assertBatchRefusal(t, err, "accepted-token-count-mismatch")
		})

		t.Run("caller-authored shortened interval refused", func(t *testing.T) {
			obs := strixConcurrentBatchTestPair(c)
			// A shortened shared completion that cannot contain the slot's own
			// prefill+decode work is a caller-forged interval, not evidence.
			obs.Candidate.Slots[0].CompletionSec = obs.Candidate.Slots[0].AdmissionSec + 0.1
			_, err := ScoreStrixConcurrentBatchPair(obs)
			assertBatchRefusal(t, err, "timing-arithmetic-invalid")
		})
	}

	t.Run("diverging arm tokens refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		// Reference slot 0 emits a different token stream than candidate slot 0.
		obs.Reference.Slots[0].AcceptedOutputTokenIDs[0] = 999
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "accepted-token-count-mismatch")
	})

	t.Run("c1 relabel as c4 refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Concurrency = 1
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "unsupported-concurrency")
	})

	t.Run("c3 unsupported refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Concurrency = 3
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "unsupported-concurrency")
	})

	t.Run("arm identity drift refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Reference.Arm.ArtifactSHA256 = strings.Repeat("d", 64)
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "artifact-mismatch")
	})

	t.Run("packet replay refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Reference.Arm.PromptPacketDigest = strings.Repeat("e", 64)
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "packet-digest-mismatch")
	})

	t.Run("missing eos observation refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Reference.Slots[0].ObservedIgnoreEOS = nil
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "eos-observation-missing")
	})

	t.Run("eos stopped refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		yes := true
		obs.Reference.Slots[0].EOSStopped = &yes
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "eos-stopped-observed")
	})

	t.Run("missing transfer accounting refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Candidate.Slots[0].QueueSubmissions = 0
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "transfer-or-submission-accounting-missing")
	})

	t.Run("non-fak-native candidate refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Candidate.Arm.Engine = "llama.cpp"
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "candidate-not-fak-native-no-fallback")
	})

	t.Run("candidate fallback refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Candidate.Arm.FallbackActive = true
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "candidate-not-fak-native-no-fallback")
	})

	t.Run("non-comparator reference refused", func(t *testing.T) {
		obs := strixConcurrentBatchTestPair(4)
		obs.Reference.Arm.ComparatorOnly = false
		_, err := ScoreStrixConcurrentBatchPair(obs)
		assertBatchRefusal(t, err, "reference-not-explicit-llamacpp-comparator")
	})
}

func assertBatchRefusal(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want refusal %q, got nil", want)
	}
	if !errors.Is(err, ErrStrixConcurrentBatch) {
		t.Fatalf("err %v does not wrap ErrStrixConcurrentBatch", err)
	}
	var refusal *BatchRefusalError
	if !errors.As(err, &refusal) {
		t.Fatalf("err %v is not a *BatchRefusalError", err)
	}
	if refusal.Reason != want {
		t.Fatalf("refusal=%q want %q (detail %q)", refusal.Reason, want, refusal.Detail)
	}
	if !slices.Contains(StrixConcurrentBatchRefusals, refusal.Reason) {
		t.Fatalf("refusal %q is not in the closed vocabulary", refusal.Reason)
	}
}
