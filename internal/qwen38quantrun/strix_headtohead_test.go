package qwen38quantrun

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func strixHeadToHeadTestPacket(t *testing.T) PromptTokenPacket {
	t.Helper()
	sha := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	packet, err := FreezePromptPacket(PromptTokenPacket{
		Schema: PromptTokenPacketSchema, PacketID: "strix-headtohead-test", ArtifactSHA256: sha,
		TokenizerIdentity: "test-tokenizer", TokenizerDigest: strings.Repeat("b", 64), TemplateDigest: strings.Repeat("c", 64),
		PromptTokenIDs: []int{1, 2, 3}, StopTokens: []string{"stop"}, StopTokenIDs: []int{4},
		ContextBudget:      ContextBudget{ContextTokens: 256, ContextBudgetBytes: 1 << 30},
		GenerationControls: GenerationControls{Temperature: 0, TopP: 1, MaxOutputTokens: 4, StopTokens: []string{"stop"}, StopTokenIDs: []int{4}},
	})
	if err != nil {
		t.Fatalf("freeze packet: %v", err)
	}
	return packet
}

// strixHeadToHeadTestArm produces one valid single-trial arm run bound to the
// shared packet. Candidate and reference differ only in identity and measured
// rates, matching the AMD scoreboard's own fixture contract.
func strixHeadToHeadTestArm(packet PromptTokenPacket, role string, pairIndex, repetition int) AMDArmReceipt {
	sha := packet.ArtifactSHA256
	arm := AMDArmReceipt{
		Name: "fak", Engine: "fak-native", Backend: "vulkan", Runtime: "native",
		ArtifactSHA256: sha, PromptSHA256: strings.Repeat("a", 64), PromptTokenIDs: slices.Clone(packet.PromptTokenIDs),
		ContextTokens: packet.ContextBudget.ContextTokens, ContextBudgetBytes: packet.ContextBudget.ContextBudgetBytes,
		KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUMemoryBudget: 6 << 30,
		HostSpillPolicy: "bounded", Temperature: 0, PrefillTokens: len(packet.PromptTokenIDs), DecodeTokens: 4,
		Hardware: "AMD Radeon 8060S / driver 26.8.1", SoftwareRevision: "internal/compute@r212+gfc6393fe90",
		BuildFlags: []string{"vulkan"}, PeakRSSBytes: 20 << 30, PeakVRAMBytes: 6 << 30, ResidentModelBytes: 1 << 30,
		TokenizerDigest: packet.TokenizerDigest, TemplateDigest: packet.TemplateDigest,
		PromptPacketDigest: packet.PacketDigest, StopTokens: slices.Clone(packet.StopTokens),
		StopTokenIDs: slices.Clone(packet.StopTokenIDs), TopP: 1, TopK: packet.GenerationControls.TopK,
		IgnoreEOS: packet.GenerationControls.IgnoreEOS,
	}
	bound := packet
	arm.PromptPacket = &bound
	tokens := []int{4, 5, 6, 7}
	logits := []float64{-1, -2, -3, -4}
	trial := AMDScoreboardTrial{
		EvidenceKind: "selected-token-logprobs", Repetition: repetition, Sequence: 2*repetition - 1,
		ColdSetupSeconds: 300, PrefillSeconds: .1, PrefillTokensPerSecond: 30,
		WarmDecodeSeconds: .1, WarmDecodeTokensPerSecond: 40,
		OutputTokenIDs: slices.Clone(tokens), Logits: slices.Clone(logits),
		H2DBytes: 1, D2HBytes: 1, D2DBytes: 1, QueueSubmissions: 1,
		NativeInferenceReceipt: &model.NativeInferenceReceipt{Engine: "inkernel", Planner: "inkernel", Owner: "fak", ForwardPath: "test-vulkan", Backend: "vulkan", PrefillSeconds: .1, DecodeSeconds: .1, TokenIDs: slices.Clone(tokens), TokenLogprobs: slices.Clone(logits)},
	}
	if role == "reference" {
		arm.Name = "llama.cpp"
		arm.Engine = "llama.cpp"
		arm.ComparatorOnly = true
		arm.SoftwareRevision = "llama.cpp@50f068ffffc3e0e4c9c2e4139281c6075224f429"
		arm.BuildFlags = []string{"GGML_VULKAN=ON"}
		arm.ResidentModelBytes = 6 << 30
		trial.PrefillSeconds = .125
		trial.WarmDecodeSeconds = .125
		trial.PrefillTokensPerSecond = 24
		trial.WarmDecodeTokensPerSecond = 32
		trial.NativeInferenceReceipt = nil
	}
	arm.Trials = []AMDScoreboardTrial{trial}
	_ = pairIndex
	return arm
}

func TestRunStrixHeadToHeadInterleavesAndHoldsLease(t *testing.T) {
	packet := strixHeadToHeadTestPacket(t)
	leasePath := filepath.Join(t.TempDir(), "fak-gpu.lease")

	var calls []StrixArmCall
	var leaseBusyDuringCall []bool
	var packetDrift bool
	runner := func(_ context.Context, call StrixArmCall) (AMDArmReceipt, error) {
		calls = append(calls, call)
		if call.Packet.PacketDigest != packet.PacketDigest {
			packetDrift = true
		}
		_, err := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
		leaseBusyDuringCall = append(leaseBusyDuringCall, errors.Is(err, gpulease.ErrBusy))
		rep := 1
		if call.PairIndex >= 0 {
			rep = call.PairIndex + 1
		}
		return strixHeadToHeadTestArm(packet, call.Role, call.PairIndex, rep), nil
	}

	res, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{
		Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath,
	}, runner)
	if err != nil {
		t.Fatalf("RunStrixHeadToHead: %v", err)
	}
	if packetDrift {
		t.Fatal("scheduler passed a packet other than the admitted packet")
	}
	if len(res.Cells) != 3 || res.Cells[0].Concurrency != 1 || res.Cells[1].Concurrency != 4 || res.Cells[2].Concurrency != 8 {
		t.Fatalf("cells=%+v", res.Cells)
	}

	// Every call observed a busy canonical lease; the lease is free afterwards.
	if len(leaseBusyDuringCall) != len(calls) || slices.Contains(leaseBusyDuringCall, false) {
		t.Fatalf("lease not held across every call: busy=%v calls=%d", leaseBusyDuringCall, len(calls))
	}
	free, err := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
	if err != nil {
		t.Fatalf("lease not released after completion: %v", err)
	}
	free.Release()

	// Exact per-cell call counts and alternating pair order.
	i := 0
	for _, c := range res.Cells {
		for w := 0; w < StrixComparisonWarmups; w++ {
			if calls[i].Role != "candidate" || !calls[i].Warmup || calls[i].Concurrency != c.Concurrency {
				t.Fatalf("warmup call %d = %+v", i, calls[i])
			}
			i++
			if calls[i].Role != "reference" || !calls[i].Warmup || calls[i].Concurrency != c.Concurrency {
				t.Fatalf("warmup call %d = %+v", i, calls[i])
			}
			i++
		}
		for p := 0; p < StrixComparisonMinimumMeasuredPairs; p++ {
			wantFirst, wantSecond := "candidate", "reference"
			if p%2 == 1 {
				wantFirst, wantSecond = "reference", "candidate"
			}
			if calls[i].Role != wantFirst || calls[i].Warmup || calls[i].PairIndex != p {
				t.Fatalf("pair %d first call = %+v want %s", p, calls[i], wantFirst)
			}
			i++
			if calls[i].Role != wantSecond || calls[i].Warmup || calls[i].PairIndex != p {
				t.Fatalf("pair %d second call = %+v want %s", p, calls[i], wantSecond)
			}
			i++
		}
	}
	if i != len(calls) {
		t.Fatalf("observed %d calls, consumed %d", len(calls), i)
	}

	// Each cell carries the canonical number of accumulated measured trials per
	// arm (one per measured pair) and a valid report.
	for _, c := range res.Cells {
		if len(c.Candidate.Trials) != StrixComparisonMinimumMeasuredPairs || len(c.Reference.Trials) != StrixComparisonMinimumMeasuredPairs {
			t.Fatalf("c=%d trials candidate=%d reference=%d", c.Concurrency, len(c.Candidate.Trials), len(c.Reference.Trials))
		}
		if err := ValidateAMDScoreboardReport(c.Report); err != nil {
			t.Fatalf("c=%d invalid report: %v", c.Concurrency, err)
		}
	}
	if !res.Cells[0].Report.Comparable {
		t.Fatalf("c=1 report not comparable: %+v", res.Cells[0].Report)
	}
	for _, c := range res.Cells[1:] {
		if c.Report.Comparable {
			t.Fatalf("c=%d manufactured concurrency credit: %+v", c.Concurrency, c.Report)
		}
		if !slices.Contains(c.Report.Reasons, "concurrent-request-observation-required") {
			t.Fatalf("c=%d missing concurrency refusal: %v", c.Concurrency, c.Report.Reasons)
		}
	}
}

func TestRunStrixHeadToHeadFailsClosed(t *testing.T) {
	packet := strixHeadToHeadTestPacket(t)
	leasePath := filepath.Join(t.TempDir(), "fak-gpu.lease")

	base := func() StrixArmRunner {
		return func(_ context.Context, call StrixArmCall) (AMDArmReceipt, error) {
			return strixHeadToHeadTestArm(packet, call.Role, call.PairIndex, 1), nil
		}
	}

	t.Run("nil runner", func(t *testing.T) {
		if _, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath}, nil); !errors.Is(err, ErrStrixHeadToHead) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("unfrozen packet", func(t *testing.T) {
		unfrozen := packet
		unfrozen.PacketDigest = ""
		if _, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: unfrozen, LogitTolerance: 1e-3, LeasePath: leasePath}, base()); !errors.Is(err, ErrStrixHeadToHead) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("too few pairs", func(t *testing.T) {
		if _, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath, MeasuredPairs: 4}, base()); !errors.Is(err, ErrStrixHeadToHead) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("non-increasing cells", func(t *testing.T) {
		if _, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath, ConcurrencyCells: []int{4, 1}}, base()); !errors.Is(err, ErrStrixHeadToHead) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("arm failure stops with no result", func(t *testing.T) {
		failing := func(_ context.Context, call StrixArmCall) (AMDArmReceipt, error) {
			if call.Role == "reference" && !call.Warmup {
				return AMDArmReceipt{}, errors.New("arm unavailable")
			}
			return strixHeadToHeadTestArm(packet, call.Role, call.PairIndex, 1), nil
		}
		res, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath}, failing)
		if !errors.Is(err, ErrStrixHeadToHead) || res != nil {
			t.Fatalf("res=%v err=%v", res, err)
		}
	})
	t.Run("empty trial refused", func(t *testing.T) {
		empty := func(_ context.Context, call StrixArmCall) (AMDArmReceipt, error) {
			arm := strixHeadToHeadTestArm(packet, call.Role, call.PairIndex, 1)
			arm.Trials = nil
			return arm, nil
		}
		if _, err := RunStrixHeadToHead(context.Background(), StrixHeadToHeadConfig{Packet: packet, LogitTolerance: 1e-3, LeasePath: leasePath}, empty); !errors.Is(err, ErrStrixHeadToHead) {
			t.Fatalf("err=%v", err)
		}
	})
}
