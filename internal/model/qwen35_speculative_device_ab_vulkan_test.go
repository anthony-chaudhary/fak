//go:build vulkan && (windows || linux) && cgo

package model_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/format"
	"hash/fnv"
	"math/rand"
	"os"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

const qwen35DeviceVerificationPath = "fak-native/device/qwen3.8-sequence-target-verify-v1"
const qwen35BoundaryRejectPath = "fak-native/boundary-target-reject-v1"

type speculativeABWorkload struct {
	Name   string
	Prompt string
}

type speculativeABSample struct {
	Arm                       string          `json:"arm"`
	Workload                  string          `json:"workload"`
	PromptMode                string          `json:"prompt_mode"`
	PrefixRegime              string          `json:"prefix_regime"`
	PromptSHA256              string          `json:"prompt_sha256"`
	Sample                    int             `json:"sample"`
	PromptTokens              int             `json:"prompt_tokens"`
	OutputTokens              int             `json:"output_tokens"`
	CompletionSHA256          string          `json:"completion_sha256"`
	Completion                string          `json:"completion"`
	TaskCorrect               *bool           `json:"task_correct,omitempty"`
	TaskCorrectReason         string          `json:"task_correct_reason,omitempty"`
	TotalNS                   int64           `json:"total_ns"`
	SetupNS                   int64           `json:"setup_ns"`
	PrefillNS                 int64           `json:"prefill_ns"`
	TTFTNS                    int64           `json:"ttft_ns"`
	DecodeNS                  int64           `json:"decode_ns"`
	TPOTNS                    int64           `json:"tpot_ns"`
	DraftNS                   int64           `json:"draft_ns"`
	TargetVerificationNS      int64           `json:"target_verification_ns"`
	SynchronizationNS         int64           `json:"synchronization_ns"`
	RollbackNS                int64           `json:"rollback_ns"`
	TargetSequenceInvocations int             `json:"target_sequence_invocations"`
	BoundaryRejectRounds      int             `json:"boundary_reject_rounds"`
	OrdinaryTargetSteps       int             `json:"ordinary_target_steps"`
	DraftedTokens             int             `json:"drafted_tokens"`
	AcceptedTokens            int             `json:"accepted_tokens"`
	RejectedTokens            int             `json:"rejected_tokens"`
	ReplayedTokens            int             `json:"replayed_tokens"`
	RecurrentRepairTokens     int             `json:"recurrent_repair_tokens"`
	ZeroAcceptRounds          int             `json:"zero_accept_rounds"`
	PartialAcceptRounds       int             `json:"partial_accept_rounds"`
	FullAcceptRounds          int             `json:"full_accept_rounds"`
	KHistogram                map[int]int     `json:"k_histogram,omitempty"`
	ExpectedD2HLogitsBytes    int64           `json:"expected_d2h_logits_bytes"`
	BackendObservation        bool            `json:"backend_observation"`
	Backend                   string          `json:"backend,omitempty"`
	Device                    string          `json:"device,omitempty"`
	Driver                    string          `json:"driver,omitempty"`
	Runtime                   string          `json:"runtime,omitempty"`
	ComputeDispatches         uint64          `json:"compute_dispatches"`
	DispatchesMeasured        bool            `json:"dispatches_measured"`
	H2DBytes                  uint64          `json:"h2d_bytes"`
	D2HBytes                  uint64          `json:"d2h_bytes"`
	D2DBytes                  uint64          `json:"d2d_bytes"`
	BackendFallbacks          uint64          `json:"backend_fallbacks"`
	PeakMemoryBytes           int64           `json:"peak_memory_bytes"`
	MemoryMeasured            bool            `json:"memory_measured"`
	SourceRevision            string          `json:"source_revision"`
	SourceModified            bool            `json:"source_modified"`
	TimestampUTC              string          `json:"timestamp_utc"`
	Output                    []int           `json:"-"`
	Receipts                  []receiptSample `json:"receipts,omitempty"`
}

type receiptSample struct {
	DraftTokens           int    `json:"draft_tokens"`
	AcceptedTokens        int    `json:"accepted_tokens"`
	Path                  string `json:"path"`
	OneOperation          bool   `json:"one_operation"`
	TargetOperations      int    `json:"target_operations"`
	WallNS                int64  `json:"wall_ns"`
	ReceiptNS             int64  `json:"receipt_ns"`
	SynchronizationNS     int64  `json:"synchronization_ns"`
	RollbackNS            int64  `json:"rollback_ns"`
	FullTargetReplaySteps int    `json:"full_target_replay_steps"`
	RecurrentRepairTokens int    `json:"recurrent_repair_tokens"`
}

type speculativeABSummary struct {
	Arm                string        `json:"arm"`
	Workload           string        `json:"workload"`
	Samples            int           `json:"samples"`
	TotalP50NS         int64         `json:"total_p50_ns"`
	TotalP90NS         int64         `json:"total_p90_ns"`
	TTFTP50NS          int64         `json:"ttft_p50_ns"`
	TTFTP90NS          int64         `json:"ttft_p90_ns"`
	DecodeP50NS        int64         `json:"decode_p50_ns"`
	DecodeP90NS        int64         `json:"decode_p90_ns"`
	TargetInvocations  int           `json:"target_sequence_invocations"`
	BoundaryRejects    int           `json:"boundary_reject_rounds"`
	OrdinarySteps      int           `json:"ordinary_target_steps"`
	DraftedTokens      int           `json:"drafted_tokens"`
	AcceptedTokens     int           `json:"accepted_tokens"`
	RejectedTokens     int           `json:"rejected_tokens"`
	FullReplaySteps    int           `json:"full_target_replay_steps"`
	RecurrentRepairs   int           `json:"recurrent_repair_tokens"`
	SynchronizationNS  int64         `json:"synchronization_ns"`
	RollbackNS         int64         `json:"rollback_ns"`
	ExpectedD2HBytes   int64         `json:"expected_d2h_logits_bytes"`
	ObservedD2HBytes   uint64        `json:"observed_d2h_bytes"`
	Fallbacks          uint64        `json:"backend_fallbacks"`
	DispatchesMeasured bool          `json:"dispatches_measured"`
	CI95Measured       bool          `json:"ci95_measured"`
	TotalP50CI95       *durationCI95 `json:"total_p50_ci95,omitempty"`
	TotalP90CI95       *durationCI95 `json:"total_p90_ci95,omitempty"`
	TTFTP50CI95        *durationCI95 `json:"ttft_p50_ci95,omitempty"`
	TTFTP90CI95        *durationCI95 `json:"ttft_p90_ci95,omitempty"`
	DecodeP50CI95      *durationCI95 `json:"decode_p50_ci95,omitempty"`
	DecodeP90CI95      *durationCI95 `json:"decode_p90_ci95,omitempty"`
}

type durationCI95 struct {
	LowNS  int64 `json:"low_ns"`
	HighNS int64 `json:"high_ns"`
}

type speculativeABPairedSummary struct {
	Workload           string        `json:"workload"`
	Samples            int           `json:"samples"`
	DeltaDefinition    string        `json:"delta_definition"`
	TotalP50DeltaNS    int64         `json:"total_p50_delta_ns"`
	TotalP90DeltaNS    int64         `json:"total_p90_delta_ns"`
	DecodeP50DeltaNS   int64         `json:"decode_p50_delta_ns"`
	DecodeP90DeltaNS   int64         `json:"decode_p90_delta_ns"`
	CI95Measured       bool          `json:"ci95_measured"`
	TotalP50DeltaCI95  *durationCI95 `json:"total_p50_delta_ci95,omitempty"`
	TotalP90DeltaCI95  *durationCI95 `json:"total_p90_delta_ci95,omitempty"`
	DecodeP50DeltaCI95 *durationCI95 `json:"decode_p50_delta_ci95,omitempty"`
	DecodeP90DeltaCI95 *durationCI95 `json:"decode_p90_delta_ci95,omitempty"`
}

func TestSpeculativeABBootstrapCI95IsDeterministicAndN1IsUnmeasured(t *testing.T) {
	values := []int64{10, 20, 30, 40, 50}
	seed := bootstrapSeed("ordinary", "copy-heavy-code-edit", "total-p50")
	first := bootstrapDurationCI95(values, 0.50, seed)
	second := bootstrapDurationCI95(values, 0.50, seed)
	if *first != *second || first.LowNS > 30 || first.HighNS < 30 {
		t.Fatalf("deterministic median CI=%+v second=%+v, want matching interval containing 30", first, second)
	}
	one := summarizeAB("ordinary", "diagnostic", []speculativeABSample{{TotalNS: 10, TTFTNS: 4, DecodeNS: 6}})
	if one.CI95Measured || one.TotalP50CI95 != nil || one.TotalP90CI95 != nil || one.TTFTP50CI95 != nil || one.TTFTP90CI95 != nil || one.DecodeP50CI95 != nil || one.DecodeP90CI95 != nil {
		t.Fatalf("N=1 summary fabricated confidence interval: %+v", one)
	}
	paired := summarizePairedAB("deterministic", []speculativeABSample{
		{TotalNS: 10, DecodeNS: 8}, {TotalNS: 20, DecodeNS: 16}, {TotalNS: 30, DecodeNS: 24}, {TotalNS: 40, DecodeNS: 32}, {TotalNS: 50, DecodeNS: 40},
	}, []speculativeABSample{
		{TotalNS: 8, DecodeNS: 7}, {TotalNS: 17, DecodeNS: 14}, {TotalNS: 26, DecodeNS: 21}, {TotalNS: 35, DecodeNS: 28}, {TotalNS: 44, DecodeNS: 35},
	})
	if !paired.CI95Measured || paired.TotalP50DeltaNS != -4 || paired.DecodeP50DeltaNS != -3 || paired.TotalP50DeltaCI95 == nil || paired.DecodeP90DeltaCI95 == nil {
		t.Fatalf("paired delta summary=%+v", paired)
	}
	good := "```go\npackage retry\nconst retryDelay=20\nfunc wait(attempt int) int { if attempt < 1 { return retryDelay }; return attempt*retryDelay }\n```"
	if ok, reason := copyEditTaskCorrect(good); !ok {
		t.Fatalf("valid copy-edit witness rejected: %s", reason)
	}
	if ok, _ := copyEditTaskCorrect(strings.Replace(good, "=20", "=10", 1)); ok {
		t.Fatal("unedited source passed copy-edit witness")
	}
}

// TestQwen35DeviceSpeculativeRealCheckpointAB is an opt-in physical qualification
// harness. It compares fresh-session ordinary and shipped n-gram speculative greedy
// decode on the same loaded checkpoint. The JSON log is a moment-in-time receipt
// backed by the selected device's transfer, dispatch, and peak-allocation counters.
func TestQwen35DeviceSpeculativeRealCheckpointAB(t *testing.T) {
	path := os.Getenv("FAK_SPECULATIVE_AB_GGUF")
	if path == "" {
		path = os.Getenv("FAK_GGUF")
	}
	if path == "" {
		t.Skip("set FAK_SPECULATIVE_AB_GGUF to the Qwen3.8 Q4_K GGUF checkpoint")
	}
	backend, ok := compute.Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_VULKAN_REQUIRE_DEVICE=1 but vulkan backend is unavailable")
		}
		t.Skip("vulkan backend unavailable")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE")
		if expected == "" {
			expected = "8060S"
		}
		if !strings.Contains(strings.ToLower(backend.Tier()), strings.ToLower(expected)) {
			t.Fatalf("device %q does not match required %q", backend.Tier(), expected)
		}
	}
	allRows, ok := backend.(compute.Qwen35SequenceAllLogitsBackend)
	if !ok || allRows.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath {
		t.Fatalf("vulkan backend lacks exact all-logits capability")
	}
	embedRows, ok := backend.(compute.Qwen35SequenceEmbeddingRowsBackend)
	if !ok || embedRows.Qwen35SequenceEmbeddingRowsPath() != compute.Qwen35SequenceEmbeddingRowsPath {
		t.Fatalf("vulkan backend lacks exact embedding-row capability")
	}

	tok := loadSpeculativeABTokenizer(t, path)
	m, err := ggufload.LoadModelQ4KProfileOptions(path, nil,
		ggufload.WithDenseKQuantResident(false),
		ggufload.WithDenseQ2KResident(true),
	)
	if err != nil {
		t.Fatalf("load Q4_K checkpoint: %v", err)
	}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close model weights: %v", err)
		}
	}()
	if !m.Cfg.IsQwen35Hybrid() {
		t.Fatalf("checkpoint architecture is not Qwen3.5/3.8 hybrid")
	}
	stopIDs := speculativeABStopIDs(t, tok, m.Cfg)

	samples := speculativeABSamples(t)
	maxNew := envIntAtLeast(t, "FAK_SPECULATIVE_AB_MAX_NEW", 128, 1)
	revision, modified := buildProvenance()
	if samples == 1 {
		t.Log(`{"qualification":false,"reason":"N=1 diagnostic pilot; performance comparison requires N>=5"}`)
	}
	workloads := []speculativeABWorkload{
		{
			Name: "copy-heavy-code-edit",
			Prompt: "Return the complete edited Go file, changing only retryDelay from 10 to 20.\n" +
				"package retry\n\nconst retryDelay = 10\n\nfunc wait(attempt int) int {\n\tif attempt < 1 { return retryDelay }\n\treturn attempt * retryDelay\n}\n",
		},
		{
			Name: "novel-reasoning-control",
			Prompt: "A botanist labels six unseen seeds A through F. Exactly two germinate in salt, " +
				"exactly three germinate in shade, and no seed has both traits. Explain whether these facts " +
				"determine the label of any seed, then give the smallest extra observation that would do so.",
		},
	}

	for _, workload := range workloads {
		renderedPrompt := "<|im_start|>user\n" + workload.Prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
		prompt, err := tok.Encode(renderedPrompt)
		if err != nil {
			t.Fatalf("encode %s: %v", workload.Name, err)
		}
		if len(prompt) == 0 {
			t.Fatalf("encode %s returned no tokens", workload.Name)
		}

		// One unreported warm-up per arm establishes weight/pipeline residency. Every
		// measured run below still uses a fresh KV and recurrent-state session.
		if _, err := runOrdinaryAB(context.Background(), m, backend, prompt, maxNew, workload.Name, -1, revision, modified, stopIDs); err != nil {
			t.Fatalf("ordinary warm-up %s: %v", workload.Name, err)
		}
		if _, err := runSpeculativeAB(context.Background(), m, backend, prompt, maxNew, workload.Name, -1, revision, modified, stopIDs); err != nil {
			t.Fatalf("speculative warm-up %s: %v", workload.Name, err)
		}

		ordinary := make([]speculativeABSample, samples)
		speculative := make([]speculativeABSample, samples)
		for i := 0; i < samples; i++ {
			var first, second speculativeABSample
			if i%2 == 0 {
				first, err = runOrdinaryAB(context.Background(), m, backend, prompt, maxNew, workload.Name, i, revision, modified, stopIDs)
				if err == nil {
					second, err = runSpeculativeAB(context.Background(), m, backend, prompt, maxNew, workload.Name, i, revision, modified, stopIDs)
				}
				ordinary[i], speculative[i] = first, second
			} else {
				first, err = runSpeculativeAB(context.Background(), m, backend, prompt, maxNew, workload.Name, i, revision, modified, stopIDs)
				if err == nil {
					second, err = runOrdinaryAB(context.Background(), m, backend, prompt, maxNew, workload.Name, i, revision, modified, stopIDs)
				}
				speculative[i], ordinary[i] = first, second
			}
			if err != nil {
				t.Fatalf("%s sample %d: %v", workload.Name, i, err)
			}
			ordinary[i].Completion, err = tok.Decode(ordinary[i].Output)
			if err != nil {
				t.Fatalf("decode %s ordinary sample %d: %v", workload.Name, i, err)
			}
			speculative[i].Completion, err = tok.Decode(speculative[i].Output)
			if err != nil {
				t.Fatalf("decode %s speculative sample %d: %v", workload.Name, i, err)
			}
			if workload.Name == "copy-heavy-code-edit" {
				ordinaryCorrect, ordinaryReason := copyEditTaskCorrect(ordinary[i].Completion)
				speculativeCorrect, speculativeReason := copyEditTaskCorrect(speculative[i].Completion)
				ordinary[i].TaskCorrect, ordinary[i].TaskCorrectReason = boolPtr(ordinaryCorrect), ordinaryReason
				speculative[i].TaskCorrect, speculative[i].TaskCorrectReason = boolPtr(speculativeCorrect), speculativeReason
			}
			logJSON(t, ordinary[i])
			logJSON(t, speculative[i])
			if !equalTokens(ordinary[i].Output, speculative[i].Output) {
				t.Fatalf("%s sample %d token mismatch: ordinary=%v speculative=%v", workload.Name, i, ordinary[i].Output, speculative[i].Output)
			}
			if workload.Name == "copy-heavy-code-edit" && samples >= 5 && (!*ordinary[i].TaskCorrect || !*speculative[i].TaskCorrect) {
				t.Fatalf("copy-heavy sample %d failed useful-task witness: ordinary=%q speculative=%q", i, ordinary[i].TaskCorrectReason, speculative[i].TaskCorrectReason)
			}
		}
		if workload.Name == "copy-heavy-code-edit" {
			invocations := 0
			for _, sample := range speculative {
				invocations += sample.TargetSequenceInvocations
			}
			if invocations == 0 {
				t.Fatal("copy-heavy speculative arm executed no target sequence verification; workload did not exercise the mechanism")
			}
		}
		logJSON(t, summarizeAB("ordinary", workload.Name, ordinary))
		logJSON(t, summarizeAB("speculative-ngram-k4", workload.Name, speculative))
		logJSON(t, summarizePairedAB(workload.Name, ordinary, speculative))
	}
}

func runOrdinaryAB(ctx context.Context, m *model.Model, backend compute.Backend, prompt []int, maxNew int, workload string, sample int, revision string, modified bool, stopIDs map[int]struct{}) (speculativeABSample, error) {
	result := speculativeABSample{Arm: "ordinary", Workload: workload, PromptMode: "qwen-chatml-no-think", PrefixRegime: "fresh-session-full-prefill", PromptSHA256: tokenHash(prompt), Sample: sample, PromptTokens: len(prompt), SourceRevision: revision, SourceModified: modified, TimestampUTC: time.Now().UTC().Format(time.RFC3339Nano)}
	window, available, err := compute.BeginBackendExecutionObservation(backend)
	if err != nil {
		return result, err
	}
	start := time.Now()
	s, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		return result, err
	}
	defer s.Close()
	s.Quant = true
	s.Q4K = true
	result.SetupNS = time.Since(start).Nanoseconds()

	prefillStart := time.Now()
	boundary := s.Prefill(prompt)
	result.PrefillNS = time.Since(prefillStart).Nanoseconds()
	result.TTFTNS = time.Since(start).Nanoseconds()
	if err := s.RequireQwen35SequencePrefillNativePerformance(); err != nil {
		return result, err
	}
	decodeStart := time.Now()
	for len(result.Output) < maxNew {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		token := argmax(boundary)
		if _, stop := stopIDs[token]; stop {
			break
		}
		result.Output = append(result.Output, token)
		if len(result.Output) == maxNew {
			break
		}
		boundary = s.Step(token)
		result.OrdinaryTargetSteps++
		if _, err := s.VerifyTokenLineage(append(append([]int(nil), prompt...), result.Output...)); err != nil {
			return result, err
		}
	}
	result.DecodeNS = time.Since(decodeStart).Nanoseconds()
	if err := finishBackendObservation(&result, window, available); err != nil {
		return result, err
	}
	result.TotalNS = time.Since(start).Nanoseconds()
	finishABSample(&result)
	return result, nil
}

func runSpeculativeAB(ctx context.Context, m *model.Model, backend compute.Backend, prompt []int, maxNew int, workload string, sample int, revision string, modified bool, stopIDs map[int]struct{}) (speculativeABSample, error) {
	result := speculativeABSample{Arm: "speculative-ngram-k4", Workload: workload, PromptMode: "qwen-chatml-no-think", PrefixRegime: "fresh-session-full-prefill", PromptSHA256: tokenHash(prompt), Sample: sample, PromptTokens: len(prompt), SourceRevision: revision, SourceModified: modified, TimestampUTC: time.Now().UTC().Format(time.RFC3339Nano), KHistogram: make(map[int]int)}
	window, available, err := compute.BeginBackendExecutionObservation(backend)
	if err != nil {
		return result, err
	}
	start := time.Now()
	s, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		return result, err
	}
	defer s.Close()
	s.Quant = true
	s.Q4K = true
	result.SetupNS = time.Since(start).Nanoseconds()

	generator := model.NewNGramProposalGenerator(model.NgramDrafter{Enabled: true, MaxDraft: 4})
	committed := append([]int(nil), prompt...)
	prefillStart := time.Now()
	boundary := s.Prefill(prompt)
	result.PrefillNS = time.Since(prefillStart).Nanoseconds()
	if err := s.RequireQwen35SequencePrefillNativePerformance(); err != nil {
		return result, err
	}
	decodeStart := time.Now()
	stopped := false
	for len(result.Output) < maxNew && !stopped {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		remaining := maxNew - len(result.Output)
		proposalStart := time.Now()
		proposal, err := generator.Propose(ctx, committed, minInt(4, remaining))
		result.DraftNS += time.Since(proposalStart).Nanoseconds()
		if err != nil {
			return result, err
		}
		draft := proposal.Tokens
		if len(draft) > remaining {
			draft = draft[:remaining]
		}
		if len(draft) == 0 {
			token := argmax(boundary)
			if _, stop := stopIDs[token]; stop {
				stopped = true
				continue
			}
			result.Output = append(result.Output, token)
			committed = append(committed, token)
			if result.TTFTNS == 0 {
				result.TTFTNS = time.Since(start).Nanoseconds()
			}
			if len(result.Output) < maxNew {
				boundary = s.Step(token)
				result.OrdinaryTargetSteps++
				if _, err := s.VerifyTokenLineage(committed); err != nil {
					return result, err
				}
			}
			continue
		}

		verifyStart := time.Now()
		verified, err := s.VerifyGreedyDeviceDraft(ctx, draft, boundary)
		verifyWall := time.Since(verifyStart)
		if err != nil {
			return result, err
		}
		r := verified.Receipt
		accepted := len(verified.Accepted)
		if r.DraftTokens != len(draft) {
			return result, fmt.Errorf("target receipt draft tokens=%d want=%d", r.DraftTokens, len(draft))
		}
		switch r.Path {
		case qwen35BoundaryRejectPath:
			if accepted != 0 || r.AcceptedTokens != 0 || r.RejectedTokens != len(draft) || r.TargetVerificationOperations != 0 || r.TargetDecodeSteps != 0 || r.FullTargetReplaySteps != 0 || r.RecurrentRepairTokens != 0 || r.OneOperation || r.DowngradeReason != "" || verified.TargetLogits != nil || verified.Correction != argmax(boundary) || !equalFloat32(verified.NextLogits, boundary) {
				return result, fmt.Errorf("non-qualifying boundary rejection: result=%+v receipt=%+v", verified, r)
			}
			result.BoundaryRejectRounds++
		case qwen35DeviceVerificationPath:
			if !r.OneOperation || r.TargetVerificationOperations != 1 || r.TargetDecodeSteps != 0 || r.DowngradeReason != "" {
				return result, fmt.Errorf("non-qualifying target receipt: %+v", r)
			}
			if len(verified.TargetLogits) != len(draft) {
				return result, fmt.Errorf("target rows=%d want draft K=%d", len(verified.TargetLogits), len(draft))
			}
			result.TargetSequenceInvocations++
			result.ExpectedD2HLogitsBytes += int64(len(draft) * m.Cfg.VocabSize * 4)
		default:
			return result, fmt.Errorf("non-qualifying target receipt path: %+v", r)
		}
		if verified.CommittedPrefixTokens != len(committed)+accepted {
			return result, fmt.Errorf("committed target prefix=%d want=%d", verified.CommittedPrefixTokens, len(committed)+accepted)
		}
		verifiedPrefix := append(append([]int(nil), committed...), verified.Accepted...)
		if _, err := s.VerifyTokenLineage(verifiedPrefix); err != nil {
			return result, err
		}
		result.DraftedTokens += len(draft)
		result.AcceptedTokens += accepted
		result.RejectedTokens += len(draft) - accepted
		wantRepair := 0
		if accepted > 0 && accepted < len(draft) {
			wantRepair = accepted
		}
		if r.FullTargetReplaySteps != 0 || r.RecurrentRepairTokens != wantRepair {
			return result, fmt.Errorf("non-qualifying partial commit accounting: accepted=%d draft=%d receipt=%+v", accepted, len(draft), r)
		}
		result.ReplayedTokens += r.FullTargetReplaySteps
		result.RecurrentRepairTokens += r.RecurrentRepairTokens
		result.KHistogram[len(draft)]++
		result.TargetVerificationNS += verifyWall.Nanoseconds()
		result.SynchronizationNS += r.Accounting.Synchronization.Nanoseconds
		result.RollbackNS += r.Accounting.Rollback.Nanoseconds
		result.Receipts = append(result.Receipts, receiptSample{DraftTokens: len(draft), AcceptedTokens: accepted, Path: r.Path, OneOperation: r.OneOperation, TargetOperations: r.TargetVerificationOperations, WallNS: verifyWall.Nanoseconds(), ReceiptNS: r.Accounting.TargetVerification.Nanoseconds, SynchronizationNS: r.Accounting.Synchronization.Nanoseconds, RollbackNS: r.Accounting.Rollback.Nanoseconds, FullTargetReplaySteps: r.FullTargetReplaySteps, RecurrentRepairTokens: r.RecurrentRepairTokens})
		switch {
		case accepted == 0:
			result.ZeroAcceptRounds++
		case accepted == len(draft):
			result.FullAcceptRounds++
		default:
			result.PartialAcceptRounds++
		}
		if result.TTFTNS == 0 {
			result.TTFTNS = time.Since(start).Nanoseconds()
		}
		for _, token := range verified.Accepted {
			if len(result.Output) == maxNew {
				break
			}
			if _, stop := stopIDs[token]; stop {
				stopped = true
				break
			}
			result.Output = append(result.Output, token)
			committed = append(committed, token)
		}
		if stopped {
			continue
		}
		if len(result.Output) < maxNew {
			if _, stop := stopIDs[verified.Correction]; stop {
				stopped = true
				continue
			}
			result.Output = append(result.Output, verified.Correction)
			committed = append(committed, verified.Correction)
			if len(result.Output) < maxNew {
				boundary = s.Step(verified.Correction)
				result.OrdinaryTargetSteps++
				if _, err := s.VerifyTokenLineage(committed); err != nil {
					return result, err
				}
			}
		} else {
			boundary = verified.NextLogits
		}
	}
	result.DecodeNS = time.Since(decodeStart).Nanoseconds()
	if err := finishBackendObservation(&result, window, available); err != nil {
		return result, err
	}
	result.TotalNS = time.Since(start).Nanoseconds()
	finishABSample(&result)
	return result, nil
}

func finishABSample(result *speculativeABSample) {
	result.OutputTokens = len(result.Output)
	result.CompletionSHA256 = tokenHash(result.Output)
	if result.OutputTokens > 0 {
		result.TPOTNS = result.DecodeNS / int64(result.OutputTokens)
	}
}

func copyEditTaskCorrect(output string) (bool, string) {
	candidate := strings.TrimSpace(output)
	if start := strings.Index(candidate, "```go"); start >= 0 {
		candidate = candidate[start+len("```go"):]
		end := strings.Index(candidate, "```")
		if end < 0 {
			return false, "truncated go fence"
		}
		candidate = candidate[:end]
	} else if start := strings.Index(candidate, "```"); start >= 0 {
		candidate = candidate[start+len("```"):]
		end := strings.Index(candidate, "```")
		if end < 0 {
			return false, "truncated code fence"
		}
		candidate = candidate[:end]
	}
	got, err := format.Source([]byte(candidate))
	if err != nil {
		return false, "completion is not a complete Go source file: " + err.Error()
	}
	want, err := format.Source([]byte("package retry\n\nconst retryDelay = 20\n\nfunc wait(attempt int) int {\n\tif attempt < 1 { return retryDelay }\n\treturn attempt * retryDelay\n}\n"))
	if err != nil {
		panic(err)
	}
	if !bytes.Equal(got, want) {
		return false, "normalized source differs from the requested one-line edit"
	}
	return true, ""
}

func boolPtr(value bool) *bool { return &value }

func finishBackendObservation(result *speculativeABSample, window compute.BackendExecutionWindow, available bool) error {
	if !available {
		return fmt.Errorf("backend execution observation unavailable")
	}
	observation, err := window.End()
	if err != nil {
		return err
	}
	result.BackendObservation = true
	result.Backend = observation.Identity.Backend
	result.Device = observation.Identity.Device
	result.Driver = observation.Identity.Driver
	result.Runtime = observation.Identity.Runtime
	result.ComputeDispatches = observation.Counters.ComputeDispatches
	result.DispatchesMeasured = os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1"
	result.H2DBytes = observation.Counters.H2DBytes
	result.D2HBytes = observation.Counters.D2HBytes
	result.D2DBytes = observation.Counters.D2DBytes
	result.BackendFallbacks = observation.Counters.Fallbacks
	if !observation.TransferCountersObserved || !observation.DeviceAllocationObserved {
		return fmt.Errorf("backend observation incomplete: transfers=%t allocation=%t", observation.TransferCountersObserved, observation.DeviceAllocationObserved)
	}
	if observation.Counters.Fallbacks != 0 {
		return fmt.Errorf("backend observation recorded %d fallbacks", observation.Counters.Fallbacks)
	}
	result.MemoryMeasured = true
	result.PeakMemoryBytes = int64(observation.DeviceAllocationPeakBytes)
	return nil
}

func loadSpeculativeABTokenizer(t *testing.T, path string) *tokenizer.Tokenizer {
	t.Helper()
	f, err := ggufload.Open(path)
	if err != nil {
		t.Fatalf("open checkpoint metadata: %v", err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("checkpoint has no embedded tokenizer")
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("construct embedded tokenizer: %v", err)
	}
	return tok
}

func envIntAtLeast(t *testing.T, name string, fallback, floor int) int {
	t.Helper()
	value := fallback
	if raw := os.Getenv(name); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			t.Fatalf("%s=%q: %v", name, raw, err)
		}
		value = parsed
	}
	if value < floor {
		t.Fatalf("%s=%d, require >= %d", name, value, floor)
	}
	return value
}

func speculativeABSamples(t *testing.T) int {
	t.Helper()
	value := envIntAtLeast(t, "FAK_SPECULATIVE_AB_SAMPLES", 5, 1)
	if value != 1 && value < 5 {
		t.Fatalf("FAK_SPECULATIVE_AB_SAMPLES=%d; use 1 for a diagnostic pilot or >=5 for qualification", value)
	}
	return value
}

func speculativeABStopIDs(t *testing.T, tok *tokenizer.Tokenizer, cfg model.Config) map[int]struct{} {
	t.Helper()
	stops := make(map[int]struct{}, len(cfg.EOSTokenIDs)+2)
	for _, id := range cfg.EOSTokenIDs {
		stops[id] = struct{}{}
	}
	if cfg.EOSTokenID >= 0 {
		stops[cfg.EOSTokenID] = struct{}{}
	}
	for _, special := range []string{"<|im_end|>", "<|endoftext|>"} {
		ids, err := tok.Encode(special)
		if err != nil {
			t.Fatalf("encode stop token %q: %v", special, err)
		}
		if len(ids) != 1 {
			t.Fatalf("stop token %q encoded as %v, want one token", special, ids)
		}
		stops[ids[0]] = struct{}{}
	}
	return stops
}

func summarizeAB(arm, workload string, samples []speculativeABSample) speculativeABSummary {
	totals := make([]int64, len(samples))
	ttfts := make([]int64, len(samples))
	decodes := make([]int64, len(samples))
	result := speculativeABSummary{Arm: arm, Workload: workload, Samples: len(samples)}
	for i, sample := range samples {
		totals[i], ttfts[i], decodes[i] = sample.TotalNS, sample.TTFTNS, sample.DecodeNS
		result.TargetInvocations += sample.TargetSequenceInvocations
		result.BoundaryRejects += sample.BoundaryRejectRounds
		result.OrdinarySteps += sample.OrdinaryTargetSteps
		result.DraftedTokens += sample.DraftedTokens
		result.AcceptedTokens += sample.AcceptedTokens
		result.RejectedTokens += sample.RejectedTokens
		result.FullReplaySteps += sample.ReplayedTokens
		result.RecurrentRepairs += sample.RecurrentRepairTokens
		result.SynchronizationNS += sample.SynchronizationNS
		result.RollbackNS += sample.RollbackNS
		result.ExpectedD2HBytes += sample.ExpectedD2HLogitsBytes
		result.ObservedD2HBytes += sample.D2HBytes
		result.Fallbacks += sample.BackendFallbacks
		result.DispatchesMeasured = result.DispatchesMeasured || sample.DispatchesMeasured
	}
	result.TotalP50NS, result.TotalP90NS = percentile(totals, 0.50), percentile(totals, 0.90)
	result.TTFTP50NS, result.TTFTP90NS = percentile(ttfts, 0.50), percentile(ttfts, 0.90)
	result.DecodeP50NS, result.DecodeP90NS = percentile(decodes, 0.50), percentile(decodes, 0.90)
	if len(samples) >= 5 {
		result.CI95Measured = true
		result.TotalP50CI95 = bootstrapDurationCI95(totals, 0.50, bootstrapSeed(arm, workload, "total-p50"))
		result.TotalP90CI95 = bootstrapDurationCI95(totals, 0.90, bootstrapSeed(arm, workload, "total-p90"))
		result.TTFTP50CI95 = bootstrapDurationCI95(ttfts, 0.50, bootstrapSeed(arm, workload, "ttft-p50"))
		result.TTFTP90CI95 = bootstrapDurationCI95(ttfts, 0.90, bootstrapSeed(arm, workload, "ttft-p90"))
		result.DecodeP50CI95 = bootstrapDurationCI95(decodes, 0.50, bootstrapSeed(arm, workload, "decode-p50"))
		result.DecodeP90CI95 = bootstrapDurationCI95(decodes, 0.90, bootstrapSeed(arm, workload, "decode-p90"))
	}
	return result
}

func summarizePairedAB(workload string, ordinary, speculative []speculativeABSample) speculativeABPairedSummary {
	n := len(ordinary)
	if len(speculative) < n {
		n = len(speculative)
	}
	totalDeltas := make([]int64, n)
	decodeDeltas := make([]int64, n)
	for i := 0; i < n; i++ {
		totalDeltas[i] = speculative[i].TotalNS - ordinary[i].TotalNS
		decodeDeltas[i] = speculative[i].DecodeNS - ordinary[i].DecodeNS
	}
	result := speculativeABPairedSummary{
		Workload:         workload,
		Samples:          n,
		DeltaDefinition:  "speculative_ngram_k4_minus_ordinary; negative_is_faster",
		TotalP50DeltaNS:  percentile(totalDeltas, 0.50),
		TotalP90DeltaNS:  percentile(totalDeltas, 0.90),
		DecodeP50DeltaNS: percentile(decodeDeltas, 0.50),
		DecodeP90DeltaNS: percentile(decodeDeltas, 0.90),
	}
	if n >= 5 {
		result.CI95Measured = true
		result.TotalP50DeltaCI95 = bootstrapDurationCI95(totalDeltas, 0.50, bootstrapSeed("paired", workload, "total-p50"))
		result.TotalP90DeltaCI95 = bootstrapDurationCI95(totalDeltas, 0.90, bootstrapSeed("paired", workload, "total-p90"))
		result.DecodeP50DeltaCI95 = bootstrapDurationCI95(decodeDeltas, 0.50, bootstrapSeed("paired", workload, "decode-p50"))
		result.DecodeP90DeltaCI95 = bootstrapDurationCI95(decodeDeltas, 0.90, bootstrapSeed("paired", workload, "decode-p90"))
	}
	return result
}

func bootstrapDurationCI95(values []int64, quantile float64, seed int64) *durationCI95 {
	const replicates = 4096
	rng := rand.New(rand.NewSource(seed))
	resample := make([]int64, len(values))
	distribution := make([]int64, replicates)
	for replicate := range distribution {
		for i := range resample {
			resample[i] = values[rng.Intn(len(values))]
		}
		distribution[replicate] = percentile(resample, quantile)
	}
	return &durationCI95{LowNS: percentile(distribution, 0.025), HighNS: percentile(distribution, 0.975)}
}

func bootstrapSeed(parts ...string) int64 {
	h := fnv.New64a()
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0})
	}
	return int64(h.Sum64() & uint64(^uint64(0)>>1))
}

func percentile(values []int64, quantile float64) int64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := int(float64(len(sorted)-1)*quantile + 0.5)
	return sorted[index]
}

func tokenHash(tokens []int) string {
	b := make([]byte, 8*len(tokens))
	for i, token := range tokens {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(token))
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func buildProvenance() (revision string, modified bool) {
	revision = "unknown"
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				if setting.Value != "" {
					revision = setting.Value
				}
			case "vcs.modified":
				modified = setting.Value == "true"
			}
		}
	}
	return revision, modified
}

func argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
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

func equalFloat32(a, b []float32) bool {
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func logJSON(t *testing.T, value any) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal benchmark receipt: %v", err)
	}
	t.Log(string(encoded))
}
