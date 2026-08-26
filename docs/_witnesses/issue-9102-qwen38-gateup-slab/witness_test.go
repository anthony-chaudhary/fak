package gateupslabwitness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	wantBaseRevision = "d306fa7e35c033943990a8241673470e115e9477"
	wantModelSHA     = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	wantPromptSHA    = "b65dd6d80ff9707468ed0d46dc20523671ac2930ec0cd8a29b4015c887eb373c"
)

type crossing struct {
	Sample                      int    `json:"sample"`
	Timestamp                   string `json:"timestamp"`
	SwapBytes                   int64  `json:"swap_bytes"`
	DeltaFromFirstRecordedBytes int64  `json:"delta_from_first_recorded_bytes"`
}

type rejectReceipt struct {
	Schema  string `json:"schema"`
	Issue   int    `json:"issue"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
	Source  struct {
		BaseRevision          string `json:"base_revision"`
		ControlBinarySHA256   string `json:"control_binary_sha256"`
		CandidateBinarySHA256 string `json:"candidate_binary_sha256"`
		CandidateDiffSHA256   string `json:"candidate_diff_sha256"`
		ConfigSHA256          string `json:"config_sha256"`
		CampaignScriptSHA256  string `json:"campaign_script_sha256"`
	} `json:"source"`
	Candidate struct {
		Executed            bool   `json:"executed"`
		ArmDirectoryCreated bool   `json:"arm_directory_created"`
		RuntimeSelector     string `json:"runtime_selector"`
		DefaultEnabled      bool   `json:"default_enabled"`
		CodeDisposition     string `json:"code_disposition"`
	} `json:"candidate"`
	Control struct {
		Executed          bool    `json:"executed"`
		RuntimeSelector   string  `json:"runtime_selector"`
		StartedAt         string  `json:"started_at"`
		CompletedAt       string  `json:"completed_at"`
		CompletedResponse bool    `json:"completed_response"`
		ValidResponse     bool    `json:"valid_response"`
		WallSeconds       float64 `json:"wall_seconds"`
		ResponseSHA256    string  `json:"response_sha256"`
		RequestSHA256     string  `json:"request_sha256"`
		NativeReceipt     struct {
			Model              string  `json:"model"`
			Engine             string  `json:"engine"`
			Backend            string  `json:"backend"`
			ForwardPath        string  `json:"forward_path"`
			Q4K                bool    `json:"q4k"`
			FallbackActive     bool    `json:"fallback_active"`
			PromptTokens       int     `json:"prompt_tokens"`
			CompletionTokens   int     `json:"completion_tokens"`
			PrefillChunkTokens int     `json:"prefill_chunk_tokens"`
			PrefillSeconds     float64 `json:"prefill_seconds"`
			TTFTSeconds        float64 `json:"ttft_seconds"`
			DecodeSeconds      float64 `json:"decode_seconds"`
		} `json:"native_receipt"`
	} `json:"control"`
	Envelope struct {
		Hardware                   string `json:"hardware"`
		ModelSHA256                string `json:"model_sha256"`
		PromptSHA256               string `json:"prompt_sha256"`
		DisplacementASHA256        string `json:"displacement_a_sha256"`
		DisplacementBSHA256        string `json:"displacement_b_sha256"`
		GGUFLoadWorkers            int    `json:"gguf_load_workers"`
		ContextBudgetTokens        int    `json:"context_budget_tokens"`
		NativeAdmissionTokenBudget int    `json:"native_admission_token_budget"`
		FreshProcess               bool   `json:"fresh_process"`
		ExclusiveMachineLease      bool   `json:"exclusive_machine_lease"`
	} `json:"envelope"`
	SafetyGate struct {
		MaximumSwapDeltaBytes          int64      `json:"maximum_swap_delta_bytes"`
		RequiredConsecutiveCrossings   int        `json:"required_consecutive_crossings"`
		MaximumPeakFootprintBytes      int64      `json:"maximum_peak_footprint_bytes"`
		MinimumFreeMemoryPercent       int        `json:"minimum_free_memory_percent"`
		FirstRecordedSwapBytes         int64      `json:"first_recorded_swap_bytes"`
		DerivedSwapLimitBytes          int64      `json:"derived_swap_limit_bytes"`
		CrossingsBeforeResponse        []crossing `json:"crossings_before_response"`
		ResponseCompletedAfterCrossing bool       `json:"response_completed_after_crossings"`
	} `json:"safety_gate"`
	ObservedMemory struct {
		PeakSampledRSSBytes          int64 `json:"peak_sampled_rss_bytes"`
		MinimumFreeMemoryPercent     int   `json:"minimum_free_memory_percent"`
		MaximumCompressorPages       int64 `json:"maximum_compressor_pages"`
		MaximumSwapBytes             int64 `json:"maximum_swap_bytes"`
		PostRequestFootprintCaptured bool  `json:"post_request_footprint_captured"`
	} `json:"observed_memory"`
	Isolation struct {
		GlobalGPULeaseHeld           bool   `json:"global_gpu_lease_held"`
		ArmsSerialized               bool   `json:"arms_serialized"`
		WatcherMatchedTerms          int    `json:"watcher_matched_terms"`
		WatcherUnmatchedSignals      int    `json:"watcher_unmatched_signals"`
		WatcherReusedPIDSignals      int    `json:"watcher_reused_pid_signals"`
		RestoredModel                string `json:"restored_model"`
		RestoredServiceCommandSHA256 string `json:"restored_service_command_sha256"`
		RestoredStableSeconds        int    `json:"restored_stable_seconds"`
		StaleHelperSamples           int    `json:"stale_helper_samples"`
	} `json:"isolation"`
	Artifacts  map[string]string `json:"artifacts"`
	Conclusion string            `json:"conclusion"`
}

type controlResponse struct {
	Schema           string `json:"schema"`
	StartedAt        string `json:"started_at"`
	CompletedAt      string `json:"completed_at"`
	PromptSHA256     string `json:"prompt_sha256"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	RequestSHA256    string `json:"request_sha256"`
	ResponseSHA256   string `json:"response_sha256"`
	Valid            bool   `json:"valid"`
	Receipt          struct {
		Model              string `json:"model"`
		Engine             string `json:"engine"`
		Backend            string `json:"backend"`
		ForwardPath        string `json:"forward_path"`
		Q4K                bool   `json:"q4k"`
		FallbackActive     bool   `json:"fallback_active"`
		PrefillChunkTokens int    `json:"prefill_chunk_tokens"`
	} `json:"receipt"`
	DecodeTrace struct {
		Schema string `json:"schema"`
		Engine string `json:"engine"`
		Events []struct {
			TokenIndex int `json:"token_index"`
		} `json:"events"`
	} `json:"decode_trace"`
}

type memorySample struct {
	Timestamp       time.Time
	Sequence        int
	Phase           string
	RSSKiB          int64
	FreePercent     int
	CompressorPages int64
	SwapBytes       int64
}

func TestGateUpSlabWitness(t *testing.T) {
	var receipt rejectReceipt
	readJSON(t, "reject.json", &receipt)
	if receipt.Schema != "fak.issue-9102-qwen38-gateup-slab-reject/1" || receipt.Issue != 9102 || receipt.Verdict != "REJECT" {
		t.Fatalf("verdict identity = %q/%d/%q", receipt.Schema, receipt.Issue, receipt.Verdict)
	}
	if receipt.Source.BaseRevision != wantBaseRevision ||
		receipt.Source.ControlBinarySHA256 != "721a153a10dfc4eb99e09b1dfcf3a1f330b7cbc16aaacc8c31a070950870f86e" ||
		receipt.Source.CandidateBinarySHA256 != "50642174deb7f528efece58f278b8796472ab6b3205b9a568e6a51ad41e17135" ||
		receipt.Source.CandidateDiffSHA256 != "370546bfcff240baf505beffdb0ad97c1698f02976ea555d01442a7c652f6e10" ||
		receipt.Source.ConfigSHA256 != "89ac20bac433b8618e22cb45642c86113f99f08c0b2436abb099155a60964e1e" ||
		receipt.Source.CampaignScriptSHA256 != "07bca43b44d9305ae068e88b8fa4e3f941578e7dd77cba5efab127bba0216934" {
		t.Fatalf("source/config identity drifted: %+v", receipt.Source)
	}
	if receipt.Candidate.Executed || receipt.Candidate.ArmDirectoryCreated || receipt.Candidate.DefaultEnabled ||
		receipt.Candidate.RuntimeSelector != "FAK_Q4K_GATEUP_SLAB=1" || !strings.Contains(receipt.Candidate.CodeDisposition, "default-off") {
		t.Fatalf("candidate disposition = %+v", receipt.Candidate)
	}
	if !receipt.Control.Executed || !receipt.Control.CompletedResponse || !receipt.Control.ValidResponse || receipt.Control.RuntimeSelector != "FAK_Q4K_GATEUP_SLAB=0" {
		t.Fatalf("control disposition = %+v", receipt.Control)
	}
	if receipt.Envelope.ModelSHA256 != wantModelSHA || receipt.Envelope.PromptSHA256 != wantPromptSHA ||
		receipt.Envelope.GGUFLoadWorkers != 12 || receipt.Envelope.ContextBudgetTokens != 65536 ||
		receipt.Envelope.NativeAdmissionTokenBudget != 65536 || !receipt.Envelope.FreshProcess || !receipt.Envelope.ExclusiveMachineLease ||
		!strings.Contains(receipt.Envelope.Hardware, "Apple M3 Pro") || !strings.Contains(receipt.Envelope.Hardware, "36 GiB") ||
		receipt.Envelope.DisplacementASHA256 != "33c45709dc2638426f1b86abc71a4dc4ecf6aae94fea583308b38459d99fee71" ||
		receipt.Envelope.DisplacementBSHA256 != "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5" {
		t.Fatalf("operating envelope drifted: %+v", receipt.Envelope)
	}

	if len(receipt.Artifacts) != 10 {
		t.Fatalf("artifact bindings=%d want=10", len(receipt.Artifacts))
	}
	for name, want := range receipt.Artifacts {
		body := readPinned(t, name, want)
		lower := strings.ToLower(string(body))
		for _, private := range []string{"/users/anthony", "/var/folders/", "anthony@"} {
			if strings.Contains(lower, private) {
				t.Fatalf("%s retained private marker %q", name, private)
			}
		}
	}

	var response controlResponse
	readJSON(t, "control-response.json", &response)
	if response.Schema != "fak.issue-9102-response/1" || !response.Valid || response.PromptSHA256 != wantPromptSHA ||
		response.PromptTokens != 512 || response.CompletionTokens != 8 || response.RequestSHA256 != receipt.Control.RequestSHA256 ||
		response.ResponseSHA256 != receipt.Control.ResponseSHA256 || response.StartedAt != receipt.Control.StartedAt || response.CompletedAt != receipt.Control.CompletedAt {
		t.Fatalf("control response binding drifted: %+v", response)
	}
	if response.Receipt.Model != "qwen38:27b" || response.Receipt.Engine != "inkernel" || response.Receipt.Backend != "metal" ||
		response.Receipt.ForwardPath != "metal/qwen35-hybrid-session-v1" || !response.Receipt.Q4K || response.Receipt.FallbackActive ||
		response.Receipt.PrefillChunkTokens != 512 || response.DecodeTrace.Schema != "fak.native-decode-trace/1" ||
		response.DecodeTrace.Engine != "fak-native" || len(response.DecodeTrace.Events) != 8 {
		t.Fatalf("native response identity drifted: receipt=%+v trace=%+v", response.Receipt, response.DecodeTrace)
	}
	for i, event := range response.DecodeTrace.Events {
		if event.TokenIndex != i+1 {
			t.Fatalf("decode event %d token_index=%d", i, event.TokenIndex)
		}
	}

	samples := readMemorySamples(t, "control-memory-samples.raw.tsv")
	if len(samples) != 132 || samples[0].Sequence != 0 || samples[len(samples)-1].Sequence != 131 {
		t.Fatalf("sample span=%d first=%d last=%d", len(samples), samples[0].Sequence, samples[len(samples)-1].Sequence)
	}
	gate := receipt.SafetyGate
	if gate.MaximumSwapDeltaBytes != 12*1024*1024*1024 || gate.RequiredConsecutiveCrossings != 3 ||
		gate.FirstRecordedSwapBytes != samples[0].SwapBytes || gate.DerivedSwapLimitBytes != samples[0].SwapBytes+gate.MaximumSwapDeltaBytes ||
		!gate.ResponseCompletedAfterCrossing || len(gate.CrossingsBeforeResponse) != 3 {
		t.Fatalf("safety gate drifted: %+v", gate)
	}
	completedAt, err := time.Parse(time.RFC3339Nano, receipt.Control.CompletedAt)
	if err != nil {
		t.Fatal(err)
	}
	var crossings []crossing
	var peakRSS, maxCompressor, maxSwap int64
	minFree := 101
	consecutive := 0
	for _, sample := range samples {
		if sample.RSSKiB*1024 > peakRSS {
			peakRSS = sample.RSSKiB * 1024
		}
		if sample.FreePercent < minFree {
			minFree = sample.FreePercent
		}
		if sample.CompressorPages > maxCompressor {
			maxCompressor = sample.CompressorPages
		}
		if sample.SwapBytes > maxSwap {
			maxSwap = sample.SwapBytes
		}
		if sample.SwapBytes > gate.DerivedSwapLimitBytes {
			consecutive++
			crossings = append(crossings, crossing{sample.Sequence, sample.Timestamp.Format(time.RFC3339), sample.SwapBytes, sample.SwapBytes - samples[0].SwapBytes})
			if !sample.Timestamp.Before(completedAt) {
				t.Fatalf("unsafe sample %d did not precede response completion", sample.Sequence)
			}
		} else {
			consecutive = 0
		}
	}
	if consecutive != 3 || len(crossings) != 3 {
		t.Fatalf("computed unsafe crossings=%d consecutive tail=%d", len(crossings), consecutive)
	}
	for i := range crossings {
		if crossings[i] != gate.CrossingsBeforeResponse[i] {
			t.Fatalf("crossing[%d]=%+v want %+v", i, crossings[i], gate.CrossingsBeforeResponse[i])
		}
	}
	if peakRSS != receipt.ObservedMemory.PeakSampledRSSBytes || minFree != receipt.ObservedMemory.MinimumFreeMemoryPercent ||
		maxCompressor != receipt.ObservedMemory.MaximumCompressorPages || maxSwap != receipt.ObservedMemory.MaximumSwapBytes ||
		receipt.ObservedMemory.PostRequestFootprintCaptured {
		t.Fatalf("memory aggregate=%d/%d/%d/%d receipt=%+v", peakRSS, minFree, maxCompressor, maxSwap, receipt.ObservedMemory)
	}

	assertContains(t, "control-server-summary.log", "control_selector FAK_Q4K_GATEUP_SLAB=0", "control_gateup_slab_execution_lines=0", "candidate_executed=false", "forward_path=metal/qwen35-hybrid-session-v1")
	assertContains(t, "campaign-state.log", "arm_directories=01-control-rep1", "candidate_arm_directories=0", "candidate_executed=false")
	assertContains(t, "watcher.log", "matched_terms=0", "unmatched_signals=0", "reused_pid_signals=0")
	assertContains(t, "isolation-summary.log", "global_gpu_lease=held_during_control", "restored_model=qwen3.6-27b", "restored_stable_seconds=90", "stale_helper_samples=30")
	assertContains(t, "restore.log", "stable_seconds=90", "stale_helper_samples=30")
	assertContains(t, "restore-models.json", "qwen3.6-27b")
	stale := strings.Split(strings.TrimSpace(string(readFile(t, "stale-helper-samples.tsv"))), "\n")
	if len(stale) != 30 {
		t.Fatalf("stale helper samples=%d want=30", len(stale))
	}
	for i, line := range stale {
		if !strings.HasSuffix(line, `\tnone`) {
			t.Fatalf("stale helper sample %d=%q", i+1, line)
		}
	}
	if !receipt.Isolation.GlobalGPULeaseHeld || !receipt.Isolation.ArmsSerialized || receipt.Isolation.WatcherMatchedTerms != 0 ||
		receipt.Isolation.WatcherUnmatchedSignals != 0 || receipt.Isolation.WatcherReusedPIDSignals != 0 ||
		receipt.Isolation.RestoredModel != "qwen3.6-27b" || receipt.Isolation.RestoredStableSeconds != 90 || receipt.Isolation.StaleHelperSamples != 30 {
		t.Fatalf("isolation receipt drifted: %+v", receipt.Isolation)
	}
	if !strings.Contains(receipt.Reason, "three consecutive") || !strings.Contains(receipt.Conclusion, "default") {
		t.Fatalf("closure is not bound to REJECT/default-off: reason=%q conclusion=%q", receipt.Reason, receipt.Conclusion)
	}
}

func readMemorySamples(t *testing.T, name string) []memorySample {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(string(readFile(t, name))), "\n")
	out := make([]memorySample, 0, len(lines))
	for lineNo, line := range lines {
		fields := strings.Split(line, `\t`)
		if len(fields) != 9 {
			t.Fatalf("%s:%d fields=%d want=9", name, lineNo+1, len(fields))
		}
		timestamp, err := time.Parse(time.RFC3339, fields[0])
		if err != nil {
			t.Fatalf("%s:%d timestamp: %v", name, lineNo+1, err)
		}
		parse := func(field int, label string) int64 {
			value, err := strconv.ParseInt(fields[field], 10, 64)
			if err != nil {
				t.Fatalf("%s:%d %s: %v", name, lineNo+1, label, err)
			}
			return value
		}
		sequence := int(parse(1, "sequence"))
		if sequence != lineNo {
			t.Fatalf("%s:%d sequence=%d want=%d", name, lineNo+1, sequence, lineNo)
		}
		out = append(out, memorySample{
			Timestamp:       timestamp,
			Sequence:        sequence,
			Phase:           fields[2],
			RSSKiB:          parse(4, "rss_kib"),
			FreePercent:     int(parse(6, "free_percent")),
			CompressorPages: parse(7, "compressor_pages"),
			SwapBytes:       parse(8, "swap_bytes"),
		})
	}
	return out
}

func readPinned(t *testing.T, name, want string) []byte {
	t.Helper()
	body := readFile(t, name)
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Fatalf("%s sha256=%s want=%s", name, got, want)
	}
	return body
}

func readFile(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func readJSON(t *testing.T, name string, dst any) {
	t.Helper()
	if err := json.Unmarshal(readFile(t, name), dst); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func assertContains(t *testing.T, name string, markers ...string) {
	t.Helper()
	body := string(readFile(t, name))
	for _, marker := range markers {
		if !strings.Contains(body, marker) {
			t.Fatalf("%s missing %q", name, marker)
		}
	}
}
