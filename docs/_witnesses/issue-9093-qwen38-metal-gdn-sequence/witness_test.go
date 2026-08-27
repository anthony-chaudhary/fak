package issue9093witness

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	wantArtifactSHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	wantForwardPath = "metal/qwen35-gdn-preprojected-sequence-v1"
	wantOwnerSHA    = "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d"
)

type closurePacket struct {
	Schema         string `json:"schema"`
	Issue          int    `json:"issue"`
	Verdict        string `json:"verdict"`
	Reason         string `json:"reason"`
	Implementation struct {
		ContractCommit     string   `json:"contract_commit"`
		PrimitiveCommit    string   `json:"primitive_commit"`
		ProductionCommit   string   `json:"production_commit"`
		RepairCommits      []string `json:"repair_commits"`
		CapabilityPath     string   `json:"capability_path"`
		ForwardPath        string   `json:"forward_path"`
		Selector           string   `json:"selector"`
		DefaultEnabled     bool     `json:"default_enabled"`
		ProductionPromoted bool     `json:"production_promoted"`
	} `json:"implementation"`
	Acceptance struct {
		CPUOracleCosineMin              float64 `json:"cpu_oracle_cosine_min"`
		CPUOracleMaxAbs                 float64 `json:"cpu_oracle_max_abs"`
		PersistentBuffersPerLayer       int     `json:"persistent_buffers_per_layer"`
		StateH2DTransfersPerOperation   int     `json:"state_h2d_transfers_per_operation"`
		StateD2HTransfersPerOperation   int     `json:"state_d2h_transfers_per_operation"`
		HostRecurrenceStepsPerOperation int     `json:"host_recurrence_steps_per_operation"`
		StableStateHandles              bool    `json:"stable_state_handles"`
		IsolatedSessions                bool    `json:"isolated_sessions"`
		DeclineBeforeMutation           bool    `json:"decline_before_mutation"`
		PostSubmitFallbackAttempts      int     `json:"post_submit_fallback_attempts"`
		ExactGreedyTokenHandoff         bool    `json:"exact_greedy_token_handoff"`
		ExactOnceCleanup                bool    `json:"exact_once_cleanup"`
	} `json:"acceptance"`
	Tests struct {
		Portable map[string][]string `json:"portable"`
		Native   map[string][]string `json:"native"`
	} `json:"tests"`
	LiveCanary struct {
		SourceIssue                  int    `json:"source_issue"`
		PromptTokens                 int    `json:"prompt_tokens"`
		OutputTokens                 int    `json:"output_tokens"`
		DeadlineSeconds              int    `json:"deadline_seconds"`
		MaximumFootprintBytes        int64  `json:"maximum_footprint_bytes"`
		MaximumSwapGrowthBytes       int64  `json:"maximum_swap_growth_bytes"`
		RequiredConsecutiveCrossings int    `json:"required_consecutive_crossings"`
		MinimumMemorystatusPercent   int64  `json:"minimum_memorystatus_percent"`
		RequiredRestorationSeconds   int    `json:"required_restoration_seconds"`
		ChildReceiptSHA256           string `json:"child_receipt_sha256"`
		MemorySamplesSHA256          string `json:"memory_samples_sha256"`
		NativeLogSHA256              string `json:"native_log_sha256"`
		ChildWitnessSHA256           string `json:"child_witness_sha256"`
	} `json:"live_canary"`
	NextBoundary struct {
		Issue       int    `json:"issue"`
		Description string `json:"description"`
	} `json:"next_boundary"`
	Conclusion string `json:"conclusion"`
}

type canaryReceipt struct {
	Schema   string `json:"schema"`
	Issue    int    `json:"issue"`
	Verdict  string `json:"verdict"`
	Artifact struct {
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Request struct {
		PromptTokens int `json:"prompt_tokens"`
		OutputTokens int `json:"output_tokens"`
		Temperature  int `json:"temperature"`
	} `json:"request"`
	Execution struct {
		Engine         string `json:"engine"`
		Backend        string `json:"backend"`
		ForwardPath    string `json:"forward_path"`
		Q4K            bool   `json:"q4k"`
		FallbackActive bool   `json:"fallback_active"`
		LlamaCPPUsed   bool   `json:"llama_cpp_used"`
	} `json:"execution"`
	Selector struct {
		Environment        string `json:"environment"`
		DefaultEnabled     bool   `json:"default_enabled"`
		ProductionPromoted bool   `json:"production_promoted"`
	} `json:"selector"`
	Timeline struct {
		RequestStartedAt        string `json:"request_started_at"`
		SafetyTripAt            string `json:"safety_trip_at"`
		ResponseComplete        bool   `json:"response_complete"`
		ResponseHeadersReceived bool   `json:"response_headers_received"`
		RerunCount              int    `json:"rerun_count"`
	} `json:"timeline"`
	Safety struct {
		MaximumFootprintBytes            int64  `json:"maximum_footprint_bytes"`
		MinimumFreeMemoryPercent         int64  `json:"minimum_free_memory_percent"`
		MinimumMemorystatusPercent       int64  `json:"minimum_memorystatus_percent"`
		MemorystatusPercentObserved      *int64 `json:"memorystatus_percent_observed"`
		MaximumSwapGrowthBytes           int64  `json:"maximum_swap_growth_bytes"`
		RequiredConsecutiveCrossings     int    `json:"required_consecutive_crossings"`
		KernelPressureEvents             int    `json:"kernel_pressure_events"`
		SampleCount                      int    `json:"sample_count"`
		PeakSampledRSSBytes              int64  `json:"peak_sampled_rss_bytes"`
		PeakSampledFootprintBytes        int64  `json:"peak_sampled_footprint_bytes"`
		MinimumFreeMemoryPercentObserved int64  `json:"minimum_free_memory_percent_observed"`
		PeakSwapGrowthBytes              int64  `json:"peak_swap_growth_bytes"`
	} `json:"safety"`
	Isolation struct {
		GPULeaseHeld              bool   `json:"gpu_lease_held"`
		CandidatePIDScoped        bool   `json:"candidate_pid_scoped"`
		TeardownSignal            string `json:"teardown_signal"`
		WatcherMatchedTerms       int    `json:"watcher_matched_terms"`
		WatcherUnmatchedWorkloads int    `json:"watcher_unmatched_workloads"`
		ServiceCommandSHA256      string `json:"service_command_sha256"`
		Restored                  bool   `json:"restored"`
		RestoredHealthStatus      int    `json:"restored_health_status"`
		RestoredModelsStatus      int    `json:"restored_models_status"`
		RestoredStableSeconds     int    `json:"restored_stable_seconds"`
	} `json:"isolation"`
}

func TestQwen38MetalGDNSequenceWitness(t *testing.T) {
	var packet closurePacket
	readJSON(t, "packet.json", &packet)
	if packet.Schema != "fak.issue-9093-qwen38-metal-gdn-sequence/1" || packet.Issue != 9093 || packet.Verdict != "REJECT" {
		t.Fatalf("closure identity=%q/%d/%q", packet.Schema, packet.Issue, packet.Verdict)
	}
	if packet.Implementation.CapabilityPath != "qwen35/gdn-preprojected-sequence-v1" ||
		packet.Implementation.ForwardPath != wantForwardPath ||
		packet.Implementation.Selector != "FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE=1" ||
		packet.Implementation.DefaultEnabled || packet.Implementation.ProductionPromoted {
		t.Fatalf("implementation disposition=%+v", packet.Implementation)
	}
	for _, commit := range append([]string{
		packet.Implementation.ContractCommit,
		packet.Implementation.PrimitiveCommit,
		packet.Implementation.ProductionCommit,
	}, packet.Implementation.RepairCommits...) {
		if len(commit) != 40 || strings.Trim(commit, "0123456789abcdef") != "" {
			t.Fatalf("invalid implementation commit identity %q", commit)
		}
	}

	a := packet.Acceptance
	if a.CPUOracleCosineMin != 0.999999 || a.CPUOracleMaxAbs != 0.0001 ||
		a.PersistentBuffersPerLayer != 2 || a.StateH2DTransfersPerOperation != 0 ||
		a.StateD2HTransfersPerOperation != 0 || a.HostRecurrenceStepsPerOperation != 0 ||
		!a.StableStateHandles || !a.IsolatedSessions || !a.DeclineBeforeMutation ||
		a.PostSubmitFallbackAttempts != 0 || !a.ExactGreedyTokenHandoff || !a.ExactOnceCleanup {
		t.Fatalf("acceptance contract drift=%+v", a)
	}
	verifyDeclaredTests(t, packet.Tests.Portable)
	verifyDeclaredTests(t, packet.Tests.Native)
	verifySourceContracts(t)
	verifyCanary(t, packet)

	if packet.NextBoundary.Issue != 9230 ||
		!strings.Contains(packet.NextBoundary.Description, "graph-scoped") ||
		!strings.Contains(packet.NextBoundary.Description, "whole Qwen forward") ||
		!strings.Contains(packet.Reason, "swap-growth guard") ||
		!strings.Contains(packet.Conclusion, "default-off") {
		t.Fatalf("REJECT routing is incomplete: next=%+v reason=%q conclusion=%q", packet.NextBoundary, packet.Reason, packet.Conclusion)
	}
}

func verifyDeclaredTests(t *testing.T, declared map[string][]string) {
	t.Helper()
	root := filepath.Join("..", "..", "..")
	for name, wants := range declared {
		path := filepath.Join(root, filepath.FromSlash(name))
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		got := make(map[string]bool)
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if ok {
				got[fn.Name.Name] = true
			}
		}
		for _, want := range wants {
			if !got[want] {
				t.Fatalf("%s is missing declared witness %s", name, want)
			}
		}
	}
}

func verifySourceContracts(t *testing.T) {
	t.Helper()
	root := filepath.Join("..", "..", "..")
	requireMarkers(t, filepath.Join(root, "internal", "metalgemm", "gdn_darwin_test.go"),
		"gdnCosineFloor = 0.999999",
		"gdnMaxAbsLimit = 0.0001",
		"accounting.StateH2DTransfers != 0",
		"accounting.StateD2HTransfers != 0",
		"accounting.HostRecurrenceSteps != 0",
		"accounting.OwnedBuffers != 2",
		"gotConv != convHandle || gotRecurrent != recurrentHandle",
	)
	requireMarkers(t, filepath.Join(root, "internal", "model", "metal_qwen35_gdn_sequence_test.go"),
		"native auxiliary identity changed across outer prefill chunks",
		"second finalize",
		"first decode greedy token",
		"idempotent close retained native buffers",
	)
	requireMarkers(t, filepath.Join(root, "internal", "model", "qwen35_hal_test.go"),
		"sessions share auxiliary identity",
		"poisoned session retried/fell back",
		"want exactly 1",
		"default session unexpectedly activated capability",
	)
	requireMarkers(t, filepath.Join(root, "internal", "model", "metal_prefill_hybrid.go"),
		"accounting.StateH2DTransfers != 0",
		"accounting.StateD2HTransfers != 0",
		"accounting.HostRecurrenceSteps != 0",
		"accounting.OwnedBuffers != 2",
		"accounting.PrivateStateBuffers != 2",
	)

	agentDir := filepath.Join(root, "internal", "agent")
	files, err := filepath.Glob(filepath.Join(agentDir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	assignments, selectorMentions := 0, 0
	for _, path := range files {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		body := string(readFile(t, path))
		assignments += strings.Count(body, "qwen35MetalGDNSequence = true")
		selectorMentions += strings.Count(body, `os.Getenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE")`)
	}
	if assignments != 1 || selectorMentions != 1 {
		t.Fatalf("selector default-off proof assignments=%d env_reads=%d, want 1/1", assignments, selectorMentions)
	}
}

func verifyCanary(t *testing.T, packet closurePacket) {
	t.Helper()
	childDir := filepath.Join("..", "issue-9097-qwen38-metal-gdn-production")
	pins := map[string]string{
		"receipt.json":       packet.LiveCanary.ChildReceiptSHA256,
		"memory-samples.tsv": packet.LiveCanary.MemorySamplesSHA256,
		"native.log":         packet.LiveCanary.NativeLogSHA256,
		"witness_test.go":    packet.LiveCanary.ChildWitnessSHA256,
	}
	for name, want := range pins {
		body := readFile(t, filepath.Join(childDir, name))
		sum := sha256.Sum256(body)
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("%s sha256=%s want=%s", name, got, want)
		}
	}

	var receipt canaryReceipt
	readJSON(t, filepath.Join(childDir, "receipt.json"), &receipt)
	if packet.LiveCanary.SourceIssue != 9097 ||
		receipt.Schema != "fak.issue-9097-qwen38-metal-gdn-production/1" ||
		receipt.Issue != packet.LiveCanary.SourceIssue || receipt.Verdict != "REJECT" {
		t.Fatalf("canary identity=%q/%d/%q source=%d", receipt.Schema, receipt.Issue, receipt.Verdict, packet.LiveCanary.SourceIssue)
	}
	if receipt.Artifact.Bytes != 17106775008 || receipt.Artifact.SHA256 != wantArtifactSHA ||
		receipt.Request.PromptTokens != packet.LiveCanary.PromptTokens ||
		receipt.Request.OutputTokens != packet.LiveCanary.OutputTokens || receipt.Request.Temperature != 0 {
		t.Fatalf("canary envelope artifact=%+v request=%+v", receipt.Artifact, receipt.Request)
	}
	if receipt.Execution.Engine != "inkernel" || receipt.Execution.Backend != "metal" ||
		receipt.Execution.ForwardPath != wantForwardPath || !receipt.Execution.Q4K ||
		receipt.Execution.FallbackActive || receipt.Execution.LlamaCPPUsed {
		t.Fatalf("native execution identity=%+v", receipt.Execution)
	}
	if receipt.Selector.Environment != packet.Implementation.Selector ||
		receipt.Selector.DefaultEnabled || receipt.Selector.ProductionPromoted {
		t.Fatalf("selector result=%+v", receipt.Selector)
	}
	if receipt.Safety.MaximumFootprintBytes != packet.LiveCanary.MaximumFootprintBytes ||
		receipt.Safety.MaximumSwapGrowthBytes != packet.LiveCanary.MaximumSwapGrowthBytes ||
		receipt.Safety.RequiredConsecutiveCrossings != packet.LiveCanary.RequiredConsecutiveCrossings ||
		receipt.Safety.MinimumMemorystatusPercent != packet.LiveCanary.MinimumMemorystatusPercent ||
		receipt.Safety.MemorystatusPercentObserved != nil || receipt.Safety.KernelPressureEvents != 0 {
		t.Fatalf("canary safety contract=%+v", receipt.Safety)
	}

	f, err := os.Open(filepath.Join(childDir, "memory-samples.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.Comma = '\t'
	rows, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != receipt.Safety.SampleCount+1 {
		t.Fatalf("memory sample rows=%d receipt=%d", len(rows)-1, receipt.Safety.SampleCount)
	}
	var maxRSS, maxFootprint, maxSwap int64
	minFree := int64(101)
	consecutive, maxConsecutive := 0, 0
	for i, row := range rows[1:] {
		if len(row) != 6 {
			t.Fatalf("memory row %d fields=%d", i+2, len(row))
		}
		rss := parseInt(t, row[1])
		footprint := parseInt(t, row[2])
		free := parseInt(t, row[3])
		swapGrowth := parseInt(t, row[5])
		maxRSS = max(maxRSS, rss)
		maxFootprint = max(maxFootprint, footprint)
		maxSwap = max(maxSwap, swapGrowth)
		minFree = min(minFree, free)
		if footprint >= receipt.Safety.MaximumFootprintBytes || free < receipt.Safety.MinimumFreeMemoryPercent {
			t.Fatalf("sample %d crossed non-swap guard footprint=%d free=%d", i+1, footprint, free)
		}
		if swapGrowth > receipt.Safety.MaximumSwapGrowthBytes {
			consecutive++
		} else {
			consecutive = 0
		}
		maxConsecutive = max(maxConsecutive, consecutive)
	}
	if maxRSS != receipt.Safety.PeakSampledRSSBytes ||
		maxFootprint != receipt.Safety.PeakSampledFootprintBytes ||
		maxSwap != receipt.Safety.PeakSwapGrowthBytes ||
		minFree != receipt.Safety.MinimumFreeMemoryPercentObserved ||
		maxConsecutive != receipt.Safety.RequiredConsecutiveCrossings {
		t.Fatalf("derived safety rss=%d footprint=%d swap=%d free=%d crossings=%d receipt=%+v",
			maxRSS, maxFootprint, maxSwap, minFree, maxConsecutive, receipt.Safety)
	}

	start := parseTime(t, receipt.Timeline.RequestStartedAt)
	trip := parseTime(t, receipt.Timeline.SafetyTripAt)
	if !trip.After(start) || trip.Sub(start) >= time.Duration(packet.LiveCanary.DeadlineSeconds)*time.Second ||
		receipt.Timeline.ResponseComplete || receipt.Timeline.ResponseHeadersReceived || receipt.Timeline.RerunCount != 0 {
		t.Fatalf("canary timeline=%+v duration=%s", receipt.Timeline, trip.Sub(start))
	}
	if !receipt.Isolation.GPULeaseHeld || !receipt.Isolation.CandidatePIDScoped ||
		receipt.Isolation.TeardownSignal != "TERM" ||
		receipt.Isolation.WatcherMatchedTerms != 0 || receipt.Isolation.WatcherUnmatchedWorkloads != 0 ||
		receipt.Isolation.ServiceCommandSHA256 != wantOwnerSHA || !receipt.Isolation.Restored ||
		receipt.Isolation.RestoredHealthStatus != 200 || receipt.Isolation.RestoredModelsStatus != 200 ||
		receipt.Isolation.RestoredStableSeconds != packet.LiveCanary.RequiredRestorationSeconds {
		t.Fatalf("canary isolation/restoration=%+v", receipt.Isolation)
	}

	nativeLog := strings.ToLower(string(readFile(t, filepath.Join(childDir, "native.log"))))
	for _, want := range []string{
		"engine=inkernel backend=metal forward_path=" + wantForwardPath,
		"q4k=true fallback=false llama_cpp=false",
		"restore_stable_complete seconds=90",
	} {
		if !strings.Contains(nativeLog, want) {
			t.Fatalf("native.log missing %q", want)
		}
	}
	for _, forbidden := range []string{"/users/", "/private/", "ssh ", "token=", "password=", "secret="} {
		if strings.Contains(nativeLog, forbidden) {
			t.Fatalf("native.log contains private marker %q", forbidden)
		}
	}
}

func requireMarkers(t *testing.T, path string, markers ...string) {
	t.Helper()
	body := string(readFile(t, path))
	for _, marker := range markers {
		if !strings.Contains(body, marker) {
			t.Fatalf("%s missing contract marker %q", path, marker)
		}
	}
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func readJSON(t *testing.T, path string, dst any) {
	t.Helper()
	if err := json.Unmarshal(readFile(t, path), dst); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func parseInt(t *testing.T, value string) int64 {
	t.Helper()
	n, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func parseTime(t *testing.T, value string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339, value)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
