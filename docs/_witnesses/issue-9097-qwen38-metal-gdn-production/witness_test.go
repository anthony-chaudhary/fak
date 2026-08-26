package issue9097witness

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	artifactSHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	promptSHA   = "eafba2115b741f1beae1a131eb11859f72c4e543d6144dd1d76e9b18a3a1fa51"
	requestSHA  = "5097d7d92e795b0f940ec0f2c3f380213ad8d0e8ef23a6e52e986d63614dc974"
	ownerSHA    = "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d"
	forwardPath = "metal/qwen35-gdn-preprojected-sequence-v1"
)

type receipt struct {
	Schema  string `json:"schema"`
	Issue   int    `json:"issue"`
	Verdict string `json:"verdict"`
	Source  struct {
		BaseRevision    string `json:"base_revision"`
		CandidateCommit string `json:"candidate_commit"`
		CandidateTree   string `json:"candidate_tree"`
		BinarySHA256    string `json:"binary_sha256"`
		BuildModified   bool   `json:"build_vcs_modified"`
	} `json:"source"`
	Artifact struct {
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Request struct {
		Fixture      string `json:"fixture"`
		PromptBytes  int    `json:"prompt_bytes"`
		PromptSHA256 string `json:"prompt_sha256"`
		BodyBytes    int    `json:"body_bytes"`
		BodySHA256   string `json:"body_sha256"`
		PromptTokens int    `json:"prompt_tokens"`
		OutputTokens int    `json:"output_tokens"`
		Temperature  int    `json:"temperature"`
		Answer       string `json:"expected_answer"`
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
		LaunchedAt              string `json:"launched_at"`
		RequestStartedAt        string `json:"request_started_at"`
		SafetyTripAt            string `json:"safety_trip_at"`
		RestorationFinishedAt   string `json:"restoration_finished_at"`
		ResponseComplete        bool   `json:"response_complete"`
		ResponseHeadersReceived bool   `json:"response_headers_received"`
		RerunCount              int    `json:"rerun_count"`
	} `json:"timeline"`
	Safety struct {
		MaximumFootprintBytes            int64  `json:"maximum_footprint_bytes"`
		MinimumFreeMemoryPercent         int64  `json:"minimum_free_memory_percent"`
		MinimumMemorystatusPercent       int64  `json:"minimum_memorystatus_percent"`
		MemorystatusPercentObserved      *int64 `json:"memorystatus_percent_observed"`
		MemorystatusSource               string `json:"memorystatus_source"`
		MaximumSwapGrowthBytes           int64  `json:"maximum_swap_growth_bytes"`
		RequiredConsecutiveCrossings     int    `json:"required_consecutive_crossings"`
		KernelPressureEvents             int    `json:"kernel_pressure_events"`
		SampleCount                      int    `json:"sample_count"`
		BaselineSwapBytes                int64  `json:"baseline_swap_bytes"`
		PeakSampledRSSBytes              int64  `json:"peak_sampled_rss_bytes"`
		TimeLMaximumRSSBytes             int64  `json:"time_l_maximum_rss_bytes"`
		PeakSampledFootprintBytes        int64  `json:"peak_sampled_footprint_bytes"`
		TimeLPeakFootprintBytes          int64  `json:"time_l_peak_footprint_bytes"`
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

func TestRejectReceiptBindsFrozenCandidateAndRequest(t *testing.T) {
	r := readReceipt(t)
	if r.Schema != "fak.issue-9097-qwen38-metal-gdn-production/1" || r.Issue != 9097 || r.Verdict != "REJECT" {
		t.Fatalf("receipt identity = schema %q issue %d verdict %q", r.Schema, r.Issue, r.Verdict)
	}
	if r.Source.BaseRevision != "1292441de633214253e79f6d3353c630ea1c9fdf" ||
		r.Source.CandidateCommit != "87b92938232ccf24d290d4c2ef08e16cf55cc0d3" ||
		r.Source.CandidateTree != "60a6c1f5f3de7687fc39d95093451dcb1b54ce83" ||
		r.Source.BinarySHA256 != "81749488237578d87e60bce82d6967e37f5200eb07eb256f64bcba3ad8355340" || r.Source.BuildModified {
		t.Fatalf("source identity drift: %+v", r.Source)
	}
	if r.Artifact.Bytes != 17106775008 || r.Artifact.SHA256 != artifactSHA {
		t.Fatalf("artifact identity drift: %+v", r.Artifact)
	}

	prompt := longPrompt(t)
	body := requestJSON(t, prompt)
	if len(prompt) != r.Request.PromptBytes || digest([]byte(prompt)) != promptSHA || r.Request.PromptSHA256 != promptSHA {
		t.Fatalf("prompt identity drift: bytes=%d sha=%s", len(prompt), digest([]byte(prompt)))
	}
	if len(body) != r.Request.BodyBytes || digest(body) != requestSHA || r.Request.BodySHA256 != requestSHA {
		t.Fatalf("request identity drift: bytes=%d sha=%s", len(body), digest(body))
	}
	if r.Request.Fixture != "long-context-needle-v1" || r.Request.PromptTokens != 32800 ||
		r.Request.OutputTokens != 8 || r.Request.Temperature != 0 || r.Request.Answer != "ORCHID-7319" {
		t.Fatalf("request shape drift: %+v", r.Request)
	}
	if r.Execution.Engine != "inkernel" || r.Execution.Backend != "metal" ||
		r.Execution.ForwardPath != forwardPath || !r.Execution.Q4K || r.Execution.FallbackActive || r.Execution.LlamaCPPUsed {
		t.Fatalf("native execution identity drift: %+v", r.Execution)
	}
	if r.Selector.Environment != "FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE=1" ||
		r.Selector.DefaultEnabled || r.Selector.ProductionPromoted {
		t.Fatalf("REJECT must leave the selector default-off: %+v", r.Selector)
	}
}

func TestRejectReceiptBindsThreeSampleSafetyTripAndRestore(t *testing.T) {
	r := readReceipt(t)
	f, err := os.Open("memory-samples.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c := csv.NewReader(f)
	c.Comma = '\t'
	rows, err := c.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != r.Safety.SampleCount+1 || r.Safety.SampleCount != 26 {
		t.Fatalf("sample rows = %d, receipt count = %d", len(rows)-1, r.Safety.SampleCount)
	}
	var maxRSS, maxFootprint, maxSwap int64
	minFree := int64(101)
	consecutive, maxConsecutive := 0, 0
	for i, row := range rows[1:] {
		if len(row) != 6 {
			t.Fatalf("row %d has %d fields", i+2, len(row))
		}
		rss := number(t, row[1])
		foot := number(t, row[2])
		free := number(t, row[3])
		sw := number(t, row[5])
		maxRSS = max(maxRSS, rss)
		maxFootprint = max(maxFootprint, foot)
		maxSwap = max(maxSwap, sw)
		minFree = min(minFree, free)
		if foot >= r.Safety.MaximumFootprintBytes || free < r.Safety.MinimumFreeMemoryPercent {
			t.Fatalf("sample %d crossed a non-swap guard: footprint=%d free=%d", i+1, foot, free)
		}
		if sw > r.Safety.MaximumSwapGrowthBytes {
			consecutive++
		} else {
			consecutive = 0
		}
		maxConsecutive = max(maxConsecutive, consecutive)
	}
	if maxRSS != r.Safety.PeakSampledRSSBytes || maxFootprint != r.Safety.PeakSampledFootprintBytes ||
		maxSwap != r.Safety.PeakSwapGrowthBytes || minFree != r.Safety.MinimumFreeMemoryPercentObserved {
		t.Fatalf("sample extrema mismatch rss=%d footprint=%d swap=%d free=%d", maxRSS, maxFootprint, maxSwap, minFree)
	}
	if maxConsecutive != r.Safety.RequiredConsecutiveCrossings || maxConsecutive != 3 || r.Safety.KernelPressureEvents != 0 {
		t.Fatalf("trip evidence = consecutive %d kernel events %d", maxConsecutive, r.Safety.KernelPressureEvents)
	}
	if r.Safety.MinimumMemorystatusPercent != 10 || r.Safety.MemorystatusPercentObserved != nil ||
		!strings.Contains(r.Safety.MemorystatusSource, "unbound") {
		t.Fatalf("REJECT must not conflate system-free percent with memorystatus: %+v", r.Safety)
	}
	if r.Timeline.ResponseComplete || r.Timeline.ResponseHeadersReceived || r.Timeline.RerunCount != 0 {
		t.Fatalf("REJECT must bind incomplete response and no rerun: %+v", r.Timeline)
	}
	start := parseTime(t, r.Timeline.RequestStartedAt)
	trip := parseTime(t, r.Timeline.SafetyTripAt)
	if !trip.After(start) || trip.Sub(start) >= 45*time.Minute {
		t.Fatalf("trip timing = %s after request start", trip.Sub(start))
	}
	if !r.Isolation.GPULeaseHeld || !r.Isolation.CandidatePIDScoped || r.Isolation.TeardownSignal != "TERM" ||
		r.Isolation.WatcherMatchedTerms != 0 || r.Isolation.WatcherUnmatchedWorkloads != 0 ||
		r.Isolation.ServiceCommandSHA256 != ownerSHA || !r.Isolation.Restored ||
		r.Isolation.RestoredHealthStatus != 200 || r.Isolation.RestoredModelsStatus != 200 || r.Isolation.RestoredStableSeconds != 90 {
		t.Fatalf("isolation/restoration drift: %+v", r.Isolation)
	}
}

func TestScrubbedNativeLogBindsPathAndContainsNoPrivatePath(t *testing.T) {
	b, err := os.ReadFile("native.log")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{"engine=inkernel backend=metal forward_path=" + forwardPath, "q4k=true fallback=false llama_cpp=false", "signal=TERM", "restore_stable_complete seconds=90", ownerSHA} {
		if !strings.Contains(s, want) {
			t.Fatalf("native.log missing %q", want)
		}
	}
	for _, forbidden := range []string{"/Users/", "/private/", "ssh", "token=", "password=", "secret="} {
		if strings.Contains(strings.ToLower(s), strings.ToLower(forbidden)) {
			t.Fatalf("native.log contains private marker %q", forbidden)
		}
	}
}

func readReceipt(t *testing.T) receipt {
	t.Helper()
	b, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	var r receipt
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func longPrompt(t *testing.T) string {
	t.Helper()
	type fixture struct {
		ID        string `json:"id"`
		Prompt    string `json:"prompt"`
		Expected  string `json:"expected_exact"`
		Generator struct {
			Kind         string `json:"kind"`
			Records      int    `json:"records"`
			NeedleRecord int    `json:"needle_record"`
			Needle       string `json:"needle"`
			Filler       string `json:"filler"`
		} `json:"generator"`
	}
	var corpus struct {
		Fixtures []fixture `json:"fixtures"`
	}
	raw, err := os.ReadFile("../../../docs/benchmarks/qwen38-quant/corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatal(err)
	}
	var f *fixture
	for i := range corpus.Fixtures {
		if corpus.Fixtures[i].ID == "long-context-needle-v1" {
			f = &corpus.Fixtures[i]
			break
		}
	}
	if f == nil {
		t.Fatal("checked-in corpus is missing long-context-needle-v1")
	}
	if f.Generator.Kind != "numbered_records_v1" || f.Generator.Records != 2048 ||
		f.Generator.NeedleRecord != 1537 || f.Generator.Needle != "ORCHID-7319" ||
		f.Generator.Filler != "record-%04d: ordinary telemetry value %04d" || f.Expected != "ORCHID-7319" {
		t.Fatalf("checked-in corpus fixture drift: %+v", *f)
	}
	var b strings.Builder
	b.WriteString(f.Prompt + "\n")
	for i := 1; i <= f.Generator.Records; i++ {
		if i == f.Generator.NeedleRecord {
			fmt.Fprintf(&b, "record-%04d: secret %s\n", i, f.Generator.Needle)
		} else {
			fmt.Fprintf(&b, f.Generator.Filler+"\n", i, i)
		}
	}
	return b.String()
}

func requestJSON(t *testing.T, prompt string) []byte {
	t.Helper()
	v := struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Temperature  int `json:"temperature"`
		MaxTokens    int `json:"max_tokens"`
		ChatTemplate struct {
			EnableThinking bool `json:"enable_thinking"`
		} `json:"chat_template_kwargs"`
	}{Model: "qwen38:27b", Temperature: 0, MaxTokens: 8}
	v.Messages = append(v.Messages, struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "user", Content: prompt})
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func number(t *testing.T, s string) int64 {
	t.Helper()
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return n
}
func parseTime(t *testing.T, s string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
