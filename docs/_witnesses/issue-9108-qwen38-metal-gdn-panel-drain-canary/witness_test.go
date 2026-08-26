package issue9108witness

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

const (
	artifactSHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	promptSHA   = "eafba2115b741f1beae1a131eb11859f72c4e543d6144dd1d76e9b18a3a1fa51"
	requestSHA  = "5097d7d92e795b0f940ec0f2c3f380213ad8d0e8ef23a6e52e986d63614dc974"
	ownerSHA    = "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d"
)

type receipt struct {
	Schema  string `json:"schema"`
	Issue   int    `json:"issue"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
	Source  struct {
		Revision                 string `json:"revision"`
		Tree                     string `json:"tree"`
		BinarySHA256             string `json:"binary_sha256"`
		BuildVCSRequested        bool   `json:"build_vcs_requested"`
		EmbeddedVCSRevisionBound bool   `json:"embedded_vcs_revision_bound"`
		WorktreeCleanAtBuild     bool   `json:"worktree_clean_at_build"`
	} `json:"source"`
	Artifact struct {
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Request struct {
		Fixture                     string `json:"fixture"`
		PromptBytes                 int    `json:"prompt_bytes"`
		PromptSHA256                string `json:"prompt_sha256"`
		BodyBytes                   int    `json:"body_bytes"`
		BodySHA256                  string `json:"body_sha256"`
		PromptTokens                int    `json:"prompt_tokens"`
		OutputTokens                int    `json:"output_tokens"`
		Temperature                 int    `json:"temperature"`
		ExpectedAnswer              string `json:"expected_answer"`
		RequestsSent                int    `json:"requests_sent"`
		RequestBodyBytesTransmitted int    `json:"request_body_bytes_transmitted"`
		ResponseHeadersReceived     int    `json:"response_headers_received"`
		ResponseBodyBytes           int    `json:"response_body_bytes"`
	} `json:"request"`
	Cardinality struct {
		AuthorizedCandidateLaunches int  `json:"authorized_candidate_launches"`
		ObservedCandidateLaunches   int  `json:"observed_candidate_launches"`
		CandidateReruns             int  `json:"candidate_reruns"`
		LaunchCardinalityConsumed   bool `json:"launch_cardinality_consumed"`
	} `json:"cardinality"`
	Candidate struct {
		LaunchedAt                   string  `json:"launched_at"`
		TerminatedAt                 string  `json:"terminated_at"`
		ReadinessObserved            bool    `json:"readiness_observed"`
		HealthProbeCount             int     `json:"health_probe_count"`
		HealthStatus                 int     `json:"health_status"`
		ModelsProbeCount             int     `json:"models_probe_count"`
		ModelsStatus                 int     `json:"models_status"`
		ServiceEngine                string  `json:"service_engine"`
		EndpointEngine               *string `json:"endpoint_engine"`
		EndpointBackend              *string `json:"endpoint_backend"`
		EndpointForwardPath          *string `json:"endpoint_forward_path"`
		EndpointQ4K                  *bool   `json:"endpoint_q4k"`
		EndpointFallbackActive       *bool   `json:"endpoint_fallback_active"`
		CandidateLlamaCPPRuntimeUsed bool    `json:"candidate_llama_cpp_runtime_used"`
		SelectorDefaultEnabled       bool    `json:"selector_default_enabled"`
		LoadWorkers                  int     `json:"load_workers"`
		GOMAXPROCS                   int     `json:"gomaxprocs"`
		TeardownSignal               string  `json:"teardown_signal"`
	} `json:"candidate"`
	Watcher struct {
		Required                            bool   `json:"required"`
		Started                             bool   `json:"started"`
		ExitBeforeFirstSample               bool   `json:"exit_before_first_sample"`
		SampleCount                         int    `json:"sample_count"`
		Failure                             string `json:"failure"`
		MinimumRequiredConsecutiveCrossings int    `json:"minimum_required_consecutive_crossings"`
		MemorystatusSource                  string `json:"memorystatus_source"`
		SystemFreeSource                    string `json:"system_free_source"`
		PeakSwapGrowthBytes                 *int64 `json:"peak_swap_growth_bytes"`
		MinimumMemorystatusObserved         *int64 `json:"minimum_memorystatus_percent_observed"`
		MinimumSystemFreeObserved           *int64 `json:"minimum_system_free_percent_observed"`
		TargetedKernelEvents                *int64 `json:"targeted_kernel_events"`
		UnmatchedWorkloads                  *int64 `json:"unmatched_workloads"`
	} `json:"watcher"`
	Isolation struct {
		GPULeaseHeld              bool   `json:"gpu_lease_held"`
		GPULeaseReleased          bool   `json:"gpu_lease_released"`
		GPULeaseReleaseSignal     string `json:"gpu_lease_release_signal"`
		PriorServiceCommandSHA256 string `json:"prior_service_command_sha256"`
		Restored                  bool   `json:"restored"`
		RestoredHealthStatus      int    `json:"restored_health_status"`
		RestoredModelsStatus      int    `json:"restored_models_status"`
		RestoredStableSeconds     int    `json:"restored_stable_seconds"`
	} `json:"isolation"`
	CorrectedScratchContext struct {
		RelaunchPerformed          bool `json:"relaunch_performed"`
		ProductionAcceptanceCredit bool `json:"production_acceptance_credit"`
	} `json:"corrected_scratch_context"`
}

func TestReceiptFailsClosedOnConsumedLaunchWithoutRequest(t *testing.T) {
	r := readReceipt(t)
	if err := validateReceipt(r); err != nil {
		t.Fatal(err)
	}
	mutants := []func(*receipt){
		func(v *receipt) { v.Cardinality.ObservedCandidateLaunches = 0 },
		func(v *receipt) { v.Request.RequestsSent = 1 },
		func(v *receipt) { v.Watcher.Failure = "" },
		func(v *receipt) { v.Isolation.RestoredStableSeconds = 89 },
		func(v *receipt) {
			s := "metal/qwen35-gdn-preprojected-sequence-v1"
			v.Candidate.EndpointForwardPath = &s
		},
	}
	for i, mutate := range mutants {
		v := r
		mutate(&v)
		if err := validateReceipt(v); err == nil {
			t.Fatalf("tamper mutant %d passed", i)
		}
	}
}

func TestCandidateLogBindsOneLaunchAndZeroRequests(t *testing.T) {
	b := mustRead(t, "candidate.log")
	s := string(b)
	if strings.Count(s, "candidate_launch count=1") != 1 || strings.Count(s, "method=POST") != 0 ||
		strings.Count(s, "method=GET path=/health status=404") != 9 ||
		strings.Count(s, "method=GET path=/v1/models status=200") != 9 {
		t.Fatalf("candidate access cardinality drift")
	}
	for _, want := range []string{
		"watcher_exit before_first_sample=true reason=bsd_awk_third_argument_match_unsupported",
		"request_sent count=0 body_bytes=0 response_headers=0 response_bytes=0",
		"candidate_teardown signal=TERM",
		"gpu_lease released=true signal=TERM",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("candidate.log missing %q", want)
		}
	}
	rows := readTSV(t, "memory-samples.tsv")
	if len(rows) != 1 {
		t.Fatalf("watcher sample rows = %d, want zero data rows", len(rows)-1)
	}
}

func TestRestorationBindsExactOwnerForNinetySeconds(t *testing.T) {
	rows := readTSV(t, "restoration-samples.tsv")
	if len(rows) != 20 {
		t.Fatalf("restoration rows = %d, want 19 data rows", len(rows)-1)
	}
	var pid string
	for i, row := range rows[1:] {
		if len(row) != 5 || row[2] != ownerSHA || row[3] != "200" || row[4] != "200" {
			t.Fatalf("restoration row %d = %v", i+2, row)
		}
		if pid == "" {
			pid = row[1]
		}
		if row[1] != pid {
			t.Fatalf("owner PID drift at row %d", i+2)
		}
	}
	start := parseTime(t, rows[1][0])
	end := parseTime(t, rows[len(rows)-1][0])
	if end.Sub(start) != 91*time.Second {
		t.Fatalf("restoration duration = %s, want 91s", end.Sub(start))
	}
}

func TestFrozenRequestRegeneratesButWasNotTransmitted(t *testing.T) {
	r := readReceipt(t)
	prompt := longPrompt(t)
	body := requestJSON(t, prompt)
	if len(prompt) != r.Request.PromptBytes || digest([]byte(prompt)) != promptSHA || r.Request.PromptSHA256 != promptSHA {
		t.Fatalf("prompt identity drift")
	}
	if len(body) != r.Request.BodyBytes || digest(body) != requestSHA || r.Request.BodySHA256 != requestSHA {
		t.Fatalf("request identity drift")
	}
	if r.Request.RequestsSent != 0 || r.Request.RequestBodyBytesTransmitted != 0 {
		t.Fatalf("generated request must remain explicitly untransmitted")
	}
}

func TestPacketIsPublicAndCorrectedDryChecksGiveNoAcceptanceCredit(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		s := string(mustRead(t, e.Name()))
		for _, forbidden := range []string{
			"/" + "Users/", "/" + "private/", "ssh" + " ",
			"token" + "=", "password" + "=", "secret" + "=",
		} {
			if strings.Contains(strings.ToLower(s), strings.ToLower(forbidden)) {
				t.Fatalf("%s contains private marker %q", e.Name(), forbidden)
			}
		}
	}
	dry := string(mustRead(t, "dry-checks.log"))
	for _, want := range []string{"watcher_swap_parser=green", "cleanup=noop", "full_dry_run=green", "context_only=true", "relaunch_performed=false"} {
		if !strings.Contains(dry, want) {
			t.Fatalf("dry-checks.log missing %q", want)
		}
	}
	r := readReceipt(t)
	if r.CorrectedScratchContext.RelaunchPerformed || r.CorrectedScratchContext.ProductionAcceptanceCredit {
		t.Fatal("corrected scratch checks cannot recover acceptance credit")
	}
}

func validateReceipt(r receipt) error {
	if r.Schema != "fak.issue-9108-qwen38-metal-gdn-panel-drain-canary/1" || r.Issue != 9108 || r.Verdict != "REJECT" {
		return errors.New("receipt identity drift")
	}
	if !strings.Contains(r.Reason, "no request was sent") || r.Source.Revision != "4d48a81c7391f3ab88228531fdb565e6f7f8352a" ||
		r.Source.Tree != "6b398fe2f112aeb22e58ae6054d7ca9d709693e6" ||
		r.Source.BinarySHA256 != "5a90de49a670e14d2636fc182182fe3f13b965d80f17de7a684a4859d765d257" ||
		!r.Source.BuildVCSRequested || r.Source.EmbeddedVCSRevisionBound || !r.Source.WorktreeCleanAtBuild {
		return errors.New("source identity/provenance drift")
	}
	if r.Artifact.Bytes != 17106775008 || r.Artifact.SHA256 != artifactSHA {
		return errors.New("artifact identity drift")
	}
	if r.Request.Fixture != "long-context-needle-v1" || r.Request.PromptTokens != 32800 || r.Request.OutputTokens != 8 ||
		r.Request.Temperature != 0 || r.Request.ExpectedAnswer != "ORCHID-7319" || r.Request.RequestsSent != 0 ||
		r.Request.RequestBodyBytesTransmitted != 0 || r.Request.ResponseHeadersReceived != 0 || r.Request.ResponseBodyBytes != 0 {
		return errors.New("request cardinality drift")
	}
	if r.Cardinality.AuthorizedCandidateLaunches != 1 || r.Cardinality.ObservedCandidateLaunches != 1 ||
		r.Cardinality.CandidateReruns != 0 || !r.Cardinality.LaunchCardinalityConsumed {
		return errors.New("launch cardinality drift")
	}
	if r.Candidate.ReadinessObserved || r.Candidate.HealthProbeCount != 9 || r.Candidate.HealthStatus != 404 ||
		r.Candidate.ModelsProbeCount != 9 || r.Candidate.ModelsStatus != 200 || r.Candidate.ServiceEngine != "inkernel" ||
		r.Candidate.EndpointEngine != nil || r.Candidate.EndpointBackend != nil || r.Candidate.EndpointForwardPath != nil ||
		r.Candidate.EndpointQ4K != nil || r.Candidate.EndpointFallbackActive != nil || r.Candidate.CandidateLlamaCPPRuntimeUsed ||
		r.Candidate.SelectorDefaultEnabled || r.Candidate.LoadWorkers != 12 || r.Candidate.GOMAXPROCS != 12 || r.Candidate.TeardownSignal != "TERM" {
		return errors.New("candidate fail-closed identity drift")
	}
	if !r.Watcher.Required || !r.Watcher.Started || !r.Watcher.ExitBeforeFirstSample || r.Watcher.SampleCount != 0 ||
		!strings.Contains(r.Watcher.Failure, "third argument") || r.Watcher.MinimumRequiredConsecutiveCrossings != 3 ||
		r.Watcher.MemorystatusSource != "sysctl -n kern.memorystatus_level" || r.Watcher.SystemFreeSource != "memory_pressure -Q" ||
		r.Watcher.PeakSwapGrowthBytes != nil || r.Watcher.MinimumMemorystatusObserved != nil ||
		r.Watcher.MinimumSystemFreeObserved != nil || r.Watcher.TargetedKernelEvents != nil || r.Watcher.UnmatchedWorkloads != nil {
		return errors.New("watcher refusal drift")
	}
	if !r.Isolation.GPULeaseHeld || !r.Isolation.GPULeaseReleased || r.Isolation.GPULeaseReleaseSignal != "TERM" ||
		r.Isolation.PriorServiceCommandSHA256 != ownerSHA || !r.Isolation.Restored ||
		r.Isolation.RestoredHealthStatus != 200 || r.Isolation.RestoredModelsStatus != 200 || r.Isolation.RestoredStableSeconds != 91 {
		return errors.New("isolation/restoration drift")
	}
	return nil
}

func readReceipt(t *testing.T) receipt {
	t.Helper()
	var r receipt
	if err := json.Unmarshal(mustRead(t, "receipt.json"), &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func readTSV(t *testing.T, name string) [][]string {
	t.Helper()
	f, err := os.Open(name)
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
	return rows
}

func mustRead(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func parseTime(t *testing.T, s string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatal(err)
	}
	return v
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
	if err := json.Unmarshal(mustRead(t, "../../../docs/benchmarks/qwen38-quant/corpus.json"), &corpus); err != nil {
		t.Fatal(err)
	}
	var f *fixture
	for i := range corpus.Fixtures {
		if corpus.Fixtures[i].ID == "long-context-needle-v1" {
			f = &corpus.Fixtures[i]
			break
		}
	}
	if f == nil || f.Generator.Kind != "numbered_records_v1" || f.Generator.Records != 2048 ||
		f.Generator.NeedleRecord != 1537 || f.Generator.Needle != "ORCHID-7319" || f.Expected != "ORCHID-7319" {
		t.Fatal("checked-in corpus fixture drift")
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
