package issue8821witness

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const (
	baseRevision = "97ab1e4e3b34b26fd9f901c0a7d12f55b6bd3722"
	artifactSHA  = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	holdDecision = "HOLD_NO_QUALIFYING_CUDA_EVIDENCE"
)

type epicBinding struct {
	Issue int `json:"issue"`
	Row   int `json:"row"`
}

type receipt struct {
	Schema        string      `json:"schema"`
	Issue         int         `json:"issue"`
	Epic          epicBinding `json:"epic"`
	BaseRevision  string      `json:"base_revision"`
	Decision      string      `json:"decision"`
	SelectedLever string      `json:"selected_lever"`
	EvidenceClass string      `json:"evidence_class"`
	Artifact      struct {
		Model        string `json:"model"`
		Repository   string `json:"repository"`
		Revision     string `json:"revision"`
		File         string `json:"file"`
		Quantization string `json:"quantization"`
		Bytes        int64  `json:"bytes"`
		SHA256       string `json:"sha256"`
	} `json:"artifact"`
	Target struct {
		Hardware          string `json:"hardware"`
		ComputeCapability string `json:"compute_capability"`
		Backend           string `json:"backend"`
	} `json:"target"`
	Workload struct {
		PromptTokens  int     `json:"prompt_tokens"`
		ContextTokens int     `json:"context_tokens"`
		OutputTokens  int     `json:"output_tokens"`
		Batch         int     `json:"batch"`
		Temperature   float64 `json:"temperature"`
		Sampling      string  `json:"sampling"`
	} `json:"workload"`
	RequiredExecution struct {
		Engine        string `json:"engine"`
		Runtime       string `json:"runtime"`
		ForwardPath   string `json:"forward_path"`
		FallbackCount int    `json:"fallback_count"`
		LlamaCppRole  string `json:"llama_cpp_role"`
	} `json:"required_execution"`
	QualityGate struct {
		CosineMinimum          float64 `json:"cosine_minimum"`
		ArgmaxExact            bool    `json:"argmax_exact"`
		MaxAbsStrictlyLessThan float64 `json:"max_abs_strictly_less_than"`
		EvidenceAvailable      bool    `json:"evidence_available"`
	} `json:"quality_gate"`
	Accounting struct {
		Inclusive                bool     `json:"inclusive"`
		RequiredPhases           []string `json:"required_phases"`
		WallClockBindingRequired bool     `json:"wall_clock_binding_required"`
		EvidenceAvailable        bool     `json:"evidence_available"`
	} `json:"accounting"`
	PriorWitness struct {
		Issue                   int    `json:"issue"`
		Path                    string `json:"path"`
		Quality                 string `json:"quality"`
		CUDACounterState        string `json:"cuda_counter_state"`
		UsableForLeverSelection bool   `json:"usable_for_lever_selection"`
	} `json:"prior_witness"`
	RouteProbe  string `json:"route_probe"`
	RerunScript string `json:"rerun_script"`
	Ablation    struct {
		MaximumChangedLevers int    `json:"maximum_changed_levers"`
		Performed            bool   `json:"performed"`
		Reason               string `json:"reason"`
	} `json:"ablation"`
	MissingEvidence []string `json:"missing_evidence"`
	Claims          struct {
		PerformanceGain bool `json:"performance_gain"`
		QualityPass     bool `json:"quality_pass"`
		LeverAttributed bool `json:"lever_attributed"`
	} `json:"claims"`
}

type routeProbe struct {
	Schema        string      `json:"schema"`
	Issue         int         `json:"issue"`
	Epic          epicBinding `json:"epic"`
	BaseRevision  string      `json:"base_revision"`
	ObservedAtUTC string      `json:"observed_at_utc"`
	PrivateBridge struct {
		Status   string `json:"status"`
		Eligible bool   `json:"eligible"`
		Detail   string `json:"detail"`
	} `json:"private_bridge"`
	GCP struct {
		Status            string           `json:"status"`
		Eligible          bool             `json:"eligible"`
		AllTiersRequested bool             `json:"all_tiers_requested"`
		Tiers             []map[string]any `json:"tiers"`
		Recommended       any              `json:"recommended"`
		Detail            string           `json:"detail"`
	} `json:"gcp"`
	DryRun struct {
		Status            string `json:"status"`
		Tier              string `json:"tier"`
		Engine            string `json:"engine"`
		Artifact          string `json:"artifact"`
		ProvisioningOnly  bool   `json:"provisioning_only"`
		QualifyingWitness bool   `json:"qualifying_witness"`
	} `json:"dry_run"`
	Decision string `json:"decision"`
	Privacy  struct {
		AccountIncluded     bool `json:"account_included"`
		ProjectIncluded     bool `json:"project_included"`
		HostnameIncluded    bool `json:"hostname_included"`
		PrivatePathIncluded bool `json:"private_path_included"`
		RawLogIncluded      bool `json:"raw_log_included"`
	} `json:"privacy"`
}

func decodeStrict[T any](t *testing.T, name string) T {
	t.Helper()
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var value T
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("strictly decode %s: %v", name, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		t.Fatalf("%s must contain exactly one JSON value, got %v", name, err)
	}
	return value
}

func TestReceiptContract(t *testing.T) {
	r := decodeStrict[receipt](t, "receipt.json")
	if r.Schema != "fak.issue-8821-qwen38-decode-lever-hold/1" || r.Issue != 8821 || r.Epic != (epicBinding{Issue: 10193, Row: 1}) {
		t.Fatalf("wrong receipt identity: schema=%q issue=%d epic=%+v", r.Schema, r.Issue, r.Epic)
	}
	if r.BaseRevision != baseRevision || r.Decision != holdDecision || r.SelectedLever != "" || r.EvidenceClass != "TYPED_HOLD" {
		t.Fatalf("receipt is not the frozen typed HOLD: %+v", r)
	}
	if r.Artifact.Model != "Qwen3.8-27B" || r.Artifact.Repository != "unsloth/Qwen3.8-27B-GGUF" || r.Artifact.Revision != "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe" || r.Artifact.File != "Qwen3.8-27B-Q4_K_M.gguf" || r.Artifact.Quantization != "Q4_K_M" || r.Artifact.Bytes != 17106775008 || r.Artifact.SHA256 != artifactSHA {
		t.Fatalf("artifact binding changed: %+v", r.Artifact)
	}
	if r.Target.Hardware != "NVIDIA A100 40GB" || r.Target.ComputeCapability != "sm_80" || r.Target.Backend != "cuda" {
		t.Fatalf("target changed: %+v", r.Target)
	}
	if r.Workload.PromptTokens != 22 || r.Workload.ContextTokens != 22 || r.Workload.OutputTokens != 6 || r.Workload.Batch != 1 || r.Workload.Temperature != 0 || r.Workload.Sampling != "greedy" {
		t.Fatalf("workload changed: %+v", r.Workload)
	}
	if r.RequiredExecution.Engine != "fak-native" || r.RequiredExecution.Runtime != "inkernel" || r.RequiredExecution.ForwardPath != "cuda/qwen35-gdn-ssm-decode-v1" || r.RequiredExecution.FallbackCount != 0 || r.RequiredExecution.LlamaCppRole != "comparator-reference-only" {
		t.Fatalf("required execution identity changed: %+v", r.RequiredExecution)
	}
	if r.QualityGate.CosineMinimum != 0.995 || !r.QualityGate.ArgmaxExact || r.QualityGate.MaxAbsStrictlyLessThan != 0.02 || r.QualityGate.EvidenceAvailable {
		t.Fatalf("quality HOLD or thresholds changed: %+v", r.QualityGate)
	}
	wantPhases := []string{"setup", "recovery", "prefill", "first-token", "steady-decode", "verification", "teardown"}
	if !r.Accounting.Inclusive || !r.Accounting.WallClockBindingRequired || r.Accounting.EvidenceAvailable || !reflect.DeepEqual(r.Accounting.RequiredPhases, wantPhases) {
		t.Fatalf("inclusive accounting contract changed: %+v", r.Accounting)
	}
	if r.PriorWitness.Issue != 8819 || r.PriorWitness.Quality != "FAILED" || r.PriorWitness.CUDACounterState != "zero counters documented as placeholders; attribution prohibited" || r.PriorWitness.UsableForLeverSelection {
		t.Fatalf("issue 8819 must remain rejected evidence: %+v", r.PriorWitness)
	}
	if r.RouteProbe != "route-probe.json" || r.RerunScript != "sanctioned-rerun.sh" || len(r.MissingEvidence) != 4 {
		t.Fatalf("incomplete recovery contract: route=%q script=%q missing=%v", r.RouteProbe, r.RerunScript, r.MissingEvidence)
	}
	if r.Ablation.MaximumChangedLevers != 1 || r.Ablation.Performed || !strings.Contains(r.Ablation.Reason, "No quality-valid real-counter CUDA profile") {
		t.Fatalf("one-variable ablation contract changed: %+v", r.Ablation)
	}
	if r.Claims.PerformanceGain || r.Claims.QualityPass || r.Claims.LeverAttributed {
		t.Fatalf("HOLD receipt must make no positive claim: %+v", r.Claims)
	}
}

func TestRouteProbeIsScrubbedAndNotEligible(t *testing.T) {
	r := decodeStrict[routeProbe](t, "route-probe.json")
	if r.Schema != "fak.issue-8821-sanctioned-route-probe/1" || r.Issue != 8821 || r.Epic != (epicBinding{Issue: 10193, Row: 1}) || r.BaseRevision != baseRevision || r.ObservedAtUTC != "2026-08-30T03:35:12Z" || r.Decision != holdDecision {
		t.Fatalf("route-probe binding changed: %+v", r)
	}
	if r.PrivateBridge.Status != "NOT_READY" || r.PrivateBridge.Eligible {
		t.Fatalf("private bridge must remain NOT_READY: %+v", r.PrivateBridge)
	}
	if r.GCP.Status != "STALE_AUTH" || r.GCP.Eligible || !r.GCP.AllTiersRequested || len(r.GCP.Tiers) != 0 || r.GCP.Recommended != nil {
		t.Fatalf("GCP route must remain stale-auth with no observed tiers: %+v", r.GCP)
	}
	if r.DryRun.Status != "RENDERED" || r.DryRun.Tier != "a2-high-a100-40gb-1g" || r.DryRun.Engine != "fak-cuda" || r.DryRun.Artifact != "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" || !r.DryRun.ProvisioningOnly || r.DryRun.QualifyingWitness {
		t.Fatalf("dry-run qualification changed: %+v", r.DryRun)
	}
	if r.Privacy.AccountIncluded || r.Privacy.ProjectIncluded || r.Privacy.HostnameIncluded || r.Privacy.PrivatePathIncluded || r.Privacy.RawLogIncluded {
		t.Fatalf("route probe contains private material: %+v", r.Privacy)
	}
}

func TestSanctionedRerunScriptFailsClosed(t *testing.T) {
	info, err := os.Stat("sanctioned-rerun.sh")
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("sanctioned-rerun.sh is not executable: mode=%v", info.Mode())
	}
	data, err := os.ReadFile("sanctioned-rerun.sh")
	if err != nil {
		t.Fatal(err)
	}
	script := string(data)
	required := []string{
		"#!/usr/bin/env bash",
		"set -euo pipefail",
		baseRevision,
		artifactSHA,
		"${GCP_PROJECT:?",
		"${GGUF_PATH:?",
		"sha256sum",
		"tools/gcp_gpu_probe.py",
		"--all-tiers",
		"a2-high-a100-40gb-1g",
		"PROVISIONABLE",
		"tools/gcp_bench.py",
		"--dry-run",
		"--engine fak-cuda",
		"tools/cuda_acceptance.sh",
		"internal/compute/build_cuda.sh",
		"go build -tags cuda",
		"--engine inkernel",
		"--backend cuda",
		"--context-budget-tokens 22",
		"current public tooling lacks full-model GDN per-stage CUDA-event capture",
		"strict cosine/argmax/max-absolute-logit triad",
		"path=cuda/qwen35-gdn-ssm-decode-v1",
		"fallback=0",
		"P/C/O=22/22/6",
		"exit 2",
	}
	for _, needle := range required {
		if !strings.Contains(script, needle) {
			t.Errorf("sanctioned-rerun.sh missing %q", needle)
		}
	}
	if strings.Contains(script, "--account") || strings.Contains(script, "--host") || strings.Contains(script, "ss"+"h ") {
		t.Fatal("sanctioned-rerun.sh must not encode account, host, or SSH details")
	}
}

func TestPublicWitnessHasExactFilesAndNoLeaksOrPositiveClaims(t *testing.T) {
	wantFiles := []string{"README.md", "go.mod", "receipt.json", "route-probe.json", "sanctioned-rerun.sh", "witness_test.go"}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var gotFiles []string
	for _, entry := range entries {
		if !entry.IsDir() {
			gotFiles = append(gotFiles, entry.Name())
		}
	}
	if !reflect.DeepEqual(gotFiles, wantFiles) {
		t.Fatalf("witness file set changed: got %v want %v", gotFiles, wantFiles)
	}

	for _, name := range wantFiles {
		data, err := os.ReadFile(filepath.Clean(name))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		privateMarkers := []string{
			"/" + "Users/",
			"/" + "home/",
			"C:" + "\\\\",
			"_scratch" + "/",
			"invalid" + "_grant",
			".iam." + "gserviceaccount.com",
			"BEGIN " + "PRIVATE KEY",
			"ss" + "h ",
		}
		for _, marker := range privateMarkers {
			if strings.Contains(text, marker) {
				t.Errorf("%s contains forbidden private marker %q", name, marker)
			}
		}
		placeholderMarkers := []string{
			"FILL" + "_ME",
			"TO_BE" + "_FILLED",
			"T" + "BD",
			"REPLACE" + "_ME",
			"<" + "exact-",
			"\"" + "pend" + "ing" + "\"",
			"\"" + "un" + "known" + "\"",
		}
		for _, marker := range placeholderMarkers {
			if strings.Contains(text, marker) {
				t.Errorf("%s contains unresolved placeholder %q", name, marker)
			}
		}
		gainFields := []string{"\"speedup\"", "\"gain_percent\"", "\"delta_percent\"", "\"tokens_per_second\""}
		for _, marker := range gainFields {
			if strings.Contains(text, marker) {
				t.Errorf("%s contains forbidden performance field %q", name, marker)
			}
		}
	}
}

func TestREADMEStatesTheBoundedHold(t *testing.T) {
	data, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	readme := strings.Join(strings.Fields(string(data)), " ")
	required := []string{
		"not a CUDA profile or a performance claim",
		"prompt/context/output token lengths `22/22/6`",
		"cosine similarity at least `0.995`",
		"maximum absolute logit error strictly less than `0.02`",
		"setup, recovery",
		"issue 8819 witness failed its quality gate",
		"zero CUDA device counters are documented placeholders",
		"`NOT_READY`",
		"`STALE_AUTH`",
		"no tiers",
		"`HOLD_NO_QUALIFYING_CUDA_EVIDENCE`",
		"`selected_lever` is empty",
	}
	for _, needle := range required {
		if !strings.Contains(readme, needle) {
			t.Errorf("README.md missing %q", needle)
		}
	}
}

func TestStrictDecoderActuallyRejectsUnknownFields(t *testing.T) {
	data, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	mutated := bytes.Replace(data, []byte("\n  \"issue\":"), []byte("\n  \"unexpected\": true,\n  \"issue\":"), 1)
	decoder := json.NewDecoder(bytes.NewReader(mutated))
	decoder.DisallowUnknownFields()
	var r receipt
	if err := decoder.Decode(&r); err == nil {
		t.Fatal("strict receipt decoder accepted an unknown field")
	}
}
