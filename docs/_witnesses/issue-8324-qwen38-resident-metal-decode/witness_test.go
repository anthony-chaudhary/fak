package issue8324witness

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
	baseRevision = "8fbba932b8128700aef41dd52ab548664a919003"
	artifactSHA  = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	holdVerdict  = "HOLD_CAPACITY_AND_CLAIM_TOPOLOGY_UNPROVEN"
	writeRoot    = "docs/_witnesses/issue-8324-qwen38-resident-metal-decode/"
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
	Verdict       string      `json:"verdict"`
	EvidenceClass string      `json:"evidence_class"`
	Reason        string      `json:"reason"`
	Stability     struct {
		Typed             bool     `json:"typed"`
		PerformanceCredit bool     `json:"performance_credit"`
		QualityCredit     bool     `json:"quality_credit"`
		DefaultCredit     bool     `json:"default_credit"`
		SupersedeOnlyWhen []string `json:"supersede_only_when"`
	} `json:"stability"`
	Artifact struct {
		Model              string `json:"model"`
		Repository         string `json:"repository"`
		Revision           string `json:"revision"`
		File               string `json:"file"`
		Quantization       string `json:"quantization"`
		Bytes              int64  `json:"bytes"`
		SHA256             string `json:"sha256"`
		Status             string `json:"status"`
		PathDisclosed      bool   `json:"path_disclosed"`
		VerificationSource string `json:"verification_source"`
	} `json:"artifact"`
	RequiredExecution struct {
		Engine                 string `json:"engine"`
		Runtime                string `json:"runtime"`
		Backend                string `json:"backend"`
		FallbackCount          int    `json:"fallback_count"`
		ExternalRuntimeAllowed bool   `json:"external_runtime_allowed"`
	} `json:"required_execution"`
	Target struct {
		OS                            string `json:"os"`
		Arch                          string `json:"arch"`
		Accelerator                   string `json:"accelerator"`
		MinimumMemoryGiB              int    `json:"minimum_memory_gib"`
		ObservedLocalMemoryGiB        int    `json:"observed_local_memory_gib"`
		ObservedLocalEligible         bool   `json:"observed_local_eligible"`
		PublicVerifiedTargetAvailable bool   `json:"public_verified_target_available"`
		PrivateBridgeStatus           string `json:"private_bridge_status"`
	} `json:"target"`
	Workload struct {
		PromptTokens      int     `json:"prompt_tokens"`
		DecodeTokens      int     `json:"decode_tokens"`
		RepetitionsPerArm int     `json:"repetitions_per_arm"`
		Arms              int     `json:"arms"`
		Batch             int     `json:"batch"`
		Temperature       float64 `json:"temperature"`
		Sampling          string  `json:"sampling"`
	} `json:"workload"`
	CurrentMechanisms []struct {
		Issue                         int    `json:"issue"`
		Commit                        string `json:"commit"`
		PresentAtBase                 bool   `json:"present_at_base"`
		Scope                         string `json:"scope"`
		LinearAttentionThroughMLP     bool   `json:"linear_attention_through_mlp"`
		PeriodicFullAttentionResident bool   `json:"periodic_full_attention_resident"`
		WholeTokenAllLayersResident   bool   `json:"whole_token_all_layers_resident"`
	} `json:"current_mechanisms"`
	CurrentReceiptTopology struct {
		EnvelopeID                            string `json:"envelope_id"`
		EnvelopeMemoryGiB                     int    `json:"envelope_memory_gib"`
		SupportsRequired64GiBTarget           bool   `json:"supports_required_64gib_target"`
		ExportsStageTiming                    bool   `json:"exports_stage_timing"`
		ExportsPeriodicFullAttentionResidency bool   `json:"exports_periodic_full_attention_residency"`
		ExportsExactMultiTokenCosine          bool   `json:"exports_exact_multi_token_cosine"`
		ExportsMatchingGreedyTokens           bool   `json:"exports_matching_greedy_tokens"`
		ExportsDistinctRecoveryAccounting     bool   `json:"exports_distinct_recovery_accounting"`
	} `json:"current_receipt_topology"`
	Acceptance struct {
		ExactArtifactRequired                  bool    `json:"exact_artifact_required"`
		StageTimingRequired                    bool    `json:"stage_timing_required"`
		ResidentDenseGDNAllLayersRequired      bool    `json:"resident_dense_gdn_all_layers_required"`
		PeriodicFullAttentionAllLayersRequired bool    `json:"periodic_full_attention_all_layers_required"`
		WholeTokenResidencyRequired            bool    `json:"whole_token_residency_required"`
		CosineMinimum                          float64 `json:"cosine_minimum"`
		ExactMultiTokenCosineRequired          bool    `json:"exact_multi_token_cosine_required"`
		MatchingGreedyTokensRequired           bool    `json:"matching_greedy_tokens_required"`
		Baseline                               struct {
			FullPrefillTokensPerSecond float64 `json:"full_prefill_tokens_per_second"`
			CachedMinTokensPerSecond   float64 `json:"cached_min_tokens_per_second"`
			CachedMaxTokensPerSecond   float64 `json:"cached_max_tokens_per_second"`
			ExactMatchedABRequired     bool    `json:"exact_matched_ab_required"`
			EvidenceAvailable          bool    `json:"evidence_available"`
		} `json:"baseline"`
		TargetTokensPerSecondMinimum             float64 `json:"target_tokens_per_second_minimum"`
		BeforeDefaultRequired                    bool    `json:"before_default_required"`
		UnsupportedGeometryLoggedDeclineRequired bool    `json:"unsupported_geometry_logged_decline_required"`
		UnsupportedGeometryClaimAllowed          bool    `json:"unsupported_geometry_claim_allowed"`
		EvidenceAvailable                        bool    `json:"evidence_available"`
	} `json:"acceptance"`
	Accounting struct {
		Inclusive                bool     `json:"inclusive"`
		RequiredCategories       []string `json:"required_categories"`
		RequiredPhases           []string `json:"required_phases"`
		WallClockBindingRequired bool     `json:"wall_clock_binding_required"`
		EvidenceAvailable        bool     `json:"evidence_available"`
	} `json:"accounting"`
	MissingEvidence []string `json:"missing_evidence"`
	Claims          struct {
		Performance                bool `json:"performance"`
		Quality                    bool `json:"quality"`
		Default                    bool `json:"default"`
		ResidentHybridTopology     bool `json:"resident_hybrid_topology"`
		UnsupportedGeometrySupport bool `json:"unsupported_geometry_support"`
	} `json:"claims"`
	PriorArt []struct {
		Project           string `json:"project"`
		Revision          string `json:"revision"`
		Route             string `json:"route"`
		Invariant         string `json:"invariant"`
		RuntimeDependency bool   `json:"runtime_dependency"`
	} `json:"prior_art"`
	Route struct {
		Probe                      string `json:"probe"`
		Script                     string `json:"script"`
		Command                    string `json:"command"`
		CommandReady               bool   `json:"command_ready"`
		PerformanceAcceptanceReady bool   `json:"performance_acceptance_ready"`
	} `json:"route"`
	Scope struct {
		WriteRoot    string   `json:"write_root"`
		ChangedPaths []string `json:"changed_paths"`
	} `json:"scope"`
}

type routeProbe struct {
	Schema        string      `json:"schema"`
	Issue         int         `json:"issue"`
	Epic          epicBinding `json:"epic"`
	BaseRevision  string      `json:"base_revision"`
	ObservedOnUTC string      `json:"observed_on_utc"`
	Artifact      struct {
		Status                           string `json:"status"`
		File                             string `json:"file"`
		Bytes                            int64  `json:"bytes"`
		SHA256                           string `json:"sha256"`
		PrivatePathIncluded              bool   `json:"private_path_included"`
		EligibleIfCopiedToSanctionedNode bool   `json:"eligible_if_copied_to_sanctioned_node"`
	} `json:"artifact"`
	LocalHost struct {
		Status             string `json:"status"`
		OS                 string `json:"os"`
		Arch               string `json:"arch"`
		Model              string `json:"model"`
		ModelIdentifier    string `json:"model_identifier"`
		MemoryBytes        int64  `json:"memory_bytes"`
		MemoryGiB          int    `json:"memory_gib"`
		MinimumRequiredGiB int    `json:"minimum_required_gib"`
		Eligible           bool   `json:"eligible"`
	} `json:"local_host"`
	PublicAppleTarget struct {
		Status             string `json:"status"`
		MinimumRequiredGiB int    `json:"minimum_required_gib"`
		Eligible           bool   `json:"eligible"`
		Detail             string `json:"detail"`
	} `json:"public_apple_target"`
	PrivateBridge struct {
		DoctorStatus string `json:"doctor_status"`
		Eligible     bool   `json:"eligible"`
		Detail       string `json:"detail"`
	} `json:"private_bridge"`
	ReceiptRoute struct {
		CurrentEnvelopeID                      string `json:"current_envelope_id"`
		CurrentEnvelopeMemoryGiB               int    `json:"current_envelope_memory_gib"`
		SupportsRequired64GiBTarget            bool   `json:"supports_required_64gib_target"`
		PeriodicFullAttentionResidencyProvable bool   `json:"periodic_full_attention_residency_provable"`
		ExactMultiTokenCosineProvable          bool   `json:"exact_multi_token_cosine_provable"`
		MatchingGreedyTokensProvable           bool   `json:"matching_greedy_tokens_provable"`
		CurrentCommandsPrintable               bool   `json:"current_commands_printable"`
		PerformanceAcceptancePossible          bool   `json:"performance_acceptance_possible"`
	} `json:"receipt_route"`
	SanctionedRoute struct {
		Script                      string `json:"script"`
		RequiredOS                  string `json:"required_os"`
		RequiredArch                string `json:"required_arch"`
		RequiredAccelerator         string `json:"required_accelerator"`
		MinimumMemoryGiB            int    `json:"minimum_memory_gib"`
		ExplicitSanctionRequired    bool   `json:"explicit_sanction_required"`
		ExactArtifactDigestRequired bool   `json:"exact_artifact_digest_required"`
		CommandReady                bool   `json:"command_ready"`
		PerformanceAcceptanceReady  bool   `json:"performance_acceptance_ready"`
	} `json:"sanctioned_route"`
	Decision string `json:"decision"`
	Privacy  struct {
		AccountIncluded       bool `json:"account_included"`
		HostnameIncluded      bool `json:"hostname_included"`
		PrivatePathIncluded   bool `json:"private_path_included"`
		CredentialIncluded    bool `json:"credential_included"`
		RawPrivateLogIncluded bool `json:"raw_private_log_included"`
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

func TestReceiptPinsTypedHoldAcceptanceAndClaims(t *testing.T) {
	r := decodeStrict[receipt](t, "receipt.json")
	if r.Schema != "fak.issue-8324-qwen38-resident-metal-decode-hold/1" || r.Issue != 8324 || r.Epic != (epicBinding{Issue: 10193, Row: 6}) || r.BaseRevision != baseRevision || r.Verdict != holdVerdict || r.EvidenceClass != "TYPED_HOLD" {
		t.Fatalf("receipt identity drift: %+v", r)
	}
	if !strings.Contains(r.Reason, "exact artifact is available and verified") || !strings.Contains(r.Reason, "36 GiB") || !strings.Contains(r.Reason, "periodic full-attention") || !strings.Contains(r.Reason, "multi-token cosine") {
		t.Fatalf("HOLD reason is incomplete: %q", r.Reason)
	}
	if !r.Stability.Typed || r.Stability.PerformanceCredit || r.Stability.QualityCredit || r.Stability.DefaultCredit || len(r.Stability.SupersedeOnlyWhen) != 3 {
		t.Fatalf("stable typed HOLD changed: %+v", r.Stability)
	}
	if r.Artifact.Model != "Qwen3.8-27B" || r.Artifact.Repository != "unsloth/Qwen3.8-27B-GGUF" || r.Artifact.Revision != "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe" || r.Artifact.File != "Qwen3.8-27B-Q4_K_M.gguf" || r.Artifact.Quantization != "Q4_K_M" || r.Artifact.Bytes != 17106775008 || r.Artifact.SHA256 != artifactSHA || r.Artifact.Status != "AVAILABLE_VERIFIED" || r.Artifact.PathDisclosed || r.Artifact.VerificationSource != "independent readback supplied to the issue worker" {
		t.Fatalf("artifact binding drift: %+v", r.Artifact)
	}
	if r.RequiredExecution.Engine != "fak-native" || r.RequiredExecution.Runtime != "inkernel" || r.RequiredExecution.Backend != "metal" || r.RequiredExecution.FallbackCount != 0 || r.RequiredExecution.ExternalRuntimeAllowed {
		t.Fatalf("execution identity drift: %+v", r.RequiredExecution)
	}
	if r.Target.OS != "darwin" || r.Target.Arch != "arm64" || r.Target.Accelerator != "Apple Metal" || r.Target.MinimumMemoryGiB != 64 || r.Target.ObservedLocalMemoryGiB != 36 || r.Target.ObservedLocalEligible || r.Target.PublicVerifiedTargetAvailable || r.Target.PrivateBridgeStatus != "NOT_READY" {
		t.Fatalf("target HOLD drift: %+v", r.Target)
	}
	if r.Workload.PromptTokens != 32 || r.Workload.DecodeTokens != 64 || r.Workload.RepetitionsPerArm != 3 || r.Workload.Arms != 6 || r.Workload.Batch != 1 || r.Workload.Temperature != 0 || r.Workload.Sampling != "greedy" {
		t.Fatalf("workload drift: %+v", r.Workload)
	}
	if len(r.CurrentMechanisms) != 2 || r.CurrentMechanisms[0].Issue != 9486 || r.CurrentMechanisms[0].Commit != "46fdd8a52fd70b3e29345cd311be3cc89443e8fc" || !r.CurrentMechanisms[0].PresentAtBase || r.CurrentMechanisms[0].LinearAttentionThroughMLP || r.CurrentMechanisms[0].PeriodicFullAttentionResident || r.CurrentMechanisms[0].WholeTokenAllLayersResident || !strings.Contains(r.CurrentMechanisms[0].Scope, "linear-attention projection") || r.CurrentMechanisms[1].Issue != 9488 || r.CurrentMechanisms[1].Commit != "99ea660ae222dd6a75dd661c54778f470904f9e7" || !r.CurrentMechanisms[1].PresentAtBase || !r.CurrentMechanisms[1].LinearAttentionThroughMLP || r.CurrentMechanisms[1].PeriodicFullAttentionResident || r.CurrentMechanisms[1].WholeTokenAllLayersResident || !strings.Contains(r.CurrentMechanisms[1].Scope, "SwiGLU MLP") {
		t.Fatalf("mechanism boundary drift: %+v", r.CurrentMechanisms)
	}
	top := r.CurrentReceiptTopology
	if top.EnvelopeID != "qwen38-27b-q4km-m3pro-p32-t64" || top.EnvelopeMemoryGiB != 36 || top.SupportsRequired64GiBTarget || !top.ExportsStageTiming || top.ExportsPeriodicFullAttentionResidency || top.ExportsExactMultiTokenCosine || top.ExportsMatchingGreedyTokens || top.ExportsDistinctRecoveryAccounting {
		t.Fatalf("receipt topology gap drift: %+v", top)
	}
	a := r.Acceptance
	if !a.ExactArtifactRequired || !a.StageTimingRequired || !a.ResidentDenseGDNAllLayersRequired || !a.PeriodicFullAttentionAllLayersRequired || !a.WholeTokenResidencyRequired || a.CosineMinimum != 0.9999 || !a.ExactMultiTokenCosineRequired || !a.MatchingGreedyTokensRequired || a.Baseline.FullPrefillTokensPerSecond != 2.9 || a.Baseline.CachedMinTokensPerSecond != 0.4 || a.Baseline.CachedMaxTokensPerSecond != 1.3 || !a.Baseline.ExactMatchedABRequired || a.Baseline.EvidenceAvailable || a.TargetTokensPerSecondMinimum != 5 || !a.BeforeDefaultRequired || !a.UnsupportedGeometryLoggedDeclineRequired || a.UnsupportedGeometryClaimAllowed || a.EvidenceAvailable {
		t.Fatalf("acceptance gate drift: %+v", a)
	}
	wantCategories := []string{"setup", "recovery", "inference", "verification", "teardown"}
	wantPhases := []string{"setup", "recovery", "prefill", "first-token", "steady-decode", "verification", "teardown"}
	if !r.Accounting.Inclusive || !reflect.DeepEqual(r.Accounting.RequiredCategories, wantCategories) || !reflect.DeepEqual(r.Accounting.RequiredPhases, wantPhases) || !r.Accounting.WallClockBindingRequired || r.Accounting.EvidenceAvailable {
		t.Fatalf("inclusive accounting drift: %+v", r.Accounting)
	}
	wantMissing := []string{
		"public verified sanctioned darwin/arm64 Apple target with at least 64 GiB",
		"claim-grade receipt support for the sanctioned >=64 GiB host geometry",
		"whole-token residency across dense GDN and periodic full-attention layers",
		"exact multi-token cosine >=0.9999 and matching greedy-token capture",
		"exact matched six-arm A/B against both frozen baselines with inclusive accounting",
		"logged unsupported-geometry decline from the claim-grade route",
	}
	if !reflect.DeepEqual(r.MissingEvidence, wantMissing) {
		t.Fatalf("missing evidence drift: %v", r.MissingEvidence)
	}
	if r.Claims.Performance || r.Claims.Quality || r.Claims.Default || r.Claims.ResidentHybridTopology || r.Claims.UnsupportedGeometrySupport {
		t.Fatalf("HOLD made a positive claim: %+v", r.Claims)
	}
	wantRouteCommand := "FAK_SANCTIONED_APPLE_NODE=YES GGUF_PATH=<absolute-exact-artifact> OUT_DIR=<absolute-output-dir> ./docs/_witnesses/issue-8324-qwen38-resident-metal-decode/sanctioned-rerun.sh"
	if r.Route.Probe != "route-probe.json" || r.Route.Script != "sanctioned-rerun.sh" || r.Route.Command != wantRouteCommand || !r.Route.CommandReady || r.Route.PerformanceAcceptanceReady {
		t.Fatalf("route drift: %+v", r.Route)
	}
}

func TestReceiptPinsBorrowOnlyPriorArtAndScopedPaths(t *testing.T) {
	r := decodeStrict[receipt](t, "receipt.json")
	wantPrior := []struct{ project, revision, invariant string }{
		{"llama.cpp", "ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0", "Qwen graph and recurrent-state ordering"},
		{"MLX", "43d2f06cb87e76895bf9a152bade4fee83408643", "caller-owned Metal CommandEncoder lifetime"},
		{"MLX-LM", "cc8521569694a3240b52c98acffd100d59b4c755", "GDN semantics and state transitions"},
	}
	if len(r.PriorArt) != len(wantPrior) {
		t.Fatalf("prior-art cardinality=%d", len(r.PriorArt))
	}
	for i, want := range wantPrior {
		got := r.PriorArt[i]
		if got.Project != want.project || got.Revision != want.revision || got.Route != "BORROW_INVARIANTS_ONLY" || got.Invariant != want.invariant || got.RuntimeDependency {
			t.Fatalf("prior-art row %d drift: %+v", i, got)
		}
	}
	wantPaths := []string{
		writeRoot + "README.md",
		writeRoot + "go.mod",
		writeRoot + "receipt.json",
		writeRoot + "route-probe.json",
		writeRoot + "sanctioned-rerun.sh",
		writeRoot + "witness_test.go",
	}
	if r.Scope.WriteRoot != writeRoot || !reflect.DeepEqual(r.Scope.ChangedPaths, wantPaths) {
		t.Fatalf("changed-path receipt drift: %+v", r.Scope)
	}
	for _, path := range r.Scope.ChangedPaths {
		if !strings.HasPrefix(path, writeRoot) || filepath.Clean(path) == filepath.Clean(strings.TrimSuffix(writeRoot, "/")) {
			t.Fatalf("out-of-scope path named as changed: %q", path)
		}
	}
}

func TestRouteProbePinsAvailabilityCapacityTopologyAndPrivacy(t *testing.T) {
	r := decodeStrict[routeProbe](t, "route-probe.json")
	if r.Schema != "fak.issue-8324-qwen38-resident-metal-decode-route-probe/1" || r.Issue != 8324 || r.Epic != (epicBinding{Issue: 10193, Row: 6}) || r.BaseRevision != baseRevision || r.ObservedOnUTC != "2026-08-29" || r.Decision != holdVerdict {
		t.Fatalf("route identity drift: %+v", r)
	}
	if r.Artifact.Status != "AVAILABLE_VERIFIED" || r.Artifact.File != "Qwen3.8-27B-Q4_K_M.gguf" || r.Artifact.Bytes != 17106775008 || r.Artifact.SHA256 != artifactSHA || r.Artifact.PrivatePathIncluded || !r.Artifact.EligibleIfCopiedToSanctionedNode {
		t.Fatalf("route artifact drift: %+v", r.Artifact)
	}
	if r.LocalHost.Status != "CAPACITY_INELIGIBLE" || r.LocalHost.OS != "darwin" || r.LocalHost.Arch != "arm64" || r.LocalHost.Model != "Apple M3 Pro" || r.LocalHost.ModelIdentifier != "Mac15,7" || r.LocalHost.MemoryBytes != 38654705664 || r.LocalHost.MemoryGiB != 36 || r.LocalHost.MinimumRequiredGiB != 64 || r.LocalHost.Eligible {
		t.Fatalf("local capacity drift: %+v", r.LocalHost)
	}
	if r.PublicAppleTarget.Status != "NO_PUBLIC_VERIFIED_GE64GIB_TARGET" || r.PublicAppleTarget.MinimumRequiredGiB != 64 || r.PublicAppleTarget.Eligible || r.PublicAppleTarget.Detail != "No public target identity and availability receipt was supplied or observed." {
		t.Fatalf("public route drift: %+v", r.PublicAppleTarget)
	}
	if r.PrivateBridge.DoctorStatus != "NOT_READY" || r.PrivateBridge.Eligible || r.PrivateBridge.Detail != "The scrubbed private bridge doctor did not yield a sanctioned Apple session." {
		t.Fatalf("private bridge drift: %+v", r.PrivateBridge)
	}
	rr := r.ReceiptRoute
	if rr.CurrentEnvelopeID != "qwen38-27b-q4km-m3pro-p32-t64" || rr.CurrentEnvelopeMemoryGiB != 36 || rr.SupportsRequired64GiBTarget || rr.PeriodicFullAttentionResidencyProvable || rr.ExactMultiTokenCosineProvable || rr.MatchingGreedyTokensProvable || !rr.CurrentCommandsPrintable || rr.PerformanceAcceptancePossible {
		t.Fatalf("receipt route drift: %+v", rr)
	}
	s := r.SanctionedRoute
	if s.Script != "sanctioned-rerun.sh" || s.RequiredOS != "darwin" || s.RequiredArch != "arm64" || s.RequiredAccelerator != "Apple Metal" || s.MinimumMemoryGiB != 64 || !s.ExplicitSanctionRequired || !s.ExactArtifactDigestRequired || !s.CommandReady || s.PerformanceAcceptanceReady {
		t.Fatalf("sanctioned route drift: %+v", s)
	}
	if r.Privacy.AccountIncluded || r.Privacy.HostnameIncluded || r.Privacy.PrivatePathIncluded || r.Privacy.CredentialIncluded || r.Privacy.RawPrivateLogIncluded {
		t.Fatalf("private data admitted: %+v", r.Privacy)
	}
}

func TestSanctionedScriptIsExecutableFailClosedExactAndPublicSafe(t *testing.T) {
	info, err := os.Stat("sanctioned-rerun.sh")
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&0o111 == 0 {
		t.Fatalf("sanctioned-rerun.sh mode %o is not executable", info.Mode().Perm())
	}
	data, err := os.ReadFile("sanctioned-rerun.sh")
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, want := range []string{
		"set -euo pipefail",
		"FAK_SANCTIONED_APPLE_NODE",
		"MINIMUM_MEMORY_BYTES",
		"EXPECTED_BYTES=\"17106775008\"",
		"EXPECTED_SHA256=\"" + artifactSHA + "\"",
		"git merge-base --is-ancestor \"$MIXER_REVISION\" HEAD",
		"git merge-base --is-ancestor \"$BLOCK_REVISION\" HEAD",
		"CGO_ENABLED=1 go test ./internal/model",
		"CGO_ENABLED=1 go test ./cmd/modelbench",
		"-native-performance-qwen35-decode-handoff=CONTROL",
		"-native-performance-qwen35-decode-handoff=MIXER",
		"-native-performance-readback=\"$p\"",
		"-native-performance-compare-phase=steady-decode",
		"-native-performance-compare-phase=end-to-end",
		"-native-performance-compare-axis=m3-decode-handoff",
		"periodic full-attention whole-token residency",
		"exact multi-token cosine >=0.9999",
		"matching greedy-token parity",
		"exit 2",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("sanctioned-rerun.sh missing %q", want)
		}
	}
	if strings.Count(s, "-native-performance-profile=\"$OUT_DIR/control-$i.json\"") != 1 || strings.Count(s, "-native-performance-profile=\"$OUT_DIR/mixer-$i.json\"") != 1 {
		t.Fatal("script does not print exactly one three-run loop per arm")
	}
	for _, forbidden := range []string{"/Users/", "/private/", "ssh ", "token=", "password=", "secret="} {
		if strings.Contains(strings.ToLower(s), strings.ToLower(forbidden)) {
			t.Fatalf("script exposes private marker %q", forbidden)
		}
	}
}
