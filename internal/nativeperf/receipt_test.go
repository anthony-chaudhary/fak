package nativeperf

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/regime"
	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

func validReceipt(t *testing.T, role, lever string) ExperimentReceipt {
	t.Helper()
	r, err := BaselineTemplate(ActiveGraph(), lever)
	if err != nil {
		t.Fatal(err)
	}
	r.Role, r.Revision, r.Machine.ScrubbedID = role, "rev-123", "lab-node-class-a"
	r.Memory = MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
	r.Quality = QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
	criterion, err := ResolveComparisonCriterion(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Comparison, err = comparisonIdentity(criterion)
	if err != nil {
		t.Fatal(err)
	}
	r.ModuleVersions = []ModuleRevision{{Module: "internal/model", Revision: "r10+gaaaaaaa"}}
	r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
	r.ProfilerArtifacts = []ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
	for i := range r.Repetitions {
		r.Repetitions[i] = Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 3 + float64(i)/10}
	}
	if role == RoleCandidate {
		r.ChangedAxes = []string{"lever:" + lever}
	}
	return r
}

func TestMetalAndCUDAReceiptsRoundTrip(t *testing.T) {
	for _, lever := range []string{"metal.command-buffer-amortization", "cuda.q8_1-activation-quant"} {
		r := validReceipt(t, RoleBaseline, lever)
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		got, err := DecodeReceipt(data)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateReceipt(ActiveGraph(), got); err != nil {
			t.Fatal(err)
		}
		if got.EnvelopeID != r.EnvelopeID || got.ChangedLeverID != lever {
			t.Fatalf("round trip drift: %+v", got)
		}
	}
}

func TestValidateReceiptFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ExperimentReceipt)
		want string
	}{
		{"envelope drift", func(r *ExperimentReceipt) { r.EnvelopeID = "other" }, "does not belong"},
		{"missing engine", func(r *ExperimentReceipt) { r.Execution.Engine = "" }, "must name an engine"},
		{"negative fallback", func(r *ExperimentReceipt) { r.Execution.FallbackCount = -1 }, "fallback count"},
		{"missing repetition", func(r *ExperimentReceipt) { r.Repetitions = nil }, "repetition count"},
		{"private host", func(r *ExperimentReceipt) { r.Machine.ScrubbedID = "user@host" }, "private path/host"},
		{"multi axis", func(r *ExperimentReceipt) { r.ChangedAxes = []string{"lever:" + r.ChangedLeverID, "batch"} }, "exactly the candidate lever"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
			tt.edit(&r)
			err := ValidateReceipt(ActiveGraph(), r)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %q", err, tt.want)
			}
		})
	}
}

func TestValidateReceiptRequiresFakNativeZeroFallbackForBothRoles(t *testing.T) {
	for _, role := range []string{RoleBaseline, RoleCandidate} {
		t.Run(role+"/other engine", func(t *testing.T) {
			r := validReceipt(t, role, "metal.command-buffer-amortization")
			r.Execution.Engine = "other"
			err := ValidateReceipt(ActiveGraph(), r)
			if err == nil || !strings.Contains(err.Error(), "engine must be fak-native") {
				t.Fatalf("err=%v", err)
			}
		})
		t.Run(role+"/nonzero fallback", func(t *testing.T) {
			r := validReceipt(t, role, "metal.command-buffer-amortization")
			r.Execution.FallbackCount = 1
			err := ValidateReceipt(ActiveGraph(), r)
			if err == nil || !strings.Contains(err.Error(), "fallback count must be zero") {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestValidateReceiptRejectsTamperedSystemBaseline(t *testing.T) {
	r := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	r.SystemBaselines = []systembaseline.Report{baselineAttestation(systembaseline.VerdictClean, true)}
	r.SystemBaselines[0].CommandExitCode = 9
	err := ValidateReceipt(ActiveGraph(), r)
	if err == nil || !strings.Contains(err.Error(), "canonical digest mismatch") {
		t.Fatalf("err=%v", err)
	}
}

func TestValidateReceiptSchemaCompatibilityAndAggregateAmbient(t *testing.T) {
	legacy := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	legacy.Schema = ReceiptSchemaV1
	if err := ValidateReceipt(ActiveGraph(), legacy); err != nil {
		t.Fatalf("legacy v1: %v", err)
	}
	legacyJSON, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	var legacyFields map[string]json.RawMessage
	if err := json.Unmarshal(legacyJSON, &legacyFields); err != nil {
		t.Fatal(err)
	}
	delete(legacyFields, "comparison")
	legacyJSON, err = json.Marshal(legacyFields)
	if err != nil {
		t.Fatal(err)
	}
	var preComparisonV1 ExperimentReceipt
	if err := json.Unmarshal(legacyJSON, &preComparisonV1); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReceipt(ActiveGraph(), preComparisonV1); err != nil {
		t.Fatalf("pre-comparison v1: %v", err)
	}
	legacy.SystemBaselines = []systembaseline.Report{baselineAttestation(systembaseline.VerdictClean, true)}
	if err := ValidateReceipt(ActiveGraph(), legacy); err == nil || !strings.Contains(err.Error(), "v1 receipts cannot carry") {
		t.Fatalf("legacy ambient err=%v", err)
	}

	v2 := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	attestation := baselineAttestation(systembaseline.VerdictClean, true)
	attestation.Policy.IncludeTopConsumers = true
	attestation.TopNonSUT = []systembaseline.Consumer{{Image: "other.exe", PID: 42, CPUAvailable: true}}
	attestation.Seal()
	v2.SystemBaselines = []systembaseline.Report{attestation}
	if err := ValidateReceipt(ActiveGraph(), v2); err == nil || !strings.Contains(err.Error(), "high-cardinality") {
		t.Fatalf("high-cardinality err=%v", err)
	}
}

func TestAttachSystemBaselineUpgradesAndPreservesRepetitionOrder(t *testing.T) {
	receipt := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	receipt.Schema = ReceiptSchemaV1
	for i := range receipt.Repetitions {
		attestation := baselineAttestation(systembaseline.VerdictClean, true)
		attestation.Window.IntervalNS = int64(i + 1)
		attestation.Seal()
		var err error
		receipt, err = AttachSystemBaseline(ActiveGraph(), receipt, attestation)
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
		if receipt.Schema != ReceiptSchemaV3 || len(receipt.SystemBaselines) != i+1 {
			t.Fatalf("append %d receipt=%+v", i, receipt)
		}
	}
	attestation := baselineAttestation(systembaseline.VerdictClean, true)
	if _, err := AttachSystemBaseline(ActiveGraph(), receipt, attestation); err == nil || !strings.Contains(err.Error(), "cover every repetition") {
		t.Fatalf("overfill err=%v", err)
	}
}

func TestLegacyAmbientEvidenceAndSystemBaselinesAreMutuallyExclusive(t *testing.T) {
	legacy := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	legacy.Schema = ReceiptSchemaV1
	addAmbient(t, &legacy, AmbientClean)
	if err := ValidateReceipt(ActiveGraph(), legacy); err != nil {
		t.Fatalf("published v1 ambient evidence: %v", err)
	}
	attestation := baselineAttestation(systembaseline.VerdictClean, true)
	if _, err := AttachSystemBaseline(ActiveGraph(), legacy, attestation); err == nil || !strings.Contains(err.Error(), "legacy ambient evidence") {
		t.Fatalf("attach over legacy evidence err=%v", err)
	}

	mixed := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	addAmbient(t, &mixed, AmbientClean)
	mixed.SystemBaselines = []systembaseline.Report{attestation}
	mixed.Repetitions[0].SystemBaselineDigest = attestation.Digest
	if err := ValidateReceipt(ActiveGraph(), mixed); err == nil || !strings.Contains(err.Error(), "cannot carry both") {
		t.Fatalf("mixed authority err=%v", err)
	}
}

func TestValidateReceiptRejectsBindingHolesMismatchAndReuse(t *testing.T) {
	base := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	attestation := baselineAttestation(systembaseline.VerdictClean, true)
	bound, err := AttachSystemBaseline(ActiveGraph(), base, attestation)
	if err != nil {
		t.Fatal(err)
	}
	mismatch := bound
	mismatch.Repetitions = append([]Repetition(nil), bound.Repetitions...)
	mismatch.Repetitions[0].SystemBaselineDigest = "sha256:" + strings.Repeat("0", 64)
	if err := ValidateReceipt(ActiveGraph(), mismatch); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatch err=%v", err)
	}
	hole := base
	hole.Repetitions = append([]Repetition(nil), base.Repetitions...)
	hole.Repetitions[1].SystemBaselineDigest = attestation.Digest
	if err := ValidateReceipt(ActiveGraph(), hole); err == nil || !strings.Contains(err.Error(), "binding hole") {
		t.Fatalf("hole err=%v", err)
	}
	if _, err := AttachSystemBaseline(ActiveGraph(), bound, attestation); err == nil || !strings.Contains(err.Error(), "reuses an attestation digest") {
		t.Fatalf("reuse err=%v", err)
	}
	legacy := base
	legacy.Schema = ReceiptSchemaV1
	legacy.Repetitions = append([]Repetition(nil), base.Repetitions...)
	legacy.Repetitions[0].SystemBaselineDigest = attestation.Digest
	if err := ValidateReceipt(ActiveGraph(), legacy); err == nil || !strings.Contains(err.Error(), "v1 repetition") {
		t.Fatalf("legacy binding err=%v", err)
	}
}

func TestCompareReceiptsDeterministicAndRejectsDrift(t *testing.T) {
	b := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	c := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	c.Revision = "rev-456"
	for i := range c.Repetitions {
		c.Repetitions[i].TokensPerSecond++
	}
	first, err := CompareReceipts(ActiveGraph(), b, c)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompareReceipts(ActiveGraph(), b, c)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || math.Abs(first.DeltaTokensPerS-1) > 1e-12 {
		t.Fatalf("comparison drift: %+v %+v", first, second)
	}
	if first.CriterionDigest != b.Comparison.CriterionDigest {
		t.Fatalf("comparison digest=%q, want %q", first.CriterionDigest, b.Comparison.CriterionDigest)
	}
	originalComparison := c.Comparison
	c.Comparison.CriterionDigest = strings.Repeat("f", 64)
	if _, err := CompareReceipts(ActiveGraph(), b, c); err == nil || !strings.Contains(err.Error(), "comparison identity") {
		t.Fatalf("comparison identity drift err=%v", err)
	}
	c.Comparison = originalComparison
	c.Controls.Batch = 2
	if _, err := CompareReceipts(ActiveGraph(), b, c); err == nil || !strings.Contains(err.Error(), "undeclared control axis drifted") {
		t.Fatalf("drift err=%v", err)
	}
	c.Controls.Batch = b.Controls.Batch
	c.Execution.FallbackCount = 1
	if _, err := CompareReceipts(ActiveGraph(), b, c); err == nil || !strings.Contains(err.Error(), "fallback count must be zero") {
		t.Fatalf("fallback drift err=%v", err)
	}
}

func TestDecodePortableReceiptBindsExactBytesAndArtifactSeparately(t *testing.T) {
	r := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	r.Execution.Engine = "fak-native"
	r.Execution.FallbackCount = 0
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodePortableReceipt(raw)
	if err != nil {
		t.Fatal(err)
	}
	wantDigest := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got.ReceiptSHA256 != wantDigest {
		t.Fatalf("receipt digest=%q want %q", got.ReceiptSHA256, wantDigest)
	}
	if got.ModelArtifactSHA256 != r.ArtifactSHA256 || got.ModelArtifactSHA256 == got.ReceiptSHA256 {
		t.Fatalf("artifact identities conflated: %+v", got)
	}
	tampered := append(append([]byte(nil), raw...), '\n')
	tamperedGot, err := DecodePortableReceipt(tampered)
	if err != nil {
		t.Fatal(err)
	}
	if tamperedGot.ReceiptSHA256 == got.ReceiptSHA256 || tamperedGot.ModelArtifactSHA256 != got.ModelArtifactSHA256 {
		t.Fatalf("exact-byte binding failed: original=%+v tampered=%+v", got, tamperedGot)
	}
}

func TestDecodePortableReceiptRejectsInvalidStructureAndRuntimeFallback(t *testing.T) {
	base := validReceipt(t, RoleCandidate, "cuda.q8_1-activation-quant")
	base.Execution.Engine = "fak-native"
	for _, tc := range []struct {
		name string
		edit func(*ExperimentReceipt)
		want string
	}{
		{"structure", func(r *ExperimentReceipt) { r.Controls.Repetitions = 0 }, "controls contain invalid dimensions"},
		{"comparator", func(r *ExperimentReceipt) { r.Execution.Engine = "llama.cpp" }, "fak-native"},
		{"fallback", func(r *ExperimentReceipt) { r.Execution.FallbackCount = 1 }, "fallback count must be zero"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			tc.edit(&r)
			raw, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodePortableReceipt(raw); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want %q", err, tc.want)
			}
		})
	}
}

func f64ptr(v float64) *float64 { return &v }
func u64ptr(v uint64) *uint64   { return &v }

// TestReceiptV3HostOverheadRoundTrip proves the v3 decomposition fields
// survive marshal/decode intact and stay queryable through the summary and
// host-share surfaces.
func TestReceiptV3HostOverheadRoundTrip(t *testing.T) {
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	if r.Schema != ReceiptSchemaV3 {
		t.Fatalf("new template stamps %q, want %q", r.Schema, ReceiptSchemaV3)
	}
	r.Repetitions[0] = Repetition{
		EndToEndMilliseconds: 100, TokensPerSecond: 10, DecodeMilliseconds: 80,
		HostPhaseUs: f64ptr(24000), KernelLaunchCount: u64ptr(1104), InterKernelGapSumUs: f64ptr(42000),
		H2DCopyBytes: u64ptr(2048), SyncRoundtrips: u64ptr(1104),
	}
	// rep 1 records host phase and launches but no decode window wall, so the
	// host-share pair covers rep 0 only.
	r.Repetitions[1] = Repetition{
		EndToEndMilliseconds: 98, TokensPerSecond: 10.2,
		HostPhaseUs: f64ptr(25000), KernelLaunchCount: u64ptr(1104),
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeReceipt(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, r) {
		t.Fatalf("round trip drift:\n got %+v\nwant %+v", got, r)
	}
	if err := ValidateReceipt(ActiveGraph(), got); err != nil {
		t.Fatalf("v3 validation: %v", err)
	}
	s := got.HostOverheadSummary()
	if s.RecordedInRepetitions != 2 || s.HostPhaseUs == nil || *s.HostPhaseUs != 24500 {
		t.Fatalf("summary host_phase_us=%v recorded=%d, want mean 24500 over 2", s.HostPhaseUs, s.RecordedInRepetitions)
	}
	if s.KernelLaunchCount == nil || *s.KernelLaunchCount != 1104 {
		t.Fatalf("summary kernel_launch_count=%v, want 1104", s.KernelLaunchCount)
	}
	// rep 1 records no gap/copy/sync data, so those means cover rep 0 alone.
	if s.InterKernelGapSumUs == nil || *s.InterKernelGapSumUs != 42000 || s.H2DCopyBytes == nil || *s.H2DCopyBytes != 2048 || s.SyncRoundtrips == nil || *s.SyncRoundtrips != 1104 {
		t.Fatalf("summary sparse fields wrong: %+v", s)
	}
	if empty := (ExperimentReceipt{}).HostOverheadSummary(); empty.RecordedInRepetitions != 0 || empty.HostPhaseUs != nil || empty.KernelLaunchCount != nil {
		t.Fatalf("field-less receipt must yield an all-absent summary: %+v", empty)
	}
	share, ok := got.HostShareFromDecomposition()
	if !ok || math.Abs(share-(24000/(80*1e3))) > 1e-12 {
		t.Fatalf("host share=%v ok=%v, want %.6f from rep 0 only (rep 1 records no gap pair)", share, ok, 24000/(80*1e3))
	}
}

// TestReceiptV2BackCompatParsesIdentically guards the mandatory back-compat:
// a committed v2 receipt decodes exactly as before the v3 extension, with the
// new decomposition reading as unknown (nil), never zero.
func TestReceiptV2BackCompatParsesIdentically(t *testing.T) {
	v2JSON := `{
  "schema": "fak-native-performance-receipt/v2",
	  "role": "candidate",
	  "envelope_id": "qwen38-apple-metal-serving",
	  "changed_lever_id": "shared.unified-memory-residency",
	  "revision": "3333333333333333333333333333333333333333",
	  "artifact_sha256": "` + strings.Repeat("c", 64) + `",
	  "machine": {"scrubbed_id": "apple-lab", "platform": "darwin/arm64", "backend": "metal"},
	  "controls": {"prompt_tokens": 64, "decode_tokens": 32, "batch": 1, "context_tokens": 4096, "temperature": 0, "sampling": "greedy", "cache_state": "warm", "warmups": 1, "repetitions": 2},
	  "unchanged_controls": ["artifact_sha256","machine","prompt_tokens","decode_tokens","batch","context_tokens","temperature","sampling","cache_state","warmups","repetitions","execution_identity"],
	  "changed_axes": ["lever:shared.unified-memory-residency"],
	  "repetitions": [
	    {"end_to_end_milliseconds": 100, "tokens_per_second": 10, "decode_milliseconds": 80},
	    {"end_to_end_milliseconds": 98, "tokens_per_second": 10.2}
	  ],
	  "memory": {"peak_bytes": 1000, "resident_bytes": 900},
	  "execution": {"engine": "fak-native", "forward_path": "gguf_lean", "fallback_count": 0},
	  "quality": {"name": "exact_match", "score": 1, "higher_is_better": true},
	  "module_versions": [{"module": "internal/model", "revision": "r10+gaaaaaaa"}],
	  "commands": ["fak run-model --native --receipt-out receipt.json"],
	  "profiler_artifacts": [{"path": "profiles/run.json", "sha256": "` + strings.Repeat("c", 64) + `"}]
	}`
	var got ExperimentReceipt
	if err := json.Unmarshal([]byte(v2JSON), &got); err != nil {
		t.Fatalf("v2 decode: %v", err)
	}
	for i, rep := range got.Repetitions {
		if rep.HasHostOverhead() {
			t.Fatalf("repetition %d: missing v3 fields must read unknown, got %+v", i, rep)
		}
	}
	if got.Repetitions[0].DecodeMilliseconds != 80 || got.Repetitions[1].TokensPerSecond != 10.2 {
		t.Fatalf("v2 core fields drifted: %+v", got.Repetitions)
	}
	// A structurally complete v2 receipt still round-trips identically.
	full := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	full.Schema = ReceiptSchemaV2
	data, err := json.Marshal(full)
	if err != nil {
		t.Fatal(err)
	}
	back, err := DecodeReceipt(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, full) {
		t.Fatalf("v2 round trip drift: %+v", back)
	}
}

// TestReceiptUnknownFieldTolerance pins both directions of the versioning
// contract: fields unknown to this reader because they belong to an older
// schema are tolerated (absent = unknown), while genuinely undeclared keys in
// a current-schema document are still refused - new fields enter only via a
// schema bump.
func TestReceiptUnknownFieldTolerance(t *testing.T) {
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	r.Schema = ReceiptSchemaV2 // v2: the v3 fields are simply absent
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeReceipt(data); err != nil {
		t.Fatalf("v2 receipt with fields unknown to v3 must decode: %v", err)
	}
	var extra map[string]any
	if err := json.Unmarshal(data, &extra); err != nil {
		t.Fatal(err)
	}
	extra["totally_undeclared_v4_field"] = 1
	strict, err := json.Marshal(extra)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeReceipt(strict); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("undeclared key must be refused, err=%v", err)
	}
}

// TestReceiptHostOverheadValidationFailsClosed keeps the decomposition honest:
// only v3 may carry it, and recorded values must be finite and non-negative.
func TestReceiptHostOverheadValidationFailsClosed(t *testing.T) {
	graph := ActiveGraph()
	mislabelled := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	mislabelled.Schema = ReceiptSchemaV2
	mislabelled.Repetitions[0].HostPhaseUs = f64ptr(1000)
	if err := ValidateReceipt(graph, mislabelled); err == nil || !strings.Contains(err.Error(), "require schema v3") {
		t.Fatalf("v2 carrying v3 fields must be refused, err=%v", err)
	}
	negative := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	negative.Repetitions[0].InterKernelGapSumUs = f64ptr(-1)
	if err := ValidateReceipt(graph, negative); err == nil || !strings.Contains(err.Error(), "finite and non-negative") {
		t.Fatalf("negative gap sum must be refused, err=%v", err)
	}
	bare := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	if err := ValidateReceipt(graph, bare); err != nil {
		t.Fatalf("v3 without decomposition data must validate like v2 did: %v", err)
	}
}

// TestFillHostOverheadProjectsOnlyMeasuredFields wires the primary writer
// path: folded decode-window totals populate exactly the fields the collector
// measured, and leave the rest absent instead of fabricating zeros.
func TestFillHostOverheadProjectsOnlyMeasuredFields(t *testing.T) {
	cpu := time.Duration(1500 * int64(time.Microsecond))
	kernels := &compute.KernelProfile{Rows: []compute.KernelRow{{Kernel: "gemm", Calls: 700}, {Kernel: "attn", Calls: 404}}}
	rep := Repetition{}
	rep.FillHostOverhead(DecodeHostWorkTotals{
		CPU: &cpu, Kernels: kernels, SyncsObserved: true,
		SyncSites: []compute.DeviceSyncSite{{Site: "sample", Stage: "decode", Syncs: 1104}},
	})
	if rep.HostPhaseUs == nil || *rep.HostPhaseUs != 1500 || rep.KernelLaunchCount == nil || *rep.KernelLaunchCount != 1104 ||
		rep.SyncRoundtrips == nil || *rep.SyncRoundtrips != 1104 {
		t.Fatalf("measured fields wrong: %+v", rep)
	}
	if rep.InterKernelGapSumUs != nil || rep.H2DCopyBytes != nil {
		t.Fatalf("unmeasured fields must stay absent: %+v", rep)
	}
	partial := Repetition{}
	partial.FillHostOverhead(DecodeHostWorkTotals{CPU: &cpu})
	if partial.HostPhaseUs == nil || partial.KernelLaunchCount != nil || partial.SyncRoundtrips != nil {
		t.Fatalf("partial totals must fill only CPU-derived fields: %+v", partial)
	}
}

// TestRegimeFlowsThroughV3Decomposition is the consumer smoke: the native v3
// inter-kernel gap field reaches the regime classifier through its alias walk
// when present, and a sparse receipt still classifies UNKNOWN rather than a
// guessed verdict.
func TestRegimeFlowsThroughV3Decomposition(t *testing.T) {
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	r.Repetitions[0] = Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 10, DecodeMilliseconds: 80, InterKernelGapSumUs: f64ptr(42000)}
	r.Repetitions[1] = Repetition{EndToEndMilliseconds: 98, TokensPerSecond: 10.2, DecodeMilliseconds: 78}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	p, err := regime.DecodeProfile(data)
	if err != nil {
		t.Fatal(err)
	}
	if p.GapSumUs != 42000 {
		t.Fatalf("gap sum did not flow through: %+v", p)
	}
	if p.TotalUsPerStep != ((80+78)/2)*1e3 {
		t.Fatalf("decode window total=%v us, want mean of decode_milliseconds", p.TotalUsPerStep)
	}
	v := regime.Classify(p)
	if v.Regime != regime.RegimeUnknown {
		t.Fatalf("sparse receipt must stay UNKNOWN, got %s (%s)", v.Regime, strings.Join(v.Evidence, "; "))
	}
	// With the execution sum annotated alongside (as a capture tool's profile
	// note carries it), the native gap field drives the verdict.
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	doc["kernel_execution_sum_us"] = 18000.0
	annotated, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := regime.DecodeProfile(annotated)
	if err != nil {
		t.Fatal(err)
	}
	v2 := regime.Classify(p2)
	if v2.Regime != regime.RegimeLaunchBound {
		t.Fatalf("gap-dominant receipt must classify LAUNCH_BOUND, got %s (%s)", v2.Regime, strings.Join(v2.Evidence, "; "))
	}
}
