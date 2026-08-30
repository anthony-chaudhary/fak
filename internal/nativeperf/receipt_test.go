package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

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
		if receipt.Schema != ReceiptSchemaV2 || len(receipt.SystemBaselines) != i+1 {
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
