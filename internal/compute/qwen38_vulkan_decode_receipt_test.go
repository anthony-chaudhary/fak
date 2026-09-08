package compute

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestQwen38VulkanDecodeReceiptValidAndComparable(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{151644, 8948, 198}, 4)
	receipt := validQwen38VulkanDecodeReceipt(t, packet)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	candidate := receipt
	candidate.Runs = slices.Clone(receipt.Runs)
	candidate.Runs[0].DecodeNanoseconds--
	candidate.Runs[0].CandidateElapsedNanoseconds--
	candidate.CandidateElapsedNanoseconds--
	candidate.Counters.DispatchSubmits++
	if err := CompareQwen38VulkanDecodeReceipts(receipt, candidate); err != nil {
		t.Fatalf("CompareQwen38VulkanDecodeReceipts() error = %v", err)
	}
}

func TestQwen38VulkanRawDecodeResultCannotBecomePhysicalReceiptWithoutIdentity(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Qwen38VulkanRawDecodeResult)
		want string
	}{
		{"source commit", func(r *Qwen38VulkanRawDecodeResult) { r.Source.GitCommit = "" }, "source commit"},
		{"source archive", func(r *Qwen38VulkanRawDecodeResult) { r.Source.SourceArchiveSHA256 = "" }, "source archive"},
		{"binary", func(r *Qwen38VulkanRawDecodeResult) { r.Source.BinarySHA256 = "" }, "binary"},
		{"tree state", func(r *Qwen38VulkanRawDecodeResult) { r.Source.Dirty = nil }, "clean/dirty"},
		{"clean with diff", func(r *Qwen38VulkanRawDecodeResult) { r.Source.DiffSHA256 = strings.Repeat("1", 64) }, "clean source cannot include"},
		{"dirty without diff", func(r *Qwen38VulkanRawDecodeResult) { dirty := true; r.Source.Dirty = &dirty }, "dirty source requires diff"},
		{"model", func(r *Qwen38VulkanRawDecodeResult) { r.Model.Name = "" }, "model identity"},
		{"artifact", func(r *Qwen38VulkanRawDecodeResult) { r.Model.ArtifactSHA256 = "" }, "artifact"},
		{"tensor inventory", func(r *Qwen38VulkanRawDecodeResult) { r.Model.TensorInventorySHA256 = "" }, "tensor inventory"},
		{"tokenizer", func(r *Qwen38VulkanRawDecodeResult) { r.Model.TokenizerSHA256 = "" }, "tokenizer"},
		{"template", func(r *Qwen38VulkanRawDecodeResult) { r.Model.TemplateSHA256 = "" }, "template"},
		{"device", func(r *Qwen38VulkanRawDecodeResult) { r.Device.Name = "" }, "device identity"},
		{"engine", func(r *Qwen38VulkanRawDecodeResult) { r.Engine.Name = "" }, "engine=fak-native"},
		{"fallback observation", func(r *Qwen38VulkanRawDecodeResult) { r.Engine.FallbackCount = nil }, "fallback_count=0"},
		{"output IDs", func(r *Qwen38VulkanRawDecodeResult) { r.OutputTokenIDs = nil }, "work boundary"},
		{"output text", func(r *Qwen38VulkanRawDecodeResult) { r.OutputText = "" }, "output text"},
		{"finite logits absent", func(r *Qwen38VulkanRawDecodeResult) { r.FiniteLogits = nil }, "finite_logits=true"},
		{"finite logits false", func(r *Qwen38VulkanRawDecodeResult) { value := false; r.FiniteLogits = &value }, "finite_logits=true"},
		{"parity absent", func(r *Qwen38VulkanRawDecodeResult) { r.CPUModelParity = nil }, "cpu_model_parity=true"},
		{"parity false", func(r *Qwen38VulkanRawDecodeResult) { value := false; r.CPUModelParity = &value }, "cpu_model_parity=true"},
		{"runs absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs = nil }, "per-repetition"},
		{"context boundary", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].ContextTokens++ }, "context boundary"},
		{"generated boundary", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].ActualGeneratedTokens--; r.Runs[0].ContextTokens-- }, "generated-token boundary"},
		{"sampler", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].Sampler = "sampled" }, "sampler=greedy"},
		{"seed policy", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].SeedPolicy = "0" }, "seed_policy=not_applicable_greedy"},
		{"EOS policy", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].IgnoreEOS = nil }, "EOS policy"},
		{"candidate includes oracle", func(r *Qwen38VulkanRawDecodeResult) {
			r.Runs[0].CandidateElapsedNanoseconds += r.Runs[0].CPUVerificationNanoseconds
		}, "candidate timing"},
		{"top-level candidate mismatch", func(r *Qwen38VulkanRawDecodeResult) { r.CandidateElapsedNanoseconds++ }, "candidate elapsed time"},
		{"top-level CPU mismatch", func(r *Qwen38VulkanRawDecodeResult) { r.CPUVerificationNanoseconds++ }, "CPU verification time"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := validQwen38VulkanRawDecodeResult()
			tt.edit(&raw)
			receipt, err := BuildQwen38VulkanDecodeReceipt(raw)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("BuildQwen38VulkanDecodeReceipt() error = %v, want %q", err, tt.want)
			}
			if !reflect.DeepEqual(receipt, Qwen38VulkanDecodeReceipt{}) {
				t.Fatalf("invalid raw result returned a nonzero receipt: %+v", receipt)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptRejectsContractInconsistencies(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{1, 2}, 4)
	base := validQwen38VulkanDecodeReceipt(t, packet)

	tests := []struct {
		name string
		edit func(*Qwen38VulkanDecodeReceipt)
		want string
	}{
		{"model identity", func(r *Qwen38VulkanDecodeReceipt) { r.Packet.ModelGGUFSHA256 = strings.Repeat("0", 64) }, "model digest"},
		{"backend", func(r *Qwen38VulkanDecodeReceipt) { r.Backend = "cpu" }, "engine identity"},
		{"boundary", func(r *Qwen38VulkanDecodeReceipt) { r.GeneratedTokens-- }, "work boundary"},
		{"output hash", func(r *Qwen38VulkanDecodeReceipt) { r.OutputTokenIDsSHA256 = strings.Repeat("0", 64) }, "output token digest"},
		{"dispatch total", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.ComputeDispatches++ }, "compute dispatch total"},
		{"transfer", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.H2D.Count = 0 }, "h2d count/bytes"},
		{"q4 stage", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.Q4KStageCalls = 0 }, "stage calls/bytes"},
		{"tensor home", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.TensorHome.CopiedBytes = r.Counters.D2D.Bytes + 1 }, "exceed d2d"},
		{"memory", func(r *Qwen38VulkanDecodeReceipt) { r.PeakDeviceMemoryBytes = r.PeakProcessMemoryBytes + 1 }, "exceeds process peak"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := base
			tt.edit(&receipt)
			if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3AggregatesExactlyReportedRuns(t *testing.T) {
	raw := validQwen38VulkanV3RawDecodeResult(5)
	receipt, err := BuildQwen38VulkanDecodeReceiptV3(raw)
	if err != nil {
		t.Fatalf("BuildQwen38VulkanDecodeReceiptV3() error = %v", err)
	}
	if receipt.Schema != Qwen38VulkanDecodeReceiptV3Schema || receipt.ResourceScope != Qwen38VulkanResourceScopeReportedRuns || receipt.ReportedRuns != 5 {
		t.Fatalf("v3 scope = schema=%q scope=%q runs=%d", receipt.Schema, receipt.ResourceScope, receipt.ReportedRuns)
	}
	if receipt.TransfersComplete == nil || !*receipt.TransfersComplete {
		t.Fatal("v3 aggregate did not preserve complete transfer observation")
	}
	if receipt.PeakProcessMemoryBytes != 104 || receipt.PeakDeviceMemoryBytes != 204 {
		t.Fatalf("v3 peaks = process=%d device=%d, want maxima 104/204", receipt.PeakProcessMemoryBytes, receipt.PeakDeviceMemoryBytes)
	}
	if receipt.Counters.H2D.Count != 15 || receipt.Counters.H2D.Bytes != 15_360 || receipt.Counters.D2D.Count != 10 || receipt.Counters.D2D.Bytes != 10_240 {
		t.Fatalf("v3 transfer aggregate = %+v", receipt.Counters)
	}
	if receipt.Counters.TensorHome.ResidentBytes != 2_048 {
		t.Fatalf("v3 tensor-home resident bytes = %d, want maximum 2048", receipt.Counters.TensorHome.ResidentBytes)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("v3 Validate() error = %v", err)
	}

	// The two peaks are independent accounting domains. Device allocation may
	// exceed resident process memory without invalidating an otherwise observed
	// v3 receipt.
	if receipt.PeakDeviceMemoryBytes <= receipt.PeakProcessMemoryBytes {
		t.Fatal("fixture did not exercise independent memory domains")
	}

	raw.Runs[0].Resources.PeakProcessMemoryBytes = 999
	if receipt.Runs[0].Resources.PeakProcessMemoryBytes == 999 {
		t.Fatal("v3 builder retained caller-owned resource pointer")
	}
}

func TestQwen38VulkanSelectedTokenLogprobsSHA256CanonicalEncoding(t *testing.T) {
	tokenIDs := []int32{11, 12, 13, 14}
	logprobs := []float64{-0.25, -1.5, 0, -12.75}
	digest, err := Qwen38VulkanSelectedTokenLogprobsSHA256(tokenIDs, logprobs)
	if err != nil {
		t.Fatalf("Qwen38VulkanSelectedTokenLogprobsSHA256() error = %v", err)
	}
	const want = "93DCC29BB9441E339A4550F7F42BAD0FBBD1F7F8A1E8FE8EE5968A93CA9DBC31"
	if digest != want {
		t.Fatalf("digest = %q, want independently computed %q", digest, want)
	}

	negativeZero := slices.Clone(logprobs)
	negativeZero[2] = math.Copysign(0, -1)
	negativeZeroDigest, err := Qwen38VulkanSelectedTokenLogprobsSHA256(tokenIDs, negativeZero)
	if err != nil {
		t.Fatalf("negative-zero digest error = %v", err)
	}
	if negativeZeroDigest != digest {
		t.Fatalf("signed-zero digests differ: positive=%q negative=%q", digest, negativeZeroDigest)
	}
	mutatedTokenIDs := slices.Clone(tokenIDs)
	mutatedTokenIDs[0]++
	mutatedDigest, err := Qwen38VulkanSelectedTokenLogprobsSHA256(mutatedTokenIDs, logprobs)
	if err != nil {
		t.Fatalf("mutated-token digest error = %v", err)
	}
	if mutatedDigest == digest {
		t.Fatal("digest did not bind token IDs")
	}
}

func TestQwen38VulkanSelectedTokenLogprobsSHA256RejectsInvalidEvidence(t *testing.T) {
	for _, tt := range []struct {
		name     string
		tokenIDs []int32
		logprobs []float64
		want     string
	}{
		{"empty", nil, nil, "nonempty"},
		{"count", []int32{1, 2}, []float64{-1}, "does not match"},
		{"positive", []int32{1}, []float64{0.01}, "finite and non-positive"},
		{"NaN", []int32{1}, []float64{math.NaN()}, "finite and non-positive"},
		{"infinity", []int32{1}, []float64{math.Inf(-1)}, "finite and non-positive"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			digest, err := Qwen38VulkanSelectedTokenLogprobsSHA256(tt.tokenIDs, tt.logprobs)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			if digest != "" {
				t.Fatalf("invalid evidence returned digest %q", digest)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3DerivesSelectedTokenLogprobIntegrity(t *testing.T) {
	raw := validQwen38VulkanV3RawDecodeResult(2)
	receipt, err := BuildQwen38VulkanDecodeReceiptV3(raw)
	if err != nil {
		t.Fatalf("BuildQwen38VulkanDecodeReceiptV3() error = %v", err)
	}
	if raw.Runs[0].SelectedTokenLogprobsSHA256 != "" || !math.Signbit(raw.Runs[0].SelectedTokenLogprobs[2]) {
		t.Fatal("v3 builder mutated caller-owned selected-token logprob evidence")
	}
	for i, run := range receipt.Runs {
		if len(run.SelectedTokenLogprobs) != run.ActualGeneratedTokens {
			t.Fatalf("run %d logprob count = %d, want %d", i+1, len(run.SelectedTokenLogprobs), run.ActualGeneratedTokens)
		}
		digest, err := Qwen38VulkanSelectedTokenLogprobsSHA256(run.OutputTokenIDs, run.SelectedTokenLogprobs)
		if err != nil {
			t.Fatalf("run %d digest error = %v", i+1, err)
		}
		if run.SelectedTokenLogprobsSHA256 != digest {
			t.Fatalf("run %d digest = %q, want %q", i+1, run.SelectedTokenLogprobsSHA256, digest)
		}
		if math.Signbit(run.SelectedTokenLogprobs[2]) {
			t.Fatalf("run %d retained negative zero", i+1)
		}
	}
	raw.Runs[0].SelectedTokenLogprobs[0] = -99
	if receipt.Runs[0].SelectedTokenLogprobs[0] == -99 {
		t.Fatal("v3 builder retained caller-owned selected-token logprob slice")
	}
}

func TestQwen38VulkanDecodeReceiptV3RejectsInvalidSelectedTokenLogprobs(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*Qwen38VulkanRawDecodeResult)
		want string
	}{
		{"digest supplied", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].SelectedTokenLogprobsSHA256 = strings.Repeat("A", 64) }, "must be derived"},
		{"observations absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].SelectedTokenLogprobs = nil }, "nonempty"},
		{"count mismatch", func(r *Qwen38VulkanRawDecodeResult) {
			r.Runs[0].SelectedTokenLogprobs = r.Runs[0].SelectedTokenLogprobs[:3]
		}, "does not match"},
		{"positive", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].SelectedTokenLogprobs[0] = 0.01 }, "finite and non-positive"},
		{"NaN", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[0].SelectedTokenLogprobs[0] = math.NaN() }, "finite and non-positive"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			raw := validQwen38VulkanV3RawDecodeResult(1)
			tt.edit(&raw)
			receipt, err := BuildQwen38VulkanDecodeReceiptV3(raw)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			if !reflect.DeepEqual(receipt, Qwen38VulkanDecodeReceipt{}) {
				t.Fatalf("invalid v3 raw result returned a nonzero receipt: %+v", receipt)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3RejectsMutatedSelectedTokenLogprobs(t *testing.T) {
	base, err := BuildQwen38VulkanDecodeReceiptV3(validQwen38VulkanV3RawDecodeResult(1))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		edit func(*Qwen38VulkanDecodeReceipt)
		want string
	}{
		{"count", func(r *Qwen38VulkanDecodeReceipt) {
			r.Runs[0].SelectedTokenLogprobs = r.Runs[0].SelectedTokenLogprobs[:3]
		}, "count"},
		{"positive", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[0].SelectedTokenLogprobs[0] = 0.01 }, "finite and non-positive"},
		{"negative zero", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[0].SelectedTokenLogprobs[2] = math.Copysign(0, -1) }, "non-canonical signed zero"},
		{"observation", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[0].SelectedTokenLogprobs[0]-- }, "digest"},
		{"digest", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[0].SelectedTokenLogprobsSHA256 = strings.Repeat("0", 64) }, "digest"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			receipt := base
			receipt.Runs = slices.Clone(base.Runs)
			receipt.Runs[0].SelectedTokenLogprobs = slices.Clone(base.Runs[0].SelectedTokenLogprobs)
			tt.edit(&receipt)
			if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3FailsClosedOnIncompleteResources(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Qwen38VulkanRawDecodeResult)
		want string
	}{
		{"scope absent", func(r *Qwen38VulkanRawDecodeResult) { r.ResourceScope = "" }, "resource scope"},
		{"reported runs zero", func(r *Qwen38VulkanRawDecodeResult) { r.ReportedRuns = 0 }, "reported runs"},
		{"reported runs mismatch", func(r *Qwen38VulkanRawDecodeResult) { r.ReportedRuns-- }, "reported runs"},
		{"run kind absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Kind = "" }, "kind"},
		{"warmup mixed in", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Kind = "warmup" }, "want \"measured\""},
		{"run resources absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Resources = nil }, "requires resource observations"},
		{"transfer completeness absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Resources.TransfersComplete = nil }, "complete transfer"},
		{"transfer completeness false", func(r *Qwen38VulkanRawDecodeResult) {
			incomplete := false
			r.Runs[2].Resources.TransfersComplete = &incomplete
		}, "complete transfer"},
		{"process peak absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Resources.PeakProcessMemoryBytes = 0 }, "positive process and device"},
		{"device peak absent", func(r *Qwen38VulkanRawDecodeResult) { r.Runs[2].Resources.PeakDeviceMemoryBytes = 0 }, "positive process and device"},
		{"legacy aggregate supplied", func(r *Qwen38VulkanRawDecodeResult) {
			peak := uint64(1)
			r.PeakProcessMemoryBytes = &peak
		}, "must be derived"},
		{"counter overflow", func(r *Qwen38VulkanRawDecodeResult) {
			r.Runs[0].Resources.Counters.ComputeDispatches = math.MaxUint64
			r.Runs[0].Resources.Counters.Q4KMatmulDispatches = math.MaxUint64
			r.Runs[0].Resources.Counters.OtherComputeDispatches = 0
		}, "overflow across reported runs"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := validQwen38VulkanV3RawDecodeResult(5)
			tt.edit(&raw)
			receipt, err := BuildQwen38VulkanDecodeReceiptV3(raw)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("BuildQwen38VulkanDecodeReceiptV3() error = %v, want %q", err, tt.want)
			}
			if !reflect.DeepEqual(receipt, Qwen38VulkanDecodeReceipt{}) {
				t.Fatalf("invalid v3 raw result returned a nonzero receipt: %+v", receipt)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3RejectsMutatedAggregation(t *testing.T) {
	base, err := BuildQwen38VulkanDecodeReceiptV3(validQwen38VulkanV3RawDecodeResult(5))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Qwen38VulkanDecodeReceipt)
		want string
	}{
		{"scope", func(r *Qwen38VulkanDecodeReceipt) { r.ResourceScope = "one_run" }, "resource scope"},
		{"run count", func(r *Qwen38VulkanDecodeReceipt) { r.ReportedRuns = 4 }, "reported runs"},
		{"aggregate completeness absent", func(r *Qwen38VulkanDecodeReceipt) { r.TransfersComplete = nil }, "complete transfer"},
		{"aggregate completeness false", func(r *Qwen38VulkanDecodeReceipt) {
			incomplete := false
			r.TransfersComplete = &incomplete
		}, "complete transfer"},
		{"process maximum", func(r *Qwen38VulkanDecodeReceipt) { r.PeakProcessMemoryBytes++ }, "peak memory"},
		{"device maximum", func(r *Qwen38VulkanDecodeReceipt) { r.PeakDeviceMemoryBytes++ }, "peak memory"},
		{"counter sum", func(r *Qwen38VulkanDecodeReceipt) { r.Counters.H2D.Bytes++ }, "counters do not match"},
		{"partial run", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[3].Resources = nil }, "requires resource observations"},
		{"warmup mixed in", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[3].Kind = "warmup" }, "want \"measured\""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := base
			receipt.Runs = slices.Clone(base.Runs)
			tt.edit(&receipt)
			if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestQwen38VulkanDecodeReceiptV3AcceptsObservedZeroTransferDirections(t *testing.T) {
	raw := validQwen38VulkanV3RawDecodeResult(1)
	raw.Runs[0].Resources.Counters.H2D = Qwen38VulkanTransferCounters{}
	raw.Runs[0].Resources.Counters.D2H = Qwen38VulkanTransferCounters{}
	receipt, err := BuildQwen38VulkanDecodeReceiptV3(raw)
	if err != nil {
		t.Fatalf("observed zero transfer direction was unavailable: %v", err)
	}
	if receipt.Counters.H2D != (Qwen38VulkanTransferCounters{}) || receipt.Counters.D2H != (Qwen38VulkanTransferCounters{}) {
		t.Fatalf("observed zero transfer direction changed: %+v", receipt.Counters)
	}
}

func TestQwen38VulkanDecodeReceiptV2RemainsCompatible(t *testing.T) {
	legacyRaw := validQwen38VulkanRawDecodeResult()
	legacyReceipt, err := BuildQwen38VulkanDecodeReceipt(legacyRaw)
	if err != nil {
		t.Fatalf("legacy BuildQwen38VulkanDecodeReceipt() error = %v", err)
	}
	legacyJSON, err := json.Marshal(legacyReceipt)
	if err != nil {
		t.Fatal(err)
	}

	raw := legacyRaw
	raw.ResourceScope = Qwen38VulkanResourceScopeReportedRuns
	raw.ReportedRuns = len(raw.Runs)
	raw.Runs[0].Kind = "warmup"
	complete := true
	raw.Runs[0].Resources = &Qwen38VulkanRunResources{TransfersComplete: &complete}
	raw.Runs[0].SelectedTokenLogprobs = []float64{-0.25, -1.5, 0, -12.75}
	raw.Runs[0].SelectedTokenLogprobsSHA256 = strings.Repeat("A", 64)
	receipt, err := BuildQwen38VulkanDecodeReceipt(raw)
	if err != nil {
		t.Fatalf("legacy BuildQwen38VulkanDecodeReceipt() error = %v", err)
	}
	if receipt.Schema != Qwen38VulkanDecodeReceiptSchema || receipt.ResourceScope != "" || receipt.ReportedRuns != 0 || receipt.TransfersComplete != nil || receipt.Runs[0].Kind != "" || receipt.Runs[0].Resources != nil || receipt.Runs[0].SelectedTokenLogprobs != nil || receipt.Runs[0].SelectedTokenLogprobsSHA256 != "" {
		t.Fatalf("v2 receipt acquired v3 fields: %+v", receipt)
	}
	gotJSON, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotJSON, legacyJSON) {
		t.Fatalf("v2 JSON changed when v3-only fields were supplied:\ngot  %s\nwant %s", gotJSON, legacyJSON)
	}
	if bytes.Contains(gotJSON, []byte("selected_token_logprobs")) {
		t.Fatalf("v2 JSON contains v3-only logprob fields: %s", gotJSON)
	}
}

func TestCompareQwen38VulkanDecodeReceiptsRejectsMismatches(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{7, 8}, 4)
	parent := validQwen38VulkanDecodeReceipt(t, packet)

	t.Run("schema", func(t *testing.T) {
		candidateRaw := validQwen38VulkanV3RawDecodeResult(1)
		candidateRaw.PromptTokenIDs = slices.Clone(packet.PromptTokenIDs)
		candidateRaw.Runs[0].ContextTokens = len(packet.PromptTokenIDs) + len(candidateRaw.OutputTokenIDs)
		candidate, err := BuildQwen38VulkanDecodeReceiptV3(candidateRaw)
		if err != nil {
			t.Fatal(err)
		}
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "schema mismatch") {
			t.Fatalf("comparison error = %v, want schema mismatch", err)
		}
	})

	t.Run("packet", func(t *testing.T) {
		candidatePacket := NewQwen38VulkanDecodePacket([]int32{7, 9}, 4)
		candidate := validQwen38VulkanDecodeReceipt(t, candidatePacket)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "packet mismatch") {
			t.Fatalf("comparison error = %v, want packet mismatch", err)
		}
	})

	t.Run("output", func(t *testing.T) {
		candidate := parent
		candidate.Runs = slices.Clone(parent.Runs)
		candidate.OutputTokenIDs = []int32{11, 12, 13, 99}
		candidate.OutputTokenIDsSHA256 = Qwen38VulkanTokenIDsSHA256(candidate.OutputTokenIDs)
		candidate.Runs[0].OutputTokenIDs = slices.Clone(candidate.OutputTokenIDs)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "output mismatch") {
			t.Fatalf("comparison error = %v, want output mismatch", err)
		}
	})

	t.Run("work boundary", func(t *testing.T) {
		candidate := parent
		candidate.Runs = slices.Clone(parent.Runs)
		candidate.Packet.GeneratedTokenLimit = 3
		candidate.OutputTokenIDs = candidate.OutputTokenIDs[:3]
		candidate.GeneratedTokens = 3
		candidate.OutputTokenIDsSHA256 = Qwen38VulkanTokenIDsSHA256(candidate.OutputTokenIDs)
		candidate.Runs[0].GeneratedTokenLimit = 3
		candidate.Runs[0].ActualGeneratedTokens = 3
		candidate.Runs[0].ContextTokens = len(candidate.Packet.PromptTokenIDs) + 3
		candidate.Runs[0].OutputTokenIDs = slices.Clone(candidate.OutputTokenIDs)
		candidate.PacketSHA256 = mustQwen38VulkanPacketDigest(t, candidate.Packet)
		if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), "packet mismatch") {
			t.Fatalf("comparison error = %v, want packet mismatch for unequal boundary", err)
		}
	})

	for _, tt := range []struct {
		name string
		edit func(*Qwen38VulkanDecodeReceipt)
		want string
	}{
		{"context limit", func(r *Qwen38VulkanDecodeReceipt) { r.Runs[0].ContextLimit++ }, "context limit mismatch"},
		{"EOS policy", func(r *Qwen38VulkanDecodeReceipt) {
			ignoreEOS := !*r.Runs[0].IgnoreEOS
			r.Runs[0].IgnoreEOS = &ignoreEOS
		}, "EOS policy mismatch"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			candidate := parent
			candidate.Runs = slices.Clone(parent.Runs)
			tt.edit(&candidate)
			if err := CompareQwen38VulkanDecodeReceipts(parent, candidate); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("comparison error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestExecuteQwen38VulkanDecodeContractCleansUpExactlyOnce(t *testing.T) {
	tests := []struct {
		name      string
		run       func() (int, error)
		wantPanic bool
	}{
		{"success", func() (int, error) { return 7, nil }, false},
		{"error", func() (int, error) { return 0, errors.New("decode failed") }, false},
		{"panic", func() (int, error) { panic("decode panic") }, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cleanups := 0
			func() {
				defer func() {
					gotPanic := recover() != nil
					if gotPanic != tt.wantPanic {
						t.Fatalf("panic = %v, want %v", gotPanic, tt.wantPanic)
					}
				}()
				_, _ = ExecuteQwen38VulkanDecodeContract(func() { cleanups++ }, tt.run)
			}()
			if cleanups != 1 {
				t.Fatalf("cleanup calls = %d, want 1", cleanups)
			}
		})
	}
}

func TestQwen38VulkanDecodePacketRequiresPositiveBoundary(t *testing.T) {
	packet := NewQwen38VulkanDecodePacket([]int32{1}, 0)
	if err := packet.Validate(); err == nil || !strings.Contains(err.Error(), "must be positive") {
		t.Fatalf("Validate() error = %v, want positive-boundary refusal", err)
	}
}

func validQwen38VulkanDecodeReceipt(t *testing.T, packet Qwen38VulkanDecodePacket) Qwen38VulkanDecodeReceipt {
	t.Helper()
	output := []int32{11, 12, 13, 14}
	if len(output) != packet.GeneratedTokenLimit {
		t.Fatalf("test output length %d != packet boundary %d", len(output), packet.GeneratedTokenLimit)
	}
	raw := validQwen38VulkanRawDecodeResult()
	raw.PromptTokenIDs = slices.Clone(packet.PromptTokenIDs)
	raw.GeneratedTokenLimit = packet.GeneratedTokenLimit
	raw.OutputTokenIDs = output
	raw.Runs[0].ContextTokens = len(packet.PromptTokenIDs) + len(output)
	raw.Runs[0].GeneratedTokenLimit = packet.GeneratedTokenLimit
	raw.Runs[0].ActualGeneratedTokens = len(output)
	raw.Runs[0].OutputTokenIDs = slices.Clone(output)
	receipt, err := BuildQwen38VulkanDecodeReceipt(raw)
	if err != nil {
		t.Fatalf("BuildQwen38VulkanDecodeReceipt() error = %v", err)
	}
	return receipt
}

func validQwen38VulkanRawDecodeResult() Qwen38VulkanRawDecodeResult {
	clean := false
	zeroFallbacks := uint64(0)
	finiteLogits, cpuModelParity := true, true
	ignoreEOS, eosStopped := true, false
	peakProcessMemoryBytes := uint64(8 << 30)
	peakDeviceMemoryBytes := uint64(6 << 30)
	counters := Qwen38VulkanDecodeCounters{
		ComputeDispatches:      40,
		Q4KMatmulDispatches:    30,
		OtherComputeDispatches: 10,
		DispatchSubmits:        8,
		H2D:                    Qwen38VulkanTransferCounters{Count: 3, Bytes: 3072},
		D2H:                    Qwen38VulkanTransferCounters{Count: 1, Bytes: 4},
		D2D:                    Qwen38VulkanTransferCounters{Count: 2, Bytes: 2048},
		Q4KStageCalls:          4,
		Q4KStageBytes:          4096,
		TensorHome: Qwen38VulkanTensorHomeCounters{
			Hits: 8, Admissions: 2, Bypasses: 1, ResidentBytes: 2048, CopiedBytes: 2048,
		},
	}
	run := Qwen38VulkanDecodeRun{
		Repetition:                  1,
		ContextLimit:                256,
		ContextTokens:               7,
		GeneratedTokenLimit:         4,
		ActualGeneratedTokens:       4,
		Sampler:                     "greedy",
		SeedPolicy:                  "not_applicable_greedy",
		IgnoreEOS:                   &ignoreEOS,
		EOSStopped:                  &eosStopped,
		OutputTokenIDs:              []int32{11, 12, 13, 14},
		SessionSetupNanoseconds:     5_000,
		PrefillNanoseconds:          10_000,
		FirstSampleNanoseconds:      5_000,
		DecodeNanoseconds:           25_000,
		TeardownNanoseconds:         5_000,
		CandidateElapsedNanoseconds: 50_000,
		CPUVerificationNanoseconds:  10_000,
	}
	return Qwen38VulkanRawDecodeResult{
		Source: Qwen38VulkanSourceIdentity{
			GitCommit:           strings.Repeat("a", 40),
			SourceArchiveSHA256: strings.Repeat("b", 64),
			BinarySHA256:        strings.Repeat("c", 64),
			Dirty:               &clean,
		},
		Model: Qwen38VulkanModelIdentity{
			Name:                  "qwen38:27b",
			ArtifactPath:          "/models/qwen3.8-27b-q4_k_m.gguf",
			ArtifactSHA256:        Qwen38VulkanDecodeGGUFSHA256,
			TensorInventorySHA256: strings.Repeat("d", 64),
			TokenizerSHA256:       strings.Repeat("e", 64),
			TemplateSHA256:        strings.Repeat("f", 64),
			Quantization:          "Q4_K_M + Q8 MTP",
		},
		Device: Qwen38VulkanDeviceIdentity{
			OS: "linux", Arch: "amd64", Kernel: "6.14", Name: "AMD Radeon 8060S Graphics|gfx1151",
			MesaVersion: "25.2", VulkanVersion: "1.4", Firmware: "amdgpu-test",
		},
		Engine: Qwen38VulkanEngineIdentity{
			Name: "fak-native", Backend: Qwen38VulkanDecodeBackend, Runtime: Qwen38VulkanDecodeRuntime,
			ExecutedPath: "modelbench/raw-decode/vulkan", FallbackCount: &zeroFallbacks,
		},
		CaptureCommand:              "fak modelbench -raw-decode",
		PromptTokenIDs:              []int32{151644, 8948, 198},
		GeneratedTokenLimit:         4,
		OutputTokenIDs:              []int32{11, 12, 13, 14},
		OutputText:                  "test output",
		FiniteLogits:                &finiteLogits,
		CPUModelParity:              &cpuModelParity,
		Runs:                        []Qwen38VulkanDecodeRun{run},
		CandidateElapsedNanoseconds: 50_000,
		CPUVerificationNanoseconds:  10_000,
		PeakProcessMemoryBytes:      &peakProcessMemoryBytes,
		PeakDeviceMemoryBytes:       &peakDeviceMemoryBytes,
		Counters:                    &counters,
	}
}

func validQwen38VulkanV3RawDecodeResult(reportedRuns int) Qwen38VulkanRawDecodeResult {
	raw := validQwen38VulkanRawDecodeResult()
	raw.PeakProcessMemoryBytes = nil
	raw.PeakDeviceMemoryBytes = nil
	raw.Counters = nil
	raw.ResourceScope = Qwen38VulkanResourceScopeReportedRuns
	raw.ReportedRuns = reportedRuns
	baseRun := raw.Runs[0]
	baseCounters := *validQwen38VulkanRawDecodeResult().Counters
	raw.Runs = make([]Qwen38VulkanDecodeRun, reportedRuns)
	for i := range raw.Runs {
		complete := true
		run := baseRun
		run.Repetition = i + 1
		run.Kind = Qwen38VulkanRunKindMeasured
		run.OutputTokenIDs = slices.Clone(baseRun.OutputTokenIDs)
		run.SelectedTokenLogprobs = []float64{-0.25, -1.5, math.Copysign(0, -1), -12.75}
		run.Resources = &Qwen38VulkanRunResources{
			PeakProcessMemoryBytes: uint64(100 + i),
			PeakDeviceMemoryBytes:  uint64(200 + i),
			TransfersComplete:      &complete,
			Counters:               baseCounters,
		}
		raw.Runs[i] = run
	}
	return raw
}

func mustQwen38VulkanPacketDigest(t *testing.T, packet Qwen38VulkanDecodePacket) string {
	t.Helper()
	digest, err := packet.Digest()
	if err != nil {
		t.Fatalf("packet.Digest() error = %v", err)
	}
	return digest
}
