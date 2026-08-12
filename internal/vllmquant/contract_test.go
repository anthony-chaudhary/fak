package vllmquant

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func boolp(v bool) *bool { return &v }
func intp(v int) *int    { return &v }

// awqServer is an Ampere build that advertises both AWQ kernels and declares no
// preference between them — the ambiguous shape most of these cases start from.
func awqServer() Server {
	return Server{
		Version:           "0.8.5",
		Kernels:           []Kernel{KernelAWQ, KernelAWQMarlin},
		ComputeCapability: 80,
		Dtype:             "float16",
	}
}

func awqArtifact() Artifact {
	return Artifact{QuantMethod: MethodAWQ, WeightBits: 4, GroupSize: intp(128), Symmetric: boolp(false)}
}

func req(a Artifact, s Server) Request {
	return Request{Schema: SchemaVersion, Artifact: a, Server: s}
}

// TestSelectsSoleAdmissibleKernel: one candidate needs no preference to pick,
// and the kernel the build never compiled is reported as excluded rather than
// silently dropped.
func TestSelectsSoleAdmissibleKernel(t *testing.T) {
	s := awqServer()
	s.Kernels = []Kernel{KernelAWQ}
	got := Adjudicate(req(awqArtifact(), s))

	if got.Outcome != OutcomeSupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
	}
	if got.Kernel() != KernelAWQ {
		t.Errorf("kernel = %q, want %q", got.Kernel(), KernelAWQ)
	}
	if !got.HasReason(ReasonAdmitted) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonAdmitted)
	}
	want := []string{"--quantization", "awq", "--dtype", "float16"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
	if got.Claim != ClaimRuntimeDelegated {
		t.Errorf("claim = %q, want %q: vLLM still resolves the concrete kernel at load time",
			got.Claim, ClaimRuntimeDelegated)
	}
	if r, ok := got.ExcludedReason(KernelAWQMarlin); !ok || r != ReasonMethodNotInBuild {
		t.Errorf("awq_marlin exclusion = %q (present %v), want %q", r, ok, ReasonMethodNotInBuild)
	}
}

// TestAmbiguousKernelSetDelegates is the anti-winner-ordering gate: two
// admissible kernels and nobody declared a preference means this leaf reports
// the candidates and hands the choice back, rather than ranking them itself.
func TestAmbiguousKernelSetDelegates(t *testing.T) {
	got := Adjudicate(req(awqArtifact(), awqServer()))

	if got.Outcome != OutcomeDelegate {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeDelegate, got.Reasons)
	}
	if !got.HasReason(ReasonKernelChoiceDelegated) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonKernelChoiceDelegated)
	}
	wantCandidates := []Kernel{KernelAWQ, KernelAWQMarlin}
	if !reflect.DeepEqual(got.CandidateMethods(), wantCandidates) {
		t.Errorf("candidates = %v, want %v", got.CandidateMethods(), wantCandidates)
	}
	if got.Kernel() != "" {
		t.Errorf("kernel = %q, want empty: a delegate names no kernel", got.Kernel())
	}
	// The non-kernel arguments are still licensed; the kernel-picking one is not.
	if want := []string{"--dtype", "float16"}; !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
	for _, a := range got.Args {
		if a == "--quantization" {
			t.Fatalf("delegate emitted --quantization; nobody licensed a ranking")
		}
	}
}

// TestServerPreferenceBreaksTie: the server's own declared order decides, and
// the first admissible kernel in THAT order wins regardless of which it is.
func TestServerPreferenceBreaksTie(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kernels []Kernel
		want    Kernel
	}{
		{"marlin first", []Kernel{KernelAWQMarlin, KernelAWQ}, KernelAWQMarlin},
		{"plain first", []Kernel{KernelAWQ, KernelAWQMarlin}, KernelAWQ},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := awqServer()
			s.Kernels = tc.kernels
			s.KernelOrderIsPreference = true

			got := Adjudicate(req(awqArtifact(), s))
			if got.Outcome != OutcomeSupported {
				t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
			}
			if got.Kernel() != tc.want {
				t.Errorf("kernel = %q, want %q", got.Kernel(), tc.want)
			}
			if !got.HasReason(ReasonSelectedByServerPreference) {
				t.Errorf("reasons = %v, want %q", got.Reasons, ReasonSelectedByServerPreference)
			}
			wantArgs := []string{"--quantization", string(tc.want), "--dtype", "float16"}
			if !reflect.DeepEqual(got.Args, wantArgs) {
				t.Errorf("args = %q, want %q", got.Args, wantArgs)
			}
		})
	}
}

// TestRuntimeSelectsDelegates: a server that says it resolves the kernel itself
// gets launch args WITHOUT --quantization even when only one kernel is
// admissible, and only a delegation claim.
func TestRuntimeSelectsDelegates(t *testing.T) {
	s := awqServer()
	s.Kernels = []Kernel{KernelAWQ}
	s.RuntimeSelects = true

	got := Adjudicate(req(awqArtifact(), s))
	if got.Outcome != OutcomeDelegate {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeDelegate, got.Reasons)
	}
	if got.Claim != ClaimRuntimeDelegated {
		t.Errorf("claim = %q, want %q", got.Claim, ClaimRuntimeDelegated)
	}
	want := []string{"--dtype", "float16"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
	for _, a := range got.Args {
		if a == "--quantization" {
			t.Fatalf("delegate emitted --quantization; the runtime owns that choice")
		}
	}
}

// TestGPTQAsymmetricDropsMarlin: an asymmetric checkpoint is inadmissible to
// gptq_marlin, so the plain kernel is left as the sole candidate and selected
// without any preference being declared.
func TestGPTQAsymmetricDropsMarlin(t *testing.T) {
	a := Artifact{QuantMethod: MethodGPTQ, WeightBits: 4, GroupSize: intp(128), Symmetric: boolp(false)}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelGPTQMarlin, KernelGPTQ}, ComputeCapability: 80, Dtype: "bfloat16"}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeSupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
	}
	if got.Kernel() != KernelGPTQ {
		t.Errorf("kernel = %q, want %q", got.Kernel(), KernelGPTQ)
	}
	if r, ok := got.ExcludedReason(KernelGPTQMarlin); !ok || r != ReasonAsymmetricUnsupported {
		t.Errorf("gptq_marlin exclusion = %q (present %v), want %q", r, ok, ReasonAsymmetricUnsupported)
	}
	want := []string{"--quantization", "gptq", "--dtype", "bfloat16"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
}

// TestGPTQOnlyMarlinAsymmetricUnsupported: when the dropped kernel was the ONLY
// one advertised, the empty candidate set reports unsupported with the specific
// reason rather than falling back to a kernel the build does not have.
func TestGPTQOnlyMarlinAsymmetricUnsupported(t *testing.T) {
	a := Artifact{QuantMethod: MethodGPTQ, WeightBits: 4, GroupSize: intp(128), Symmetric: boolp(false)}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelGPTQMarlin}, ComputeCapability: 80}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeUnsupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeUnsupported, got.Reasons)
	}
	if !got.HasReason(ReasonAsymmetricUnsupported) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonAsymmetricUnsupported)
	}
	if got.Claim != ClaimArtifactDescribed {
		t.Errorf("claim = %q, want %q: the artifact was read, the build fell short", got.Claim, ClaimArtifactDescribed)
	}
	if len(got.Args) != 0 {
		t.Errorf("args = %q, want empty: nothing was licensed", got.Args)
	}
}

// TestFP8NeedsAdaOrHopper: an Ampere card cannot run native W8A8 fp8, and the
// verdict must name compute capability rather than the method.
func TestFP8NeedsAdaOrHopper(t *testing.T) {
	a := Artifact{QuantMethod: MethodFP8, WeightBits: 8, ActivationScheme: "dynamic"}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelFP8}, ComputeCapability: 80}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeUnsupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeUnsupported, got.Reasons)
	}
	if !got.HasReason(ReasonComputeCapabilityTooLow) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonComputeCapabilityTooLow)
	}
}

// TestFP8OnHopperSelects with the KV cache dtype carried through to the args.
func TestFP8OnHopperSelects(t *testing.T) {
	a := Artifact{QuantMethod: MethodFP8, WeightBits: 8, ActivationScheme: "static", KVCacheDtype: "fp8_e4m3", Symmetric: boolp(true)}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelFP8}, ComputeCapability: 90, Dtype: "bfloat16"}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeSupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
	}
	want := []string{"--quantization", "fp8", "--dtype", "bfloat16", "--kv-cache-dtype", "fp8_e4m3"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
}

// TestFP8WeightBitsAreEntailed: fp8 is 8-bit by construction, so an artifact
// that never spells out `bits` is complete, not undescribed. A method with more
// than one legal width does not get that pass (see TestUnsupportedCombinations).
func TestFP8WeightBitsAreEntailed(t *testing.T) {
	a := Artifact{QuantMethod: MethodFP8, ActivationScheme: "static"}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelFP8}, ComputeCapability: 90}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeSupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
	}
}

// TestFP8ActivationSchemeUnknownAbstains: fp8 without a declared activation
// scheme is not a static-by-default fp8 artifact, it is an undescribed one.
func TestFP8ActivationSchemeUnknownAbstains(t *testing.T) {
	a := Artifact{QuantMethod: MethodFP8, WeightBits: 8}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelFP8}, ComputeCapability: 90}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeAbstain {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeAbstain, got.Reasons)
	}
	if !got.HasReason(ReasonActivationSchemeUnknown) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonActivationSchemeUnknown)
	}
}

// TestBitsAndBytesSelects on a build new enough to carry the kernel. The kernel
// needs its own --load-format, which the candidate carries rather than the
// caller having to know it.
func TestBitsAndBytesSelects(t *testing.T) {
	a := Artifact{QuantMethod: MethodBitsAndBytes, WeightBits: 4, Symmetric: boolp(false)}
	s := Server{Version: "0.8.5", Kernels: []Kernel{KernelBitsAndBytes}, ComputeCapability: 75}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeSupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeSupported, got.Reasons)
	}
	want := []string{"--quantization", "bitsandbytes", "--load-format", "bitsandbytes"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Errorf("args = %q, want %q", got.Args, want)
	}
}

// TestBitsAndBytesBelowMinimumVersionUnsupported: the build predates the kernel
// it claims to advertise, so nothing it advertises can serve the artifact.
func TestBitsAndBytesBelowMinimumVersionUnsupported(t *testing.T) {
	a := Artifact{QuantMethod: MethodBitsAndBytes, WeightBits: 4}
	s := Server{Version: "0.4.0", Kernels: []Kernel{KernelBitsAndBytes}, ComputeCapability: 80}

	got := Adjudicate(req(a, s))
	if got.Outcome != OutcomeUnsupported {
		t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, OutcomeUnsupported, got.Reasons)
	}
	if !got.HasReason(ReasonVersionBelowMinimum) {
		t.Errorf("reasons = %v, want %q", got.Reasons, ReasonVersionBelowMinimum)
	}
}

// TestUnsupportedCombinations walks the abstain/unsupported/refuse surface:
// every one of these must be typed, none may silently degrade to a nearby
// kernel, and each must land on the right rung of the claim ladder — an
// artifact-stage failure claims nothing, a build-stage failure still claims the
// artifact was read.
func TestUnsupportedCombinations(t *testing.T) {
	base := awqServer()
	base.Kernels = []Kernel{KernelAWQ}

	for _, tc := range []struct {
		name    string
		mutate  func(*Artifact, *Server)
		outcome Outcome
		reason  Reason
		claim   Claim
	}{
		{"method undeclared", func(a *Artifact, _ *Server) { a.QuantMethod = "" },
			OutcomeAbstain, ReasonArtifactUndeclared, ClaimNone},
		{"method outside vocabulary", func(a *Artifact, _ *Server) { a.QuantMethod = "hqq" },
			OutcomeAbstain, ReasonMethodUnknown, ClaimNone},
		{"weight bits undeclared for a multi-width method", func(a *Artifact, _ *Server) {
			a.QuantMethod = MethodGPTQ
			a.WeightBits = 0
		}, OutcomeAbstain, ReasonWeightBitsUndeclared, ClaimNone},
		{"weight bits off the method", func(a *Artifact, _ *Server) { a.WeightBits = 3 },
			OutcomeRefuse, ReasonMethodBitsConflict, ClaimNone},
		{"group size undeclared", func(a *Artifact, _ *Server) { a.GroupSize = nil },
			OutcomeAbstain, ReasonGroupSizeUndeclared, ClaimNone},
		{"group size declared zero", func(a *Artifact, _ *Server) { a.GroupSize = intp(0) },
			OutcomeRefuse, ReasonGroupSizeMalformed, ClaimNone},
		{"group size off the grid", func(a *Artifact, _ *Server) { a.GroupSize = intp(96) },
			OutcomeRefuse, ReasonGroupSizeUnsupported, ClaimNone},
		{"kv cache dtype outside vocabulary", func(a *Artifact, _ *Server) { a.KVCacheDtype = "int3" },
			OutcomeAbstain, ReasonKVCacheDtypeUnknown, ClaimNone},
		{"checkpoint format outside vocabulary", func(a *Artifact, _ *Server) { a.CheckpointFormat = "exl2" },
			OutcomeAbstain, ReasonCheckpointFormatUnknown, ClaimNone},
		{"server wholly undeclared", func(_ *Artifact, s *Server) { *s = Server{} },
			OutcomeAbstain, ReasonServerUndeclared, ClaimArtifactDescribed},
		{"version unparseable", func(_ *Artifact, s *Server) { s.Version = "0.8.5.post1" },
			OutcomeAbstain, ReasonVersionUnknown, ClaimArtifactDescribed},
		{"version absent", func(_ *Artifact, s *Server) { s.Version = "" },
			OutcomeAbstain, ReasonVersionUnknown, ClaimArtifactDescribed},
		{"version past the known window", func(_ *Artifact, s *Server) { s.Version = "1.0.0" },
			OutcomeAbstain, ReasonVersionUnknown, ClaimArtifactDescribed},
		{"no kernel advertised", func(_ *Artifact, s *Server) { s.Kernels = nil },
			OutcomeRefuse, ReasonNoKernelAdvertised, ClaimArtifactDescribed},
		{"no kernel serves the method", func(_ *Artifact, s *Server) { s.Kernels = []Kernel{KernelFP8} },
			OutcomeUnsupported, ReasonMethodNotInBuild, ClaimArtifactDescribed},
		{"kernel outside vocabulary", func(_ *Artifact, s *Server) { s.Kernels = []Kernel{"awq_experimental"} },
			OutcomeAbstain, ReasonKernelUnknown, ClaimArtifactDescribed},
		{"compute capability unknown", func(_ *Artifact, s *Server) { s.ComputeCapability = CapabilityUndeclared },
			OutcomeAbstain, ReasonComputeCapabilityUnknown, ClaimArtifactDescribed},
		{"compute capability malformed", func(_ *Artifact, s *Server) { s.ComputeCapability = CapabilityMalformed },
			OutcomeRefuse, ReasonCapabilityMalformed, ClaimArtifactDescribed},
		{"compute capability too low", func(_ *Artifact, s *Server) { s.ComputeCapability = 70 },
			OutcomeUnsupported, ReasonComputeCapabilityTooLow, ClaimArtifactDescribed},
		{"marlin cannot take the declared width", func(a *Artifact, s *Server) {
			a.QuantMethod = MethodGPTQ
			a.WeightBits = 3
			s.Kernels = []Kernel{KernelGPTQMarlin}
		}, OutcomeUnsupported, ReasonMarlinRequirementUnmet, ClaimArtifactDescribed},
		{"symmetry undeclared for marlin", func(a *Artifact, s *Server) {
			a.Symmetric = nil
			s.Kernels = []Kernel{KernelGPTQMarlin}
			a.QuantMethod = MethodGPTQ
		}, OutcomeAbstain, ReasonSymmetryUndeclared, ClaimArtifactDescribed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, s := awqArtifact(), base
			s.Kernels = append([]Kernel(nil), base.Kernels...)
			tc.mutate(&a, &s)

			got := Adjudicate(req(a, s))
			if got.Outcome != tc.outcome {
				t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, tc.outcome, got.Reasons)
			}
			if !got.HasReason(tc.reason) {
				t.Errorf("reasons = %v, want %q", got.Reasons, tc.reason)
			}
			if len(got.Args) != 0 {
				t.Errorf("args = %q, want empty: nothing was selected", got.Args)
			}
			if len(got.Candidates) != 0 {
				t.Errorf("candidates = %v, want empty: nothing was admitted", got.CandidateMethods())
			}
			if got.Claim != tc.claim {
				t.Errorf("claim = %q, want %q", got.Claim, tc.claim)
			}
			if got.Kernel() != "" {
				t.Errorf("kernel = %q, want empty", got.Kernel())
			}
		})
	}
}

// TestSchemaAndParseErrorsAreTyped: a caller never has to string-match a Go
// error to tell "could not read" from "read and refused".
func TestSchemaAndParseErrorsAreTyped(t *testing.T) {
	got := ParseAndAdjudicate([]byte("{not json"))
	if got.Outcome != OutcomeRefuse || !got.HasReason(ReasonInvalidJSON) {
		t.Errorf("malformed bytes: outcome %q reasons %v, want refuse/%s", got.Outcome, got.Reasons, ReasonInvalidJSON)
	}
	if got.ArtifactMethod != MethodUnknown {
		t.Errorf("artifact method = %q, want %q", got.ArtifactMethod, MethodUnknown)
	}

	other := req(awqArtifact(), awqServer())
	other.Schema = "vllmquant/v99"
	if got := Adjudicate(other); got.Outcome != OutcomeAbstain || !got.HasReason(ReasonSchemaUnknown) {
		t.Errorf("unknown schema: outcome %q reasons %v, want abstain/%s", got.Outcome, got.Reasons, ReasonSchemaUnknown)
	}
}

// TestProducerFieldNamesAreAccepted: the descriptor is the producer's own
// metadata, so a Hugging Face quantization_config (bits/sym) and this
// contract's older spelling (weight_bits/symmetric) must adjudicate identically,
// as must a build reporting its kernel set as `methods` or as `kernels`.
func TestProducerFieldNamesAreAccepted(t *testing.T) {
	producer := []byte(`{"schema":"vllmquant/v1",
		"artifact":{"quant_method":"gptq","bits":4,"group_size":128,"sym":true},
		"server":{"version":"0.6.3","methods":["gptq_marlin"],"compute_capability":"8.0"}}`)
	legacy := []byte(`{"schema":"vllmquant/v1",
		"artifact":{"quant_method":"gptq","weight_bits":4,"group_size":128,"symmetric":true},
		"server":{"version":"0.6.3","kernels":["gptq_marlin"],"compute_capability":80}}`)

	got, want := ParseAndAdjudicate(producer), ParseAndAdjudicate(legacy)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("producer spelling adjudicated differently:\n producer: %+v\n legacy:   %+v", got, want)
	}
	if got.Outcome != OutcomeSupported || got.Kernel() != KernelGPTQMarlin {
		t.Fatalf("outcome %q kernel %q, want supported/%s (reasons %v)", got.Outcome, got.Kernel(), KernelGPTQMarlin, got.Reasons)
	}
}

// TestCapabilityIsReadNotGuessed: a compute capability is declared either the
// way vLLM prints it or already packed; anything else is malformed, and absent
// is a different fact from malformed.
func TestCapabilityIsReadNotGuessed(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want Capability
	}{
		{`"8.0"`, 80},
		{`"7.5"`, 75},
		{`"9.0"`, 90},
		{`80`, 80},
		{`75`, 75},
		{`"80"`, 80},
		{`""`, CapabilityUndeclared},
		{`null`, CapabilityUndeclared},
		{`"eight"`, CapabilityMalformed},
		{`"8.x"`, CapabilityMalformed},
		{`"8.0.1"`, CapabilityMalformed},
		{`true`, CapabilityMalformed},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			var got Capability
			if err := json.Unmarshal([]byte(tc.raw), &got); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.raw, err)
			}
			if got != tc.want {
				t.Errorf("capability(%s) = %d, want %d", tc.raw, got, tc.want)
			}
		})
	}
}

// TestKernelTableAgreesWithMethodFamilies keeps the two tables from drifting: a
// kernel listed under a method's family must declare that it serves it, and
// every kernel in the table must belong to exactly one family.
func TestKernelTableAgreesWithMethodFamilies(t *testing.T) {
	seen := map[Kernel]int{}
	for method, mr := range methodTable {
		if len(mr.kernels) == 0 {
			t.Errorf("method %q declares no kernel family", method)
		}
		for _, k := range mr.kernels {
			kr, ok := kernelTable[k]
			if !ok {
				t.Errorf("method %q lists unknown kernel %q", method, k)
				continue
			}
			if kr.method != method {
				t.Errorf("kernel %q is listed under %q but serves %q", k, method, kr.method)
			}
			seen[k]++
		}
	}
	for k := range kernelTable {
		if seen[k] != 1 {
			t.Errorf("kernel %q appears in %d method families, want exactly 1", k, seen[k])
		}
	}
}

// TestEveryEmittedReasonIsPublished keeps the reason vocabulary closed: a code
// a caller can receive but cannot look up is not routable.
func TestEveryEmittedReasonIsPublished(t *testing.T) {
	for _, sel := range allFixtureSelections(t) {
		reasons := append([]Reason(nil), sel.Reasons...)
		for _, e := range sel.Excluded {
			reasons = append(reasons, e.Reason)
		}
		for _, r := range reasons {
			if !r.Known() {
				t.Errorf("emitted unpublished reason %q", r)
			}
			if !strings.HasPrefix(string(r), "VLLMQUANT_") {
				t.Errorf("reason %q lacks the leaf prefix", r)
			}
		}
		if !sel.Outcome.Known() {
			t.Errorf("emitted unpublished outcome %q", sel.Outcome)
		}
	}
}

// --- the fixture corpus ----------------------------------------------------
//
// testdata carries two conventions, and the walk below keeps them apart:
//
//   - a PAIR — <name>.input.json holds the request, <name>.golden.json holds the
//     whole expected Selection on the wire. The golden is the spec; the test
//     compares the full serialized selection, so an added, dropped, or renamed
//     field is a failure rather than a silent pass.
//   - a SELF-DESCRIBING file — a bare <name>.json that carries its own
//     expectation inline in _want_* keys alongside the request it describes.
//
// A golden is never fed to ParseAndAdjudicate: it is a result, not a request,
// and adjudicating one would "pass" by finding no schema in it.

const testdataDir = "testdata"

type fixturePair struct{ name, input, golden string }

// fixturePairs returns every input/golden pair, and fails if either half of a
// pair is missing — a golden with no input is dead weight, and an input with no
// golden is an unasserted case.
func fixturePairs(t *testing.T) []fixturePair {
	t.Helper()
	inputs, err := filepath.Glob(filepath.Join(testdataDir, "*.input.json"))
	if err != nil {
		t.Fatalf("glob inputs: %v", err)
	}
	goldens, err := filepath.Glob(filepath.Join(testdataDir, "*.golden.json"))
	if err != nil {
		t.Fatalf("glob goldens: %v", err)
	}
	unmatched := map[string]bool{}
	for _, g := range goldens {
		unmatched[g] = true
	}
	var out []fixturePair
	for _, in := range inputs {
		golden := strings.TrimSuffix(in, ".input.json") + ".golden.json"
		if !unmatched[golden] {
			t.Errorf("%s has no matching %s", filepath.Base(in), filepath.Base(golden))
			continue
		}
		delete(unmatched, golden)
		out = append(out, fixturePair{
			name:   strings.TrimSuffix(filepath.Base(in), ".input.json"),
			input:  in,
			golden: golden,
		})
	}
	for g := range unmatched {
		t.Errorf("%s has no matching input fixture", filepath.Base(g))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	if len(out) == 0 {
		t.Fatal("found no input/golden fixture pairs")
	}
	return out
}

// selfDescribingFixtures returns the bare *.json fixtures — the ones that are
// neither half of a pair.
func selfDescribingFixtures(t *testing.T) []string {
	t.Helper()
	all, err := filepath.Glob(filepath.Join(testdataDir, "*.json"))
	if err != nil {
		t.Fatalf("glob testdata: %v", err)
	}
	var out []string
	for _, p := range all {
		base := filepath.Base(p)
		if strings.HasSuffix(base, ".input.json") || strings.HasSuffix(base, ".golden.json") {
			continue
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		t.Fatal("found no self-describing fixtures")
	}
	return out
}

// TestFixtureCorpusIsWhollyExercised: every file under testdata is claimed by
// exactly one convention, so a fixture can never be added and then silently
// skipped by the walk.
func TestFixtureCorpusIsWhollyExercised(t *testing.T) {
	all, err := filepath.Glob(filepath.Join(testdataDir, "*.json"))
	if err != nil {
		t.Fatalf("glob testdata: %v", err)
	}
	claimed := map[string]int{}
	for _, p := range fixturePairs(t) {
		claimed[p.input]++
		claimed[p.golden]++
	}
	for _, p := range selfDescribingFixtures(t) {
		claimed[p]++
	}
	for _, p := range all {
		if claimed[p] != 1 {
			t.Errorf("%s is claimed by %d fixture conventions, want exactly 1", filepath.Base(p), claimed[p])
		}
	}
	if len(all) != len(claimed) {
		t.Errorf("walked %d fixtures, testdata holds %d", len(claimed), len(all))
	}
}

// fixtureCase is one self-describing testdata file: bytes in, the outcome and
// reason the file itself declares out. The expectation is read from the fixture
// rather than written in Go, so the fixture is an independently readable record
// of the contract's surface.
type fixtureCase struct {
	Note        string   `json:"_note"`
	WantOutcome Outcome  `json:"_want_outcome"`
	WantReason  Reason   `json:"_want_reason"`
	WantKernel  Kernel   `json:"_want_kernel"`
	WantArgs    []string `json:"_want_launch_args"`
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return raw
}

// allFixtureSelections adjudicates every REQUEST in the corpus — the inputs of
// each pair plus the self-describing files. Goldens are excluded by
// construction: they are results.
func allFixtureSelections(t *testing.T) []Selection {
	t.Helper()
	var out []Selection
	for _, p := range fixturePairs(t) {
		out = append(out, ParseAndAdjudicate(readFixture(t, p.input)))
	}
	for _, p := range selfDescribingFixtures(t) {
		out = append(out, ParseAndAdjudicate(readFixture(t, p)))
	}
	return out
}

// TestFixturesAgainstGoldens adjudicates each paired input and compares the
// WHOLE serialized selection against its golden.
func TestFixturesAgainstGoldens(t *testing.T) {
	for _, p := range fixturePairs(t) {
		t.Run("pair/"+p.name, func(t *testing.T) {
			got := ParseAndAdjudicate(readFixture(t, p.input))

			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal selection: %v", err)
			}
			var gotAny, wantAny any
			if err := json.Unmarshal(gotJSON, &gotAny); err != nil {
				t.Fatalf("re-read selection: %v", err)
			}
			wantRaw := readFixture(t, p.golden)
			if err := json.Unmarshal(wantRaw, &wantAny); err != nil {
				t.Fatalf("read golden %s: %v", filepath.Base(p.golden), err)
			}
			if !reflect.DeepEqual(gotAny, wantAny) {
				t.Errorf("selection does not match %s\n got: %s\nwant: %s",
					filepath.Base(p.golden), canonical(t, gotAny), canonical(t, wantAny))
			}
		})
	}
}

// TestSelfDescribingFixtures adjudicates every bare testdata file against the
// expectation it carries in its own _want_* keys.
func TestSelfDescribingFixtures(t *testing.T) {
	for _, path := range selfDescribingFixtures(t) {
		t.Run("self/"+strings.TrimSuffix(filepath.Base(path), ".json"), func(t *testing.T) {
			raw := readFixture(t, path)
			var want fixtureCase
			if err := json.Unmarshal(raw, &want); err != nil {
				t.Fatalf("read expectation: %v", err)
			}
			if want.WantOutcome == "" {
				t.Fatalf("fixture declares no _want_outcome")
			}

			got := ParseAndAdjudicate(raw)
			if got.Outcome != want.WantOutcome {
				t.Fatalf("outcome = %q, want %q (reasons %v)", got.Outcome, want.WantOutcome, got.Reasons)
			}
			if want.WantReason != "" && !got.HasReason(want.WantReason) {
				t.Errorf("reasons = %v, want %q", got.Reasons, want.WantReason)
			}
			if got.Kernel() != want.WantKernel {
				t.Errorf("kernel = %q, want %q", got.Kernel(), want.WantKernel)
			}
			if len(want.WantArgs) == 0 {
				if len(got.Args) != 0 {
					t.Errorf("args = %q, want empty", got.Args)
				}
			} else if !reflect.DeepEqual(got.Args, want.WantArgs) {
				t.Errorf("args = %q, want %q", got.Args, want.WantArgs)
			}
		})
	}
}

func canonical(t *testing.T, v any) string {
	t.Helper()
	out, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("canonicalize: %v", err)
	}
	return string(out)
}
