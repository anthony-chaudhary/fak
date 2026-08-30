package modelsetresolve_test

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var evaluationTime = time.Date(2026, 8, 19, 20, 0, 0, 0, time.UTC)

type candidateSpec struct {
	id              string
	family          string
	quantization    string
	local           bool
	available       bool
	runtime         string
	servingProtocol string
	os              string
	architecture    string
	accelerator     string
	memoryBytes     int64
	locality        string
	privacy         string
	license         string
	toolCalling     bool
	structuredJSON  bool
	toolProtocol    string
	contextTokens   int64
	observedAt      time.Time
}

func compatibleSpec(id string) candidateSpec {
	return candidateSpec{
		id:              id,
		family:          "qwen3.8",
		quantization:    "q4_k_m",
		available:       true,
		runtime:         "vllm",
		servingProtocol: "openai-compatible",
		os:              "linux",
		architecture:    "amd64",
		accelerator:     "cuda",
		memoryBytes:     12 << 30,
		locality:        "local",
		privacy:         "no-egress",
		license:         "apache-2.0",
		toolCalling:     true,
		structuredJSON:  true,
		toolProtocol:    "openai-tools",
		contextTokens:   131072,
		observedAt:      evaluationTime.Add(-time.Hour),
	}
}

func exactAlternative(id string) harnessmodelset.Alternative {
	return harnessmodelset.Alternative{
		ID: id,
		Capabilities: harnessmodelset.ModelRequirements{
			ToolCalling:        boolPointer(true),
			StructuredOutput:   boolPointer(true),
			ToolProtocol:       harnessmodelset.ToolProtocolOpenAI,
			MinimumInputTokens: int64Pointer(32768),
			Modalities:         []harnessmodelset.Modality{harnessmodelset.ModalityText},
		},
		Operational: harnessmodelset.OperationalConstraints{
			Runtime:          "vllm",
			ServingProtocol:  harnessmodelset.ServingProtocolOpenAI,
			Platforms:        []string{"linux/amd64"},
			Accelerators:     []string{"cuda"},
			MaxMemoryBytes:   int64Pointer(16 << 30),
			Locality:         harnessmodelset.LocalityLocalOnly,
			Privacy:          harnessmodelset.PrivacyNoEgress,
			LicenseAllowlist: []string{"Apache-2.0"},
		},
	}
}

func role(id string, required bool, alternatives ...harnessmodelset.Alternative) harnessmodelset.Role {
	return harnessmodelset.Role{
		ID:           id,
		Required:     required,
		Alternatives: alternatives,
		Preference:   &harnessmodelset.SelectionPreference{Mode: harnessmodelset.PreferenceDeclaredOrder},
		Evidence: harnessmodelset.EvidencePolicy{
			MaxAgeHours: 4,
			RequiredKinds: []harnessmodelset.EvidenceKind{
				harnessmodelset.EvidenceModelBehaviorProbe,
				harnessmodelset.EvidenceRuntimeProbe,
				harnessmodelset.EvidenceOperatorAttestation,
			},
		},
	}
}

func TestResolveExactMatch(t *testing.T) {
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{
		role("planner", true, exactAlternative("exact")),
	}}
	inventory := normalize(t, compatibleSpec("planner-exact"))

	resolution, err := modelsetresolve.Resolve(intent, inventory, evaluationTime)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if resolution.Schema != modelsetresolve.Schema || resolution.EvaluatedAt != "2026-08-19T20:00:00Z" {
		t.Fatalf("resolution envelope = %+v", resolution)
	}
	if len(resolution.Roles) != 1 || resolution.Roles[0].Status != modelsetresolve.StatusSelected {
		t.Fatalf("roles = %+v", resolution.Roles)
	}
	want := &modelsetresolve.Selection{AlternativeID: "exact", CandidateID: "planner-exact"}
	if !reflect.DeepEqual(resolution.Roles[0].Selection, want) {
		t.Fatalf("selection = %+v, want %+v", resolution.Roles[0].Selection, want)
	}
	if got := resolution.Rejections(); len(got) != 0 {
		t.Fatalf("exact candidate rejected: %+v", got)
	}
}

func TestResolveComposesFamilyAndQuantizationConstraints(t *testing.T) {
	preferred := compatibleSpec("preferred")
	preferred.family = "Qwen3.8"
	preferred.quantization = "Q4_K_M"
	otherFamily := compatibleSpec("other-family")
	otherFamily.family = "llama"
	otherQuant := compatibleSpec("other-quant")
	otherQuant.quantization = "q8_0"

	alternative := exactAlternative("composed")
	alternative.Capabilities.Family = "qwen3.8"
	alternative.Capabilities.Quantization = "q4_k_m"
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{role("executor", true, alternative)}}
	resolution, err := modelsetresolve.Resolve(intent, normalize(t, otherFamily, otherQuant, preferred), evaluationTime)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got := resolution.Roles[0].Selection.CandidateID; got != "preferred" {
		t.Fatalf("selected %q, want preferred", got)
	}
	if !hasCode(resolution.Rejections(), modelsetresolve.CodeFamily) || !hasCode(resolution.Rejections(), modelsetresolve.CodeQuantization) {
		t.Fatalf("rejections = %+v, want family and quantization mismatches", resolution.Rejections())
	}
}

func TestResolveFamilyAndQuantizationFactsFailClosed(t *testing.T) {
	alternative := exactAlternative("composed")
	alternative.Capabilities.Family = "qwen3.8"
	alternative.Capabilities.Quantization = "q4_k_m"
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{role("executor", true, alternative)}}

	missing := compatibleSpec("missing")
	missingEvidence := evidenceFor(missing)
	missingEvidence.Capabilities = missingEvidence.Capabilities[2:]
	wrongType := compatibleSpec("wrong-type")
	wrongEvidence := evidenceFor(wrongType)
	wrongEvidence.Capabilities[0].Value = modelinventory.Bool(true)
	wrongEvidence.Capabilities[1].Value = modelinventory.Integer(4)
	inventory := normalizeEvidence(t, missing, missingEvidence, wrongType, wrongEvidence)

	resolution, err := modelsetresolve.Resolve(intent, inventory, evaluationTime)
	if err == nil {
		t.Fatal("Resolve succeeded, want required role unresolved")
	}
	if !hasCode(resolution.Rejections(), modelsetresolve.CodeFactUnknown) || !hasCode(resolution.Rejections(), modelsetresolve.CodeFactType) {
		t.Fatalf("rejections = %+v, want missing and wrong-type facts", resolution.Rejections())
	}
}

func TestResolveHonorsOrderedAlternatives(t *testing.T) {
	first := exactAlternative("local-runtime")
	first.Operational.Runtime = "llamacpp"
	second := exactAlternative("compatible-runtime")
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{
		role("executor", true, first, second),
	}}

	resolution, err := modelsetresolve.Resolve(intent, normalize(t, compatibleSpec("executor")), evaluationTime)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got := resolution.Roles[0].Selection.AlternativeID; got != "compatible-runtime" {
		t.Fatalf("alternative = %q, want compatible-runtime", got)
	}
	if !hasCode(resolution.Rejections(), modelsetresolve.CodeRuntime) {
		t.Fatalf("rejections = %+v, want runtime reason for the first alternative", resolution.Rejections())
	}
}

func TestResolveUsesPreferenceThenStableCandidateKey(t *testing.T) {
	alpha := compatibleSpec("alpha")
	zulu := compatibleSpec("zulu")
	firstInventory := normalize(t, zulu, alpha)
	secondInventory := firstInventory
	secondInventory.Candidates = append([]modelinventory.Candidate(nil), firstInventory.Candidates...)
	secondInventory.Candidates[0], secondInventory.Candidates[1] = secondInventory.Candidates[1], secondInventory.Candidates[0]
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{
		role("planner", true, exactAlternative("any-compatible")),
	}}

	first, err := modelsetresolve.Resolve(intent, firstInventory, evaluationTime)
	if err != nil {
		t.Fatalf("first Resolve: %v", err)
	}
	second, err := modelsetresolve.Resolve(intent, secondInventory, evaluationTime)
	if err != nil {
		t.Fatalf("second Resolve: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("candidate permutation changed resolution:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if got := first.Roles[0].Selection.CandidateID; got != "alpha" {
		t.Fatalf("stable tie-break selected %q, want alpha", got)
	}

	local := compatibleSpec("z-local")
	local.local = true
	localPreference := intent
	localPreference.Roles[0].Preference.Mode = harnessmodelset.PreferenceLocalFirst
	preferred, err := modelsetresolve.Resolve(localPreference, normalize(t, alpha, local), evaluationTime)
	if err != nil {
		t.Fatalf("local-first Resolve: %v", err)
	}
	if got := preferred.Roles[0].Selection.CandidateID; got != "z-local" {
		t.Fatalf("local-first selected %q, want z-local", got)
	}

	lowMemory := compatibleSpec("z-low-memory")
	lowMemory.memoryBytes = 8 << 30
	memoryPreference := intent
	memoryPreference.Roles[0].Preference.Mode = harnessmodelset.PreferenceLowestMemory
	preferred, err = modelsetresolve.Resolve(memoryPreference, normalize(t, alpha, lowMemory), evaluationTime)
	if err != nil {
		t.Fatalf("lowest-memory Resolve: %v", err)
	}
	if got := preferred.Roles[0].Selection.CandidateID; got != "z-low-memory" {
		t.Fatalf("lowest-memory selected %q, want z-low-memory", got)
	}
}

func TestResolveReportsEachRoleAndFailsOnlyRequiredAbsence(t *testing.T) {
	planner := role("planner", true, exactAlternative("planner-compatible"))
	executorAlternative := exactAlternative("executor-cpu")
	executorAlternative.Operational.Accelerators = []string{"cpu"}
	executor := role("executor", true, executorAlternative)
	optionalAlternative := exactAlternative("observer-llamacpp")
	optionalAlternative.Operational.Runtime = "llamacpp"
	observer := role("observer", false, optionalAlternative)
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{planner, observer, executor}}

	resolution, err := modelsetresolve.Resolve(intent, normalize(t, compatibleSpec("planner-candidate")), evaluationTime)
	var requiredErr *modelsetresolve.RequiredRolesError
	if !errors.As(err, &requiredErr) {
		t.Fatalf("error = %T %v, want *RequiredRolesError", err, err)
	}
	if !reflect.DeepEqual(requiredErr.RoleIDs, []string{"executor"}) {
		t.Fatalf("failed required roles = %v, want executor", requiredErr.RoleIDs)
	}
	if got := roleStatus(t, resolution, "planner"); got != modelsetresolve.StatusSelected {
		t.Fatalf("planner status = %s", got)
	}
	if got := roleStatus(t, resolution, "executor"); got != modelsetresolve.StatusRequiredUnresolved {
		t.Fatalf("executor status = %s", got)
	}
	if got := roleStatus(t, resolution, "observer"); got != modelsetresolve.StatusOptionalUnresolved {
		t.Fatalf("observer status = %s", got)
	}
}

func TestResolveRejectionsAreCompleteAndStable(t *testing.T) {
	alpha := incompatibleSpec("alpha-rejected")
	beta := incompatibleSpec("beta-rejected")
	firstInventory := normalize(t, beta, alpha)
	secondInventory := firstInventory
	secondInventory.Candidates = append([]modelinventory.Candidate(nil), firstInventory.Candidates...)
	for index := range secondInventory.Candidates {
		candidate := &secondInventory.Candidates[index]
		reverseFacts(candidate.Evidence.Serving)
		reverseFacts(candidate.Evidence.Platform)
		reverseFacts(candidate.Evidence.Policy)
		reverseFacts(candidate.Evidence.Capabilities)
	}
	secondInventory.Candidates[0], secondInventory.Candidates[1] = secondInventory.Candidates[1], secondInventory.Candidates[0]
	staleRole := role("executor", true, exactAlternative("strict"))
	staleRole.Evidence.MaxAgeHours = 1
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{staleRole}}

	first, firstErr := modelsetresolve.Resolve(intent, firstInventory, evaluationTime)
	second, secondErr := modelsetresolve.Resolve(intent, secondInventory, evaluationTime)
	if firstErr == nil || secondErr == nil {
		t.Fatalf("incompatible inventories resolved: first=%v second=%v", firstErr, secondErr)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("permutation changed rejection output:\nfirst=%+v\nsecond=%+v", first, second)
	}
	rejections := first.Rejections()
	for _, code := range []modelsetresolve.RejectionCode{
		modelsetresolve.CodeToolProtocol,
		modelsetresolve.CodeServingProtocol,
		modelsetresolve.CodePlatform,
		modelsetresolve.CodeMemory,
		modelsetresolve.CodeLocality,
		modelsetresolve.CodePrivacy,
		modelsetresolve.CodeLicense,
		modelsetresolve.CodeEvidenceStale,
	} {
		if !hasCode(rejections, code) {
			t.Errorf("rejections omit %s: %+v", code, rejections)
		}
	}
	if !sort.SliceIsSorted(rejections, func(i, j int) bool { return rejectionKey(rejections[i]) < rejectionKey(rejections[j]) }) {
		t.Fatalf("rejections are not in stable order: %+v", rejections)
	}
}

func TestResolvePreservesSourceValidationFailures(t *testing.T) {
	invalidIntent := harnessmodelset.Intent{Schema: "unknown"}
	inventory := normalize(t, compatibleSpec("candidate"))
	_, err := modelsetresolve.Resolve(invalidIntent, inventory, evaluationTime.Add(48*time.Hour))
	var inputErr *modelsetresolve.InputError
	if !errors.As(err, &inputErr) {
		t.Fatalf("error = %T %v, want *InputError", err, err)
	}
	if len(inputErr.IntentDiagnostics) == 0 || !hasInventoryCode(inputErr.InventoryDiagnostics, modelinventory.CodeEvidenceStale) {
		t.Fatalf("source diagnostics were not preserved: %+v", inputErr)
	}
}

func incompatibleSpec(id string) candidateSpec {
	spec := compatibleSpec(id)
	spec.servingProtocol = "grpc"
	spec.os = "windows"
	spec.architecture = "arm64"
	spec.memoryBytes = 64 << 30
	spec.locality = "remote"
	spec.privacy = "public-endpoint"
	spec.license = "proprietary"
	spec.toolProtocol = "anthropic-tools"
	spec.observedAt = evaluationTime.Add(-10 * time.Hour)
	return spec
}

func normalize(t *testing.T, specs ...candidateSpec) modelinventory.Inventory {
	t.Helper()
	observations := modelinventory.Observations{}
	for _, spec := range specs {
		identityWitness := witness(modelinventory.EvidenceProbe, "identity://"+spec.id, spec.observedAt)
		evidence := evidenceFor(spec)
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(spec.id)))
		if spec.local {
			observations.Locals = append(observations.Locals, modelinventory.LocalObservation{
				ID: spec.id, Artifact: "models/" + spec.id + ".gguf", Digest: digest, Format: "gguf",
				IdentityEvidence: []modelinventory.Witness{identityWitness}, Evidence: evidence,
			})
			continue
		}
		revision := fmt.Sprintf("%x", sha256.Sum256([]byte("revision:"+spec.id)))[:40]
		observations.Providers = append(observations.Providers, modelinventory.ProviderObservation{
			ID: spec.id, Provider: "example", Repository: "models/" + spec.id, Revision: revision, Digest: digest, Format: "safetensors",
			IdentityEvidence: []modelinventory.Witness{{
				Kind: modelinventory.EvidenceArtifactMetadata, Source: "metadata://" + spec.id,
				ObservedAt: spec.observedAt.Format(time.RFC3339), ExpiresAt: evaluationTime.Add(24 * time.Hour).Format(time.RFC3339),
			}}, Evidence: evidence,
		})
	}
	inventory, diagnostics := modelinventory.Normalize(observations, evaluationTime)
	if len(diagnostics) != 0 {
		t.Fatalf("Normalize diagnostics:\n%s", diagnostics)
	}
	return inventory
}

func normalizeEvidence(t *testing.T, specsAndEvidence ...any) modelinventory.Inventory {
	t.Helper()
	observations := modelinventory.Observations{}
	for index := 0; index < len(specsAndEvidence); index += 2 {
		spec := specsAndEvidence[index].(candidateSpec)
		evidence := specsAndEvidence[index+1].(modelinventory.EvidenceSet)
		identityWitness := witness(modelinventory.EvidenceProbe, "identity://"+spec.id, spec.observedAt)
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(spec.id)))
		revision := fmt.Sprintf("%x", sha256.Sum256([]byte("revision:"+spec.id)))[:40]
		observations.Providers = append(observations.Providers, modelinventory.ProviderObservation{
			ID: spec.id, Provider: "example", Repository: "models/" + spec.id, Revision: revision, Digest: digest, Format: "safetensors",
			IdentityEvidence: []modelinventory.Witness{identityWitness}, Evidence: evidence,
		})
	}
	inventory, diagnostics := modelinventory.Normalize(observations, evaluationTime)
	if len(diagnostics) != 0 {
		t.Fatalf("Normalize diagnostics:\n%s", diagnostics)
	}
	return inventory
}

func evidenceFor(spec candidateSpec) modelinventory.EvidenceSet {
	probe := func(name string, value modelinventory.Value) modelinventory.Fact {
		return fact(name, value, modelinventory.EvidenceProbe, spec.id+"/"+name, spec.observedAt)
	}
	attest := func(name string, value modelinventory.Value) modelinventory.Fact {
		return fact(name, value, modelinventory.EvidenceOperatorAttestation, spec.id+"/"+name, spec.observedAt)
	}
	return modelinventory.EvidenceSet{
		Availability: probe("available", modelinventory.Bool(spec.available)),
		Serving: []modelinventory.Fact{
			probe("runtime", modelinventory.Text(spec.runtime)),
			probe("protocol", modelinventory.Text(spec.servingProtocol)),
		},
		Platform: []modelinventory.Fact{
			probe("os", modelinventory.Text(spec.os)),
			probe("architecture", modelinventory.Text(spec.architecture)),
			probe("accelerator", modelinventory.Text(spec.accelerator)),
			probe("accelerator_memory_bytes", modelinventory.Integer(spec.memoryBytes)),
		},
		Policy: []modelinventory.Fact{
			attest("locality", modelinventory.Text(spec.locality)),
			attest("privacy", modelinventory.Text(spec.privacy)),
			attest("license", modelinventory.Text(spec.license)),
		},
		Capabilities: []modelinventory.Fact{
			probe("model.family", modelinventory.Text(spec.family)),
			probe("weights.quantization", modelinventory.Text(spec.quantization)),
			probe("tool_calling", modelinventory.Bool(spec.toolCalling)),
			probe("structured_json", modelinventory.Bool(spec.structuredJSON)),
			probe("tool_protocol", modelinventory.Text(spec.toolProtocol)),
			probe("context_tokens", modelinventory.Integer(spec.contextTokens)),
			probe("modality.text", modelinventory.Bool(true)),
		},
	}
}

func fact(name string, value modelinventory.Value, kind modelinventory.WitnessKind, source string, observed time.Time) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{witness(kind, source, observed)}}
}

func witness(kind modelinventory.WitnessKind, source string, observed time.Time) modelinventory.Witness {
	return modelinventory.Witness{
		Kind: kind, Source: source, ObservedAt: observed.Format(time.RFC3339),
		ExpiresAt: evaluationTime.Add(24 * time.Hour).Format(time.RFC3339),
	}
}

func boolPointer(value bool) *bool    { return &value }
func int64Pointer(value int64) *int64 { return &value }

func roleStatus(t *testing.T, resolution modelsetresolve.Resolution, roleID string) modelsetresolve.RoleStatus {
	t.Helper()
	for _, role := range resolution.Roles {
		if role.RoleID == roleID {
			return role.Status
		}
	}
	t.Fatalf("role %q absent from %+v", roleID, resolution.Roles)
	return ""
}

func hasCode(rejections []modelsetresolve.Rejection, code modelsetresolve.RejectionCode) bool {
	for _, rejection := range rejections {
		if rejection.Code == code {
			return true
		}
	}
	return false
}

func hasInventoryCode(diagnostics modelinventory.Diagnostics, code modelinventory.Code) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == code {
			return true
		}
	}
	return false
}

func reverseFacts(facts []modelinventory.Fact) {
	for left, right := 0, len(facts)-1; left < right; left, right = left+1, right-1 {
		facts[left], facts[right] = facts[right], facts[left]
	}
}

func rejectionKey(rejection modelsetresolve.Rejection) string {
	return fmt.Sprintf("%s\x00%08d\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s",
		rejection.RoleID, rejection.AlternativeIndex, rejection.CandidateID, rejection.Code, rejection.Constraint,
		rejection.EvidenceSource, rejection.Expected, rejection.Actual, rejection.Remediation)
}
