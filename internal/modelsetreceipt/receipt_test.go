package modelsetreceipt_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var evaluationTime = time.Date(2030, 1, 2, 12, 0, 0, 0, time.UTC)

func TestStartupReceiptBindsTwoRoleSpineDeterministically(t *testing.T) {
	requirements, inventory := compatibleInputs(t)
	resolution, err := modelsetresolve.Resolve(requirements, inventory, evaluationTime)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	expectation, err := modelsetreceipt.Bind(requirements, resolution, inventory)
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	receipt, err := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, evaluationTime)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if receipt.Status != modelsetreceipt.StatusCompatible || len(receipt.Roles) != 2 || len(receipt.Failures) != 0 {
		t.Fatalf("receipt = %+v, want compatible two-role receipt", receipt)
	}
	first, err := receipt.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON(first): %v", err)
	}
	repeated, err := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, evaluationTime)
	if err != nil {
		t.Fatalf("Evaluate(repeated): %v", err)
	}
	second, err := repeated.CanonicalJSON()
	if err != nil {
		t.Fatalf("CanonicalJSON(second): %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("receipt changed between runs:\nfirst=%s\nsecond=%s", first, second)
	}
	readBack, err := readEmittedReceipt(t, first)
	if err != nil {
		t.Fatalf("ParseJSON: %v\n%s", err, first)
	}
	for _, role := range readBack.Roles {
		if role.Expected == nil || role.Observed == nil || role.Reevaluated == nil || len(role.FactBindings) == 0 {
			t.Fatalf("role lacks bound selection/evidence: %+v", role)
		}
	}
	expectationRaw, err := expectation.CanonicalJSON()
	if err != nil {
		t.Fatalf("expectation CanonicalJSON: %v", err)
	}
	if _, err := modelsetreceipt.ParseExpectationJSON(expectationRaw); err != nil {
		t.Fatalf("ParseExpectationJSON: %v", err)
	}
}

func TestStartupReceiptFailsClosedAcrossRequiredEnvelope(t *testing.T) {
	requirements, inventory := compatibleInputs(t)
	resolution, err := modelsetresolve.Resolve(requirements, inventory, evaluationTime)
	if err != nil {
		t.Fatal(err)
	}
	expectation, err := modelsetreceipt.Bind(requirements, resolution, inventory)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		mutate     func(*modelsetreceipt.Expectation, *harnessmodelset.Intent, *modelsetresolve.Resolution, *modelinventory.Inventory, *time.Time)
		want       modelsetreceipt.Code
		wantSource string
		wantRole   string
	}{
		{
			name: "changed digest",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, _ *modelsetresolve.Resolution, inventory *modelinventory.Inventory, _ *time.Time) {
				setIntegerFact(inventory, "executor-model", "context_tokens", 30000)
			},
			want: modelsetreceipt.CodeInventoryDigest,
		},
		{
			name: "immutable identity mismatch",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, _ *modelsetresolve.Resolution, inventory *modelinventory.Inventory, _ *time.Time) {
				for candidateIndex := range inventory.Candidates {
					if inventory.Candidates[candidateIndex].ID == "executor-model" {
						inventory.Candidates[candidateIndex].Identity.Digest = digestText("replacement-executor-artifact")
					}
				}
			},
			want: modelsetreceipt.CodeIdentityMismatch, wantRole: "executor",
		},
		{
			name: "missing role",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, resolution *modelsetresolve.Resolution, _ *modelinventory.Inventory, _ *time.Time) {
				resolution.Roles = resolution.Roles[:1]
			},
			want: modelsetreceipt.CodeRoleMissing, wantRole: "planner",
		},
		{
			name: "selection mismatch",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, resolution *modelsetresolve.Resolution, _ *modelinventory.Inventory, _ *time.Time) {
				for roleIndex := range resolution.Roles {
					if resolution.Roles[roleIndex].RoleID == "executor" {
						resolution.Roles[roleIndex].Selection.CandidateID = "planner-model"
					}
				}
			},
			want: modelsetreceipt.CodeSelectionMismatch, wantRole: "executor",
		},
		{
			name: "expired evidence",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, _ *modelsetresolve.Resolution, _ *modelinventory.Inventory, asOf *time.Time) {
				*asOf = evaluationTime.Add(72 * time.Hour)
			},
			want: modelsetreceipt.CodeEvidenceStale, wantSource: string(modelinventory.CodeEvidenceStale),
		},
		{
			name: "runtime mismatch",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, _ *modelsetresolve.Resolution, inventory *modelinventory.Inventory, _ *time.Time) {
				setTextFact(inventory, "executor-model", "runtime", "other-runtime")
			},
			want: modelsetreceipt.CodeRuntimeMismatch, wantSource: string(modelsetresolve.CodeRuntime), wantRole: "executor",
		},
		{
			name: "unknown capability",
			mutate: func(_ *modelsetreceipt.Expectation, _ *harnessmodelset.Intent, _ *modelsetresolve.Resolution, inventory *modelinventory.Inventory, _ *time.Time) {
				removeCapability(inventory, "planner-model", "tool_calling")
			},
			want: modelsetreceipt.CodeRequiredFactUnknown, wantSource: string(modelsetresolve.CodeFactUnknown), wantRole: "planner",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotExpectation := clone(t, expectation)
			gotRequirements := clone(t, requirements)
			gotResolution := clone(t, resolution)
			gotInventory := cloneInventory(t, inventory)
			asOf := evaluationTime
			test.mutate(&gotExpectation, &gotRequirements, &gotResolution, &gotInventory, &asOf)

			receipt, err := modelsetreceipt.Evaluate(gotExpectation, gotRequirements, gotResolution, gotInventory, asOf)
			var incompatible *modelsetreceipt.IncompatibleError
			if !errors.As(err, &incompatible) {
				t.Fatalf("error = %T %v, want *IncompatibleError", err, err)
			}
			if receipt.Status != modelsetreceipt.StatusIncompatible || len(receipt.Failures) < 1 {
				t.Fatalf("receipt = %+v, want typed incompatibilities", receipt)
			}
			if !hasFailure(receipt.Failures, test.want, test.wantSource, test.wantRole) {
				t.Fatalf("failures = %+v, want code=%s source=%s role=%s", receipt.Failures, test.want, test.wantSource, test.wantRole)
			}
			raw, canonicalErr := receipt.CanonicalJSON()
			if canonicalErr != nil {
				t.Fatalf("CanonicalJSON: %v", canonicalErr)
			}
			readBack, parseErr := readEmittedReceipt(t, raw)
			if parseErr != nil || !reflect.DeepEqual(readBack, receipt) {
				t.Fatalf("independent read-back changed receipt: err=%v\nraw=%s\nread=%+v\nwant=%+v", parseErr, raw, readBack, receipt)
			}
		})
	}
}

func TestStartupReceiptDoesNotMutateInputsOrSerializeCredentials(t *testing.T) {
	requirements, inventory := compatibleInputs(t)
	resolution, err := modelsetresolve.Resolve(requirements, inventory, evaluationTime)
	if err != nil {
		t.Fatal(err)
	}
	expectation, err := modelsetreceipt.Bind(requirements, resolution, inventory)
	if err != nil {
		t.Fatal(err)
	}

	wantExpectation := clone(t, expectation)
	wantRequirements := clone(t, requirements)
	wantResolution := clone(t, resolution)
	wantInventory := cloneInventory(t, inventory)
	if _, err := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, evaluationTime); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expectation, wantExpectation) || !reflect.DeepEqual(requirements, wantRequirements) ||
		!reflect.DeepEqual(resolution, wantResolution) || !reflect.DeepEqual(inventory, wantInventory) {
		t.Fatal("Bind or Evaluate mutated caller-owned input")
	}

	const secret = "sk-proj-do-not-serialize"
	inventory.Candidates[0].Identity.Witnesses[0].Source = secret
	receipt, err := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, evaluationTime)
	var incompatible *modelsetreceipt.IncompatibleError
	if !errors.As(err, &incompatible) {
		t.Fatalf("credential-bearing inventory error = %T %v, want *IncompatibleError", err, err)
	}
	raw, err := receipt.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(secret)) {
		t.Fatalf("receipt leaked credential material:\n%s", raw)
	}
	if !hasFailure(receipt.Failures, modelsetreceipt.CodeInventoryInvalid, "", "") {
		t.Fatalf("failures = %+v, want inventory refusal", receipt.Failures)
	}
}

func TestReceiptStrictParsingRejectsMalformedUnknownAndDuplicateJSON(t *testing.T) {
	requirements, inventory := compatibleInputs(t)
	resolution, _ := modelsetresolve.Resolve(requirements, inventory, evaluationTime)
	expectation, _ := modelsetreceipt.Bind(requirements, resolution, inventory)
	receipt, _ := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, evaluationTime)
	raw, _ := receipt.CanonicalJSON()

	cases := [][]byte{
		[]byte(`{"schema":`),
		append(append([]byte(nil), raw...), []byte(`{}`)...),
		bytes.Replace(raw, []byte(`"schema": "`+modelsetreceipt.ReceiptSchema+`"`), []byte(`"schema": "`+modelsetreceipt.ReceiptSchema+`", "schema": "`+modelsetreceipt.ReceiptSchema+`"`), 1),
		bytes.Replace(raw, []byte(`"evaluated_at":`), []byte(`"unknown": true, "evaluated_at":`), 1),
	}
	for _, invalid := range cases {
		if parsed, err := modelsetreceipt.ParseJSON(invalid); err == nil || !reflect.DeepEqual(parsed, modelsetreceipt.Receipt{}) {
			t.Fatalf("ParseJSON accepted malformed input: err=%v parsed=%+v\n%s", err, parsed, invalid)
		}
	}

	credentialReceipt := clone(t, receipt)
	credentialReceipt.Roles[0].FactBindings[0].Witnesses[0].Source = "Bearer secret-value"
	if _, err := credentialReceipt.CanonicalJSON(); err == nil || !strings.Contains(err.Error(), "credential material") {
		t.Fatalf("credential-bearing receipt validation error = %v", err)
	}
}

func TestBindingAndStartupClockFailClosed(t *testing.T) {
	requirements, inventory := compatibleInputs(t)
	resolution, err := modelsetresolve.Resolve(requirements, inventory, evaluationTime)
	if err != nil {
		t.Fatal(err)
	}
	forged := clone(t, resolution)
	for roleIndex := range forged.Roles {
		if forged.Roles[roleIndex].RoleID == "executor" {
			forged.Roles[roleIndex].Selection.CandidateID = "planner-model"
		}
	}
	if expectation, err := modelsetreceipt.Bind(requirements, forged, inventory); err == nil || !reflect.DeepEqual(expectation, modelsetreceipt.Expectation{}) {
		t.Fatalf("Bind accepted forged resolution: expectation=%+v err=%v", expectation, err)
	}

	expectation, err := modelsetreceipt.Bind(requirements, resolution, inventory)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := modelsetreceipt.Evaluate(expectation, requirements, resolution, inventory, time.Time{})
	var incompatible *modelsetreceipt.IncompatibleError
	if !errors.As(err, &incompatible) || !hasFailure(receipt.Failures, modelsetreceipt.CodeAsOfRequired, "", "") {
		t.Fatalf("zero clock did not fail closed: receipt=%+v err=%v", receipt, err)
	}
	if raw, err := receipt.CanonicalJSON(); err != nil {
		t.Fatalf("zero-clock receipt is not canonical: %v", err)
	} else if _, err := modelsetreceipt.ParseJSON(raw); err != nil {
		t.Fatalf("zero-clock receipt cannot be read back: %v", err)
	}

	invalidExpectation := clone(t, expectation)
	invalidExpectation.Digests.Inventory = "sha256:invalid"
	if receipt, err := modelsetreceipt.Evaluate(invalidExpectation, requirements, resolution, inventory, evaluationTime); err == nil || !reflect.DeepEqual(receipt, modelsetreceipt.Receipt{}) {
		t.Fatalf("Evaluate accepted malformed expectation: receipt=%+v err=%v", receipt, err)
	}
}

func hasFailure(failures []modelsetreceipt.Failure, code modelsetreceipt.Code, source, role string) bool {
	for _, failure := range failures {
		if failure.Code == code && (source == "" || failure.SourceCode == source) && (role == "" || failure.RoleID == role) {
			return true
		}
	}
	return false
}

func readEmittedReceipt(t *testing.T, raw []byte) (modelsetreceipt.Receipt, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "harness.model-set.receipt.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	emitted, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return modelsetreceipt.ParseJSON(emitted)
}

func clone[T any](t *testing.T, value T) T {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func cloneInventory(t *testing.T, inventory modelinventory.Inventory) modelinventory.Inventory {
	t.Helper()
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	out, diagnostics := modelinventory.ParseJSON(raw, evaluationTime)
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	return out
}

func setIntegerFact(inventory *modelinventory.Inventory, candidateID, name string, value int64) {
	fact := findFact(inventory, candidateID, name)
	fact.Value = modelinventory.Integer(value)
}

func setTextFact(inventory *modelinventory.Inventory, candidateID, name, value string) {
	fact := findFact(inventory, candidateID, name)
	fact.Value = modelinventory.Text(value)
}

func findFact(inventory *modelinventory.Inventory, candidateID, name string) *modelinventory.Fact {
	for candidateIndex := range inventory.Candidates {
		candidate := &inventory.Candidates[candidateIndex]
		if candidate.ID != candidateID {
			continue
		}
		groups := [][]modelinventory.Fact{candidate.Evidence.Serving, candidate.Evidence.Platform, candidate.Evidence.Policy, candidate.Evidence.Capabilities}
		for _, group := range groups {
			for factIndex := range group {
				if group[factIndex].Name == name {
					return &group[factIndex]
				}
			}
		}
	}
	panic("fact not found: " + candidateID + "/" + name)
}

func removeCapability(inventory *modelinventory.Inventory, candidateID, name string) {
	for candidateIndex := range inventory.Candidates {
		candidate := &inventory.Candidates[candidateIndex]
		if candidate.ID != candidateID {
			continue
		}
		filtered := candidate.Evidence.Capabilities[:0]
		for _, fact := range candidate.Evidence.Capabilities {
			if fact.Name != name {
				filtered = append(filtered, fact)
			}
		}
		candidate.Evidence.Capabilities = filtered
		return
	}
}

func compatibleInputs(t *testing.T) (harnessmodelset.Intent, modelinventory.Inventory) {
	t.Helper()
	requirements := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{
		{ID: "executor", Required: true, Evidence: harnessmodelset.EvidencePolicy{MaxAgeHours: 24}, Alternatives: []harnessmodelset.Alternative{{
			ID: "local-executor", Capabilities: harnessmodelset.ModelRequirements{ToolCalling: boolPointer(true)},
			Operational: harnessmodelset.OperationalConstraints{Runtime: "llama.cpp", MaxMemoryBytes: int64Pointer(8 << 30), Locality: harnessmodelset.LocalityLocalOnly},
		}}},
		{ID: "planner", Required: true, Evidence: harnessmodelset.EvidencePolicy{MaxAgeHours: 24}, Alternatives: []harnessmodelset.Alternative{{
			ID: "tool-planner", Capabilities: harnessmodelset.ModelRequirements{ToolCalling: boolPointer(true), StructuredOutput: boolPointer(true), MinimumInputTokens: int64Pointer(65536)},
			Operational: harnessmodelset.OperationalConstraints{Runtime: "llama.cpp", ServingProtocol: harnessmodelset.ServingProtocolOpenAI},
		}}},
	}}
	observations := modelinventory.Observations{Locals: []modelinventory.LocalObservation{
		candidate("executor-model", 4<<30, 32768, false),
		candidate("planner-model", 16<<30, 131072, true),
	}}
	inventory, diagnostics := modelinventory.Normalize(observations, evaluationTime)
	if len(diagnostics) != 0 {
		t.Fatalf("Normalize: %v", diagnostics)
	}
	return requirements, inventory
}

func candidate(id string, memory, context int64, structured bool) modelinventory.LocalObservation {
	return modelinventory.LocalObservation{
		ID: id, Artifact: "models/" + id + ".gguf", Digest: digestText(id), Format: "gguf",
		IdentityEvidence: []modelinventory.Witness{witness("identity/" + id)},
		Evidence: modelinventory.EvidenceSet{
			Availability: fact("available", modelinventory.Bool(true), "availability/"+id),
			Serving: []modelinventory.Fact{
				fact("protocol", modelinventory.Text("openai-compatible"), "serving/protocol/"+id),
				fact("runtime", modelinventory.Text("llama.cpp"), "serving/runtime/"+id),
			},
			Platform: []modelinventory.Fact{
				fact("accelerator", modelinventory.Text("cuda"), "platform/accelerator/"+id),
				fact("accelerator_memory_bytes", modelinventory.Integer(memory), "platform/memory/"+id),
				fact("architecture", modelinventory.Text("amd64"), "platform/arch/"+id),
				fact("os", modelinventory.Text("linux"), "platform/os/"+id),
			},
			Policy: []modelinventory.Fact{
				fact("license", modelinventory.Text("apache-2.0"), "policy/license/"+id),
				fact("locality", modelinventory.Text("local"), "policy/locality/"+id),
				fact("privacy", modelinventory.Text("no-egress"), "policy/privacy/"+id),
			},
			Capabilities: []modelinventory.Fact{
				fact("context_tokens", modelinventory.Integer(context), "capability/context/"+id),
				fact("structured_json", modelinventory.Bool(structured), "capability/json/"+id),
				fact("tool_calling", modelinventory.Bool(true), "capability/tools/"+id),
			},
		},
	}
}

func fact(name string, value modelinventory.Value, source string) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{witness(source)}}
}

func witness(source string) modelinventory.Witness {
	return modelinventory.Witness{Kind: modelinventory.EvidenceProbe, Source: source, ObservedAt: evaluationTime.Add(-time.Hour).Format(time.RFC3339), ExpiresAt: evaluationTime.Add(48 * time.Hour).Format(time.RFC3339)}
}

func digestText(value string) string  { return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(value))) }
func boolPointer(value bool) *bool    { return &value }
func int64Pointer(value int64) *int64 { return &value }
