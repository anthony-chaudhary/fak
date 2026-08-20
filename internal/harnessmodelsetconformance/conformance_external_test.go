package harnessmodelsetconformance_test

import (
	"bytes"
	"crypto/sha256"
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
	"github.com/anthony-chaudhary/fak/internal/modelsetlock"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var conformanceTime = time.Date(2030, 1, 2, 12, 0, 0, 0, time.UTC)

const selectionPolicy = "{\"schema\":\"fak.harness-model-set-selection-policy/1\"}\n"

func TestTwoRoleModelSetConformanceSpine(t *testing.T) {
	intent := readIntent(t)
	inventory, inventoryRaw := normalize(t, successObservations())
	assertGolden(t, "model-inventory.json", inventoryRaw)

	permuted := successObservations()
	permuted.Locals[0], permuted.Locals[1] = permuted.Locals[1], permuted.Locals[0]
	_, permutedRaw := normalize(t, permuted)
	if !bytes.Equal(inventoryRaw, permutedRaw) {
		t.Fatalf("normalized inventory depends on observation order:\nfirst=%s\nsecond=%s", inventoryRaw, permutedRaw)
	}

	resolution := resolve(t, intent, inventory)
	repeated := resolve(t, intent, inventory)
	if !reflect.DeepEqual(resolution, repeated) {
		t.Fatalf("resolver changed between identical runs:\nfirst=%+v\nsecond=%+v", resolution, repeated)
	}
	assertSelected(t, resolution, "executor", "executor-exact")
	assertSelected(t, resolution, "planner", "planner-exact")

	inputs := lockInputs(intent, inventory)
	lock, err := modelsetlock.New(inputs, resolution)
	if err != nil {
		t.Fatalf("modelsetlock.New: %v", err)
	}
	lockRaw, err := modelsetlock.CanonicalJSON(lock)
	if err != nil {
		t.Fatalf("modelsetlock.CanonicalJSON: %v", err)
	}
	assertGolden(t, "harness.model-set.lock.json", lockRaw)

	expectation, err := modelsetreceipt.Bind(intent, resolution, inventory)
	if err != nil {
		t.Fatalf("modelsetreceipt.Bind: %v", err)
	}
	expectationRaw, err := expectation.CanonicalJSON()
	if err != nil {
		t.Fatalf("expectation.CanonicalJSON: %v", err)
	}
	assertGolden(t, "harness.model-set.expectation.json", expectationRaw)

	receipt, err := modelsetreceipt.Evaluate(expectation, intent, resolution, inventory, conformanceTime)
	if err != nil {
		t.Fatalf("modelsetreceipt.Evaluate: %v", err)
	}
	if receipt.Status != modelsetreceipt.StatusCompatible || len(receipt.Roles) != 2 || len(receipt.Failures) != 0 {
		t.Fatalf("success receipt = %+v", receipt)
	}
	receiptRaw, err := receipt.CanonicalJSON()
	if err != nil {
		t.Fatalf("receipt.CanonicalJSON: %v", err)
	}
	assertGolden(t, "harness.model-set.receipt.json", receiptRaw)

	artifactDir := t.TempDir()
	lockPath := filepath.Join(artifactDir, modelsetlock.DefaultFileName)
	if err := modelsetlock.WriteFile(lockPath, lock); err != nil {
		t.Fatalf("first lock write: %v", err)
	}
	firstLock := readFile(t, lockPath)
	if err := modelsetlock.WriteFile(lockPath, lock); err != nil {
		t.Fatalf("second lock write: %v", err)
	}
	if secondLock := readFile(t, lockPath); !bytes.Equal(firstLock, secondLock) || !bytes.Equal(secondLock, lockRaw) {
		t.Fatal("canonical lock changed across deterministic writes")
	}
	if _, err := modelsetlock.ReadFile(lockPath); err != nil {
		t.Fatalf("independent lock read-back: %v", err)
	}
	receiptPath := filepath.Join(artifactDir, "harness.model-set.receipt.json")
	if err := os.WriteFile(receiptPath, receiptRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	readReceipt, err := modelsetreceipt.ParseJSON(readFile(t, receiptPath))
	if err != nil || !reflect.DeepEqual(readReceipt, receipt) {
		t.Fatalf("independent receipt read-back = (%+v, %v), want %+v", readReceipt, err, receipt)
	}
	repeatedReceipt, err := modelsetreceipt.Evaluate(expectation, intent, resolution, inventory, conformanceTime)
	if err != nil {
		t.Fatalf("repeated receipt: %v", err)
	}
	repeatedReceiptRaw, _ := repeatedReceipt.CanonicalJSON()
	if !bytes.Equal(receiptRaw, repeatedReceiptRaw) {
		t.Fatal("compatible startup receipt changed across identical runs")
	}

	t.Run("exact immutable model drift", func(t *testing.T) {
		drifted := cloneInventory(t, inventory)
		candidateByID(t, &drifted, "executor-exact").Identity.Digest = digest("executor-exact-repacked")
		got, evaluateErr := modelsetreceipt.Evaluate(expectation, intent, resolution, drifted, conformanceTime)
		requireIncompatible(t, got, evaluateErr)
		assertReceiptFailure(t, got.Failures, modelsetreceipt.CodeIdentityMismatch, "", "executor", false)
	})

	t.Run("compatible substitute cannot replace exact lock", func(t *testing.T) {
		replacement, _ := normalize(t, replacementObservations())
		got, evaluateErr := modelsetreceipt.Evaluate(expectation, intent, resolution, replacement, conformanceTime)
		requireIncompatible(t, got, evaluateErr)
		failure := assertReceiptFailure(t, got.Failures, modelsetreceipt.CodeSelectionMismatch, "", "executor", false)
		if !strings.Contains(failure.Expected, "executor-exact") || !strings.Contains(failure.Actual, "z-executor-compatible") {
			t.Fatalf("selection mismatch = %+v, want exact and substitute candidate IDs", failure)
		}
	})

	t.Run("incompatible alternatives fail closed", func(t *testing.T) {
		incompatible, _ := normalize(t, incompatibleObservations())
		failedResolution, resolveErr := modelsetresolve.Resolve(intent, incompatible, conformanceTime)
		var requiredErr *modelsetresolve.RequiredRolesError
		if !errors.As(resolveErr, &requiredErr) || !reflect.DeepEqual(requiredErr.RoleIDs, []string{"executor"}) {
			t.Fatalf("resolver error = %T %v roles=%v, want executor RequiredRolesError", resolveErr, resolveErr, requiredErr)
		}
		for _, code := range []modelsetresolve.RejectionCode{
			modelsetresolve.CodeServingProtocol,
			modelsetresolve.CodeMemory,
			modelsetresolve.CodeEvidenceStale,
		} {
			assertRejection(t, failedResolution.Rejections(), code, code == modelsetresolve.CodeEvidenceStale)
		}

		got, evaluateErr := modelsetreceipt.Evaluate(expectation, intent, resolution, incompatible, conformanceTime)
		requireIncompatible(t, got, evaluateErr)
		assertReceiptFailure(t, got.Failures, modelsetreceipt.CodeRuntimeMismatch, string(modelsetresolve.CodeServingProtocol), "executor", false)
		assertReceiptFailure(t, got.Failures, modelsetreceipt.CodeRequiredFactMismatch, string(modelsetresolve.CodeMemory), "executor", false)
		assertReceiptFailure(t, got.Failures, modelsetreceipt.CodeEvidenceStale, string(modelsetresolve.CodeEvidenceStale), "executor", true)
		if after := readFile(t, lockPath); !bytes.Equal(after, firstLock) {
			t.Fatal("failed startup evaluation mutated the canonical lock")
		}
	})
}

func readIntent(t *testing.T) harnessmodelset.Intent {
	t.Helper()
	raw := readFile(t, filepath.Join("testdata", "harness.model-set.json"))
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		t.Fatalf("harnessmodelset.ParseJSON: %v", err)
	}
	canonical, err := harnessmodelset.CanonicalJSON(intent)
	if err != nil || !bytes.Equal(canonical, raw) {
		t.Fatalf("intent fixture is not canonical: err=%v\nwant=%s\ngot=%s", err, raw, canonical)
	}
	if len(intent.Roles) != 2 || intent.Roles[0].ID != "executor" || intent.Roles[1].ID != "planner" {
		t.Fatalf("intent roles = %+v, want executor and planner", intent.Roles)
	}
	return intent
}

func normalize(t *testing.T, observations modelinventory.Observations) (modelinventory.Inventory, []byte) {
	t.Helper()
	inventory, diagnostics := modelinventory.Normalize(observations, conformanceTime)
	if len(diagnostics) != 0 {
		t.Fatalf("modelinventory.Normalize: %s", diagnostics)
	}
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatalf("inventory.CanonicalJSON: %s", diagnostics)
	}
	readBack, diagnostics := modelinventory.ParseJSON(raw, conformanceTime)
	if len(diagnostics) != 0 || !reflect.DeepEqual(readBack, inventory) {
		t.Fatalf("inventory read-back diagnostics=%s\nread=%+v\nwant=%+v", diagnostics, readBack, inventory)
	}
	return inventory, raw
}

func resolve(t *testing.T, intent harnessmodelset.Intent, inventory modelinventory.Inventory) modelsetresolve.Resolution {
	t.Helper()
	resolution, err := modelsetresolve.Resolve(intent, inventory, conformanceTime)
	if err != nil {
		t.Fatalf("modelsetresolve.Resolve: %v", err)
	}
	return resolution
}

func lockInputs(intent harnessmodelset.Intent, inventory modelinventory.Inventory) modelsetlock.Inputs {
	return modelsetlock.Inputs{
		Intent: intent, Inventory: inventory, RuleBytes: []byte(selectionPolicy),
		Target:          modelsetlock.Target{OS: "linux", Architecture: "amd64", Accelerator: "cpu", Runtime: "mixed-runtime"},
		ResolverVersion: modelsetresolve.Schema,
	}
}

func successObservations() modelinventory.Observations {
	return modelinventory.Observations{Locals: []modelinventory.LocalObservation{
		candidate("executor-exact", 4<<30, 32768, false, "openai-compatible", conformanceTime.Add(-time.Hour)),
		candidate("planner-exact", 16<<30, 131072, true, "openai-compatible", conformanceTime.Add(-time.Hour)),
	}}
}

func replacementObservations() modelinventory.Observations {
	return modelinventory.Observations{Locals: []modelinventory.LocalObservation{
		candidate("z-executor-compatible", 4<<30, 32768, false, "openai-compatible", conformanceTime.Add(-time.Hour)),
		candidate("planner-exact", 16<<30, 131072, true, "openai-compatible", conformanceTime.Add(-time.Hour)),
	}}
}

func incompatibleObservations() modelinventory.Observations {
	return modelinventory.Observations{Locals: []modelinventory.LocalObservation{
		candidate("executor-memory", 16<<30, 32768, false, "openai-compatible", conformanceTime.Add(-time.Hour)),
		candidate("executor-protocol", 4<<30, 32768, false, "grpc", conformanceTime.Add(-time.Hour)),
		candidate("executor-stale", 4<<30, 32768, false, "openai-compatible", conformanceTime.Add(-8*time.Hour)),
		candidate("planner-exact", 16<<30, 131072, true, "openai-compatible", conformanceTime.Add(-time.Hour)),
	}}
}

func candidate(id string, memory, context int64, structured bool, protocol string, observedAt time.Time) modelinventory.LocalObservation {
	probe := func(name string, value modelinventory.Value, source string) modelinventory.Fact {
		return inventoryFact(name, value, modelinventory.EvidenceProbe, source, observedAt)
	}
	attest := func(name string, value modelinventory.Value, source string) modelinventory.Fact {
		return inventoryFact(name, value, modelinventory.EvidenceOperatorAttestation, source, observedAt)
	}
	return modelinventory.LocalObservation{
		ID: id, Artifact: "models/" + id + ".gguf", Digest: digest(id), Format: "gguf",
		IdentityEvidence: []modelinventory.Witness{inventoryWitness(modelinventory.EvidenceProbe, "identity/"+id, observedAt)},
		Evidence: modelinventory.EvidenceSet{
			Availability: probe("available", modelinventory.Bool(true), "availability/"+id),
			Serving: []modelinventory.Fact{
				probe("protocol", modelinventory.Text(protocol), "serving/protocol/"+id),
				probe("runtime", modelinventory.Text("llama.cpp"), "serving/runtime/"+id),
			},
			Platform: []modelinventory.Fact{
				probe("accelerator", modelinventory.Text("cuda"), "platform/accelerator/"+id),
				probe("accelerator_memory_bytes", modelinventory.Integer(memory), "platform/memory/"+id),
				probe("architecture", modelinventory.Text("amd64"), "platform/architecture/"+id),
				probe("os", modelinventory.Text("linux"), "platform/os/"+id),
			},
			Policy: []modelinventory.Fact{
				attest("license", modelinventory.Text("apache-2.0"), "policy/license/"+id),
				attest("locality", modelinventory.Text("local"), "policy/locality/"+id),
				attest("privacy", modelinventory.Text("no-egress"), "policy/privacy/"+id),
			},
			Capabilities: []modelinventory.Fact{
				probe("context_tokens", modelinventory.Integer(context), "capability/context/"+id),
				probe("structured_json", modelinventory.Bool(structured), "capability/structured/"+id),
				probe("tool_calling", modelinventory.Bool(true), "capability/tools/"+id),
			},
		},
	}
}

func inventoryFact(name string, value modelinventory.Value, kind modelinventory.WitnessKind, source string, observedAt time.Time) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{inventoryWitness(kind, source, observedAt)}}
}

func inventoryWitness(kind modelinventory.WitnessKind, source string, observedAt time.Time) modelinventory.Witness {
	return modelinventory.Witness{
		Kind: kind, Source: source, ObservedAt: observedAt.Format(time.RFC3339),
		ExpiresAt: conformanceTime.Add(48 * time.Hour).Format(time.RFC3339),
	}
}

func cloneInventory(t *testing.T, inventory modelinventory.Inventory) modelinventory.Inventory {
	t.Helper()
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	cloned, diagnostics := modelinventory.ParseJSON(raw, conformanceTime)
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	return cloned
}

func candidateByID(t *testing.T, inventory *modelinventory.Inventory, id string) *modelinventory.Candidate {
	t.Helper()
	for index := range inventory.Candidates {
		if inventory.Candidates[index].ID == id {
			return &inventory.Candidates[index]
		}
	}
	t.Fatalf("candidate %q not found", id)
	return nil
}

func assertSelected(t *testing.T, resolution modelsetresolve.Resolution, roleID, candidateID string) {
	t.Helper()
	for _, role := range resolution.Roles {
		if role.RoleID == roleID && role.Status == modelsetresolve.StatusSelected && role.Selection != nil && role.Selection.CandidateID == candidateID {
			return
		}
	}
	t.Fatalf("role %s did not select %s: %+v", roleID, candidateID, resolution.Roles)
}

func requireIncompatible(t *testing.T, receipt modelsetreceipt.Receipt, err error) {
	t.Helper()
	var incompatible *modelsetreceipt.IncompatibleError
	if !errors.As(err, &incompatible) || receipt.Status != modelsetreceipt.StatusIncompatible || len(receipt.Failures) == 0 {
		t.Fatalf("startup result = (%+v, %T %v), want typed incompatible receipt", receipt, err, err)
	}
	raw, canonicalErr := receipt.CanonicalJSON()
	if canonicalErr != nil {
		t.Fatalf("incompatible receipt is not canonical: %v", canonicalErr)
	}
	if _, parseErr := modelsetreceipt.ParseJSON(raw); parseErr != nil {
		t.Fatalf("incompatible receipt read-back: %v", parseErr)
	}
}

func assertRejection(t *testing.T, rejections []modelsetresolve.Rejection, code modelsetresolve.RejectionCode, requireEvidence bool) modelsetresolve.Rejection {
	t.Helper()
	for _, rejection := range rejections {
		if rejection.RoleID != "executor" || rejection.Code != code {
			continue
		}
		if rejection.Constraint == "" || rejection.Remediation == "" || (requireEvidence && rejection.EvidenceSource == "") {
			t.Fatalf("rejection lacks actionable envelope: %+v", rejection)
		}
		return rejection
	}
	t.Fatalf("rejections omit executor code %s: %+v", code, rejections)
	return modelsetresolve.Rejection{}
}

func assertReceiptFailure(t *testing.T, failures []modelsetreceipt.Failure, code modelsetreceipt.Code, sourceCode, roleID string, requireEvidence bool) modelsetreceipt.Failure {
	t.Helper()
	for _, failure := range failures {
		if failure.Code != code || (sourceCode != "" && failure.SourceCode != sourceCode) || (roleID != "" && failure.RoleID != roleID) {
			continue
		}
		if failure.Field == "" || failure.Remediation == "" || (requireEvidence && failure.EvidenceSource == "") {
			t.Fatalf("receipt failure lacks actionable envelope: %+v", failure)
		}
		return failure
	}
	t.Fatalf("receipt failures omit code=%s source=%s role=%s: %+v", code, sourceCode, roleID, failures)
	return modelsetreceipt.Failure{}
}

func assertGolden(t *testing.T, name string, got []byte) {
	t.Helper()
	want := readFile(t, filepath.Join("testdata", name))
	if !bytes.Equal(got, want) {
		t.Fatalf("%s differs from captured golden:\nwant=%s\ngot=%s", name, want, got)
	}
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func digest(value string) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(value)))
}
