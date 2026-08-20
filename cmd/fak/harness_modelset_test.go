package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetlock"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
)

var harnessModelSetTestTime = time.Date(2030, 1, 2, 12, 0, 0, 0, time.UTC)

func TestHarnessModelSetCLIResolveInspectAndOfflineSelfcheck(t *testing.T) {
	dir := t.TempDir()
	intentPath := filepath.Join(dir, harnessModelSetIntentFile)
	inventoryPath := filepath.Join(dir, "model-inventory.json")
	lockPath := filepath.Join(dir, modelsetlock.DefaultFileName)
	receiptPath := filepath.Join(dir, "harness.model-set.receipt.json")
	writeHarnessModelSetIntentFixture(t, intentPath)
	writeHarnessModelSetInventoryFixture(t, inventoryPath, harnessModelSetInventoryFixture(t))

	resolveArgs := []string{
		"resolve", "--intent", intentPath, "--inventory", inventoryPath, "--out", lockPath,
		"--as-of", harnessModelSetTestTime.Format(time.RFC3339), "--os", "linux", "--arch", "amd64",
		"--accelerator", "cpu", "--runtime", "mixed-runtime", "--json",
	}
	code, stdout, stderr := captureHarnessModelSet(t, resolveArgs...)
	if code != 0 {
		t.Fatalf("resolve exit=%d stderr=%s stdout=%s", code, stderr, stdout)
	}
	var resolved harnessModelSetResolveResult
	if err := json.Unmarshal([]byte(stdout), &resolved); err != nil {
		t.Fatalf("decode resolve output: %v\n%s", err, stdout)
	}
	if resolved.Schema != "fak.harness-model-set-resolve/1" || resolved.Status != "resolved" || resolved.Roles != 2 {
		t.Fatalf("resolve output = %+v", resolved)
	}

	lock, err := modelsetlock.ReadFile(lockPath)
	if err != nil {
		t.Fatalf("independent lock read-back: %v", err)
	}
	if len(lock.Roles) != 2 || lock.Roles[0].Selected == nil || lock.Roles[1].Selected == nil {
		t.Fatalf("resolved lock roles = %+v", lock.Roles)
	}
	lockFirst := readHarnessModelSetFile(t, lockPath)
	expectationPath := defaultModelSetExpectationPath(lockPath)
	expectationFirst := readHarnessModelSetFile(t, expectationPath)
	if _, err := modelsetreceipt.ParseExpectationJSON(expectationFirst); err != nil {
		t.Fatalf("independent expectation read-back: %v", err)
	}

	code, _, stderr = captureHarnessModelSet(t, resolveArgs...)
	if code != 0 {
		t.Fatalf("deterministic rerun exit=%d stderr=%s", code, stderr)
	}
	if !bytes.Equal(lockFirst, readHarnessModelSetFile(t, lockPath)) || !bytes.Equal(expectationFirst, readHarnessModelSetFile(t, expectationPath)) {
		t.Fatal("resolve rerun changed canonical lock or expectation bytes")
	}

	beforeInspectLock := readHarnessModelSetFile(t, lockPath)
	beforeInspectExpectation := readHarnessModelSetFile(t, expectationPath)
	code, stdout, stderr = captureHarnessModelSet(t, "inspect", "--lock", lockPath, "--json")
	if code != 0 {
		t.Fatalf("inspect exit=%d stderr=%s", code, stderr)
	}
	var inspection harnessModelSetInspection
	if err := json.Unmarshal([]byte(stdout), &inspection); err != nil {
		t.Fatalf("decode inspect output: %v\n%s", err, stdout)
	}
	if inspection.Schema != "fak.harness-model-set-inspect/1" || inspection.ContentDigest != lock.ContentDigest || len(inspection.Roles) != 2 {
		t.Fatalf("inspection = %+v", inspection)
	}
	if !bytes.Equal(beforeInspectLock, readHarnessModelSetFile(t, lockPath)) || !bytes.Equal(beforeInspectExpectation, readHarnessModelSetFile(t, expectationPath)) {
		t.Fatal("inspect mutated model-set artifacts")
	}

	// A dead proxy makes accidental provider/network use fail while the command
	// still drives the complete local startup read-back path.
	t.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	t.Setenv("HTTPS_PROXY", "http://127.0.0.1:1")
	t.Setenv("NO_PROXY", "")
	code, stdout, stderr = captureHarnessModelSet(t,
		"selfcheck", "--lock", lockPath, "--inventory", inventoryPath, "--receipt", receiptPath,
		"--as-of", harnessModelSetTestTime.Format(time.RFC3339), "--json",
	)
	if code != 0 {
		t.Fatalf("offline selfcheck exit=%d stderr=%s stdout=%s", code, stderr, stdout)
	}
	var selfcheck harnessModelSetSelfcheckResult
	if err := json.Unmarshal([]byte(stdout), &selfcheck); err != nil {
		t.Fatalf("decode selfcheck output: %v\n%s", err, stdout)
	}
	if selfcheck.Status != modelsetreceipt.StatusCompatible || selfcheck.Roles != 2 || len(selfcheck.Failures) != 0 {
		t.Fatalf("selfcheck output = %+v", selfcheck)
	}
	receiptRaw := readHarnessModelSetFile(t, receiptPath)
	receipt, err := modelsetreceipt.ParseJSON(receiptRaw)
	if err != nil {
		t.Fatalf("independent receipt read-back: %v", err)
	}
	if receipt.Status != modelsetreceipt.StatusCompatible || len(receipt.Roles) != 2 {
		t.Fatalf("receipt = %+v", receipt)
	}

	code, stdout, stderr = captureHarnessModelSet(t, "inspect", "--lock", lockPath, "--receipt", receiptPath, "--json")
	if code != 0 {
		t.Fatalf("inspect receipt exit=%d stderr=%s", code, stderr)
	}
	if err := json.Unmarshal([]byte(stdout), &inspection); err != nil || inspection.Receipt == nil || inspection.Receipt.Status != modelsetreceipt.StatusCompatible {
		t.Fatalf("receipt inspection = %+v err=%v\n%s", inspection, err, stdout)
	}
}

func TestHarnessModelSetCLIFailsClosedAcrossEnvelopeAndPreservesLock(t *testing.T) {
	dir := t.TempDir()
	intentPath := filepath.Join(dir, harnessModelSetIntentFile)
	inventoryPath := filepath.Join(dir, "model-inventory.json")
	lockPath := filepath.Join(dir, modelsetlock.DefaultFileName)
	writeHarnessModelSetIntentFixture(t, intentPath)
	baseInventory := harnessModelSetInventoryFixture(t)
	writeHarnessModelSetInventoryFixture(t, inventoryPath, baseInventory)
	resolve := []string{
		"resolve", "--intent", intentPath, "--inventory", inventoryPath, "--out", lockPath,
		"--as-of", harnessModelSetTestTime.Format(time.RFC3339), "--os", "linux", "--arch", "amd64",
		"--accelerator", "cpu", "--runtime", "mixed-runtime",
	}
	if code, stdout, stderr := captureHarnessModelSet(t, resolve...); code != 0 {
		t.Fatalf("fixture resolve exit=%d stdout=%s stderr=%s", code, stdout, stderr)
	}
	priorLock := readHarnessModelSetFile(t, lockPath)

	tests := []struct {
		name       string
		asOf       time.Time
		mutate     func(*modelinventory.Inventory)
		wantCode   modelsetreceipt.Code
		wantSource string
	}{
		{
			name: "stale evidence", asOf: harnessModelSetTestTime.Add(72 * time.Hour),
			mutate:   func(*modelinventory.Inventory) {},
			wantCode: modelsetreceipt.CodeEvidenceStale, wantSource: string(modelinventory.CodeEvidenceStale),
		},
		{
			name: "runtime mismatch", asOf: harnessModelSetTestTime,
			mutate: func(inventory *modelinventory.Inventory) {
				harnessModelSetFact(t, inventory, "executor-model", "runtime").Value = modelinventory.Text("other-runtime")
			},
			wantCode: modelsetreceipt.CodeRuntimeMismatch,
		},
		{
			name: "unknown required capability", asOf: harnessModelSetTestTime,
			mutate: func(inventory *modelinventory.Inventory) {
				candidate := harnessModelSetCandidate(t, inventory, "planner-model")
				filtered := candidate.Evidence.Capabilities[:0]
				for _, fact := range candidate.Evidence.Capabilities {
					if fact.Name != "tool_calling" {
						filtered = append(filtered, fact)
					}
				}
				candidate.Evidence.Capabilities = filtered
			},
			wantCode: modelsetreceipt.CodeRequiredFactUnknown,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			current := cloneHarnessModelSetInventory(t, baseInventory)
			test.mutate(&current)
			currentPath := filepath.Join(dir, strings.ReplaceAll(test.name, " ", "-")+".json")
			writeHarnessModelSetInventoryFixture(t, currentPath, current)
			receiptPath := currentPath + ".receipt.json"
			code, stdout, stderr := captureHarnessModelSet(t,
				"selfcheck", "--lock", lockPath, "--intent", intentPath, "--inventory", currentPath,
				"--receipt", receiptPath, "--as-of", test.asOf.Format(time.RFC3339), "--json",
			)
			if code != 3 {
				t.Fatalf("selfcheck exit=%d, want 3\nstdout=%s\nstderr=%s", code, stdout, stderr)
			}
			receipt, err := modelsetreceipt.ParseJSON(readHarnessModelSetFile(t, receiptPath))
			if err != nil {
				t.Fatalf("parse incompatible receipt: %v", err)
			}
			if receipt.Status != modelsetreceipt.StatusIncompatible || !harnessModelSetHasFailure(receipt.Failures, test.wantCode, test.wantSource) {
				t.Fatalf("failures = %+v, want code=%s source=%s", receipt.Failures, test.wantCode, test.wantSource)
			}
		})
	}

	broken := cloneHarnessModelSetInventory(t, baseInventory)
	harnessModelSetFact(t, &broken, "executor-model", "runtime").Value = modelinventory.Text("other-runtime")
	brokenPath := filepath.Join(dir, "resolution-failure.json")
	writeHarnessModelSetInventoryFixture(t, brokenPath, broken)
	failedResolve := append([]string(nil), resolve...)
	for index := range failedResolve {
		if failedResolve[index] == inventoryPath {
			failedResolve[index] = brokenPath
		}
	}
	code, stdout, stderr := captureHarnessModelSet(t, append(failedResolve, "--json")...)
	if code != 3 {
		t.Fatalf("failed resolve exit=%d, want 3\nstdout=%s\nstderr=%s", code, stdout, stderr)
	}
	if !bytes.Equal(priorLock, readHarnessModelSetFile(t, lockPath)) {
		t.Fatal("failed resolution changed the prior canonical lock")
	}
}

func TestHarnessModelSetCLIUsageIsTyped(t *testing.T) {
	if code, _, stderr := captureHarnessModelSet(t, "selfcheck"); code != 2 || !strings.Contains(stderr, "--lock, --inventory, and --receipt are required") {
		t.Fatalf("selfcheck usage exit=%d stderr=%s", code, stderr)
	}
	if code, _, stderr := captureHarnessModelSet(t, "unknown"); code != 2 || !strings.Contains(stderr, "unknown command") {
		t.Fatalf("unknown usage exit=%d stderr=%s", code, stderr)
	}
}

func captureHarnessModelSet(t *testing.T, argv ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runHarnessModelSet(&stdout, &stderr, argv)
	return code, stdout.String(), stderr.String()
}

func writeHarnessModelSetIntentFixture(t *testing.T, path string) {
	t.Helper()
	intent := harnessmodelset.Intent{Schema: harnessmodelset.SchemaV1, Roles: []harnessmodelset.Role{
		{
			ID: "executor", Required: true, Evidence: harnessmodelset.EvidencePolicy{MaxAgeHours: 24},
			Alternatives: []harnessmodelset.Alternative{{
				ID: "local-executor", Capabilities: harnessmodelset.ModelRequirements{ToolCalling: harnessModelSetBool(true)},
				Operational: harnessmodelset.OperationalConstraints{
					Runtime: "llama.cpp", MaxMemoryBytes: harnessModelSetInt64(8 << 30), Locality: harnessmodelset.LocalityLocalOnly,
				},
			}},
		},
		{
			ID: "planner", Required: true, Evidence: harnessmodelset.EvidencePolicy{MaxAgeHours: 24},
			Alternatives: []harnessmodelset.Alternative{{
				ID: "tool-planner", Capabilities: harnessmodelset.ModelRequirements{
					ToolCalling: harnessModelSetBool(true), StructuredOutput: harnessModelSetBool(true), MinimumInputTokens: harnessModelSetInt64(65536),
				},
				Operational: harnessmodelset.OperationalConstraints{Runtime: "llama.cpp", ServingProtocol: harnessmodelset.ServingProtocolOpenAI},
			}},
		},
	}}
	raw, err := harnessmodelset.CanonicalJSON(intent)
	if err != nil {
		t.Fatalf("canonical intent: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func harnessModelSetInventoryFixture(t *testing.T) modelinventory.Inventory {
	t.Helper()
	observations := modelinventory.Observations{Locals: []modelinventory.LocalObservation{
		harnessModelSetLocalCandidate("executor-model", 4<<30, 32768, false),
		harnessModelSetLocalCandidate("planner-model", 16<<30, 131072, true),
	}}
	inventory, diagnostics := modelinventory.Normalize(observations, harnessModelSetTestTime)
	if len(diagnostics) != 0 {
		t.Fatalf("normalize inventory: %s", diagnostics)
	}
	return inventory
}

func harnessModelSetLocalCandidate(id string, memory, context int64, structured bool) modelinventory.LocalObservation {
	return modelinventory.LocalObservation{
		ID: id, Artifact: "models/" + id + ".gguf", Digest: harnessModelSetDigest(id), Format: "gguf",
		IdentityEvidence: []modelinventory.Witness{harnessModelSetWitness("identity/" + id)},
		Evidence: modelinventory.EvidenceSet{
			Availability: harnessModelSetInventoryFact("available", modelinventory.Bool(true), "availability/"+id),
			Serving: []modelinventory.Fact{
				harnessModelSetInventoryFact("protocol", modelinventory.Text("openai-compatible"), "serving/protocol/"+id),
				harnessModelSetInventoryFact("runtime", modelinventory.Text("llama.cpp"), "serving/runtime/"+id),
			},
			Platform: []modelinventory.Fact{
				harnessModelSetInventoryFact("accelerator", modelinventory.Text("cuda"), "platform/accelerator/"+id),
				harnessModelSetInventoryFact("accelerator_memory_bytes", modelinventory.Integer(memory), "platform/memory/"+id),
				harnessModelSetInventoryFact("architecture", modelinventory.Text("amd64"), "platform/arch/"+id),
				harnessModelSetInventoryFact("os", modelinventory.Text("linux"), "platform/os/"+id),
			},
			Policy: []modelinventory.Fact{
				harnessModelSetInventoryFact("license", modelinventory.Text("apache-2.0"), "policy/license/"+id),
				harnessModelSetInventoryFact("locality", modelinventory.Text("local"), "policy/locality/"+id),
				harnessModelSetInventoryFact("privacy", modelinventory.Text("no-egress"), "policy/privacy/"+id),
			},
			Capabilities: []modelinventory.Fact{
				harnessModelSetInventoryFact("context_tokens", modelinventory.Integer(context), "capability/context/"+id),
				harnessModelSetInventoryFact("structured_json", modelinventory.Bool(structured), "capability/json/"+id),
				harnessModelSetInventoryFact("tool_calling", modelinventory.Bool(true), "capability/tools/"+id),
			},
		},
	}
}

func harnessModelSetInventoryFact(name string, value modelinventory.Value, source string) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{harnessModelSetWitness(source)}}
}

func harnessModelSetWitness(source string) modelinventory.Witness {
	return modelinventory.Witness{
		Kind: modelinventory.EvidenceProbe, Source: source,
		ObservedAt: harnessModelSetTestTime.Add(-time.Hour).Format(time.RFC3339),
		ExpiresAt:  harnessModelSetTestTime.Add(48 * time.Hour).Format(time.RFC3339),
	}
}

func writeHarnessModelSetInventoryFixture(t *testing.T, path string, inventory modelinventory.Inventory) {
	t.Helper()
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatalf("canonical inventory: %s", diagnostics)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func cloneHarnessModelSetInventory(t *testing.T, inventory modelinventory.Inventory) modelinventory.Inventory {
	t.Helper()
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	cloned, diagnostics := modelinventory.ParseJSON(raw, harnessModelSetTestTime)
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	return cloned
}

func harnessModelSetCandidate(t *testing.T, inventory *modelinventory.Inventory, id string) *modelinventory.Candidate {
	t.Helper()
	for index := range inventory.Candidates {
		if inventory.Candidates[index].ID == id {
			return &inventory.Candidates[index]
		}
	}
	t.Fatalf("candidate %q not found", id)
	return nil
}

func harnessModelSetFact(t *testing.T, inventory *modelinventory.Inventory, candidateID, name string) *modelinventory.Fact {
	t.Helper()
	candidate := harnessModelSetCandidate(t, inventory, candidateID)
	groups := [][]modelinventory.Fact{candidate.Evidence.Serving, candidate.Evidence.Platform, candidate.Evidence.Policy, candidate.Evidence.Capabilities}
	for _, group := range groups {
		for index := range group {
			if group[index].Name == name {
				return &group[index]
			}
		}
	}
	t.Fatalf("fact %s/%s not found", candidateID, name)
	return nil
}

func harnessModelSetHasFailure(failures []modelsetreceipt.Failure, code modelsetreceipt.Code, source string) bool {
	for _, failure := range failures {
		if failure.Code == code && (source == "" || failure.SourceCode == source) {
			return true
		}
	}
	return false
}

func readHarnessModelSetFile(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func harnessModelSetDigest(value string) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(value)))
}

func harnessModelSetBool(value bool) *bool    { return &value }
func harnessModelSetInt64(value int64) *int64 { return &value }
