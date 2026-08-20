package modelsetlock_test

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetlock"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

var evaluatedAt = time.Date(2026, 8, 19, 20, 0, 0, 0, time.UTC)

func TestCanonicalLockGoldenAndAtomicRoundTrip(t *testing.T) {
	inputs, resolution := fixture(t)
	lock, err := modelsetlock.New(inputs, resolution)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	canonical, err := modelsetlock.CanonicalJSON(lock)
	if err != nil {
		t.Fatalf("CanonicalJSON: %v", err)
	}
	want, err := os.ReadFile("testdata/two-role.lock.json")
	if err != nil {
		t.Fatalf("read golden: %v\ngot:\n%s", err, canonical)
	}
	if !bytes.Equal(canonical, want) {
		t.Fatalf("canonical lock differs from golden:\nwant:\n%s\ngot:\n%s", want, canonical)
	}

	path := filepath.Join(t.TempDir(), modelsetlock.DefaultFileName)
	if err := modelsetlock.WriteFile(path, lock); err != nil {
		t.Fatalf("first WriteFile: %v", err)
	}
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelsetlock.WriteFile(path, lock); err != nil {
		t.Fatalf("second WriteFile: %v", err)
	}
	second, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) || !bytes.Equal(first, want) {
		t.Fatalf("atomic rerun changed bytes:\nfirst=%s\nsecond=%s", first, second)
	}
	readBack, err := modelsetlock.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !reflect.DeepEqual(readBack, lock) {
		t.Fatalf("read-back changed lock:\nwant=%+v\ngot=%+v", lock, readBack)
	}
	if err := modelsetlock.Compare(readBack, inputs); err != nil {
		t.Fatalf("fresh lock compared stale: %v", err)
	}
	if len(readBack.Roles) != 2 || readBack.Roles[0].Selected == nil || readBack.Roles[1].Selected == nil {
		t.Fatalf("two selected role identities were not persisted: %+v", readBack.Roles)
	}
	if len(readBack.Roles[0].Rejections) == 0 || len(readBack.Roles[1].Rejections) == 0 {
		t.Fatalf("ordered alternative rejection evidence was not persisted: %+v", readBack.Roles)
	}

	changed := inputs
	changed.RuleBytes = []byte("{\"mode\":\"declared-order\"}\n")
	changedLock, err := modelsetlock.New(changed, resolution)
	if err != nil {
		t.Fatalf("New(changed policy): %v", err)
	}
	if err := modelsetlock.WriteFile(path, changedLock); err != nil {
		t.Fatalf("replacement WriteFile: %v", err)
	}
	replaced, err := modelsetlock.ReadFile(path)
	if err != nil {
		t.Fatalf("replacement ReadFile: %v", err)
	}
	if !reflect.DeepEqual(replaced, changedLock) || bytes.Equal(first, mustCanonical(t, changedLock)) {
		t.Fatalf("atomic replacement did not publish exactly the new lock: %+v", replaced)
	}
	if temps, err := filepath.Glob(filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")); err != nil || len(temps) != 0 {
		t.Fatalf("temporary lock residue = %v, err=%v", temps, err)
	}
}

func TestReadFailsClosedOnTamperUnknownSchemaAndNonCanonicalBytes(t *testing.T) {
	inputs, resolution := fixture(t)
	lock, err := modelsetlock.New(inputs, resolution)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := modelsetlock.CanonicalJSON(lock)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		raw  []byte
		code modelsetlock.Code
	}{
		{
			name: "payload tamper",
			raw:  bytes.Replace(raw, []byte(`"candidate_id": "executor-local"`), []byte(`"candidate_id": "executor-other"`), 1),
			code: modelsetlock.CodeDigestMismatch,
		},
		{
			name: "digest tamper",
			raw:  bytes.Replace(raw, []byte(lock.ContentDigest), []byte("sha256:"+strings.Repeat("0", 64)), 1),
			code: modelsetlock.CodeDigestMismatch,
		},
		{
			name: "unknown schema",
			raw:  bytes.Replace(raw, []byte(modelsetlock.Schema), []byte("fak.model-set-lock/2"), 1),
			code: modelsetlock.CodeSchemaUnknown,
		},
		{
			name: "non canonical",
			raw:  bytes.TrimSuffix(raw, []byte("\n")),
			code: modelsetlock.CodeNonCanonical,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parsed, err := modelsetlock.ParseJSON(tc.raw)
			if err == nil || !modelsetlock.HasCode(err, tc.code) || !reflect.DeepEqual(parsed, modelsetlock.Lock{}) {
				t.Fatalf("ParseJSON = (%+v, %v), want empty lock and %s", parsed, err, tc.code)
			}
		})
	}

	path := filepath.Join(t.TempDir(), modelsetlock.DefaultFileName)
	if err := modelsetlock.WriteFile(path, lock); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	invalid := lock
	invalid.ContentDigest = "sha256:" + strings.Repeat("0", 64)
	if err := modelsetlock.WriteFile(path, invalid); !modelsetlock.HasCode(err, modelsetlock.CodeDigestMismatch) {
		t.Fatalf("invalid WriteFile error = %v, want digest mismatch", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("failed write changed prior lock:\nbefore=%s\nafter=%s", before, after)
	}
}

func TestCompareClassifiesEveryStaleInput(t *testing.T) {
	inputs, resolution := fixture(t)
	lock, err := modelsetlock.New(inputs, resolution)
	if err != nil {
		t.Fatal(err)
	}
	changed, _ := fixture(t)
	changed.Intent.Roles[0].Alternatives[0].ID = "changed-runtime"
	changed.Inventory.AsOf = evaluatedAt.Add(time.Minute).Format(time.RFC3339)
	changed.RuleBytes = []byte("{\"mode\":\"declared-order\"}\n")
	changed.Target.Accelerator = "cuda"
	changed.ResolverVersion = "fak.model-set-resolution/2"

	err = modelsetlock.Compare(lock, changed)
	want := []modelsetlock.StaleReason{
		modelsetlock.StaleIntent,
		modelsetlock.StaleInventory,
		modelsetlock.StaleSelectionRules,
		modelsetlock.StalePlatform,
		modelsetlock.StaleResolverVersion,
	}
	if got := modelsetlock.StaleReasons(err); !reflect.DeepEqual(got, want) {
		t.Fatalf("stale reasons = %v, want %v (err=%v)", got, want, err)
	}
}

func TestNewRejectsForgedResolution(t *testing.T) {
	inputs, resolution := fixture(t)
	resolution.Roles[0].Selection.CandidateID = "forged-candidate"
	lock, err := modelsetlock.New(inputs, resolution)
	if err == nil || !modelsetlock.HasCode(err, modelsetlock.CodeResolutionMismatch) || !reflect.DeepEqual(lock, modelsetlock.Lock{}) {
		t.Fatalf("New(forged) = (%+v, %v), want empty lock and resolution mismatch", lock, err)
	}
}

func fixture(t *testing.T) (modelsetlock.Inputs, modelsetresolve.Resolution) {
	t.Helper()
	intent := harnessmodelset.Intent{
		Schema: harnessmodelset.SchemaV1,
		Roles: []harnessmodelset.Role{
			role("planner", "planner-vllm", "vllm"),
			role("executor", "executor-llamacpp", "llamacpp"),
		},
	}
	inventory, diagnostics := modelinventory.Normalize(modelinventory.Observations{
		Providers: []modelinventory.ProviderObservation{{
			ID: "planner-provider", Provider: "example", Repository: "models/planner",
			Revision: strings.Repeat("c", 40), Digest: "sha256:" + strings.Repeat("a", 64), Format: "safetensors",
			IdentityEvidence: []modelinventory.Witness{witness(modelinventory.EvidenceArtifactMetadata, "metadata://planner")},
			Evidence:         evidence("planner", "vllm"),
		}},
		Locals: []modelinventory.LocalObservation{{
			ID: "executor-local", Artifact: "models/executor.gguf",
			Digest: "sha256:" + strings.Repeat("b", 64), Format: "gguf",
			IdentityEvidence: []modelinventory.Witness{witness(modelinventory.EvidenceProbe, "digest://executor")},
			Evidence:         evidence("executor", "llamacpp"),
		}},
	}, evaluatedAt)
	if len(diagnostics) != 0 {
		t.Fatalf("Normalize: %s", diagnostics)
	}
	resolution, err := modelsetresolve.Resolve(intent, inventory, evaluatedAt)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	return modelsetlock.Inputs{
		Intent:    intent,
		Inventory: inventory,
		RuleBytes: []byte("{\"mode\":\"local-first\"}\n"),
		Target: modelsetlock.Target{
			OS: "linux", Architecture: "amd64", Accelerator: "cpu", Runtime: "mixed-runtime",
		},
		ResolverVersion: modelsetresolve.Schema,
	}, resolution
}

func role(id, alternativeID, runtime string) harnessmodelset.Role {
	return harnessmodelset.Role{
		ID: id, Required: true,
		Alternatives: []harnessmodelset.Alternative{{
			ID:          alternativeID,
			Operational: harnessmodelset.OperationalConstraints{Runtime: runtime},
		}},
		Preference: &harnessmodelset.SelectionPreference{Mode: harnessmodelset.PreferenceDeclaredOrder},
		Evidence: harnessmodelset.EvidencePolicy{
			MaxAgeHours:   4,
			RequiredKinds: []harnessmodelset.EvidenceKind{harnessmodelset.EvidenceRuntimeProbe},
		},
	}
}

func evidence(prefix, runtime string) modelinventory.EvidenceSet {
	return modelinventory.EvidenceSet{
		Availability: fact("available", modelinventory.Bool(true), prefix+"/availability"),
		Serving: []modelinventory.Fact{
			fact("runtime", modelinventory.Text(runtime), prefix+"/runtime"),
			fact("protocol", modelinventory.Text("openai-compatible"), prefix+"/protocol"),
		},
		Platform: []modelinventory.Fact{
			fact("os", modelinventory.Text("linux"), prefix+"/os"),
			fact("architecture", modelinventory.Text("amd64"), prefix+"/architecture"),
			fact("accelerator", modelinventory.Text("cpu"), prefix+"/accelerator"),
		},
		Policy: []modelinventory.Fact{
			fact("locality", modelinventory.Text("local"), prefix+"/locality"),
			fact("license", modelinventory.Text("apache-2.0"), prefix+"/license"),
		},
		Capabilities: []modelinventory.Fact{
			fact("tool_calling", modelinventory.Bool(true), prefix+"/tool-calling"),
		},
	}
}

func fact(name string, value modelinventory.Value, source string) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{witness(modelinventory.EvidenceProbe, source)}}
}

func witness(kind modelinventory.WitnessKind, source string) modelinventory.Witness {
	return modelinventory.Witness{
		Kind: kind, Source: source,
		ObservedAt: evaluatedAt.Add(-time.Hour).Format(time.RFC3339),
		ExpiresAt:  evaluatedAt.Add(24 * time.Hour).Format(time.RFC3339),
	}
}

func mustCanonical(t *testing.T, lock modelsetlock.Lock) []byte {
	t.Helper()
	raw, err := modelsetlock.CanonicalJSON(lock)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
