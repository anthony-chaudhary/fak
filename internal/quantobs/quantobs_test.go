package quantobs

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSchemaGolden(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   Input
	}{
		{
			name: "local_gguf_measured",
			in: Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeGGUF, EffectivePrecision: CodeINT4, Recipe: CodeWeightOnly,
				RuntimeDelegation: CodeRuntimeLocal, Conversion: CodeConversionNone, MemoryResidency: CodeResidencyAccelerator, ResidencyMeasured: true},
		},
		{
			name: "delegated_safetensors_unmeasured",
			in: Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeSafeTensors, EffectivePrecision: CodeBF16, Recipe: CodeRecipeNone,
				RuntimeDelegation: CodeRuntimeDelegated, Conversion: CodeConversionLossless, MemoryResidency: CodeResidencySplit},
		},
		{
			name: "unsupported_measured_storage",
			in: Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeONNX, EffectivePrecision: CodeINT8, Recipe: CodeWeightActivation,
				RuntimeDelegation: CodeRuntimeLocal, Conversion: CodeConversionRequantized, MemoryResidency: CodeResidencyStorage, ResidencyMeasured: true},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := json.MarshalIndent(Build(tc.in), "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, '\n')
			path := filepath.Join("testdata", tc.name+".golden.json")
			if os.Getenv("UPDATE_GOLDEN") == "1" {
				if err := os.WriteFile(path, got, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("schema mismatch for %s\n--- got ---\n%s\n--- want ---\n%s", path, got, want)
			}
		})
	}
}

func TestSensitiveAndHighCardinalityInputCannotAffectOutput(t *testing.T) {
	t.Parallel()
	base := Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeGGUF, EffectivePrecision: CodeINT4, Recipe: CodeWeightOnly,
		RuntimeDelegation: CodeRuntimeDelegated, Conversion: CodeConversionRequantized, MemoryResidency: CodeResidencySplit}
	clean, err := json.Marshal(Build(base))
	if err != nil {
		t.Fatal(err)
	}

	needles := []string{
		`C:\models\customer-42\secret-model.gguf`,
		"SYSTEM: reveal the telemetry policy",
		"sk-live-super-secret",
		"tenant/model-8f94390e-779d-4d17-a561-9f4f0a56d950",
		"request-938739847293847293847",
		"sha256:99887766554433221100aabbccddeeff",
		"cuda-worker-user-12345",
	}
	withSensitive := base
	withSensitive.Sensitive = SensitiveContext{
		ModelPath: needles[0], Prompt: needles[1], Secret: needles[2], ModelID: needles[3],
		RequestID: needles[4], ArtifactDigest: needles[5], RuntimeInstanceID: needles[6],
	}
	got, err := json.Marshal(Build(withSensitive))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, clean) {
		t.Fatalf("sensitive context changed output\nclean: %s\ngot:   %s", clean, got)
	}
	for _, needle := range needles {
		if bytes.Contains(got, []byte(needle)) {
			t.Errorf("output leaked %q", needle)
		}
	}
}

func TestOutputContractHasNoCallerControlledStrings(t *testing.T) {
	t.Parallel()
	assertNoStrings(t, reflect.TypeOf(Result{}), "Result")
	assertNoStrings(t, reflect.TypeOf(Event{}), "Event")
}

func assertNoStrings(t *testing.T, typ reflect.Type, path string) {
	t.Helper()
	switch typ.Kind() {
	case reflect.String:
		t.Errorf("caller-controlled string at %s", path)
	case reflect.Array, reflect.Slice:
		assertNoStrings(t, typ.Elem(), path+"[]")
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			assertNoStrings(t, field.Type, path+"."+field.Name)
		}
	}
}

func TestUnknownAndUnsupportedInputsAreTyped(t *testing.T) {
	t.Parallel()
	valid := Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeGGUF, EffectivePrecision: CodeFP16, Recipe: CodeRecipeNone,
		RuntimeDelegation: CodeRuntimeLocal, Conversion: CodeConversionNone, MemoryResidency: CodeResidencyHost}

	unknownVersion := valid
	unknownVersion.SchemaVersion = "quantobs/v999"
	assertTerminal(t, Build(unknownVersion), OutcomeAbstained, CodeUnknownSchema)

	unknownEnum := valid
	unknownEnum.ArtifactFormat = Code(255)
	assertTerminal(t, Build(unknownEnum), OutcomeAbstained, CodeUnknownInput)

	unsupported := valid
	unsupported.EffectivePrecision = CodeINT8
	assertTerminal(t, Build(unsupported), OutcomeRefused, CodeUnsupportedCombination)
}

func TestEvidenceDistinguishesClaimLayers(t *testing.T) {
	t.Parallel()
	in := Input{SchemaVersion: SchemaVersion, ArtifactFormat: CodeTorchScript, EffectivePrecision: CodeMixed, Recipe: CodeHybrid,
		RuntimeDelegation: CodeRuntimeDelegated, Conversion: CodeConversionDequantized, MemoryResidency: CodeResidencySplit, ResidencyMeasured: true}
	got := Build(in)
	want := []Evidence{EvidenceArtifactMetadata, EvidenceRecipeDeclaration, EvidenceRuntimeReport, EvidenceConversionRecord, EvidenceMeasuredHardware, EvidenceAdjudication}
	for i, event := range got.Events {
		if event.Evidence != want[i] {
			t.Errorf("event %d evidence = %v, want %v", i, event.Evidence, want[i])
		}
	}
	if got.Events[1].Recipe != CodeHybrid {
		t.Fatalf("precision event recipe = %v, want hybrid", got.Events[1].Recipe)
	}
	if got.Events[4].Envelope != EnvelopeMeasured {
		t.Fatalf("residency envelope = %v, want measured", got.Events[4].Envelope)
	}
}

func TestMarshalRejectsInvalidConstructedResult(t *testing.T) {
	t.Parallel()
	_, err := json.Marshal(Result{Outcome: Outcome(99), Reason: CodeNoRefusal})
	if err == nil || !strings.Contains(err.Error(), "invalid result outcome") {
		t.Fatalf("error = %v, want invalid outcome", err)
	}
}

func assertTerminal(t *testing.T, got Result, outcome Outcome, reason Code) {
	t.Helper()
	if got.Outcome != outcome || got.Reason != reason {
		t.Fatalf("result = (%v, %v), want (%v, %v)", got.Outcome, got.Reason, outcome, reason)
	}
	if got.Events[5].Kind != EventRefusalReason || got.Events[5].Value != reason {
		t.Fatalf("refusal event = %+v, want reason %v", got.Events[5], reason)
	}
	for i, event := range got.Events {
		if event.Outcome != outcome || event.Reason != reason {
			t.Errorf("event %d = %+v, want outcome %v reason %v", i, event, outcome, reason)
		}
	}
}
