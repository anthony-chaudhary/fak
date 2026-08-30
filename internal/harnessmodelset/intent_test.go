package harnessmodelset_test

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
)

func TestTwoRoleIntentRoundTripsCanonicalJSON(t *testing.T) {
	raw := readFixture(t, "testdata/two-role.json")
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		t.Fatalf("ParseJSON(two-role): %v", err)
	}
	if len(intent.Roles) != 2 || intent.Roles[0].ID != "executor" || intent.Roles[1].ID != "planner" {
		t.Fatalf("roles = %+v, want canonical executor/planner order", intent.Roles)
	}

	first, err := harnessmodelset.CanonicalJSON(intent)
	if err != nil {
		t.Fatalf("CanonicalJSON(first): %v", err)
	}
	second, err := harnessmodelset.CanonicalJSON(intent)
	if err != nil {
		t.Fatalf("CanonicalJSON(second): %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("canonical JSON changed between runs:\nfirst=%s\nsecond=%s", first, second)
	}
	if !bytes.Equal(first, raw) {
		t.Fatalf("fixture is not canonical:\nwant=%s\ngot=%s", raw, first)
	}

	roundTrip, err := harnessmodelset.ParseJSON(first)
	if err != nil {
		t.Fatalf("ParseJSON(canonical): %v", err)
	}
	if !reflect.DeepEqual(roundTrip, intent) {
		t.Fatalf("semantic round trip changed:\nfirst=%+v\nsecond=%+v", intent, roundTrip)
	}
}

func TestParserAcceptsGenericFamilyAndQuantizationConstraints(t *testing.T) {
	raw := []byte(`{"schema":"fak-harness-model-set-intent/1","roles":[{"id":"executor","required":true,"alternatives":[{"id":"native","capabilities":{"family":"Qwen3.8","quantization":"Q4_K_M"}}],"evidence":{"max_age_hours":24}}]}`)
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	got := intent.Roles[0].Alternatives[0].Capabilities
	if got.Family != "Qwen3.8" || got.Quantization != "Q4_K_M" {
		t.Fatalf("capabilities = %+v", got)
	}
}

func TestParserRejectsWhitespaceFamilyAndQuantizationTokens(t *testing.T) {
	for _, field := range []string{"family", "quantization"} {
		raw := []byte(fmt.Sprintf(`{"schema":"fak-harness-model-set-intent/1","roles":[{"id":"executor","required":true,"alternatives":[{"id":"native","capabilities":{"%s":" q4 "}}],"evidence":{"max_age_hours":24}}]}`, field))
		_, err := harnessmodelset.ParseJSON(raw)
		diagnostics := harnessmodelset.Diagnostics(err)
		if len(diagnostics) != 1 || diagnostics[0].Code != harnessmodelset.CodeValueInvalid || diagnostics[0].Path != "$.roles[0].alternatives[0].capabilities."+field {
			t.Fatalf("%s diagnostics = %+v", field, diagnostics)
		}
	}
}

func TestParserRejectsRequiredFailureClassesWithStableDiagnostics(t *testing.T) {
	valid := string(readFixture(t, "testdata/two-role.json"))
	cases := []struct {
		name     string
		raw      string
		wantCode harnessmodelset.DiagnosticCode
		wantPath string
	}{
		{
			name:     "unknown capability",
			raw:      strings.Replace(valid, `"tool_calling": true,`, "\"tool_calling\": true,\n          \"semantic_quality\": \"excellent\",", 1),
			wantCode: harnessmodelset.CodeFieldUnknown,
			wantPath: "$.roles[0].alternatives[0].capabilities.semantic_quality",
		},
		{
			name:     "duplicate role",
			raw:      strings.Replace(valid, `"id": "planner",`, `"id": "executor",`, 1),
			wantCode: harnessmodelset.CodeRoleDuplicate,
			wantPath: "$.roles[1].id",
		},
		{
			name:     "empty alternatives",
			raw:      replaceRoleAlternatives(valid, "executor"),
			wantCode: harnessmodelset.CodeAlternativesEmpty,
			wantPath: "$.roles[0].alternatives",
		},
		{
			name:     "invalid freshness",
			raw:      strings.Replace(valid, `"max_age_hours": 24`, `"max_age_hours": 0`, 1),
			wantCode: harnessmodelset.CodeFreshnessInvalid,
			wantPath: "$.roles[0].evidence.max_age_hours",
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := harnessmodelset.ParseJSON([]byte(test.raw))
			validationErr := requireValidationError(t, err)
			if !hasDiagnostic(validationErr.Diagnostics, test.wantCode, test.wantPath) {
				t.Fatalf("diagnostics = %+v, want %s at %s", validationErr.Diagnostics, test.wantCode, test.wantPath)
			}
			first := validationErr.CanonicalJSON()
			_, repeatedErr := harnessmodelset.ParseJSON([]byte(test.raw))
			second := requireValidationError(t, repeatedErr).CanonicalJSON()
			if !bytes.Equal(first, second) || !bytes.HasSuffix(first, []byte("\n")) {
				t.Fatalf("independent parses produced unstable diagnostic JSON:\nfirst=%s\nsecond=%s", first, second)
			}
		})
	}
}

func TestParserFailsClosedOnMalformedAmbiguousAndUnknownValues(t *testing.T) {
	valid := string(readFixture(t, "testdata/two-role.json"))
	cases := []struct {
		name     string
		raw      string
		wantCode harnessmodelset.DiagnosticCode
		wantPath string
	}{
		{"malformed", `{"schema":`, harnessmodelset.CodeJSONInvalid, "$"},
		{"trailing document", valid + `{}`, harnessmodelset.CodeJSONTrailing, "$"},
		{"duplicate object field", strings.Replace(valid, `"required": true,`, "\"required\": true,\n      \"required\": false,", 1), harnessmodelset.CodeFieldDuplicate, "$.roles[0].required"},
		{"ambiguous negative requirement", strings.Replace(valid, `"tool_calling": true`, `"tool_calling": false`, 1), harnessmodelset.CodeConstraintAmbiguous, "$.roles[0].alternatives[0].capabilities.tool_calling"},
		{"unknown locality", strings.Replace(valid, `"locality": "local-only"`, `"locality": "nearby"`, 1), harnessmodelset.CodeValueInvalid, "$.roles[0].alternatives[0].operational.locality"},
		{"unknown intent version", strings.Replace(valid, harnessmodelset.SchemaV1, "fak-harness-model-set-intent/2", 1), harnessmodelset.CodeIntentVersionUnknown, "$.schema"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			intent, err := harnessmodelset.ParseJSON([]byte(test.raw))
			if !reflect.DeepEqual(intent, harnessmodelset.Intent{}) {
				t.Fatalf("failed parse returned partial intent: %+v", intent)
			}
			validationErr := requireValidationError(t, err)
			if !hasDiagnostic(validationErr.Diagnostics, test.wantCode, test.wantPath) {
				t.Fatalf("diagnostics = %+v, want %s at %s", validationErr.Diagnostics, test.wantCode, test.wantPath)
			}
		})
	}
}

func requireValidationError(t *testing.T, err error) *harnessmodelset.ValidationError {
	t.Helper()
	if err == nil {
		t.Fatal("ParseJSON accepted invalid intent")
	}
	var validationErr *harnessmodelset.ValidationError
	if !errors.As(err, &validationErr) {
		t.Fatalf("error type = %T, want *harnessmodelset.ValidationError: %v", err, err)
	}
	return validationErr
}

func hasDiagnostic(diagnostics []harnessmodelset.Diagnostic, code harnessmodelset.DiagnosticCode, path string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == code && diagnostic.Path == path {
			return true
		}
	}
	return false
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func replaceRoleAlternatives(raw, roleID string) string {
	needle := `"id": "` + roleID + `",`
	roleStart := strings.Index(raw, needle)
	if roleStart < 0 {
		return raw
	}
	alternativesStart := strings.Index(raw[roleStart:], `"alternatives": [`)
	if alternativesStart < 0 {
		return raw
	}
	alternativesStart += roleStart
	arrayStart := alternativesStart + len(`"alternatives": `)
	depth := 0
	inString := false
	escaped := false
	for index := arrayStart; index < len(raw); index++ {
		char := raw[index]
		if inString {
			if escaped {
				escaped = false
			} else if char == '\\' {
				escaped = true
			} else if char == '"' {
				inString = false
			}
			continue
		}
		if char == '"' {
			inString = true
			continue
		}
		switch char {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return raw[:arrayStart] + "[]" + raw[index+1:]
			}
		}
	}
	return raw
}
