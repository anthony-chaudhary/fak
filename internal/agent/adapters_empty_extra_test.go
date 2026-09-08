package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

type extraBodyTestStruct struct {
	Zebra string `json:"zebra"`
	Alpha string `json:"alpha"`
	Beta  int    `json:"beta"`
}

func TestMarshalWithExtraBodyEmptyFastPath(t *testing.T) {
	base := extraBodyTestStruct{
		Zebra: "last",
		Alpha: "first",
		Beta:  42,
	}
	raw, err := json.Marshal(base)
	if err != nil {
		t.Fatalf("json.Marshal(base): %v", err)
	}

	// Verify that struct field order is preserved by raw JSON and is not alphabetical.
	// Map serialization sorts keys alphabetically ("alpha" before "zebra"), so if
	// marshalWithExtraBody re-marshals through a map, the byte representation changes.
	if !strings.HasPrefix(string(raw), `{"zebra":`) {
		t.Fatalf("expected raw JSON to begin with zebra, got: %s", string(raw))
	}

	emptyCases := []struct {
		name  string
		extra json.RawMessage
	}{
		{"nil", nil},
		{"empty_bytes", []byte("")},
		{"empty_object", []byte("{}")},
		{"whitespace_object", []byte("  {  }  ")},
		{"newlines_whitespace_object", []byte("\n\t{ \n } \t\n")},
	}

	for _, tc := range emptyCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := marshalWithExtraBody(base, tc.extra)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !bytes.Equal(got, raw) {
				t.Fatalf("got %s, want exact raw bytes %s", string(got), string(raw))
			}
		})
	}
}

func TestMarshalWithExtraBodyNonObjectRejection(t *testing.T) {
	base := extraBodyTestStruct{
		Zebra: "last",
		Alpha: "first",
		Beta:  42,
	}

	nonObjectCases := []struct {
		name  string
		extra json.RawMessage
	}{
		{"array", []byte("[]")},
		{"number", []byte("123")},
		{"string", []byte(`"not-an-object"`)},
		{"boolean", []byte("true")},
		{"invalid_json", []byte("invalid")},
	}

	for _, tc := range nonObjectCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := marshalWithExtraBody(base, tc.extra)
			if err == nil {
				t.Fatalf("expected error for extra %s, got nil", string(tc.extra))
			}
			wantPrefix := "provider extra body:"
			if !strings.Contains(err.Error(), wantPrefix) {
				t.Fatalf("got error %q, want prefix/substring %q", err.Error(), wantPrefix)
			}
		})
	}
}

func TestMarshalWithExtraBodyOverrideCollision(t *testing.T) {
	base := extraBodyTestStruct{
		Zebra: "last",
		Alpha: "first",
		Beta:  42,
	}

	collisionCases := []struct {
		key   string
		extra string
	}{
		{"zebra", `{"zebra":"override"}`},
		{"alpha", `{"alpha":"override"}`},
		{"beta", `{"beta":99}`},
	}

	for _, tc := range collisionCases {
		t.Run(tc.key, func(t *testing.T) {
			_, err := marshalWithExtraBody(base, json.RawMessage(tc.extra))
			if err == nil {
				t.Fatalf("expected collision error for key %q, got nil", tc.key)
			}
			wantErr := fmt.Sprintf("provider extra body must not override %q", tc.key)
			if !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("got error %q, want %q", err.Error(), wantErr)
			}
		})
	}
}

func TestMarshalWithEmptyExtraBodyDifferential(t *testing.T) {
	base := extraBodyTestStruct{
		Zebra: "value-z",
		Alpha: "value-a",
		Beta:  42,
	}

	allocsEmpty := testing.AllocsPerRun(100, func() {
		_, err := marshalWithExtraBody(base, json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
	})

	allocsWithFields := testing.AllocsPerRun(100, func() {
		_, err := marshalWithExtraBody(base, json.RawMessage(`{"custom_field":123,"extra_string":"val"}`))
		if err != nil {
			t.Fatal(err)
		}
	})

	if allocsEmpty >= allocsWithFields {
		t.Fatalf("expected allocs for empty extra (%v) to be strictly less than with fields (%v)", allocsEmpty, allocsWithFields)
	}
}

func TestTranscriptAdaptersWithEmptyExtraBody(t *testing.T) {
	for _, provider := range []Provider{ProviderOpenAI, ProviderOpenAIResponses} {
		t.Run(string(provider), func(t *testing.T) {
			adapter, err := NewTranscriptAdapter(provider)
			if err != nil {
				t.Fatalf("NewTranscriptAdapter: %v", err)
			}

			reqNoExtra := adapterRequest{
				Model:       "Qwen/Qwen3.6-27B",
				Messages:    []Message{{Role: RoleUser, Content: "hello"}},
				MaxTokens:   64,
				Temperature: 0,
			}
			bodyNoExtra, err := adapter.MarshalRequest(reqNoExtra)
			if err != nil {
				t.Fatalf("MarshalRequest without extra: %v", err)
			}

			reqEmptyExtra := reqNoExtra
			reqEmptyExtra.ExtraBody = json.RawMessage("{}")
			bodyEmptyExtra, err := adapter.MarshalRequest(reqEmptyExtra)
			if err != nil {
				t.Fatalf("MarshalRequest with empty extra: %v", err)
			}

			if !bytes.Equal(bodyNoExtra, bodyEmptyExtra) {
				t.Fatalf("marshaled body with empty extra diverges from body without extra:\nno extra:    %s\nempty extra: %s",
					string(bodyNoExtra), string(bodyEmptyExtra))
			}
		})
	}
}
