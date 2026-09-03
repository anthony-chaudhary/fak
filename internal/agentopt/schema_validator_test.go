package agentopt

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGrammarConstrainedToolCall(t *testing.T) {
	schema := ToolSchema{
		Name:        "readFile",
		Description: "Reads bytes from an absolute path",
		Properties: map[string]PropertySchema{
			"filePath": {Type: TypeString},
			"offset":   {Type: TypeInteger},
			"limit":    {Type: TypeInteger},
			"mode":     {Type: TypeString, Enum: []string{"text", "binary"}},
		},
		Required:             []string{"filePath"},
		AdditionalProperties: false,
	}

	validator := NewSchemaValidator(schema)

	t.Run("conforming tool call passes with 100% validity", func(t *testing.T) {
		args := []byte(`{"filePath":"/var/log/app.log","offset":10,"limit":50,"mode":"text"}`)
		res := validator.ValidateToolCall("readFile", args)
		if !res.Valid || !res.Conforming {
			t.Fatalf("expected valid call, got violations: %v", res.Violations)
		}
		if !strings.HasPrefix(res.Attestation, "attest:sha256:") {
			t.Fatalf("expected attestation hash, got: %q", res.Attestation)
		}
	})

	t.Run("missing required property fails validation", func(t *testing.T) {
		args := []byte(`{"offset":0,"limit":10}`)
		res := validator.ValidateToolCall("readFile", args)
		if res.Valid || res.Conforming {
			t.Fatal("expected invalid call when required property missing")
		}
		found := false
		for _, v := range res.Violations {
			if strings.Contains(v, `missing required property "filePath"`) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing required violation text: %v", res.Violations)
		}
	})

	t.Run("hallucinated property rejected when additional properties false", func(t *testing.T) {
		args := []byte(`{"filePath":"a.txt","hallucinatedKey":"bad"}`)
		res := validator.ValidateToolCall("readFile", args)
		if res.Valid || res.Conforming {
			t.Fatal("expected invalid call when unrecognized property present")
		}
		found := false
		for _, v := range res.Violations {
			if strings.Contains(v, `unrecognized property "hallucinatedKey"`) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing unrecognized property violation text: %v", res.Violations)
		}
	})

	t.Run("type mismatch rejected", func(t *testing.T) {
		args := []byte(`{"filePath":"a.txt","offset":"not-an-integer"}`)
		res := validator.ValidateToolCall("readFile", args)
		if res.Valid || res.Conforming {
			t.Fatal("expected type mismatch failure")
		}
	})

	t.Run("enum value violation rejected", func(t *testing.T) {
		args := []byte(`{"filePath":"a.txt","mode":"invalid-mode"}`)
		res := validator.ValidateToolCall("readFile", args)
		if res.Valid || res.Conforming {
			t.Fatal("expected enum violation failure")
		}
	})

	t.Run("nested object and array validation", func(t *testing.T) {
		nestedSchema := ToolSchema{
			Name: "batchEdit",
			Properties: map[string]PropertySchema{
				"edits": {
					Type: TypeArray,
					Items: &PropertySchema{
						Type: TypeObject,
						Properties: map[string]PropertySchema{
							"old": {Type: TypeString},
							"new": {Type: TypeString},
						},
						Required: []string{"old", "new"},
					},
				},
			},
			Required:             []string{"edits"},
			AdditionalProperties: false,
		}
		validator.RegisterSchema(nestedSchema)

		good := []byte(`{"edits":[{"old":"foo","new":"bar"},{"old":"baz","new":"qux"}]}`)
		if res := validator.ValidateToolCall("batchEdit", good); !res.Valid {
			t.Fatalf("nested valid call failed: %v", res.Violations)
		}

		bad := []byte(`{"edits":[{"old":"foo"}]}`) // missing "new"
		if res := validator.ValidateToolCall("batchEdit", bad); res.Valid {
			t.Fatal("nested call missing required field should fail")
		}
	})

	t.Run("malformed JSON arguments rejected cleanly", func(t *testing.T) {
		res := validator.ValidateToolCall("readFile", []byte(`{not-json`))
		if res.Valid || len(res.Violations) == 0 {
			t.Fatal("expected malformed JSON violation")
		}
	})

	t.Run("unregistered tool name rejected", func(t *testing.T) {
		var dummy map[string]any
		b, _ := json.Marshal(dummy)
		res := validator.ValidateToolCall("nonExistentTool", b)
		if res.Valid {
			t.Fatal("expected unknown tool failure")
		}
	})
}
