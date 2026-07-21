package sharedtask

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// repoRootForContractTest walks up from the package dir (go test CWD) until it
// finds go.mod, so the contract witnesses can read the committed schemas, doc,
// and example fixtures.
func repoRootForContractTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repo root (go.mod) above the package dir")
		}
		dir = parent
	}
}

func contractSchemaDir(t *testing.T) string {
	t.Helper()
	return filepath.Join(repoRootForContractTest(t), "tools", "schemas")
}

// The expected counts below are the exact output of the retired Python validator
// (tools/shared_task_contract.py) on the same tracked fixtures, captured at port
// time — the parity witness for the Go rewrite.

func TestContractDocExamplesValidate(t *testing.T) {
	root := repoRootForContractTest(t)
	counts, err := ValidateContractDoc(contractSchemaDir(t), filepath.Join(root, "docs", "shared-task-record-contract.md"))
	if err != nil {
		t.Fatalf("validate-doc: %v", err)
	}
	want := "fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=1, fak.shared-patch.v1=1, fak.shared-task-journal.v1=1, fak.shared-task.v1=1"
	if got := FormatContractCounts(counts); got != want {
		t.Fatalf("doc example counts drifted:\n got %s\nwant %s", got, want)
	}
}

func TestContractSequenceFixtureValidates(t *testing.T) {
	root := repoRootForContractTest(t)
	counts, err := ValidateContractSequence(contractSchemaDir(t), filepath.Join(root, "examples", "shared-task-record"))
	if err != nil {
		t.Fatalf("validate-sequence: %v", err)
	}
	want := "fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=7, fak.shared-patch.v1=6, fak.shared-task-journal.v1=2, fak.shared-task.v1=1"
	if got := FormatContractCounts(counts); got != want {
		t.Fatalf("sequence fixture counts drifted:\n got %s\nwant %s", got, want)
	}
}

func TestContractVerdictsFixtureValidates(t *testing.T) {
	root := repoRootForContractTest(t)
	counts, err := ValidateContractVerdicts(contractSchemaDir(t), filepath.Join(root, "examples", "shared-task-record-verdicts"))
	if err != nil {
		t.Fatalf("validate-verdicts: %v", err)
	}
	if got, want := FormatContractCounts(counts), "fak.shared-patch-result.v1=5"; got != want {
		t.Fatalf("verdict fixture counts drifted:\n got %s\nwant %s", got, want)
	}
}

// mustDecode builds a decoded-JSON value (json.Number numbers) from a literal, so
// the pure validator tests exercise exactly what the file loaders produce.
func mustDecode(t *testing.T, src string) any {
	t.Helper()
	v, err := decodeContractJSON([]byte(src))
	if err != nil {
		t.Fatalf("decode %q: %v", src, err)
	}
	return v
}

func mustDecodeObj(t *testing.T, src string) map[string]any {
	t.Helper()
	obj, ok := mustDecode(t, src).(map[string]any)
	if !ok {
		t.Fatalf("not an object: %q", src)
	}
	return obj
}

func TestValidateValueRejections(t *testing.T) {
	cases := []struct {
		name     string
		schema   string
		instance string
		wantErr  string
	}{
		{"object type", `{"type":"object"}`, `[]`, "want object"},
		{"array type", `{"type":"array"}`, `{}`, "want array"},
		{"string type", `{"type":"string"}`, `3`, "want string"},
		{"integer type", `{"type":"integer"}`, `"3"`, "want integer"},
		{"float is not integer", `{"type":"integer"}`, `3.5`, "want integer"},
		{"bool is not integer", `{"type":"integer"}`, `true`, "want integer"},
		{"const mismatch", `{"const":"fak.shared-task.v1"}`, `"other"`, "want fak.shared-task.v1"},
		{"enum violation", `{"enum":["trusted","tainted"]}`, `"radioactive"`, "not in enum"},
		{"pattern violation", `{"type":"string","pattern":"^sha256:.+"}`, `"md5:abc"`, "does not match"},
		{"minLength violation", `{"type":"string","minLength":1}`, `""`, "string too short"},
		{"minimum violation", `{"type":"integer","minimum":0}`, `-4`, "below minimum"},
		{"missing required", `{"type":"object","required":["task_id"]}`, `{}`, "missing [task_id]"},
		{"minItems violation", `{"type":"array","minItems":1}`, `[]`, "too few items"},
		{"nested property", `{"type":"object","properties":{"rev":{"type":"string","pattern":"^sha256:.+"}}}`, `{"rev":"bad"}`, "$.rev"},
		{"item recursion", `{"type":"array","items":{"type":"integer"}}`, `[1,"x"]`, "$[1]: want integer"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			schema := mustDecodeObj(t, tc.schema)
			err := validateValue(mustDecode(t, tc.instance), schema, schema, "$")
			if err == nil {
				t.Fatalf("want error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want error containing %q, got %q", tc.wantErr, err.Error())
			}
		})
	}
}

func TestValidateValueAccepts(t *testing.T) {
	schema := mustDecodeObj(t, `{
		"type":"object",
		"required":["kind","bytes"],
		"properties":{
			"kind":{"type":"string","enum":["inline","blob"]},
			"bytes":{"type":"integer","minimum":0},
			"tags":{"type":"array","minItems":1,"items":{"type":"string","minLength":1}}
		}
	}`)
	instance := mustDecode(t, `{"kind":"blob","bytes":42,"tags":["a","b"]}`)
	if err := validateValue(instance, schema, schema, "$"); err != nil {
		t.Fatalf("valid instance rejected: %v", err)
	}
}

func TestValidateValueResolvesRefs(t *testing.T) {
	schema := mustDecodeObj(t, `{
		"type":"object",
		"properties":{"actor":{"$ref":"#/definitions/actor"}},
		"definitions":{"actor":{"type":"object","required":["id"],"properties":{"id":{"type":"string","minLength":1}}}}
	}`)
	if err := validateValue(mustDecode(t, `{"actor":{"id":"a1"}}`), schema, schema, "$"); err != nil {
		t.Fatalf("ref-valid instance rejected: %v", err)
	}
	err := validateValue(mustDecode(t, `{"actor":{}}`), schema, schema, "$")
	if err == nil || !strings.Contains(err.Error(), "$.actor: missing [id]") {
		t.Fatalf("want $.actor missing-id error, got %v", err)
	}
}

func TestValidateEnvelopeRejectsUnknownAndMissingSchema(t *testing.T) {
	dir := contractSchemaDir(t)
	if _, err := ValidateEnvelope(map[string]any{}, dir); err == nil || !strings.Contains(err.Error(), "missing string schema") {
		t.Fatalf("want missing-schema error, got %v", err)
	}
	if _, err := ValidateEnvelope(map[string]any{"schema": "fak.unknown.v9"}, dir); err == nil || !strings.Contains(err.Error(), "unknown schema") {
		t.Fatalf("want unknown-schema error, got %v", err)
	}
}

func TestValidateContractFilesCountsAndFails(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "ref.json")
	if err := os.WriteFile(good, []byte(`{
		"schema":"fak.shared-artifact-ref.v1","artifact_id":"art_1",
		"ref":"sha256:abc","media_type":"text/plain","taint":"trusted",
		"scope":"fleet","store":"blob"
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	counts, err := ValidateContractFiles(contractSchemaDir(t), []string{good})
	if err != nil {
		t.Fatalf("valid artifact ref rejected: %v", err)
	}
	if counts[SchemaArtifactRef] != 1 {
		t.Fatalf("want one artifact ref counted, got %v", counts)
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{
		"schema":"fak.shared-artifact-ref.v1","artifact_id":"art_2",
		"ref":"md5:nope","media_type":"text/plain","taint":"trusted",
		"scope":"fleet","store":"blob"
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ValidateContractFiles(contractSchemaDir(t), []string{bad}); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("want pattern violation for md5 ref, got %v", err)
	}
}

func TestSequenceValidationRejectsEmptyDir(t *testing.T) {
	if _, err := ValidateContractSequence(contractSchemaDir(t), t.TempDir()); err == nil || !strings.Contains(err.Error(), "no JSON files") {
		t.Fatalf("want no-JSON-files error, got %v", err)
	}
}
