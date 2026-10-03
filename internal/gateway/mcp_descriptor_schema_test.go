package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestToolDescriptorsInputSchemasAreValidJSON pins the wire-safety invariant of the
// MCP descriptor table: every toolDescriptors() entry's `inputSchema` must be a
// parseable JSON object.
//
// WHY THIS IS PINNED. The `inputSchema` is carried as raw json.RawMessage and is
// only ever marshalled — never validated — on the way to tools/list. A malformed
// literal therefore has no local failure site: it propagates into json.Marshal of
// the whole descriptor map, so the ENTIRE tools/list response fails (or, where the
// literal is unmarshalled first, degrades to a nil schema). A one-character
// descriptor edit is enough to break tool advertisement for every client, and
// nothing in this package catches it — the existing tests read descriptor NAMES
// only. That exact corruption shipped once (an unbalanced brace in the fak_read
// schema) and was caught only incidentally, by a test in another package that
// happened to assert on that one tool.
func TestToolDescriptorsInputSchemasAreValidJSON(t *testing.T) {
	descs := toolDescriptors()
	if len(descs) == 0 {
		t.Fatal("toolDescriptors() returned no descriptors; the guard would be vacuous")
	}
	for _, d := range descs {
		name, _ := d["name"].(string)
		if name == "" {
			t.Errorf("descriptor with no name: %+v", d)
			continue
		}
		raw, ok := d["inputSchema"].(json.RawMessage)
		if !ok {
			t.Errorf("tool %q: inputSchema is %T, want json.RawMessage", name, d["inputSchema"])
			continue
		}
		if len(raw) == 0 {
			t.Errorf("tool %q: inputSchema is empty", name)
			continue
		}
		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Errorf("tool %q: MALFORMED inputSchema JSON literal (%v); tools/list marshalling of the whole descriptor map would fail, so pin this before editing a descriptor", name, err)
			continue
		}
		if decoded == nil {
			t.Errorf("tool %q: inputSchema decoded to null", name)
		}
	}
}

// TestFakReadDescriptorDocumentsWidenedReadRoots pins the contract the fak_read
// descriptor advertises to the model: `file_path` is a string, and its description
// tells the model which roots are admitted and what a refusal looks like. A
// descriptor that under-advertises its confinement makes a model guess and retry
// a path the floor will refuse.
func TestFakReadDescriptorDocumentsWidenedReadRoots(t *testing.T) {
	for _, d := range toolDescriptors() {
		if name, _ := d["name"].(string); name != "fak_read" {
			continue
		}
		raw, ok := d["inputSchema"].(json.RawMessage)
		if !ok || len(raw) == 0 {
			t.Fatalf("fak_read inputSchema is %T/%d bytes, want a non-empty json.RawMessage", d["inputSchema"], len(raw))
		}
		var schema struct {
			Properties struct {
				FilePath struct {
					Type        string `json:"type"`
					Description string `json:"description"`
				} `json:"file_path"`
				FilePaths *struct {
					Type string `json:"type"`
				} `json:"file_paths"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(raw, &schema); err != nil {
			t.Fatalf("fak_read inputSchema does not decode into the expected shape: %v", err)
		}
		if got := schema.Properties.FilePath.Type; got != "string" {
			t.Errorf("fak_read file_path type = %q, want \"string\"", got)
		}
		desc := schema.Properties.FilePath.Description
		// The model must be able to tell, from the descriptor alone, that the
		// companion public checkout is in view and what a refusal is called.
		lowerDesc := strings.ToLower(desc)
		for _, marker := range []string{"read root", "companion public fak checkout", "path_escape"} {
			if !strings.Contains(lowerDesc, marker) {
				t.Errorf("fak_read file_path description does not mention %q, so the widened-root contract is undocumented for the model; description = %q", marker, desc)
			}
		}
		if schema.Properties.FilePaths == nil {
			t.Error("fak_read no longer declares file_paths; the batch-read form must stay advertised")
		}
		return
	}
	t.Fatal("fak_read missing from toolDescriptors()")
}
