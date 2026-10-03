package agent

// toolssearch_schema_test.go — the guard against the silent tool-schema regression.
//
// internal/agent/mcptools.go handleToolsSearch does `_ = json.Unmarshal(d.Function.Parameters,
// &p)` on every searchable tool. A malformed JSON literal in a tool descriptor therefore
// degrades to a nil inputSchema with NO error anywhere: the model is handed a tool whose
// schema is silently missing, the catalog still "works", and CI stays green. That exact
// failure shipped once on the fak_read schema (a bad edit to the widened-root description) and
// was caught only incidentally, by a test that happened to assert on fak_read's schema.
//
// The invariant pinned here is cheap and total: EVERY schema literal in the searchable catalog
// must parse. handleToolsSearch's error-swallowing is left exactly as it is — this is a test,
// not a behaviour change — but a malformed literal can no longer reach it unnoticed.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/systools"
)

// schemaEntry is one tool's name plus its raw parameter schema, so catalogs from different
// packages (agent, codetools, systools) can be checked by one helper.
type schemaEntry struct {
	name       string
	parameters json.RawMessage
}

// assertToolSchemasParse fails loudly, naming the tool, for every non-empty Parameters literal
// that does not decode as a JSON object.
func assertToolSchemasParse(t *testing.T, catalog string, entries []schemaEntry) int {
	t.Helper()
	parsed := 0
	for _, e := range entries {
		if len(e.parameters) == 0 {
			// An absent schema is honest (inputSchema stays nil) rather than wrong, so this is
			// reported but not failed on.
			t.Logf("%s: tool %q advertises no parameters schema (empty literal)", catalog, e.name)
			continue
		}
		var decoded map[string]any
		if err := json.Unmarshal(e.parameters, &decoded); err != nil {
			t.Errorf("%s: tool %q has a MALFORMED parameters JSON literal (%v); handleToolsSearch swallows this as a nil inputSchema, so the model would silently lose the schema: %s",
				catalog, e.name, err, string(e.parameters))
			continue
		}
		if decoded == nil {
			t.Errorf("%s: tool %q decodes to JSON null, not an object schema: %s", catalog, e.name, string(e.parameters))
			continue
		}
		parsed++
	}
	return parsed
}

func entriesOf(defs []ToolDef) []schemaEntry {
	out := make([]schemaEntry, 0, len(defs))
	for _, d := range defs {
		out = append(out, schemaEntry{name: d.Function.Name, parameters: d.Function.Parameters})
	}
	return out
}

// TestToolsSearchCatalogParametersAreValidJSON is the regression guard. It walks the FULL
// searchable catalog in two passes, because allSearchableTools() is gated on arming: unarmed,
// it returns 6 travel-domain tools and does not even contain fak_read, which would make a
// guard over it vacuous.
func TestToolsSearchCatalogParametersAreValidJSON(t *testing.T) {
	t.Cleanup(func() { DisarmMCPTools() })
	Configure()
	if _, err := ArmMCPTools(); err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}

	t.Run("armed allSearchableTools", func(t *testing.T) {
		tools := allSearchableTools()
		if len(tools) == 0 {
			t.Fatal("allSearchableTools() is empty after arming: the catalog gate changed and this guard would pass vacuously")
		}
		if parsed := assertToolSchemasParse(t, "allSearchableTools", entriesOf(tools)); parsed != len(tools) {
			t.Fatalf("only %d of %d searchable tools carried a schema", parsed, len(tools))
		}
		// Non-vacuity: the tool whose schema was corrupted once must be IN the catalog the
		// guard walks, or the guard proves nothing about it.
		names := make(map[string]bool, len(tools))
		for _, d := range tools {
			names[d.Function.Name] = true
		}
		for _, want := range []string{"fak_read", "fak_tools_search"} {
			if !names[want] {
				t.Fatalf("tool %q is missing from allSearchableTools() after ArmMCPTools: the schema guard would not cover it (catalog = %v)", want, names)
			}
		}
	})

	// The catalogs allSearchableTools() composes but that stay empty until their own arming
	// (each needs configuration this guard has no business supplying). They are walked here
	// directly so the guard cannot be narrowed to a subset of the catalog by an unrelated
	// arming change. These Catalog()/XxxDefs() functions are pure literal returns.
	t.Run("raw mcp catalog", func(t *testing.T) {
		if parsed := assertToolSchemasParse(t, "rawMCPToolCatalog", entriesOf(rawMCPToolCatalog())); parsed == 0 {
			t.Fatal("rawMCPToolCatalog carries no parsable schema")
		}
	})
	t.Run("task catalog", func(t *testing.T) {
		if parsed := assertToolSchemasParse(t, "taskToolDefs", entriesOf(taskToolDefs())); parsed == 0 {
			t.Fatal("taskToolDefs carries no parsable schema")
		}
	})
	t.Run("todo catalog", func(t *testing.T) {
		if parsed := assertToolSchemasParse(t, "todoToolDefs", entriesOf(todoToolDefs())); parsed == 0 {
			t.Fatal("todoToolDefs carries no parsable schema")
		}
	})
	t.Run("code tool catalog", func(t *testing.T) {
		defs := codetools.Catalog()
		entries := make([]schemaEntry, 0, len(defs))
		for _, d := range defs {
			entries = append(entries, schemaEntry{name: d.Name, parameters: d.Parameters})
		}
		if parsed := assertToolSchemasParse(t, "codetools.Catalog", entries); parsed == 0 {
			t.Fatal("codetools.Catalog carries no parsable schema")
		}
	})
	t.Run("sys tool catalog", func(t *testing.T) {
		defs := systools.Catalog()
		entries := make([]schemaEntry, 0, len(defs))
		for _, d := range defs {
			entries = append(entries, schemaEntry{name: d.Name, parameters: d.Parameters})
		}
		if parsed := assertToolSchemasParse(t, "systools.Catalog", entries); parsed == 0 {
			t.Fatal("systools.Catalog carries no parsable schema")
		}
	})
}

// TestToolsSearchFakReadSchemaDocumentsWidenedReadRoots pins the descriptor text this ticket
// changed. The description is the only place the widened read-root contract is stated to the
// model, and the corrupted literal that motivated this guard was a corrupted DESCRIPTION
// inside it: the schema still parsed, so only the wording was wrong and nothing noticed.
func TestToolsSearchFakReadSchemaDocumentsWidenedReadRoots(t *testing.T) {
	t.Cleanup(func() { DisarmMCPTools() })
	Configure()
	if _, err := ArmMCPTools(); err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}

	var found bool
	for _, d := range allSearchableTools() {
		if d.Function.Name != "fak_read" {
			continue
		}
		found = true
		var schema struct {
			Properties map[string]struct {
				Type        string `json:"type"`
				Description string `json:"description"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(d.Function.Parameters, &schema); err != nil {
			t.Fatalf("fak_read parameters do not parse: %v", err)
		}
		filePath, ok := schema.Properties["file_path"]
		if !ok {
			t.Fatalf("fak_read schema has no file_path property: %s", string(d.Function.Parameters))
		}
		if filePath.Type != "string" {
			t.Fatalf("fak_read file_path type = %q, want string", filePath.Type)
		}
		for _, marker := range []string{"read root", "companion public fak checkout", "path_escape"} {
			if !strings.Contains(filePath.Description, marker) {
				t.Fatalf("fak_read file_path description does not mention %q, so the widened-root contract is undocumented for the model: %q",
					marker, filePath.Description)
			}
		}
		if _, ok := schema.Properties["file_paths"]; !ok {
			t.Fatalf("fak_read schema dropped the batch form file_paths: %s", string(d.Function.Parameters))
		}
	}
	if !found {
		t.Fatal("fak_read is not in allSearchableTools() after ArmMCPTools")
	}
}
