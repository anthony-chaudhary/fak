// fromtools.go — scaffold a starter capability floor from an APPLICATION'S OWN
// tool catalog (issue #5153). The default floor is authored for Claude Code; a
// product embedding fak in front of its own model call has tools the floor has
// never heard of, so fail-closed denies every one of them until the developer
// hand-authors a manifest. ScaffoldFromTools is the importer that closes that
// gap: it reads the tool/function catalog the developer ALREADY passes to the
// model — an OpenAI `tools: [{type:"function", function:{name, parameters}}]`
// array or an Anthropic `tools: [{name, input_schema}]` array — and emits a
// starter Manifest that round-trips through `fak policy --check` unchanged.
//
// The scaffold keeps the floor's fail-closed discipline:
//   - Read-shaped tool names (get_/list_/lookup_/... prefixes) go into allow.
//   - Every other tool lands in the explicit deny section citing POLICY_BLOCK —
//     the author PROMOTES a tool to allow after review; nothing verb-y is
//     silently allowed.
//   - Every string-typed top-level argument gets a placeholder deny_regex stub
//     citing the closed refusal vocabulary, with a TODO fix line, so the author
//     tightens existing structure instead of inventing it.
//
// Deterministic and offline: pure schema → manifest, no model, no key, no GPU,
// stable ordering — so the output is CI-checkable and golden-testable.
package policy

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// scaffoldPlaceholderRegex is the inert deny_regex stub emitted for each
// string-typed argument: it compiles under RE2 and matches only the literal
// placeholder token, so the starter floor validates and enforces nothing on
// that argument until the author replaces it with a real forbidden-value
// pattern (or deletes the rule).
const scaffoldPlaceholderRegex = `\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\z`

// readShapedPrefixes are the tool-name verb prefixes the scaffold treats as
// read-ish and therefore safe to place on the starter allow-list. Matching is
// on the lowercased name at a word boundary (exact prefix, or prefix followed
// by `_`, `-`, `.`, or an uppercase rune), so `get_user` and `getUser` qualify
// but `getaway_car` does NOT silently ride on `get` — see isReadShaped.
var readShapedPrefixes = []string{
	"browse", "check", "count", "describe", "fetch", "find", "get",
	"inspect", "list", "lookup", "peek", "query", "read", "retrieve",
	"search", "show", "status", "view",
}

// rawCatalogTool is the union of the two wire shapes ScaffoldFromTools accepts,
// per entry: Anthropic ({name, input_schema}) and OpenAI ({type:"function",
// function:{name, parameters}}). Unknown sibling fields (description, strict,
// cache_control, ...) are ignored — the catalog is the developer's existing
// model-call payload, not a fak-owned schema.
type rawCatalogTool struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	Function    *struct {
		Name       string          `json:"name"`
		Parameters json.RawMessage `json:"parameters,omitempty"`
	} `json:"function,omitempty"`
}

// toolSchema is one normalized catalog entry: the tool's exact name plus its
// JSON-Schema parameter object (nil when the tool declares no parameters).
type toolSchema struct {
	Name   string
	Schema json.RawMessage
}

// ScaffoldFromTools reads a tool-catalog JSON document — a bare array of tools,
// or an object wrapping one under a "tools" key — in either the OpenAI or the
// Anthropic shape (entries may mix), and returns a starter Manifest:
// read-shaped names allowed, everything else explicitly denied with
// POLICY_BLOCK, and a placeholder deny_regex stub per string-typed top-level
// argument. The returned manifest is validated through ToRuntime before it is
// handed back, so the scaffold can never emit a floor `fak policy --check`
// would reject.
func ScaffoldFromTools(data []byte) (*Manifest, error) {
	tools, err := parseToolCatalog(data)
	if err != nil {
		return nil, err
	}
	if len(tools) == 0 {
		return nil, fmt.Errorf("from-tools: catalog declares no tools; nothing to scaffold a floor from")
	}

	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })

	policyBlock := abi.ReasonName(abi.ReasonPolicyBlock)
	m := &Manifest{
		Version: Version,
		Posture: postureFailClosed,
	}
	for _, t := range tools {
		if isReadShaped(t.Name) {
			m.Allow = append(m.Allow, t.Name)
		} else {
			if m.Deny == nil {
				m.Deny = map[string]string{}
			}
			m.Deny[t.Name] = policyBlock
		}
		for _, arg := range stringArgs(t.Schema) {
			m.ArgRules = append(m.ArgRules, ArgRule{
				Tool:      t.Name,
				Arg:       arg,
				DenyRegex: scaffoldPlaceholderRegex,
				Reason:    policyBlock,
				Fix: fmt.Sprintf(
					"TODO: replace the placeholder deny_regex for %s.%s with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing",
					t.Name, arg),
			})
		}
	}

	// Round-trip through the same validator `fak policy --check` uses, so a
	// scaffold that would not load is a bug caught HERE, not at the adopter's
	// boundary.
	if _, err := m.ToRuntime(); err != nil {
		return nil, fmt.Errorf("from-tools: scaffolded manifest failed validation (bug): %w", err)
	}
	return m, nil
}

// parseToolCatalog decodes the raw catalog bytes into normalized entries,
// accepting a bare JSON array or a {"tools": [...]} wrapper, with each entry in
// either wire shape. A tool with no resolvable name fails loud with its index —
// a floor scaffolded from a half-read catalog would silently deny the missing
// tool forever.
func parseToolCatalog(data []byte) ([]toolSchema, error) {
	var raw []rawCatalogTool
	if err := json.Unmarshal(data, &raw); err != nil {
		var wrapper struct {
			Tools []rawCatalogTool `json:"tools"`
		}
		if werr := json.Unmarshal(data, &wrapper); werr != nil || wrapper.Tools == nil {
			return nil, fmt.Errorf("from-tools: catalog is neither a JSON tool array nor an object with a \"tools\" array: %w", err)
		}
		raw = wrapper.Tools
	}

	out := make([]toolSchema, 0, len(raw))
	seen := make(map[string]bool, len(raw))
	for i, r := range raw {
		name, schema := r.Name, r.InputSchema
		if name == "" && r.Function != nil {
			name, schema = r.Function.Name, r.Function.Parameters
		}
		name = strings.TrimSpace(name)
		if name == "" {
			return nil, fmt.Errorf("from-tools: tools[%d] has no name (expected Anthropic {name, input_schema} or OpenAI {function:{name, parameters}})", i)
		}
		if seen[name] {
			continue // duplicate declaration: first schema wins, deterministically
		}
		seen[name] = true
		out = append(out, toolSchema{Name: name, Schema: schema})
	}
	return out, nil
}

// isReadShaped reports whether a tool name starts with a read-ish verb prefix
// at a word boundary: the whole name, or the prefix followed by `_`, `-`, `.`,
// or an uppercase rune (camelCase). Conservative on purpose — anything not
// provably read-shaped lands in the deny section for the author to promote.
func isReadShaped(name string) bool {
	lower := strings.ToLower(name)
	for _, p := range readShapedPrefixes {
		if !strings.HasPrefix(lower, p) {
			continue
		}
		if len(name) == len(p) {
			return true
		}
		next := name[len(p)]
		if next == '_' || next == '-' || next == '.' || (next >= 'A' && next <= 'Z') {
			return true
		}
	}
	return false
}

// stringArgs extracts the sorted names of string-typed top-level properties
// from a JSON-Schema parameter object. Anything unparseable or non-object is
// skipped silently: the argument stubs are an authoring convenience, and a
// catalog with exotic schemas still deserves a name-level floor.
func stringArgs(schema json.RawMessage) []string {
	if len(schema) == 0 {
		return nil
	}
	var s struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(schema, &s); err != nil {
		return nil
	}
	var out []string
	for name, prop := range s.Properties {
		if prop.Type == "string" {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}
