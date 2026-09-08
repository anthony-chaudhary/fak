package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"
)

// ToolDescriptorsForResolver returns the tool descriptors for use by the
// protocol-generic capindex MCP resolver. This function is exported to allow
// the resolver to index MCP tools as generic Capabilities, proving the loader
// is protocol-blind (issue #1108, C5).
func ToolDescriptorsForResolver() []map[string]any {
	return toolDescriptors()
}

// validateToolDescriptors verifies that every registered MCP tool provides a
// well-formed, provider-conforming OpenAPI 3.0 schema at startup (#10769).
// Fails loud rather than discover schema dialect incompatibilities at runtime.
func validateToolDescriptors() error {
	for _, td := range toolDescriptors() {
		name, _ := td["name"].(string)
		if name == "" {
			return errors.New("gateway: tool descriptor missing name")
		}
		desc, _ := td["description"].(string)
		if desc == "" {
			return fmt.Errorf("gateway: tool %q missing description", name)
		}
		raw, ok := td["inputSchema"].(json.RawMessage)
		if !ok || len(raw) == 0 {
			continue
		}
		var schemaObj map[string]any
		if err := json.Unmarshal(raw, &schemaObj); err != nil {
			return fmt.Errorf("gateway: tool %q inputSchema is not valid JSON: %w", name, err)
		}
		if err := validateOpenAPISchemaNode(name, schemaObj); err != nil {
			return err
		}
	}
	return nil
}

func validateOpenAPISchemaNode(path string, schema map[string]any) error {
	stype, _ := schema["type"].(string)
	if req, ok := schema["required"].([]any); ok {
		if !strings.EqualFold(stype, "object") {
			return fmt.Errorf("gateway: schema %s.required only allowed for type object (got %q)", path, stype)
		}
		props, _ := schema["properties"].(map[string]any)
		for _, r := range req {
			rStr, _ := r.(string)
			if props == nil || props[rStr] == nil {
				return fmt.Errorf("gateway: schema %s.required property %q not declared in properties", path, rStr)
			}
		}
	}
	for _, key := range []string{"anyOf", "any_of", "oneOf", "one_of", "allOf", "all_of"} {
		if alts, ok := schema[key].([]any); ok {
			for i, alt := range alts {
				if altMap, ok := alt.(map[string]any); ok {
					subPath := fmt.Sprintf("%s.%s[%d]", path, key, i)
					if err := validateOpenAPISchemaNode(subPath, altMap); err != nil {
						return err
					}
				}
			}
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for k, child := range props {
			if childMap, ok := child.(map[string]any); ok {
				if err := validateOpenAPISchemaNode(path+".properties."+k, childMap); err != nil {
					return err
				}
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		if err := validateOpenAPISchemaNode(path+".items", items); err != nil {
			return err
		}
	}
	return nil
}

// toolRegistryNames returns the bare names of every tool in the built-in
// registry, in registry order. It is the universe compileToolExposeAllow
// validates --expose patterns against and exposedToolDescriptors filters.
func toolRegistryNames() []string {
	all := toolDescriptors()
	names := make([]string, 0, len(all))
	for _, td := range all {
		if n, _ := td["name"].(string); n != "" {
			names = append(names, n)
		}
	}
	return names
}

// compileToolExposeAllow turns the raw --expose patterns (Config.ExposeTools)
// into an allowlist predicate over tool names. Each raw entry may be
// comma-separated; entries are trimmed and empties dropped, so `--expose a,b`
// and `--expose a --expose b` compile identically. An empty result set returns
// (nil, nil) — the full-surface default. Otherwise every pattern is validated
// as a path.Match glob that matches ≥1 registered tool; a malformed glob or a
// zero-match pattern is a fail-loud error (a typo must never silently hide the
// whole surface). The returned predicate reports whether a tool NAME matches
// any pattern.
func compileToolExposeAllow(raw []string) (func(string) bool, error) {
	var patterns []string
	for _, entry := range raw {
		for _, p := range strings.Split(entry, ",") {
			if p = strings.TrimSpace(p); p != "" {
				patterns = append(patterns, p)
			}
		}
	}
	if len(patterns) == 0 {
		return nil, nil
	}
	names := toolRegistryNames()
	for _, p := range patterns {
		matched := false
		for _, n := range names {
			ok, err := path.Match(p, n)
			if err != nil {
				return nil, fmt.Errorf("gateway: --expose pattern %q is not a valid glob: %w", p, err)
			}
			if ok {
				matched = true
			}
		}
		if !matched {
			return nil, fmt.Errorf("gateway: --expose pattern %q matches no known tool (have: %s)", p, strings.Join(names, ", "))
		}
	}
	return func(name string) bool {
		for _, p := range patterns {
			if ok, _ := path.Match(p, name); ok {
				return true
			}
		}
		return false
	}, nil
}

func (s *Server) supportsVision() bool {
	if s == nil {
		return false
	}
	if s.modelVision {
		return true
	}
	if envEnabled("FAK_MODEL_VISION") {
		return true
	}
	m := strings.ToLower(s.model)
	if m != "" && (strings.Contains(m, "vision") || strings.Contains(m, "-vl") || strings.Contains(m, "4o") || strings.Contains(m, "claude") || strings.Contains(m, "gemini")) {
		return true
	}
	return false
}

// exposedToolDescriptors is toolDescriptors filtered by the optional --expose
// allowlist and model capability constraints (e.g. vision support for view_image).
// It is the SINGLE choke point every discovery view routes through
// (tools/list, fak_tools_search, the capabilities resource + self-feature
// catalog, fak_feature_query, fak_capabilities), so a hidden tool never appears
// in any of them. When no allowlist is in force (s.exposeAllow == nil) it
// returns the eligible registry filtered by model capabilities.
func (s *Server) exposedToolDescriptors() []map[string]any {
	all := toolDescriptors()
	hasVision := s.supportsVision()
	out := make([]map[string]any, 0, len(all))
	for _, td := range all {
		name, _ := td["name"].(string)
		if !hasVision && (name == "view_image" || name == "functions.view_image") {
			continue
		}
		if s != nil && s.exposeAllow != nil && !s.exposeAllow(name) {
			continue
		}
		out = append(out, td)
	}
	return out
}
