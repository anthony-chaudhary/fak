// Package contextq derives a skill's tool-schema fault set from its declared
// allowed-tools fence. The two concepts remain distinct types even while this
// first spine intentionally maps the fence one-for-one to the fault set.
package contextq

import (
	"bufio"
	"fmt"
	"sort"
	"strings"
)

// ToolCatalog is the set of tool schema names available to a skill.
type ToolCatalog map[string]struct{}

// FaultSet is the resolved schema pages a skill may need. It is deliberately a
// distinct type from the frontmatter fence: future paging policy must not
// silently change the capability maximum.
type FaultSet struct{ Tools []string }

// ResolveAllowedTools parses SKILL.md frontmatter and resolves allowed-tools
// exactly against catalog. Unknown tools and absent declarations fail loudly.
func ResolveAllowedTools(src []byte, catalog ToolCatalog) (FaultSet, error) {
	s := bufio.NewScanner(strings.NewReader(string(src)))
	if !s.Scan() || strings.TrimSpace(s.Text()) != "---" {
		return FaultSet{}, fmt.Errorf("skill frontmatter: opening delimiter missing")
	}
	var raw string
	closed := false
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "---" {
			closed = true
			break
		}
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "allowed-tools" {
			raw = strings.TrimSpace(value)
		}
	}
	if err := s.Err(); err != nil {
		return FaultSet{}, fmt.Errorf("skill frontmatter: %w", err)
	}
	if !closed {
		return FaultSet{}, fmt.Errorf("skill frontmatter: closing delimiter missing")
	}
	if raw == "" {
		return FaultSet{}, fmt.Errorf("skill frontmatter: allowed-tools missing or empty")
	}
	seen := map[string]bool{}
	tools := make([]string, 0)
	for _, item := range strings.Split(raw, ",") {
		name := strings.TrimSpace(item)
		if name == "" {
			return FaultSet{}, fmt.Errorf("skill frontmatter: empty allowed-tools entry")
		}
		if _, ok := catalog[name]; !ok {
			return FaultSet{}, fmt.Errorf("skill frontmatter: unknown allowed-tool %q", name)
		}
		if !seen[name] {
			seen[name] = true
			tools = append(tools, name)
		}
	}
	sort.Strings(tools)
	return FaultSet{Tools: tools}, nil
}
