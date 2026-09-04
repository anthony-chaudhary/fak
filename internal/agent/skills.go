package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/frontmatter"
)

// skills.go — native Agent Skills parser and discovery engine (#11141).
//
// Agent Skills conform to the standard YAML frontmatter specification:
// - frontmatter enclosed by '---' on line 1 and closing '---'
// - metadata fields: name, description, allowed-tools (or allowed_tools), metadata (and canonical)
// - followed by the markdown instructions/procedure body.

const (
	// ToolSkill is the loop-facing tool name for dynamic skill loading.
	ToolSkill = "skill"
	// SkillDriverID is the registered kernel engine id for skill execution.
	SkillDriverID = "agent.skill"
)

// Skill represents a parsed and resolved agent skill definition.
type Skill struct {
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	AllowedTools []string          `json:"allowed_tools,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	Body         string            `json:"body,omitempty"`
	Path         string            `json:"path"`
	Canonical    string            `json:"canonical,omitempty"`
}

// Format renders the loaded skill into a structured context payload for the agent.
func (s *Skill) Format() string {
	var sb strings.Builder
	sb.WriteString("# Skill: ")
	sb.WriteString(s.Name)
	sb.WriteString("\n\n")
	if s.Description != "" {
		sb.WriteString(s.Description)
		sb.WriteString("\n\n")
	}
	if len(s.AllowedTools) > 0 {
		sb.WriteString("Allowed Tools: ")
		sb.WriteString(strings.Join(s.AllowedTools, ", "))
		sb.WriteString("\n\n")
	}
	if s.Body != "" {
		sb.WriteString(s.Body)
		sb.WriteString("\n")
	}
	return sb.String()
}

// SkillRegistry holds discovered skills with thread-safe access.
type SkillRegistry struct {
	mu     sync.RWMutex
	skills map[string]*Skill
}

// NewSkillRegistry creates an empty SkillRegistry.
func NewSkillRegistry() *SkillRegistry {
	return &SkillRegistry{
		skills: make(map[string]*Skill),
	}
}

// Register adds or updates a skill in the registry.
func (r *SkillRegistry) Register(skill *Skill) {
	if skill == nil || skill.Name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.skills[skill.Name] = skill
}

// Get looks up a skill by name, falling back to case-insensitive matching.
func (r *SkillRegistry) Get(name string) (*Skill, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if s, ok := r.skills[name]; ok {
		return s, true
	}
	lower := strings.ToLower(name)
	for k, v := range r.skills {
		if strings.ToLower(k) == lower {
			return v, true
		}
	}
	return nil, false
}

// List returns all skills in deterministic alphabetical order.
func (r *SkillRegistry) List() []*Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.skills))
	for name := range r.skills {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]*Skill, 0, len(names))
	for _, name := range names {
		out = append(out, r.skills[name])
	}
	return out
}

// ListNames returns the sorted names of all registered skills.
func (r *SkillRegistry) ListNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.skills))
	for name := range r.skills {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Len returns the count of registered skills.
func (r *SkillRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.skills)
}

// Summary returns a concise human/model-readable overview of available skills.
func (r *SkillRegistry) Summary() string {
	skills := r.List()
	if len(skills) == 0 {
		return "No skills available."
	}
	var sb strings.Builder
	for i, s := range skills {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("- ")
		sb.WriteString(s.Name)
		if s.Description != "" {
			sb.WriteString(": ")
			sb.WriteString(s.Description)
		}
	}
	return sb.String()
}

// SkillToolDef returns the loop ToolDef for the native "skill" tool.
func SkillToolDef() ToolDef {
	return ToolDef{
		Type: "function",
		Function: ToolDefFunction{
			Name:        ToolSkill,
			Description: "Load a specialized skill when the task at hand matches one of the available skills. Injects the skill's instructions and procedures into context.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"name":{"type":"string","description":"The name of the skill to load"}},"required":["name"],"additionalProperties":false}`),
		},
	}
}

// ToolDef returns the tool definition for loading skills.
func (r *SkillRegistry) ToolDef() ToolDef {
	return SkillToolDef()
}

// Execute loads the requested skill and formats its instructions. If not found,
// returns a descriptive error listing the available skill names.
func (r *SkillRegistry) Execute(name string) (string, error) {
	s, ok := r.Get(name)
	if !ok {
		available := r.ListNames()
		if len(available) == 0 {
			return "", fmt.Errorf("skill %q not found (no skills available)", name)
		}
		return "", fmt.Errorf("skill %q not found; available skills: %s", name, strings.Join(available, ", "))
	}
	return s.Format(), nil
}

// ParseSkill parses a SKILL.md file content and resolves canonical paths.
func ParseSkill(content []byte, path string) (*Skill, error) {
	return parseSkillRaw(content, path, 0)
}

// ParseSkillFile reads and parses a SKILL.md file from disk.
func ParseSkillFile(path string) (*Skill, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseSkill(data, path)
}

func parseSkillRaw(content []byte, path string, depth int) (*Skill, error) {
	raw := string(content)
	lines := strings.Split(raw, "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("empty skill file")
	}

	firstLine := strings.TrimSpace(strings.TrimRight(lines[0], "\r"))
	if firstLine != "---" {
		return nil, fmt.Errorf("invalid frontmatter: expected opening '---', got %q", firstLine)
	}

	closingIdx := -1
	for i := 1; i < len(lines); i++ {
		line := strings.TrimSpace(strings.TrimRight(lines[i], "\r"))
		if line == "---" {
			closingIdx = i
			break
		}
	}
	if closingIdx == -1 {
		return nil, fmt.Errorf("invalid frontmatter: missing closing '---'")
	}

	fmLines := lines[1:closingIdx]
	bodyLines := lines[closingIdx+1:]
	body := strings.TrimSpace(strings.Join(bodyLines, "\n"))

	skill := &Skill{
		Path:     path,
		Metadata: make(map[string]string),
		Body:     body,
	}

	currentKey := ""
	for _, line := range fmLines {
		trimmed := strings.TrimRight(line, "\r")
		if len(trimmed) > 0 && (trimmed[0] == ' ' || trimmed[0] == '\t') {
			indentVal := strings.TrimSpace(trimmed)
			if indentVal == "" || strings.HasPrefix(indentVal, "#") {
				continue
			}
			switch currentKey {
			case "allowed-tools", "allowed_tools":
				if strings.HasPrefix(indentVal, "- ") || strings.HasPrefix(indentVal, "-\t") {
					item := strings.TrimSpace(indentVal[1:])
					item = stripInlineComment(item)
					dec, _ := frontmatter.DecodeScalar(item)
					dec = strings.TrimSpace(dec)
					if dec != "" {
						skill.AllowedTools = append(skill.AllowedTools, dec)
					}
				}
			case "metadata":
				k, v, ok := strings.Cut(indentVal, ":")
				if ok {
					k = strings.TrimSpace(k)
					v = stripInlineComment(strings.TrimSpace(v))
					decV, _ := frontmatter.DecodeScalar(v)
					skill.Metadata[k] = decV
				}
			case "description":
				if skill.Description != "" {
					skill.Description += " " + indentVal
				} else {
					skill.Description = indentVal
				}
			}
			continue
		}

		lineClean := strings.TrimSpace(trimmed)
		if lineClean == "" || strings.HasPrefix(lineClean, "#") {
			continue
		}

		key, val, ok := strings.Cut(lineClean, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = stripInlineComment(strings.TrimSpace(val))
		currentKey = key

		switch key {
		case "name":
			skill.Name, _ = frontmatter.DecodeScalar(val)
		case "description":
			skill.Description, _ = frontmatter.DecodeScalar(val)
		case "allowed-tools", "allowed_tools":
			if val != "" {
				skill.AllowedTools = parseToolsList(val)
			}
		case "metadata":
			// Nested indented keys follow.
		default:
			decV, _ := frontmatter.DecodeScalar(val)
			skill.Metadata[key] = decV
		}
	}

	// Canonical resolution: if metadata["canonical"] is present, resolve relative to the file's dir.
	if canonicalRel, ok := skill.Metadata["canonical"]; ok && strings.TrimSpace(canonicalRel) != "" && path != "" && depth < 5 {
		fileDir := filepath.Dir(path)
		canonicalAbs := filepath.Clean(filepath.Join(fileDir, canonicalRel))
		if data, err := os.ReadFile(canonicalAbs); err == nil {
			if canonicalSkill, err := parseSkillRaw(data, canonicalAbs, depth+1); err == nil {
				skill.Body = canonicalSkill.Body
				skill.Canonical = canonicalAbs
				if skill.Description == "" {
					skill.Description = canonicalSkill.Description
				}
				if len(skill.AllowedTools) == 0 {
					skill.AllowedTools = canonicalSkill.AllowedTools
				}
			}
		}
	}

	return skill, nil
}

func parseToolsList(val string) []string {
	val = strings.TrimSpace(val)
	if strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]") {
		val = strings.TrimSpace(val[1 : len(val)-1])
	}
	parts := strings.Split(val, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = stripInlineComment(p)
		p = strings.TrimSpace(p)
		dec, _ := frontmatter.DecodeScalar(p)
		dec = strings.TrimSpace(dec)
		if dec != "" {
			out = append(out, dec)
		}
	}
	return out
}

func stripInlineComment(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "'") || strings.HasPrefix(s, "\"") {
		return s
	}
	if idx := strings.Index(s, " #"); idx != -1 {
		return strings.TrimSpace(s[:idx])
	}
	if strings.HasPrefix(s, "#") {
		return ""
	}
	return s
}

// DiscoverSkills scans .agents/skills/ and .claude/skills/ under workspaceRoot, plus extraDirs.
// Subdirectories with SKILL.md are loaded and deduplicated by name, preferring canonical/complete bodies.
func DiscoverSkills(workspaceRoot string, extraDirs ...string) (*SkillRegistry, error) {
	if workspaceRoot == "" {
		if cwd, err := os.Getwd(); err == nil {
			workspaceRoot = cwd
		}
	}
	workspaceRoot = filepath.Clean(workspaceRoot)

	searchDirs := []string{
		filepath.Join(workspaceRoot, ".agents", "skills"),
		filepath.Join(workspaceRoot, ".claude", "skills"),
	}
	for _, d := range extraDirs {
		d = strings.TrimSpace(d)
		if d == "" {
			continue
		}
		if !filepath.IsAbs(d) {
			d = filepath.Join(workspaceRoot, d)
		}
		searchDirs = append(searchDirs, filepath.Clean(d))
	}

	discovered := make(map[string]*Skill)
	for _, dir := range searchDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			selfSkill := filepath.Join(dir, "SKILL.md")
			if s, err := ParseSkillFile(selfSkill); err == nil {
				addDiscoveredSkill(discovered, s)
			}
			continue
		}
		selfSkill := filepath.Join(dir, "SKILL.md")
		if s, err := ParseSkillFile(selfSkill); err == nil {
			addDiscoveredSkill(discovered, s)
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			skillPath := filepath.Join(dir, entry.Name(), "SKILL.md")
			s, err := ParseSkillFile(skillPath)
			if err != nil {
				skillPath = filepath.Join(dir, entry.Name(), "skill.md")
				s, err = ParseSkillFile(skillPath)
				if err != nil {
					continue
				}
			}
			if s.Name == "" {
				s.Name = entry.Name()
			}
			addDiscoveredSkill(discovered, s)
		}
	}

	reg := NewSkillRegistry()
	for _, s := range discovered {
		reg.Register(s)
	}
	return reg, nil
}

func addDiscoveredSkill(discovered map[string]*Skill, candidate *Skill) {
	if candidate == nil || candidate.Name == "" {
		return
	}
	existing, ok := discovered[candidate.Name]
	if !ok {
		discovered[candidate.Name] = candidate
		return
	}
	if isBetterSkill(candidate, existing) {
		discovered[candidate.Name] = candidate
	}
}

func isBetterSkill(candidate, existing *Skill) bool {
	candStub := isStubBody(candidate.Body)
	existStub := isStubBody(existing.Body)
	if existStub && !candStub {
		return true
	}
	if !existStub && candStub {
		return false
	}
	if existing.Canonical != "" && candidate.Canonical == "" {
		return true
	}
	if existing.Canonical == "" && candidate.Canonical != "" {
		return false
	}
	if len(candidate.Body) > len(existing.Body) {
		return true
	}
	return false
}

func isStubBody(body string) bool {
	lower := strings.ToLower(body)
	return strings.Contains(lower, "canonical project skill adapter") ||
		strings.Contains(lower, "contains no maintained workflow body")
}

// ---------------------------------------------------------------------------
// In-kernel driver implementation for ToolSkill.
// ---------------------------------------------------------------------------

type skillDriver struct{}

func (skillDriver) Caps() []abi.Capability { return nil }

func (skillDriver) WeightBearing() bool { return false }

func (skillDriver) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, m := decodeCallArgs(ctx, c.Args)
	name, _ := m["name"].(string)
	name = strings.TrimSpace(name)
	if name == "" {
		out, _ := json.Marshal(map[string]any{"error": "missing required field: name"})
		return engineResult(ctx, c, body, out, true, SkillDriverID), nil
	}
	reg := ActiveSkills()
	if reg == nil {
		out, _ := json.Marshal(map[string]any{"error": "no skill registry active"})
		return engineResult(ctx, c, body, out, true, SkillDriverID), nil
	}
	res, err := reg.Execute(name)
	if err != nil {
		out, _ := json.Marshal(map[string]any{"error": err.Error()})
		return engineResult(ctx, c, body, out, true, SkillDriverID), nil
	}
	return engineResult(ctx, c, body, []byte(res), false, SkillDriverID), nil
}

// RegisterSkillDriver registers the in-kernel engine driver for ToolSkill.
func RegisterSkillDriver() {
	abi.RegisterEngine(SkillDriverID, skillDriver{})
}

var (
	// armedSkills stores the active SkillRegistry for the loop.
	armedSkills atomic.Pointer[SkillRegistry]
)

// ActiveSkills returns the currently armed SkillRegistry, or nil.
func ActiveSkills() *SkillRegistry {
	return armedSkills.Load()
}

// ArmSkills sets the active SkillRegistry.
func ArmSkills(reg *SkillRegistry) {
	armedSkills.Store(reg)
}

// DisarmSkills clears the active SkillRegistry.
func DisarmSkills() {
	armedSkills.Store(nil)
}

func init() {
	RegisterSkillDriver()
}
