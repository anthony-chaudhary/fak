package agent

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/frontmatter"
	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// Mode constants for declarative agent descriptors.
const (
	AgentModePrimary  = "primary"
	AgentModeSubagent = "subagent"
)

// Model tier aliases.
const (
	ModelTier1 = "tier1"
	ModelTier2 = "tier2"
	ModelTier3 = "tier3"
)

// Variant options for reasoning/output style.
const (
	VariantDefault  = "default"
	VariantHigh     = "high"
	VariantAdaptive = "adaptive"
)

// DefaultDescriptorMaxTurns is the default turn budget if unspecified.
const DefaultDescriptorMaxTurns = 10

// ErrAuthorityWidened is returned when a child subagent capability envelope
// attempts to widen the authority granted by its parent.
var ErrAuthorityWidened = errors.New("agent: child capability envelope widens parent authority")

// AgentCapabilities defines the bounded capability envelope granted to an agent.
type AgentCapabilities struct {
	Tools         []string `json:"tools,omitempty"`
	Paths         []string `json:"paths,omitempty"`
	AllowMutation bool     `json:"allow_mutation"`
}

// Clone returns a deep copy of the capability envelope.
func (c AgentCapabilities) Clone() AgentCapabilities {
	return AgentCapabilities{
		Tools:         append([]string(nil), c.Tools...),
		Paths:         append([]string(nil), c.Paths...),
		AllowMutation: c.AllowMutation,
	}
}

// AgentDescriptor represents a parsed declarative agent specification loaded from
// a markdown descriptor file (.fak/agents/*.md or .agents/*.md).
type AgentDescriptor struct {
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	Mode         string            `json:"mode"`      // "primary" | "subagent"
	Model        string            `json:"model"`     // model name or tier1|tier2|tier3
	Variant      string            `json:"variant"`   // "default" | "high" | "adaptive"
	MaxTurns     int               `json:"max_turns"` // maximum turn budget
	Capabilities AgentCapabilities `json:"capabilities"`
	Prompt       string            `json:"prompt"` // markdown persona system prompt overlay
	Path         string            `json:"path"`   // origin file path
}

// FormatPrompt renders the agent descriptor as a persona system prompt overlay.
func (d *AgentDescriptor) FormatPrompt() string {
	var sb strings.Builder
	sb.WriteString("# Agent Persona: ")
	sb.WriteString(d.Name)
	if d.Mode != "" {
		sb.WriteString(" (")
		sb.WriteString(d.Mode)
		sb.WriteString(")")
	}
	sb.WriteString("\n\n")

	if d.Description != "" {
		sb.WriteString(d.Description)
		sb.WriteString("\n\n")
	}

	sb.WriteString("## Capability Envelope\n")
	sb.WriteString(fmt.Sprintf("- Model: %s\n", d.Model))
	sb.WriteString(fmt.Sprintf("- Variant: %s\n", d.Variant))
	sb.WriteString(fmt.Sprintf("- Max Turns: %d\n", d.MaxTurns))
	sb.WriteString(fmt.Sprintf("- Allow Mutation: %t\n", d.Capabilities.AllowMutation))

	if len(d.Capabilities.Tools) > 0 {
		sb.WriteString(fmt.Sprintf("- Authorized Tools: %s\n", strings.Join(d.Capabilities.Tools, ", ")))
	}
	if len(d.Capabilities.Paths) > 0 {
		sb.WriteString(fmt.Sprintf("- Scoped Paths: %s\n", strings.Join(d.Capabilities.Paths, ", ")))
	}
	sb.WriteString("\n")

	if d.Prompt != "" {
		sb.WriteString("## Persona Instructions\n\n")
		sb.WriteString(d.Prompt)
		sb.WriteString("\n")
	}

	return sb.String()
}

// PromptOverlayBytes returns the formatted persona prompt as raw bytes for overlay injection.
func (d *AgentDescriptor) PromptOverlayBytes() []byte {
	return []byte(d.FormatPrompt())
}

// AsPromptEdit wraps the descriptor's formatted prompt overlay as a syspromptmmu.BaseEdit
// targeting TierOverlay with EditAdd.
func (d *AgentDescriptor) AsPromptEdit() syspromptmmu.BaseEdit {
	return syspromptmmu.BaseEdit{
		Op:      syspromptmmu.EditAdd,
		Tier:    syspromptmmu.TierOverlay,
		Content: d.PromptOverlayBytes(),
		Version: "v1",
	}
}

// BuildSystemBlock creates a complete SystemBlock incorporating fak's owned base spine
// and this descriptor's persona prompt overlay.
func (d *AgentDescriptor) BuildSystemBlock(additionalOverlays [][]byte, witness func(syspromptmmu.BaseEdit) bool) SystemBlock {
	items := make([][]byte, 0, len(additionalOverlays)+1)
	items = append(items, d.PromptOverlayBytes())
	items = append(items, additionalOverlays...)
	return BuildOwnedSystemBlock(items, witness)
}

// ApplyToSegments applies the descriptor's persona overlay edit to an existing syspromptmmu.Segment collection.
func (d *AgentDescriptor) ApplyToSegments(segments []syspromptmmu.Segment, witness func(syspromptmmu.BaseEdit) bool) ([]syspromptmmu.Segment, syspromptmmu.EditVerdict) {
	return syspromptmmu.ApplyEdit(segments, d.AsPromptEdit(), witness)
}

// Narrow returns a copy of the descriptor with its capabilities monotonically narrowed
// by parent capabilities and turn budget capped.
func (d *AgentDescriptor) Narrow(parent AgentCapabilities, maxTurnsBudget ...int) *AgentDescriptor {
	cloned := *d
	cloned.Capabilities = IntersectCapabilities(parent, d.Capabilities)
	if len(maxTurnsBudget) > 0 && maxTurnsBudget[0] > 0 {
		if cloned.MaxTurns <= 0 || cloned.MaxTurns > maxTurnsBudget[0] {
			cloned.MaxTurns = maxTurnsBudget[0]
		}
	}
	return &cloned
}

// NarrowWithParent narrows this descriptor against a parent descriptor.
func (d *AgentDescriptor) NarrowWithParent(parent *AgentDescriptor) (*AgentDescriptor, error) {
	if parent == nil {
		return d, nil
	}
	budget := parent.MaxTurns
	return d.Narrow(parent.Capabilities, budget), nil
}

// IntersectCapabilities monotonically narrows requested child capabilities to the authority of parent.
// - Mutation: child can mutate only if parent allows mutation.
// - Tools: if parent specifies tools, child tools are restricted to parent's tools.
// - Paths: if parent specifies paths, child paths must fall within parent's scope.
func IntersectCapabilities(parent, requested AgentCapabilities) AgentCapabilities {
	out := AgentCapabilities{
		AllowMutation: parent.AllowMutation && requested.AllowMutation,
	}

	if len(parent.Tools) == 0 {
		out.Tools = append([]string(nil), requested.Tools...)
	} else {
		parentSet := make(map[string]bool, len(parent.Tools))
		for _, t := range parent.Tools {
			parentSet[t] = true
		}
		for _, t := range requested.Tools {
			if parentSet[t] {
				out.Tools = append(out.Tools, t)
			}
		}
	}

	if len(parent.Paths) == 0 {
		out.Paths = append([]string(nil), requested.Paths...)
	} else {
		for _, cp := range requested.Paths {
			for _, pp := range parent.Paths {
				if pathWithin(pp, cp) {
					out.Paths = append(out.Paths, cp)
					break
				}
			}
		}
	}

	return out
}

// ValidateChildCapabilities returns an error if child requests authority not granted to parent.
func ValidateChildCapabilities(parent, child AgentCapabilities) error {
	if !parent.AllowMutation && child.AllowMutation {
		return fmt.Errorf("%w: child requested allow_mutation=true but parent is false", ErrAuthorityWidened)
	}

	if len(parent.Tools) > 0 {
		parentSet := make(map[string]bool, len(parent.Tools))
		for _, t := range parent.Tools {
			parentSet[t] = true
		}
		for _, t := range child.Tools {
			if !parentSet[t] {
				return fmt.Errorf("%w: child requested tool %q not granted to parent", ErrAuthorityWidened, t)
			}
		}
	}

	if len(parent.Paths) > 0 && len(child.Paths) > 0 {
		for _, cp := range child.Paths {
			allowed := false
			for _, pp := range parent.Paths {
				if pathWithin(pp, cp) {
					allowed = true
					break
				}
			}
			if !allowed {
				return fmt.Errorf("%w: child requested path %q outside parent scope", ErrAuthorityWidened, cp)
			}
		}
	}

	return nil
}

func pathWithin(root, requested string) bool {
	root = filepath.Clean(strings.TrimSpace(root))
	requested = filepath.Clean(strings.TrimSpace(requested))
	if root == "." || root == "*" || root == "**" || root == requested {
		return true
	}

	rootNorm := filepath.ToSlash(root)
	reqNorm := filepath.ToSlash(requested)

	rootTrim := strings.TrimSuffix(rootNorm, "/**")
	rootTrim = strings.TrimSuffix(rootTrim, "/*")
	rootTrim = strings.TrimSuffix(rootTrim, "/")
	if rootTrim == "" || rootTrim == "." {
		return true
	}

	if reqNorm == rootTrim || strings.HasPrefix(reqNorm, rootTrim+"/") {
		return true
	}

	rel, err := filepath.Rel(filepath.FromSlash(rootTrim), filepath.FromSlash(reqNorm))
	if err != nil || filepath.IsAbs(rel) {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// ParseAgentDescriptor parses an agent markdown document with YAML frontmatter.
func ParseAgentDescriptor(content []byte, path string) (*AgentDescriptor, error) {
	raw := strings.ReplaceAll(string(content), "\r\n", "\n")
	lines := strings.Split(raw, "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("agent descriptor %s: empty content", path)
	}

	firstLine := strings.TrimSpace(lines[0])
	if firstLine != "---" {
		return nil, fmt.Errorf("agent descriptor %s: expected opening '---', got %q", path, firstLine)
	}

	closingIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closingIdx = i
			break
		}
	}
	if closingIdx == -1 {
		return nil, fmt.Errorf("agent descriptor %s: missing closing '---'", path)
	}

	fmLines := lines[1:closingIdx]
	bodyLines := lines[closingIdx+1:]
	prompt := strings.TrimSpace(strings.Join(bodyLines, "\n"))

	desc := &AgentDescriptor{
		Path:   path,
		Prompt: prompt,
	}

	var currentTopKey string
	var currentSubKey string

	for _, line := range fmLines {
		trimmedLine := strings.TrimRight(line, "\r")
		trimmed := strings.TrimSpace(trimmedLine)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		indent := len(trimmedLine) - len(strings.TrimLeft(trimmedLine, " \t"))

		if indent == 0 {
			key, val, ok := strings.Cut(trimmed, ":")
			if !ok {
				continue
			}
			key = strings.ToLower(strings.TrimSpace(key))
			val = strings.TrimSpace(val)
			currentTopKey = key
			currentSubKey = ""

			switch key {
			case "name":
				desc.Name, _ = frontmatter.DecodeScalar(val)
			case "description":
				desc.Description, _ = frontmatter.DecodeScalar(val)
			case "mode":
				dec, _ := frontmatter.DecodeScalar(val)
				desc.Mode = strings.ToLower(strings.TrimSpace(dec))
			case "model":
				desc.Model, _ = frontmatter.DecodeScalar(val)
			case "variant":
				dec, _ := frontmatter.DecodeScalar(val)
				desc.Variant = strings.ToLower(strings.TrimSpace(dec))
			case "max_turns", "max-turns", "maxturns":
				val = stripInlineComment(val)
				dec, _ := frontmatter.DecodeScalar(val)
				if n, err := strconv.Atoi(strings.TrimSpace(dec)); err == nil {
					desc.MaxTurns = n
				}
			case "tools":
				if val != "" {
					desc.Capabilities.Tools = append(desc.Capabilities.Tools, parseStringList(val)...)
				}
			case "paths":
				if val != "" {
					desc.Capabilities.Paths = append(desc.Capabilities.Paths, parseStringList(val)...)
				}
			case "allow_mutation", "allow-mutation", "allowmutation":
				desc.Capabilities.AllowMutation = parseBool(val)
			case "capabilities":
				// Sub-keys follow on subsequent indented lines
			}
		} else {
			// Indented line
			if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "-\t") || strings.HasPrefix(trimmed, "-") {
				item := parseListItem(trimmed)
				if item != "" {
					if currentTopKey == "capabilities" {
						switch currentSubKey {
						case "tools":
							desc.Capabilities.Tools = append(desc.Capabilities.Tools, item)
						case "paths":
							desc.Capabilities.Paths = append(desc.Capabilities.Paths, item)
						}
					} else if currentTopKey == "tools" {
						desc.Capabilities.Tools = append(desc.Capabilities.Tools, item)
					} else if currentTopKey == "paths" {
						desc.Capabilities.Paths = append(desc.Capabilities.Paths, item)
					}
				}
				continue
			}

			subKey, subVal, ok := strings.Cut(trimmed, ":")
			if ok {
				subKey = strings.ToLower(strings.TrimSpace(subKey))
				subVal = strings.TrimSpace(subVal)
				if currentTopKey == "capabilities" {
					currentSubKey = subKey
					switch subKey {
					case "tools":
						if subVal != "" {
							desc.Capabilities.Tools = append(desc.Capabilities.Tools, parseStringList(subVal)...)
						}
					case "paths":
						if subVal != "" {
							desc.Capabilities.Paths = append(desc.Capabilities.Paths, parseStringList(subVal)...)
						}
					case "allow_mutation", "allow-mutation", "allowmutation":
						desc.Capabilities.AllowMutation = parseBool(subVal)
					}
				} else if currentTopKey == "description" {
					if desc.Description != "" {
						desc.Description += " " + trimmed
					} else {
						desc.Description = trimmed
					}
				}
			} else if currentTopKey == "description" {
				if desc.Description != "" {
					desc.Description += " " + trimmed
				} else {
					desc.Description = trimmed
				}
			}
		}
	}

	if desc.Name == "" && path != "" {
		base := filepath.Base(path)
		desc.Name = strings.TrimSuffix(base, filepath.Ext(base))
	}
	if desc.Mode == "" {
		desc.Mode = AgentModeSubagent
	}
	if desc.Model == "" {
		desc.Model = ModelTier1
	}
	if desc.Variant == "" {
		desc.Variant = VariantDefault
	}
	if desc.MaxTurns <= 0 {
		desc.MaxTurns = DefaultDescriptorMaxTurns
	}

	desc.Capabilities.Tools = dedupeStrings(desc.Capabilities.Tools)
	desc.Capabilities.Paths = dedupeStrings(desc.Capabilities.Paths)

	return desc, nil
}

// LoadAgentDescriptor reads and parses an agent descriptor from a markdown file path.
func LoadAgentDescriptor(path string) (*AgentDescriptor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load agent descriptor %s: %w", path, err)
	}
	return ParseAgentDescriptor(data, path)
}

// ScanAgentDescriptors reads all markdown files (*.md) in dir and parses valid agent descriptors.
func ScanAgentDescriptors(dir string) ([]*AgentDescriptor, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := make([]*AgentDescriptor, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".md") {
			continue
		}
		if strings.EqualFold(name, "readme.md") || strings.EqualFold(name, "skill.md") {
			continue
		}
		path := filepath.Join(dir, name)
		desc, err := LoadAgentDescriptor(path)
		if err != nil {
			continue
		}
		out = append(out, desc)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// FindWorkspaceAgentDescriptors scans canonical workspace directories (.fak/agents/*.md and .agents/*.md)
// for declarative agent descriptors, deduplicating by descriptor name (.fak/agents precedence).
func FindWorkspaceAgentDescriptors(workspace string) ([]*AgentDescriptor, error) {
	if workspace == "" {
		workspace = "."
	}
	scanDirs := []string{
		filepath.Join(workspace, ".fak", "agents"),
		filepath.Join(workspace, ".agents"),
		filepath.Join(workspace, ".agents", "agents"),
	}

	seen := make(map[string]bool)
	out := make([]*AgentDescriptor, 0)

	for _, dir := range scanDirs {
		descs, err := ScanAgentDescriptors(dir)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		for _, d := range descs {
			key := strings.ToLower(d.Name)
			if !seen[key] {
				seen[key] = true
				out = append(out, d)
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// AgentDescriptorRegistry holds discovered agent descriptors with thread-safe access.
type AgentDescriptorRegistry struct {
	mu          sync.RWMutex
	descriptors map[string]*AgentDescriptor
}

// NewAgentDescriptorRegistry creates an empty AgentDescriptorRegistry.
func NewAgentDescriptorRegistry() *AgentDescriptorRegistry {
	return &AgentDescriptorRegistry{
		descriptors: make(map[string]*AgentDescriptor),
	}
}

// Register adds or updates an agent descriptor in the registry.
func (r *AgentDescriptorRegistry) Register(d *AgentDescriptor) {
	if d == nil || d.Name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.descriptors[strings.ToLower(d.Name)] = d
}

// Get looks up an agent descriptor by name (case-insensitive).
func (r *AgentDescriptorRegistry) Get(name string) (*AgentDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.descriptors[strings.ToLower(name)]
	return d, ok
}

// List returns all registered descriptors in deterministic alphabetical order.
func (r *AgentDescriptorRegistry) List() []*AgentDescriptor {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*AgentDescriptor, 0, len(r.descriptors))
	for _, d := range r.descriptors {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

// Len returns the number of registered descriptors.
func (r *AgentDescriptorRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.descriptors)
}

func parseListItem(line string) string {
	line = strings.TrimSpace(line)
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "-\t") {
		line = strings.TrimSpace(line[2:])
	} else if strings.HasPrefix(line, "-") {
		line = strings.TrimSpace(line[1:])
	}
	line = stripInlineComment(line)
	dec, _ := frontmatter.DecodeScalar(line)
	return strings.TrimSpace(dec)
}

func parseStringList(val string) []string {
	val = strings.TrimSpace(val)
	if strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]") {
		val = strings.TrimSpace(val[1 : len(val)-1])
	}
	if val == "" {
		return nil
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

func parseBool(val string) bool {
	val = strings.TrimSpace(val)
	val = stripInlineComment(val)
	dec, _ := frontmatter.DecodeScalar(val)
	dec = strings.ToLower(strings.TrimSpace(dec))
	return dec == "true" || dec == "1" || dec == "yes" || dec == "on"
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return in
	}
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}
