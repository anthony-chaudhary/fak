package evebridge

import (
	"encoding/json"
	"errors"
	"io/fs"
	"path"
	"sort"
	"strings"
)

const SchemaInspect = "fak-eve-inspect/1"

const (
	CodeLayoutUnsupported          = "EVE_LAYOUT_UNSUPPORTED"
	CodeNameCollision              = "EVE_RUNTIME_NAME_COLLISION"
	CodeRootOnlyMisplaced          = "EVE_ROOT_ONLY_MISPLACED"
	CodeManifestVersionUnsupported = "EVE_MANIFEST_VERSION_UNSUPPORTED"
	CodeManifestMalformed          = "EVE_MANIFEST_MALFORMED"
)

type InspectDiagnostic struct {
	Code         string `json:"code"`
	Severity     string `json:"severity"`
	Message      string `json:"message"`
	EvidencePath string `json:"evidence_path,omitempty"`
}
type Surface struct {
	Name string `json:"name"`
	Path string `json:"path"`
}
type BuiltinTool struct {
	Name   string `json:"name"`
	Action string `json:"action"`
	Path   string `json:"path,omitempty"`
}
type SandboxMount struct {
	RuntimePath string `json:"runtime_path"`
	SourcePath  string `json:"source_path"`
	ReadOnly    bool   `json:"read_only"`
}
type PolicyImplications struct {
	DefaultDeny            bool     `json:"default_deny"`
	Shell                  bool     `json:"shell"`
	File                   bool     `json:"file"`
	Network                bool     `json:"network"`
	WriteRoots             []string `json:"write_roots,omitempty"`
	ConnectionToolPatterns []string `json:"connection_tool_patterns,omitempty"`
	RootOnlySurfaces       []string `json:"root_only_surfaces,omitempty"`
}
type InspectManifest struct {
	Schema        string              `json:"schema"`
	OK            bool                `json:"ok"`
	SourceMode    string              `json:"source_mode,omitempty"`
	Tools         []Surface           `json:"tools,omitempty"`
	Connections   []Surface           `json:"connections,omitempty"`
	Subagents     []Surface           `json:"subagents,omitempty"`
	Schedules     []Surface           `json:"schedules,omitempty"`
	Channels      []Surface           `json:"channels,omitempty"`
	EvalIDs       []Surface           `json:"eval_ids,omitempty"`
	BuiltinTools  []BuiltinTool       `json:"builtin_tools,omitempty"`
	SandboxMounts []SandboxMount      `json:"sandbox_mounts,omitempty"`
	Policy        PolicyImplications  `json:"policy_implications"`
	Diagnostics   []InspectDiagnostic `json:"diagnostics,omitempty"`
}

func (m InspectManifest) JSON() []byte {
	b, _ := json.MarshalIndent(m, "", "  ")
	return append(b, '\n')
}

type PolicyDraft struct {
	Version  string            `json:"version"`
	Posture  string            `json:"posture"`
	Allow    []string          `json:"allow,omitempty"`
	ArgRules []PolicyArgRule   `json:"arg_rules,omitempty"`
	Sources  map[string]string `json:"sources,omitempty"`
}
type PolicyArgRule struct {
	Tool      string `json:"tool"`
	Arg       string `json:"arg"`
	AllowGlob string `json:"allow_glob"`
	Reason    string `json:"reason,omitempty"`
}

func (p PolicyDraft) JSON() []byte { b, _ := json.MarshalIndent(p, "", "  "); return append(b, '\n') }
func (m InspectManifest) PolicyDraft() (PolicyDraft, error) {
	if !m.OK {
		return PolicyDraft{}, errors.New("cannot draft policy from failed Eve inspection")
	}
	p := PolicyDraft{Version: "fak-policy/v1", Posture: "fail_closed", Sources: map[string]string{}}
	for _, t := range m.Tools {
		p.Allow = append(p.Allow, t.Name)
		p.Sources[t.Name] = "trusted_local"
	}
	for _, mount := range m.SandboxMounts {
		if !mount.ReadOnly {
			p.Allow = append(p.Allow, "write_file")
			p.ArgRules = append(p.ArgRules, PolicyArgRule{Tool: "write_file", Arg: "path", AllowGlob: strings.TrimSuffix(mount.RuntimePath, "/") + "/**", Reason: "POLICY_BLOCK"})
		}
	}
	sort.Strings(p.Allow)
	return p, nil
}

func InspectFS(root fs.FS) (InspectManifest, error) {
	m := InspectManifest{Schema: SchemaInspect, Policy: PolicyImplications{DefaultDeny: true, RootOnlySurfaces: []string{"connections", "schedules", "channels", "sandbox"}}}
	if _, err := fs.Stat(root, "."); err != nil {
		return m, err
	}
	if exists(root, "agent/agent.ts") || exists(root, "agent/agent.js") {
		m.SourceMode = "source"
		inspectSource(root, &m)
	} else if p := compiledManifestPath(root); p != "" {
		m.SourceMode = "compiled"
		inspectCompiled(root, p, &m)
	} else {
		fail(&m, CodeLayoutUnsupported, "expected agent/agent.ts or a compiled Eve manifest", ".")
	}
	sortManifest(&m)
	m.OK = !hasFailure(m.Diagnostics)
	return m, nil
}
func inspectSource(root fs.FS, m *InspectManifest) {
	maps := map[string]map[string]string{"tools": {}, "connections": {}, "subagents": {}, "schedules": {}, "channels": {}, "evals": {}}
	_ = fs.WalkDir(root, "agent", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		clean := strings.TrimSuffix(strings.TrimSuffix(p, path.Ext(p)), "/index")
		switch {
		case strings.HasPrefix(p, "agent/tools/") && sourceFile(p):
			addSurface(m, &m.Tools, runtimeName(strings.TrimPrefix(clean, "agent/tools/")), p, maps["tools"])
		case strings.HasPrefix(p, "agent/subagents/") && strings.Contains(p, "/tools/") && sourceFile(p):
			rel := strings.TrimPrefix(clean, "agent/subagents/")
			parts := strings.SplitN(rel, "/tools/", 2)
			if len(parts) == 2 {
				addSurface(m, &m.Tools, runtimeName(parts[0]+"/"+parts[1]), p, maps["tools"])
			}
		case strings.HasPrefix(p, "agent/connections/") && sourceFile(p):
			n := runtimeName(strings.TrimPrefix(clean, "agent/connections/"))
			addSurface(m, &m.Connections, n, p, maps["connections"])
			m.Policy.Network = true
			m.Policy.ConnectionToolPatterns = append(m.Policy.ConnectionToolPatterns, n+"__*")
		case strings.HasPrefix(p, "agent/subagents/") && (strings.HasSuffix(p, "/agent.ts") || strings.HasSuffix(p, "/agent.js")):
			n := strings.TrimSuffix(strings.TrimSuffix(strings.TrimPrefix(p, "agent/subagents/"), "/agent.ts"), "/agent.js")
			addSurface(m, &m.Subagents, runtimeName(n), p, maps["subagents"])
		case strings.HasPrefix(p, "agent/schedules/"):
			addSurface(m, &m.Schedules, runtimeName(strings.TrimPrefix(clean, "agent/schedules/")), p, maps["schedules"])
		case strings.HasPrefix(p, "agent/channels/") && sourceFile(p):
			addSurface(m, &m.Channels, runtimeName(strings.TrimPrefix(clean, "agent/channels/")), p, maps["channels"])
		case strings.HasPrefix(p, "agent/evals/"):
			addSurface(m, &m.EvalIDs, runtimeName(strings.TrimPrefix(clean, "agent/evals/")), p, maps["evals"])
		}
		if strings.HasPrefix(p, "agent/subagents/") && (strings.Contains(p, "/schedules/") || strings.Contains(p, "/connections/") || strings.Contains(p, "/channels/") || strings.Contains(p, "/sandbox/")) {
			fail(m, CodeRootOnlyMisplaced, "root-only Eve surface appears under a subagent", p)
		}
		return nil
	})
	if exists(root, "agent/sandbox/workspace") {
		m.SandboxMounts = append(m.SandboxMounts, SandboxMount{RuntimePath: "/workspace", SourcePath: "agent/sandbox/workspace"})
		m.Policy.File = true
		m.Policy.WriteRoots = append(m.Policy.WriteRoots, "/workspace")
	}
	m.Policy.Shell = exists(root, "agent/sandbox/sandbox.ts") || exists(root, "agent/sandbox/sandbox.js")
}

type rawCompiled struct {
	Kind                   string            `json:"kind"`
	Version                json.RawMessage   `json:"version"`
	Tools                  []json.RawMessage `json:"tools"`
	Connections            []json.RawMessage `json:"connections"`
	Subagents              []json.RawMessage `json:"subagents"`
	Schedules              []json.RawMessage `json:"schedules"`
	Channels               []json.RawMessage `json:"channels"`
	Evals                  []json.RawMessage `json:"eval_ids"`
	SandboxWorkspaces      []json.RawMessage `json:"sandboxWorkspaces"`
	LegacyMounts           []json.RawMessage `json:"workspace_mounts"`
	DisabledFrameworkTools []string          `json:"disabledFrameworkTools"`
}

func inspectCompiled(root fs.FS, p string, m *InspectManifest) {
	b, err := fs.ReadFile(root, p)
	if err != nil {
		fail(m, CodeManifestMalformed, err.Error(), p)
		return
	}
	var d rawCompiled
	if json.Unmarshal(b, &d) != nil {
		fail(m, CodeManifestMalformed, "compiled manifest is not valid JSON", p)
		return
	}
	v := rawVersion(d.Version)
	if !supportedVersion(d.Kind, v) {
		fail(m, CodeManifestVersionUnsupported, "unsupported Eve manifest version "+v, p)
		return
	}
	toolNames := map[string]string{}
	addEntries(m, &m.Tools, d.Tools, "name", p, toolNames)
	addEntries(m, &m.Connections, d.Connections, "connectionName", p, map[string]string{})
	addEntries(m, &m.Schedules, d.Schedules, "name", p, map[string]string{})
	addEntries(m, &m.Channels, d.Channels, "name", p, map[string]string{})
	addEntries(m, &m.EvalIDs, d.Evals, "name", p, map[string]string{})
	for _, c := range m.Connections {
		m.Policy.Network = true
		m.Policy.ConnectionToolPatterns = append(m.Policy.ConnectionToolPatterns, c.Name+"__*")
	}
	subs := map[string]string{}
	for _, r := range d.Subagents {
		n, e, ok := entry(r, "name", p)
		if !ok {
			fail(m, CodeManifestMalformed, "subagent entry has no runtime identity", p)
			continue
		}
		n = runtimeName(n)
		addSurface(m, &m.Subagents, n, e, subs)
		var x struct {
			Agent struct {
				Tools []json.RawMessage `json:"tools"`
			} `json:"agent"`
		}
		_ = json.Unmarshal(r, &x)
		for _, tr := range x.Agent.Tools {
			tn, te, ok := entry(tr, "name", e)
			if ok {
				addSurface(m, &m.Tools, runtimeName(n+"/"+tn), te, toolNames)
			}
		}
	}
	for _, r := range append(d.SandboxWorkspaces, d.LegacyMounts...) {
		src, runtime, ro, ok := mountEntry(r)
		if !ok {
			fail(m, CodeManifestMalformed, "sandbox workspace has no source path", p)
			continue
		}
		m.SandboxMounts = append(m.SandboxMounts, SandboxMount{RuntimePath: runtime, SourcePath: src, ReadOnly: ro})
		m.Policy.File = true
		if !ro {
			m.Policy.WriteRoots = append(m.Policy.WriteRoots, runtime)
		}
	}
	for _, n := range d.DisabledFrameworkTools {
		action := "disable"
		if _, ok := toolNames[runtimeName(n)]; ok {
			action = "override"
		}
		m.BuiltinTools = append(m.BuiltinTools, BuiltinTool{Name: n, Action: action, Path: p})
	}
}
func addEntries(m *InspectManifest, dst *[]Surface, raw []json.RawMessage, key, p string, names map[string]string) {
	for _, r := range raw {
		n, e, ok := entry(r, key, p)
		if !ok {
			fail(m, CodeManifestMalformed, key+" entry has no runtime identity", p)
			continue
		}
		addSurface(m, dst, runtimeName(n), e, names)
	}
}
func entry(raw json.RawMessage, key, fallback string) (string, string, bool) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, fallback, s != ""
	}
	var x map[string]any
	if json.Unmarshal(raw, &x) != nil {
		return "", fallback, false
	}
	n := first(x, key, "name", "connectionName", "subagentId", "logicalPath", "sourceId")
	e := first(x, "logicalPath", "sourcePath", "entryPath", "rootPath", "sourceId")
	if e == "" {
		e = fallback
	}
	return n, e, n != ""
}
func mountEntry(raw json.RawMessage) (string, string, bool, bool) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, "/workspace", false, s != ""
	}
	var x map[string]any
	if json.Unmarshal(raw, &x) != nil {
		return "", "", false, false
	}
	src := first(x, "sourcePath", "logicalPath")
	runtime := first(x, "runtimePath", "runtime_path")
	if runtime == "" {
		runtime = "/workspace"
	}
	ro, _ := x["readOnly"].(bool)
	if v, ok := x["read_only"].(bool); ok {
		ro = v
	}
	return src, runtime, ro, src != ""
}
func first(x map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := x[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
}
func rawVersion(r json.RawMessage) string {
	var v any
	if json.Unmarshal(r, &v) != nil {
		return "<malformed>"
	}
	return strings.TrimSpace(toString(v))
}
func supportedVersion(kind, v string) bool {
	switch kind {
	case "eve-agent-compiled-manifest":
		return v == "36" || v == "1"
	case "eve-agent-discovery-manifest":
		return v == "12" || v == "1"
	case "":
		return v == "1" || v == "v1"
	default:
		return false
	}
}
func addSurface(m *InspectManifest, dst *[]Surface, n, p string, names map[string]string) {
	if old, ok := names[n]; ok && old != p {
		fail(m, CodeNameCollision, "path-derived runtime name collides with "+old, p)
		return
	}
	names[n] = p
	*dst = append(*dst, Surface{Name: n, Path: p})
}
func fail(m *InspectManifest, c, msg, p string) {
	m.Diagnostics = append(m.Diagnostics, InspectDiagnostic{Code: c, Severity: "fail", Message: msg, EvidencePath: p})
}
func sortManifest(m *InspectManifest) {
	sortSurfaces := func(s []Surface) {
		sort.Slice(s, func(i, j int) bool {
			if s[i].Name == s[j].Name {
				return s[i].Path < s[j].Path
			}
			return s[i].Name < s[j].Name
		})
	}
	sortSurfaces(m.Tools)
	sortSurfaces(m.Connections)
	sortSurfaces(m.Subagents)
	sortSurfaces(m.Schedules)
	sortSurfaces(m.Channels)
	sortSurfaces(m.EvalIDs)
	sort.Slice(m.BuiltinTools, func(i, j int) bool { return m.BuiltinTools[i].Name < m.BuiltinTools[j].Name })
	sort.Slice(m.SandboxMounts, func(i, j int) bool { return m.SandboxMounts[i].SourcePath < m.SandboxMounts[j].SourcePath })
	sort.Slice(m.Diagnostics, func(i, j int) bool {
		if m.Diagnostics[i].Code == m.Diagnostics[j].Code {
			return m.Diagnostics[i].EvidencePath < m.Diagnostics[j].EvidencePath
		}
		return m.Diagnostics[i].Code < m.Diagnostics[j].Code
	})
	sort.Strings(m.Policy.ConnectionToolPatterns)
	sort.Strings(m.Policy.WriteRoots)
}
func sourceFile(p string) bool {
	e := path.Ext(p)
	return e == ".ts" || e == ".js" || e == ".tsx" || e == ".jsx"
}
func runtimeName(s string) string {
	s = strings.Trim(s, "/")
	s = strings.ReplaceAll(s, "/", "__")
	s = strings.ReplaceAll(s, "-", "_")
	return s
}
func exists(root fs.FS, p string) bool { _, err := fs.Stat(root, p); return err == nil }
func compiledManifestPath(root fs.FS) string {
	for _, p := range []string{".eve/compile/compiled-agent-manifest.json", ".eve/discovery/agent-discovery-manifest.json", "compile/compiled-agent-manifest.json", "discovery/agent-discovery-manifest.json"} {
		if exists(root, p) {
			return p
		}
	}
	return ""
}
func hasFailure(ds []InspectDiagnostic) bool {
	for _, d := range ds {
		if d.Severity == "fail" {
			return true
		}
	}
	return false
}
func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strings.TrimSuffix(strings.TrimSuffix(jsonNumber(x), "0"), ".")
	default:
		return ""
	}
}
func jsonNumber(v float64) string { b, _ := json.Marshal(v); return string(b) }
