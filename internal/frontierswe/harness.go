package frontierswe

import (
	"fmt"
	"sort"
	"strings"
)

// FrontierHarness is one FrontierSWE CLI harness class that fak can route
// through its gateway with only the model base URL overridden. FrontierSWE's
// job.yaml drives the same task through several such harbor_ext agent classes;
// routing each through the FakRoutedAgent proves the time-to-solution win is a
// property of the serving substrate, not of any one CLI (issue #1725, epic
// #1706 C19). The only per-harness delta the routing introduces is the base
// URL — model, budget, and no-internet boundary are unchanged.
type FrontierHarness struct {
	// Name is the FrontierSWE job.yaml agent short name (e.g. "codex").
	Name string
	// ImportPath is the harbor_ext agent class the FakRoutedAgent wraps.
	ImportPath string
}

// frontierHarnesses is the set of harness classes wired for fak routing.
// claude-code and codex are the classes named in the committed FrontierSWE
// job.yaml / issue #1725; gemini-cli and qwen-code follow the documented
// harbor_ext.<harness>:<Class>ApiKeyNoSearch convention. kimi-cli and opencode
// are NOT yet wired — see HarnessCoverage for the honest remaining set.
var frontierHarnesses = []FrontierHarness{
	{Name: "claude-code", ImportPath: DefaultWrappedAgent},
	{Name: "codex", ImportPath: "harbor_ext.codex:CodexApiKeyNoSearch"},
	{Name: "gemini-cli", ImportPath: "harbor_ext.gemini_cli:GeminiCliApiKeyNoSearch"},
	{Name: "qwen-code", ImportPath: "harbor_ext.qwen_code:QwenCodeApiKeyNoSearch"},
}

// frontierHarnessRemaining are FrontierSWE harness classes not yet wired for fak
// routing — kept explicit so coverage is honest, not a blanket "all harnesses"
// claim. FrontierSWE's job.yaml can name six harness classes; four are wired.
var frontierHarnessRemaining = []string{"kimi-cli", "opencode"}

// FrontierHarnesses returns the harness classes wired for fak routing, in order.
func FrontierHarnesses() []FrontierHarness {
	out := make([]FrontierHarness, len(frontierHarnesses))
	copy(out, frontierHarnesses)
	return out
}

// WrappedAgentForHarness resolves a FrontierSWE harness short name to the
// harbor_ext class the FakRoutedAgent wraps. The lookup is case-insensitive and
// treats "gemini_cli" and "gemini-cli" as the same harness.
func WrappedAgentForHarness(name string) (string, bool) {
	key := normalizeHarnessName(name)
	for _, h := range frontierHarnesses {
		if normalizeHarnessName(h.Name) == key {
			return h.ImportPath, true
		}
	}
	return "", false
}

// harnessNameForWrapped is the reverse lookup: the job.yaml short name for a
// wrapped harbor_ext class. Unknown classes fall back to a slugged module name
// so a not-yet-registered harness still routes with a stable agent name.
func harnessNameForWrapped(importPath string) string {
	for _, h := range frontierHarnesses {
		if h.ImportPath == importPath {
			return h.Name
		}
	}
	return slugWrappedAgent(importPath)
}

// HarnessCoverage reports which FrontierSWE harness classes are wired for fak
// routing and which remain — an honest coverage statement, never a blanket
// "all harnesses" claim.
func HarnessCoverage() (wired []string, remaining []string) {
	for _, h := range frontierHarnesses {
		wired = append(wired, h.Name)
	}
	remaining = append(remaining, frontierHarnessRemaining...)
	return wired, remaining
}

// HarnessRouting is the raw-vs-fak-routed shim comparison for one harness. The
// fak-routed job.yaml wraps the harness in the FakRoutedAgent and points it at
// the fak gateway; the raw job.yaml points the same harness at the upstream base
// URL directly. By construction the ONLY per-line difference is the base URL —
// model, wrapped class, budget, and no-internet boundary are identical.
type HarnessRouting struct {
	Harness      string `json:"harness"`
	WrappedAgent string `json:"wrapped_agent"`
	RawBaseURL   string `json:"raw_base_url"`
	FakBaseURL   string `json:"fak_base_url"`
	RawJobYAML   string `json:"raw_job_yaml"`
	FakJobYAML   string `json:"fak_job_yaml"`
}

// BuildHarnessRouting returns the raw and fak-routed job.yaml for a harness,
// differing only in the model base URL. An unregistered harness name falls
// through to a wrapped=<name> passthrough so callers can still route a
// not-yet-registered class honestly (the short name is slugged from the class).
func BuildHarnessRouting(harness, gatewayBase, upstreamBase string, allowInternet bool) HarnessRouting {
	wrapped, ok := WrappedAgentForHarness(harness)
	if !ok {
		wrapped = strings.TrimSpace(harness)
	}
	name := harnessNameForWrapped(wrapped)
	return HarnessRouting{
		Harness:      name,
		WrappedAgent: wrapped,
		RawBaseURL:   upstreamBase,
		FakBaseURL:   gatewayBase,
		RawJobYAML:   routedJobYAML(name, wrapped, upstreamBase, allowInternet),
		FakJobYAML:   routedJobYAML(name, wrapped, gatewayBase, allowInternet),
	}
}

// BaseURLOnlyDelta returns the differing (raw, fak) line pairs between the raw
// and fak-routed job.yaml and whether the only change is the base URL. It is the
// machine-checkable proof for #1725: routing a harness through fak overrides
// nothing but the model base URL.
func (r HarnessRouting) BaseURLOnlyDelta() (diffs [][2]string, baseURLOnly bool) {
	rawLines := strings.Split(r.RawJobYAML, "\n")
	fakLines := strings.Split(r.FakJobYAML, "\n")
	if len(rawLines) != len(fakLines) {
		return [][2]string{{r.RawJobYAML, r.FakJobYAML}}, false
	}
	for i := range rawLines {
		if rawLines[i] == fakLines[i] {
			continue
		}
		diffs = append(diffs, [2]string{rawLines[i], fakLines[i]})
	}
	baseURLOnly = len(diffs) == 1 && strings.Contains(diffs[0][0], "fak_base_url")
	return diffs, baseURLOnly
}

// RenderHarnessCoverageMarkdown documents the wired-vs-remaining FrontierSWE
// harness classes as a table, so coverage is a concrete artifact and never a
// blanket "all harnesses" claim.
func RenderHarnessCoverageMarkdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# FrontierSWE Multi-Harness Routing Coverage\n\n")
	fmt.Fprintf(&b, "fak routes each harness class through its gateway with only the model base URL overridden (issue #1725, epic #1706 C19).\n\n")
	fmt.Fprintf(&b, "## Wired\n\n")
	fmt.Fprintf(&b, "| Harness | Wrapped harbor_ext class |\n")
	fmt.Fprintf(&b, "|---|---|\n")
	for _, h := range frontierHarnesses {
		fmt.Fprintf(&b, "| `%s` | `%s` |\n", h.Name, h.ImportPath)
	}
	remaining := append([]string(nil), frontierHarnessRemaining...)
	sort.Strings(remaining)
	fmt.Fprintf(&b, "\n## Not yet wired\n\n")
	for _, name := range remaining {
		fmt.Fprintf(&b, "- `%s`\n", name)
	}
	return b.String()
}

func normalizeHarnessName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	return strings.ReplaceAll(s, "_", "-")
}

func slugWrappedAgent(importPath string) string {
	// harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch -> claude-code
	p := strings.TrimSpace(importPath)
	if i := strings.Index(p, ":"); i >= 0 {
		p = p[:i]
	}
	if i := strings.LastIndex(p, "."); i >= 0 {
		p = p[i+1:]
	}
	p = strings.ReplaceAll(p, "_", "-")
	if p == "" {
		return "harness"
	}
	return p
}

// routedJobYAML renders a FrontierSWE job.yaml that wraps one harness in the
// FakRoutedAgent, pointing it at base. It is the single source of the shim's
// job.yaml so every harness — and the raw-vs-fak comparison — shares one shape.
func routedJobYAML(harnessName, wrapped, base string, allowInternet bool) string {
	return fmt.Sprintf(`agents:
  - name: fak-routed-%s
    import_path: harbor_ext.fak_routed:FakRoutedAgent
    model_name: ${FRONTIERSWE_MODEL}
    override_timeout_sec: 72000
    kwargs:
      wrapped: %s
      fak_base_url: %s
      allow_internet: %t
`, harnessName, wrapped, base, allowInternet)
}
