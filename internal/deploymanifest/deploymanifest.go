// Package deploymanifest defines the unified `fak.toml` all-in-one deployment
// manifest (issue #3421, Workstream E of epic #3256) and its fail-closed loader.
//
// A real all-in-one deployment is otherwise assembled from scattered,
// non-declarative inputs: flag-soup on `fak serve`/`fak guard`, `policy.json`
// (the capability floor), env vars, and scheduled-task/systemd/compose
// registration. `fak.toml` is the single declarative artifact that describes
// the whole deployment so the same reviewable file boots an identical topology
// on a laptop and in a locked-down VPC — the diff between the two becomes a
// handful of values in a PR, not a set of remembered flag deltas.
//
// This leaf owns the manifest SHAPE and its load-time CONTRACT, not the deep
// per-section runtime behavior (budgets → #3273, tenants → #3263, agent
// templates → #3283) and not the consumers that boot/validate a live topology
// (`fak up` → E1, deployment `fak doctor` → E4). It deliberately does NOT change
// `dos.toml`'s role: `dos.toml` stays the workspace/lane descriptor, `fak.toml`
// is the deployment descriptor — distinct and cross-referenced.
//
// The load-time contract is fail-closed: an unknown/typo'd section or key
// REFUSES at load with a structured reason from the closed vocabulary (see
// Reason) — never a silent default. A mistyped `requre_key_env` must refuse, not
// quietly disable auth. Precedence is explicit-flags > manifest > built-in
// defaults (see Defaults and Manifest.WithOverrides).
//
// The parser is a small hand-rolled TOML subset (section headers + quoted
// strings, bare booleans, and integers), matching the existing dos.toml readers
// in this repo (e.g. internal/branchrole) rather than pulling in a TOML
// dependency the module does not carry.
package deploymanifest

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Reason is a closed-vocabulary refusal reason. The loader never returns a
// free-text failure for a contract violation: every refusal names one of these,
// so an operator (or `fak doctor`) can branch on the reason rather than parse a
// message. Mirrors the guard `[reasons.*]` grammar shape.
type Reason string

const (
	// ReasonUnknownSection: a [section] header not in the closed section set.
	ReasonUnknownSection Reason = "UNKNOWN_SECTION"
	// ReasonUnknownKey: a key that does not exist in its (known) section — the
	// typo case (`requre_key_env`) that must refuse rather than silently drift.
	ReasonUnknownKey Reason = "UNKNOWN_KEY"
	// ReasonDuplicateKey: the same key set twice in one section.
	ReasonDuplicateKey Reason = "DUPLICATE_KEY"
	// ReasonMalformedLine: a non-empty, non-header line with no `key = value`.
	ReasonMalformedLine Reason = "MALFORMED_LINE"
	// ReasonBadValue: a value whose type/shape is wrong for its key.
	ReasonBadValue Reason = "BAD_VALUE"
	// ReasonBareKey: a key = value pair before any [section] header.
	ReasonBareKey Reason = "BARE_KEY"
)

// LoadError is the structured refusal the loader returns for any contract
// violation. It carries the closed-vocabulary Reason plus enough locus (section,
// key, line number) to point the operator at the exact spot — no silent drift.
type LoadError struct {
	Reason  Reason
	Section string
	Key     string
	Line    int    // 1-based source line, 0 if not line-specific
	Message string // human-readable detail; the Reason is the machine-readable part
}

func (e *LoadError) Error() string {
	loc := ""
	if e.Line > 0 {
		loc = fmt.Sprintf(" (line %d)", e.Line)
	}
	sec := ""
	if e.Section != "" {
		sec = " [" + e.Section + "]"
	}
	return fmt.Sprintf("fak.toml refused: %s%s%s: %s", e.Reason, sec, loc, e.Message)
}

// Runtimes declares which server-side runtimes the all-in-one starts and where
// the model comes from. Deep per-runtime wiring stays with `fak up` (E1).
type Runtimes struct {
	Gateway      bool   // start the gateway (serve) runtime
	AgentRuntime bool   // start the guard/agent runtime
	Model        string // "upstream" | "in-kernel" — where completions resolve
}

// Policy is the capability floor: a path to a policy.json, or an inline body.
// This leaf declares the shape; enforcement stays where policy already lives.
type Policy struct {
	Floor  string // path to a policy.json floor
	Inline string // inline policy body (mutually exclusive with Floor in practice)
}

// Auth names the environment variable that must hold the gateway key. Naming the
// env var (not the secret) keeps the manifest reviewable and secret-free.
type Auth struct {
	RequireKeyEnv string // e.g. "FAK_GATEWAY_KEY"; empty = auth not required
	AllowLAN      bool   // when true, allow unauthenticated access from local/private network (RFC 1918, link-local, loopback)
}

// Budgets declares the per-scope token-cap SHAPE. Deep budget semantics → #3273.
type Budgets struct {
	DefaultTokens int // per-scope default token cap; 0 = unset (no cap declared)
}

// Audit declares the audit journal path and retention. Deep retention semantics
// live with the audit journal itself; this is the deployment-level shape.
type Audit struct {
	Journal       string // audit journal path
	RetentionDays int    // 0 = keep indefinitely
}

// Tenants declares multi-tenancy SHAPE only. Deep tenant semantics → #3263.
type Tenants struct {
	Enabled bool
}

// AgentTemplates declares where agent templates live. Deep semantics → #3283.
type AgentTemplates struct {
	Dir string
}

// Observability declares the metrics/bind SHAPE for the all-in-one.
type Observability struct {
	Metrics bool   // expose the metrics endpoint
	Bind    string // host:port the metrics endpoint binds
}

type PluginSelection struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}
type PreferenceLayer struct {
	RequireWitness     bool   `json:"require_witness"`
	WitnessRoute       string `json:"witness_route"`
	WaitMode           string `json:"wait_mode"`
	TransformMode      string `json:"transform_mode"`
	Disclosure         string `json:"disclosure"`
	Timeout            string `json:"timeout"`
	ResumeNotification string `json:"resume_notification"`
}
type ToolPlugins struct {
	Plugins      []PluginSelection `json:"plugins"`
	Organization PreferenceLayer   `json:"organization"`
	Project      PreferenceLayer   `json:"project"`
	User         PreferenceLayer   `json:"user"`
}

// Manifest is the parsed, defaulted `fak.toml`. Every field is typed so a
// consumer reads a value, never re-parses text.
type Manifest struct {
	Runtimes       Runtimes
	Policy         Policy
	Auth           Auth
	Budgets        Budgets
	Audit          Audit
	Tenants        Tenants
	AgentTemplates AgentTemplates
	Observability  Observability
	ToolPlugins    ToolPlugins
	present        map[string]bool
}

// Defaults returns the built-in defaults — the lowest-precedence layer. A
// manifest overlays these; explicit flags overlay the manifest.
func Defaults() Manifest {
	return Manifest{
		Runtimes:       Runtimes{Gateway: true, AgentRuntime: true, Model: "upstream"},
		Policy:         Policy{},
		Auth:           Auth{},
		Budgets:        Budgets{},
		Audit:          Audit{Journal: "", RetentionDays: 0},
		Tenants:        Tenants{},
		AgentTemplates: AgentTemplates{},
		Observability:  Observability{Metrics: false, Bind: "127.0.0.1:9090"},
	}
}

// Present reports whether the operator explicitly declared section.key in the
// manifest, as distinct from receiving the same value from built-in defaults.
// Consumers use it to preserve truthful provenance and avoid treating omitted
// zero values as customization.
func (m Manifest) Present(section, key string) bool {
	return m.present[section+"."+key]
}

// Key names one field in the closed fak.toml vocabulary. Consumers iterate
// KnownKeys instead of copying the parser's schema and silently missing a new
// opinion when the manifest grows.
type Key struct {
	Section string `json:"section"`
	Name    string `json:"key"`
}

func (k Key) Dotted() string { return k.Section + "." + k.Name }

// Descriptor is the discoverability contract for one fak.toml opinion. Every
// parser key must have one, so customization never degenerates into a mystery
// knob whose default or purpose lives only in source code.
type Descriptor struct {
	Key         Key    `json:"key"`
	Default     any    `json:"default"`
	Description string `json:"description"`
	Secret      bool   `json:"secret"`
}

var keyDescriptions = map[string]string{
	"agent_templates.dir":    "directory containing agent templates for the all-in-one runtime",
	"audit.journal":          "path of the deployment audit journal",
	"audit.retention_days":   "days to retain audit records; zero keeps them indefinitely",
	"auth.allow_lan":         "when true, allow unauthenticated requests from private/local networks (RFC 1918 IPv4, link-local, loopback)",
	"auth.require_key_env":   "name of the environment variable containing the gateway token; never the token itself",
	"budgets.default_tokens": "default per-session context token ceiling; zero leaves it unset",
	"observability.bind":     "gateway listener address",
	"observability.metrics":  "whether the all-in-one topology exposes metrics",
	"policy.floor":           "path to the reviewed capability-floor policy",
	"policy.inline":          "inline capability-floor body for orchestrators that materialize it safely",
	"runtimes.agent_runtime": "whether the all-in-one topology starts the agent runtime",
	"runtimes.gateway":       "whether the all-in-one topology starts the gateway",
	"runtimes.model":         "completion source: upstream or in-kernel",
	"tenants.enabled":        "whether the all-in-one topology enables tenant isolation",

	"tool_plugins.plugins":                      "pinned built-in tool-plugin profiles selected for the monotone gateway host",
	"tool_plugins.organization.wait_mode":       "organization preference for local execution within the capability floor",
	"tool_plugins.organization.transform_mode":  "organization preference for admitted cached results",
	"tool_plugins.organization.require_witness": "organization witness requirement; lower layers cannot remove it",
	"tool_plugins.organization.disclosure":      "organization-selected model route within deployment policy",
	"tool_plugins.organization.witness_route":   "organization-selected witness route",
	"tool_plugins.organization.timeout":         "organization-selected cost class",
	"tool_plugins.project.wait_mode":            "project preference for local execution within the capability floor",
	"tool_plugins.project.transform_mode":       "project preference for admitted cached results",
	"tool_plugins.project.require_witness":      "project witness requirement; the user layer cannot remove it",
	"tool_plugins.project.disclosure":           "project-selected model route within deployment policy",
	"tool_plugins.project.witness_route":        "project-selected witness route",
	"tool_plugins.project.timeout":              "project-selected cost class",
	"tool_plugins.user.wait_mode":               "user preference for local execution within higher-authority policy",
	"tool_plugins.user.transform_mode":          "user preference for admitted cached results",
	"tool_plugins.user.require_witness":         "user-requested witness requirement",
	"tool_plugins.user.disclosure":              "user-selected model route within higher-authority policy",
	"tool_plugins.user.witness_route":           "user-selected witness route",
	"tool_plugins.user.timeout":                 "user-selected cost class",
}

// Descriptors returns the complete configuration vocabulary with witnessed
// built-in defaults in stable key order.
func Descriptors() []Descriptor {
	defaults := Defaults()
	out := make([]Descriptor, 0, len(knownSections))
	for _, key := range KnownKeys() {
		out = append(out, Descriptor{Key: key, Default: defaults.Value(key), Description: keyDescriptions[key.Dotted()]})
	}
	return out
}

// Value projects one typed manifest field by its closed-vocabulary key.
func (m Manifest) Value(key Key) any {
	switch key.Dotted() {
	case "runtimes.gateway":
		return m.Runtimes.Gateway
	case "runtimes.agent_runtime":
		return m.Runtimes.AgentRuntime
	case "runtimes.model":
		return m.Runtimes.Model
	case "policy.floor":
		return m.Policy.Floor
	case "policy.inline":
		return m.Policy.Inline
	case "auth.require_key_env":
		return m.Auth.RequireKeyEnv
	case "auth.allow_lan":
		return m.Auth.AllowLAN
	case "budgets.default_tokens":
		return m.Budgets.DefaultTokens
	case "audit.journal":
		return m.Audit.Journal
	case "audit.retention_days":
		return m.Audit.RetentionDays
	case "tenants.enabled":
		return m.Tenants.Enabled
	case "tool_plugins.plugins":
		return append([]PluginSelection(nil), m.ToolPlugins.Plugins...)
	case "agent_templates.dir":
		return m.AgentTemplates.Dir
	case "observability.metrics":
		return m.Observability.Metrics
	case "observability.bind":
		return m.Observability.Bind
	default:
		if strings.HasPrefix(key.Section, "tool_plugins.") {
			return preferenceValue(m, key.Section, key.Name)
		}
		panic("unknown fak.toml key: " + key.Dotted())
	}
}

func preferenceValue(m Manifest, section, key string) any {
	var p PreferenceLayer
	switch section {
	case "tool_plugins.organization":
		p = m.ToolPlugins.Organization
	case "tool_plugins.project":
		p = m.ToolPlugins.Project
	case "tool_plugins.user":
		p = m.ToolPlugins.User
	}
	switch key {
	case "wait_mode":
		return p.WaitMode
	case "transform_mode":
		return p.TransformMode
	case "require_witness":
		return p.RequireWitness
	case "disclosure":
		return p.Disclosure
	case "witness_route":
		return p.WitnessRoute
	case "timeout":
		return p.Timeout
	case "resume_notification":
		return p.ResumeNotification
	}
	return nil
}

// KnownKeys returns the complete closed vocabulary in stable dotted order.
func KnownKeys() []Key {
	keys := make([]Key, 0)
	for section, sectionKeys := range knownSections {
		for key := range sectionKeys {
			keys = append(keys, Key{Section: section, Name: key})
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].Dotted() < keys[j].Dotted() })
	return keys
}

// DeclaredKeys returns only fields explicitly written by the operator, in the
// same stable order as KnownKeys.
func (m Manifest) DeclaredKeys() []Key {
	declared := make([]Key, 0, len(m.present))
	for _, key := range KnownKeys() {
		if m.Present(key.Section, key.Name) {
			declared = append(declared, key)
		}
	}
	return declared
}

// knownSections maps each closed-vocabulary section to its closed set of keys.
// The parser refuses any section or key not in this table — the fail-closed
// contract that kills silent drift.
var knownSections = map[string]map[string]bool{
	"runtimes":                  {"gateway": true, "agent_runtime": true, "model": true},
	"policy":                    {"floor": true, "inline": true},
	"auth":                      {"require_key_env": true, "allow_lan": true},
	"budgets":                   {"default_tokens": true},
	"audit":                     {"journal": true, "retention_days": true},
	"tenants":                   {"enabled": true},
	"agent_templates":           {"dir": true},
	"observability":             {"metrics": true, "bind": true},
	"tool_plugins":              {"plugins": true},
	"tool_plugins.organization": {"wait_mode": true, "transform_mode": true, "require_witness": true, "disclosure": true, "witness_route": true, "timeout": true},
	"tool_plugins.project":      {"wait_mode": true, "transform_mode": true, "require_witness": true, "disclosure": true, "witness_route": true, "timeout": true},
	"tool_plugins.user":         {"wait_mode": true, "transform_mode": true, "require_witness": true, "disclosure": true, "witness_route": true, "timeout": true},
}

// validModel is the closed set of runtime model sources.
var validModel = map[string]bool{"upstream": true, "in-kernel": true}

// Load reads and parses a `fak.toml` from disk, fail-closed. A missing file is
// returned as the underlying os error (callers decide whether absence is fatal);
// a present-but-invalid file returns a *LoadError with a named Reason.
func Load(path string) (Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Defaults(), err
	}
	return Parse(b)
}

// Parse parses manifest bytes over the built-in Defaults, fail-closed. Any
// unknown section, unknown key, duplicate, malformed line, or bad value refuses
// with a *LoadError naming a closed-vocabulary Reason.
func Parse(b []byte) (Manifest, error) {
	m := Defaults()
	m.present = make(map[string]bool)
	section := ""
	seen := map[string]bool{} // "section.key" already set
	scanner := bufio.NewScanner(bytes.NewReader(b))
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(stripComment(scanner.Text()))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			name := strings.TrimSpace(line[1 : len(line)-1])
			if _, ok := knownSections[name]; !ok {
				return m, &LoadError{Reason: ReasonUnknownSection, Section: name, Line: lineNo,
					Message: fmt.Sprintf("unknown section %q — not in the fak.toml vocabulary", name)}
			}
			section = name
			continue
		}
		eq := strings.Index(line, "=")
		if eq < 0 {
			return m, &LoadError{Reason: ReasonMalformedLine, Section: section, Line: lineNo,
				Message: fmt.Sprintf("expected `key = value`, got %q", line)}
		}
		key := strings.TrimSpace(line[:eq])
		rawVal := strings.TrimSpace(line[eq+1:])
		if section == "" {
			return m, &LoadError{Reason: ReasonBareKey, Key: key, Line: lineNo,
				Message: fmt.Sprintf("key %q appears before any [section] header", key)}
		}
		keys := knownSections[section]
		if !keys[key] {
			return m, &LoadError{Reason: ReasonUnknownKey, Section: section, Key: key, Line: lineNo,
				Message: fmt.Sprintf("unknown key %q in [%s] — refusing rather than silently defaulting", key, section)}
		}
		dotted := section + "." + key
		if seen[dotted] {
			return m, &LoadError{Reason: ReasonDuplicateKey, Section: section, Key: key, Line: lineNo,
				Message: fmt.Sprintf("key %q set twice in [%s]", key, section)}
		}
		seen[dotted] = true
		m.present[dotted] = true
		if err := assign(&m, section, key, rawVal, lineNo); err != nil {
			return m, err
		}
	}
	if err := scanner.Err(); err != nil {
		return m, err
	}
	return m, nil
}

// assign coerces rawVal to the field's type and stores it, refusing with a
// ReasonBadValue LoadError on a type/shape mismatch.
func assign(m *Manifest, section, key, rawVal string, lineNo int) error {
	badValue := func(msg string) error {
		return &LoadError{Reason: ReasonBadValue, Section: section, Key: key, Line: lineNo, Message: msg}
	}
	// Every field below is one of exactly three shapes. Naming the three coercions once —
	// each already carrying this line's section/key/line in its refusal — lets the arms show
	// what a reader is here for: which manifest key lands in which struct field. A count is
	// its own shape rather than "an int": a negative budget and a negative retention are the
	// same class of nonsense and must refuse identically.
	boolVal := func() (bool, error) {
		v, err := parseBool(rawVal)
		if err != nil {
			return false, badValue(err.Error())
		}
		return v, nil
	}
	stringVal := func(what string) (string, error) {
		v, ok := parseString(rawVal)
		if !ok {
			return "", badValue(what + " must be a quoted string")
		}
		return v, nil
	}
	countVal := func(what string) (int, error) {
		v, err := parseInt(rawVal)
		if err != nil {
			return 0, badValue(err.Error())
		}
		if v < 0 {
			return 0, badValue(what + " must not be negative")
		}
		return v, nil
	}
	switch section {
	case "runtimes":
		switch key {
		case "gateway":
			v, err := boolVal()
			if err != nil {
				return err
			}
			m.Runtimes.Gateway = v
		case "agent_runtime":
			v, err := boolVal()
			if err != nil {
				return err
			}
			m.Runtimes.AgentRuntime = v
		case "model":
			v, err := stringVal("model")
			if err != nil {
				return err
			}
			if !validModel[v] {
				return badValue(fmt.Sprintf("model %q must be one of upstream|in-kernel", v))
			}
			m.Runtimes.Model = v
		}
	case "policy":
		v, err := stringVal(key)
		if err != nil {
			return err
		}
		if key == "floor" {
			m.Policy.Floor = v
		} else {
			m.Policy.Inline = v
		}
	case "auth":
		if key == "require_key_env" {
			v, err := stringVal("require_key_env")
			if err != nil {
				return err
			}
			m.Auth.RequireKeyEnv = v
		} else if key == "allow_lan" {
			v, err := boolVal()
			if err != nil {
				return err
			}
			m.Auth.AllowLAN = v
		}
	case "budgets":
		v, err := countVal("default_tokens")
		if err != nil {
			return err
		}
		m.Budgets.DefaultTokens = v
	case "audit":
		if key == "journal" {
			v, err := stringVal("journal")
			if err != nil {
				return err
			}
			m.Audit.Journal = v
		} else {
			v, err := countVal("retention_days")
			if err != nil {
				return err
			}
			m.Audit.RetentionDays = v
		}
	case "tenants":
		v, err := boolVal()
		if err != nil {
			return err
		}
		m.Tenants.Enabled = v
	case "agent_templates":
		v, err := stringVal("dir")
		if err != nil {
			return err
		}
		m.AgentTemplates.Dir = v
	case "tool_plugins":
		v, err := parsePluginSelections(rawVal, lineNo)
		if err != nil {
			return err
		}
		m.ToolPlugins.Plugins = v
	case "tool_plugins.organization", "tool_plugins.project", "tool_plugins.user":
		var v any
		var err error
		switch key {
		case "require_witness":
			v, err = boolVal()
		default:
			v, err = stringVal(key)
		}
		if err != nil {
			return err
		}
		setPreferenceValue(m, section, key, v)
	case "observability":
		if key == "metrics" {
			v, err := boolVal()
			if err != nil {
				return err
			}
			m.Observability.Metrics = v
		} else {
			v, err := stringVal("bind")
			if err != nil {
				return err
			}
			m.Observability.Bind = v
		}
	}
	return nil
}

// Overrides carries explicit-flag values that override the manifest. A nil
// pointer means "flag not set — keep the manifest value" (which itself may be a
// default). This encodes the flags > manifest > defaults precedence without
// conflating "flag absent" with "flag set to the zero value".
type Overrides struct {
	Gateway       *bool
	AgentRuntime  *bool
	Model         *string
	PolicyFloor   *string
	RequireKeyEnv *string
	AllowLAN      *bool
	DefaultTokens *int
	AuditJournal  *string
	RetentionDays *int
	TenantsOn     *bool
	TemplatesDir  *string
	Metrics       *bool
	MetricsBind   *string
}

// WithOverrides applies explicit-flag overrides on top of the manifest, honoring
// the flags > manifest > defaults precedence. Only non-nil override fields win.
func (m Manifest) WithOverrides(o Overrides) Manifest {
	if o.Gateway != nil {
		m.Runtimes.Gateway = *o.Gateway
	}
	if o.AgentRuntime != nil {
		m.Runtimes.AgentRuntime = *o.AgentRuntime
	}
	if o.Model != nil {
		m.Runtimes.Model = *o.Model
	}
	if o.PolicyFloor != nil {
		m.Policy.Floor = *o.PolicyFloor
	}
	if o.RequireKeyEnv != nil {
		m.Auth.RequireKeyEnv = *o.RequireKeyEnv
	}
	if o.AllowLAN != nil {
		m.Auth.AllowLAN = *o.AllowLAN
	}
	if o.DefaultTokens != nil {
		m.Budgets.DefaultTokens = *o.DefaultTokens
	}
	if o.AuditJournal != nil {
		m.Audit.Journal = *o.AuditJournal
	}
	if o.RetentionDays != nil {
		m.Audit.RetentionDays = *o.RetentionDays
	}
	if o.TenantsOn != nil {
		m.Tenants.Enabled = *o.TenantsOn
	}
	if o.TemplatesDir != nil {
		m.AgentTemplates.Dir = *o.TemplatesDir
	}
	if o.Metrics != nil {
		m.Observability.Metrics = *o.Metrics
	}
	if o.MetricsBind != nil {
		m.Observability.Bind = *o.MetricsBind
	}
	return m
}

// Minimal returns the bytes of a minimal, valid `fak.toml` — the artifact
// `fak init` emits. It is intentionally the smallest reviewable starting point:
// it declares the runtimes and auth an operator will almost always edit, with
// the rest carried by Defaults. It round-trips: Parse(Minimal()) succeeds.
func setPreferenceValue(m *Manifest, section, key string, value any) {
	var p *PreferenceLayer
	switch section {
	case "tool_plugins.organization":
		p = &m.ToolPlugins.Organization
	case "tool_plugins.project":
		p = &m.ToolPlugins.Project
	case "tool_plugins.user":
		p = &m.ToolPlugins.User
	}
	if p == nil {
		return
	}
	switch key {
	case "wait_mode":
		p.WaitMode = value.(string)
	case "transform_mode":
		p.TransformMode = value.(string)
	case "require_witness":
		p.RequireWitness = value.(bool)
	case "disclosure":
		p.Disclosure = value.(string)
	case "witness_route":
		p.WitnessRoute = value.(string)
	case "timeout":
		p.Timeout = value.(string)
	case "resume_notification":
		p.ResumeNotification = value.(string)
	}
}

func Minimal() []byte {
	return []byte(`# fak.toml — unified all-in-one deployment manifest (#3421).
# One declarative file for the whole deployment: the same reviewable artifact
# boots an identical topology on a laptop and in a locked-down VPC — the diff is
# a handful of values, not a set of remembered flag deltas.
#
# Precedence: explicit flags > this manifest > built-in defaults.
# Unknown/typo'd keys REFUSE at load with a named reason — no silent drift.
# This is the DEPLOYMENT descriptor; dos.toml stays the workspace/lane descriptor.

[runtimes]
gateway = true
agent_runtime = true
model = "upstream"        # upstream | in-kernel

[auth]
require_key_env = "FAK_GATEWAY_KEY"

[observability]
metrics = true
bind = "127.0.0.1:9090"
`)
}

// --- small hand-rolled TOML-subset value scanners (dos.toml-style) ---

// stripComment removes a trailing `# comment`, respecting a quoted string so a
// `#` inside quotes is not mistaken for a comment.
func stripComment(s string) string {
	inQuote := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			inQuote = !inQuote
		case '#':
			if !inQuote {
				return s[:i]
			}
		}
	}
	return s
}

// parseString accepts a double-quoted string and returns its contents.
func parsePluginSelections(raw string, line int) ([]PluginSelection, error) {
	bad := func(reason Reason, key, msg string) error {
		return &LoadError{Reason: reason, Section: "tool_plugins", Key: key, Line: line, Message: msg}
	}
	var out []PluginSelection
	raw = strings.TrimSpace(raw)
	if raw == "[]" {
		return out, nil
	}
	if !strings.HasPrefix(raw, "[") || !strings.HasSuffix(raw, "]") {
		return nil, bad(ReasonBadValue, "plugins", "expected array of inline tables")
	}
	body := strings.TrimSpace(raw[1 : len(raw)-1])
	depth, inString, escaped, start := 0, false, false, -1
	var entries []string
	for i, r := range body {
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				inString = false
			}
			continue
		}
		switch r {
		case '"':
			inString = true
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth < 0 {
				return nil, bad(ReasonBadValue, "plugins", "unbalanced inline table")
			}
			if depth == 0 && start >= 0 {
				entries = append(entries, body[start:i+1])
				start = -1
			}
		}
	}
	if depth != 0 || inString {
		return nil, bad(ReasonBadValue, "plugins", "unterminated inline table")
	}
	for _, entry := range entries {
		fields := map[string]string{}
		for _, part := range splitInlineFields(strings.TrimSpace(entry[1 : len(entry)-1])) {
			k, v, ok := strings.Cut(part, "=")
			if !ok {
				return nil, bad(ReasonBadValue, "plugins", "invalid inline field")
			}
			key := strings.TrimSpace(k)
			if key != "id" && key != "version" && key != "digest" {
				return nil, bad(ReasonUnknownKey, "plugins."+key, "unknown plugin field")
			}
			decoded, ok := parseString(strings.TrimSpace(v))
			if !ok {
				return nil, bad(ReasonBadValue, "plugins."+key, "must be a quoted string")
			}
			fields[key] = decoded
		}
		out = append(out, PluginSelection{ID: fields["id"], Version: fields["version"], Digest: fields["digest"]})
	}
	return out, nil
}
func splitInlineFields(s string) []string {
	var out []string
	inString, escaped, start := false, false, 0
	for i, r := range s {
		if inString {
			if escaped {
				escaped = false
			} else if r == '\\' {
				escaped = true
			} else if r == '"' {
				inString = false
			}
			continue
		}
		if r == '"' {
			inString = true
		} else if r == ',' {
			out = append(out, strings.TrimSpace(s[start:i]))
			start = i + 1
		}
	}
	if strings.TrimSpace(s[start:]) != "" {
		out = append(out, strings.TrimSpace(s[start:]))
	}
	return out
}

func parseString(s string) (string, bool) {
	if len(s) >= 2 && strings.HasPrefix(s, "\"") && strings.HasSuffix(s, "\"") {
		return s[1 : len(s)-1], true
	}
	return "", false
}

// parseBool accepts bare `true`/`false` (TOML booleans are unquoted).
func parseBool(s string) (bool, error) {
	switch s {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}
	return false, fmt.Errorf("expected true|false, got %q", s)
}

// parseInt accepts a bare base-10 integer (TOML integers are unquoted).
func parseInt(s string) (int, error) {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("expected an integer, got %q", s)
	}
	return n, nil
}
