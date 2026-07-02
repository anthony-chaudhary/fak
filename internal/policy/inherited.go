package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

// InheritedRule is one per-child-launch capability inheritance grant in the
// manifest's inherited_capabilities block. It names the child-launch tool that
// may receive a bounded subset of the parent's environment and runtime scope.
type InheritedRule struct {
	Tool             string           `json:"tool"`
	Env              []string         `json:"env,omitempty"`
	SecretRefs       []SecretRefGrant `json:"secret_refs,omitempty"`
	CWD              string           `json:"cwd,omitempty"`
	WritablePaths    []string         `json:"writable_paths,omitempty"`
	PersistencePaths []string         `json:"persistence_paths,omitempty"`
	EgressRefs       []string         `json:"egress_refs,omitempty"`
}

// SecretRefGrant passes a reference to a secret, not the secret value itself.
type SecretRefGrant struct {
	Env string `json:"env"`
	Ref string `json:"ref"`
}

// InheritedParent is the scope held by the parent process at spawn time.
type InheritedParent struct {
	Env              map[string]string
	SecretEnv        map[string]bool
	CWD              string
	WritablePaths    []string
	PersistencePaths []string
	EgressRefs       []string
}

// InheritedEnvelope is the materialized subset a child may receive.
type InheritedEnvelope struct {
	Env              map[string]string `json:"env,omitempty"`
	SecretRefs       []SecretRefGrant  `json:"secret_refs,omitempty"`
	CWD              string            `json:"cwd,omitempty"`
	WritablePaths    []string          `json:"writable_paths,omitempty"`
	PersistencePaths []string          `json:"persistence_paths,omitempty"`
	EgressRefs       []string          `json:"egress_refs,omitempty"`
	Audit            InheritedAudit    `json:"audit,omitempty"`
}

// InheritedAudit is intentionally bounded: it names env variables, but hashes
// values, secret refs, and path scopes so the audit can be logged safely.
type InheritedAudit struct {
	EnvNames               []string          `json:"env_names,omitempty"`
	EnvValueDigests        map[string]string `json:"env_value_digests,omitempty"`
	SecretRefEnvNames      []string          `json:"secret_ref_env_names,omitempty"`
	SecretRefDigests       map[string]string `json:"secret_ref_digests,omitempty"`
	CWDDigest              string            `json:"cwd_digest,omitempty"`
	WritablePathDigests    []string          `json:"writable_path_digests,omitempty"`
	PersistencePathDigests []string          `json:"persistence_path_digests,omitempty"`
	EgressRefDigests       []string          `json:"egress_ref_digests,omitempty"`
}

// InheritedTable is the compiled child-launch inheritance lookup. A nil table
// resolves to an empty envelope, preserving default-deny inheritance.
type InheritedTable struct {
	rules map[string]InheritedRule
}

// compileInheritedCapabilities validates declared inheritance rows at policy
// load. Absent block means no child inherits anything.
func compileInheritedCapabilities(rules []InheritedRule) (*InheritedTable, error) {
	if len(rules) == 0 {
		return nil, nil
	}
	t := &InheritedTable{rules: make(map[string]InheritedRule, len(rules))}
	for i, r := range rules {
		n, err := normalizeInheritedRule(i, r)
		if err != nil {
			return nil, err
		}
		if _, dup := t.rules[n.Tool]; dup {
			return nil, fmt.Errorf("inherited_capabilities[%d]: duplicate row for tool %q", i, n.Tool)
		}
		t.rules[n.Tool] = n
	}
	return t, nil
}

// Resolve returns the inherited envelope for tool. Exact rows win over the
// catch-all "*" row. Nil-safe: no table means default-deny.
func (t *InheritedTable) Resolve(tool string, parent InheritedParent) InheritedEnvelope {
	r, ok := t.ruleFor(tool)
	if !ok {
		return InheritedEnvelope{}
	}
	env := InheritedEnvelope{}
	for _, name := range r.Env {
		value, ok := parent.Env[name]
		if !ok || parent.SecretEnv[name] || secretShapedValue(value) {
			continue
		}
		if env.Env == nil {
			env.Env = map[string]string{}
		}
		env.Env[name] = value
	}
	env.SecretRefs = append([]SecretRefGrant(nil), r.SecretRefs...)
	if r.CWD != "" && strings.TrimSpace(parent.CWD) == r.CWD {
		env.CWD = r.CWD
	}
	env.WritablePaths = intersectStrings(parent.WritablePaths, r.WritablePaths)
	env.PersistencePaths = intersectStrings(parent.PersistencePaths, r.PersistencePaths)
	env.EgressRefs = intersectStrings(parent.EgressRefs, r.EgressRefs)
	env.Audit = auditInheritedEnvelope(env)
	return env
}

// ResolveLaunch names the child-launch call site. It is the same default-deny
// parent intersection as Resolve.
func (t *InheritedTable) ResolveLaunch(tool string, parent InheritedParent) InheritedEnvelope {
	return t.Resolve(tool, parent)
}

func (t *InheritedTable) ruleFor(tool string) (InheritedRule, bool) {
	if t == nil {
		return InheritedRule{}, false
	}
	tool = strings.TrimSpace(tool)
	if r, ok := t.rules[tool]; ok {
		return r, true
	}
	if r, ok := t.rules[ToolRuntimeCatchAll]; ok {
		return r, true
	}
	return InheritedRule{}, false
}

// Rules returns compiled rows sorted by tool name, for summaries/tests. Nil-safe.
func (t *InheritedTable) Rules() []InheritedRule {
	if t == nil || len(t.rules) == 0 {
		return nil
	}
	out := make([]InheritedRule, 0, len(t.rules))
	for _, r := range t.rules {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Tool < out[j].Tool })
	return out
}

// Environ renders the environment variables to hand to a child process. Secret
// refs are passed as reference strings, never raw secret values.
func (e InheritedEnvelope) Environ() []string {
	if len(e.Env) == 0 && len(e.SecretRefs) == 0 {
		return nil
	}
	out := make([]string, 0, len(e.Env)+len(e.SecretRefs))
	names := make([]string, 0, len(e.Env))
	for name := range e.Env {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		out = append(out, name+"="+e.Env[name])
	}
	for _, ref := range e.SecretRefs {
		out = append(out, ref.Env+"="+ref.Ref)
	}
	return out
}

func normalizeInheritedRule(i int, r InheritedRule) (InheritedRule, error) {
	tool := strings.TrimSpace(r.Tool)
	if tool == "" {
		return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d]: tool is required", i)
	}
	env, err := normalizeEnvNames(i, r.Env)
	if err != nil {
		return InheritedRule{}, err
	}
	usedEnv := map[string]bool{}
	for _, name := range env {
		usedEnv[name] = true
	}
	secretRefs := make([]SecretRefGrant, 0, len(r.SecretRefs))
	for j, sr := range r.SecretRefs {
		name := strings.TrimSpace(sr.Env)
		if err := validateEnvName(name); err != nil {
			return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].secret_refs[%d].env: %w", i, j, err)
		}
		if usedEnv[name] {
			return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].secret_refs[%d]: env %q already used by env grant", i, j, name)
		}
		if containsSecretRefEnv(secretRefs, name) {
			return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].secret_refs[%d]: duplicate env %q", i, j, name)
		}
		ref := strings.TrimSpace(sr.Ref)
		if ref == "" {
			return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].secret_refs[%d]: ref is required", i, j)
		}
		if strings.ContainsRune(ref, '\x00') {
			return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].secret_refs[%d].ref: contains NUL", i, j)
		}
		secretRefs = append(secretRefs, SecretRefGrant{Env: name, Ref: ref})
	}
	cwd := strings.TrimSpace(r.CWD)
	if strings.ContainsRune(cwd, '\x00') {
		return InheritedRule{}, fmt.Errorf("inherited_capabilities[%d].cwd: contains NUL", i)
	}
	writable, err := normalizeScopes(i, "writable_paths", r.WritablePaths)
	if err != nil {
		return InheritedRule{}, err
	}
	persistence, err := normalizeScopes(i, "persistence_paths", r.PersistencePaths)
	if err != nil {
		return InheritedRule{}, err
	}
	egress, err := normalizeScopes(i, "egress_refs", r.EgressRefs)
	if err != nil {
		return InheritedRule{}, err
	}
	return InheritedRule{
		Tool:             tool,
		Env:              env,
		SecretRefs:       secretRefs,
		CWD:              cwd,
		WritablePaths:    writable,
		PersistencePaths: persistence,
		EgressRefs:       egress,
	}, nil
}

func normalizeEnvNames(row int, names []string) ([]string, error) {
	out := make([]string, 0, len(names))
	seen := map[string]bool{}
	for j, raw := range names {
		name := strings.TrimSpace(raw)
		if err := validateEnvName(name); err != nil {
			return nil, fmt.Errorf("inherited_capabilities[%d].env[%d]: %w", row, j, err)
		}
		if seen[name] {
			return nil, fmt.Errorf("inherited_capabilities[%d].env[%d]: duplicate env %q", row, j, name)
		}
		seen[name] = true
		out = append(out, name)
	}
	return out, nil
}

func validateEnvName(name string) error {
	if name == "" {
		return fmt.Errorf("invalid env: name is required")
	}
	for i, r := range name {
		switch {
		case r == '_' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z'):
		case i > 0 && r >= '0' && r <= '9':
		default:
			return fmt.Errorf("invalid env %q", name)
		}
	}
	return nil
}

func normalizeScopes(row int, field string, vals []string) ([]string, error) {
	out := make([]string, 0, len(vals))
	seen := map[string]bool{}
	for j, raw := range vals {
		v := strings.TrimSpace(raw)
		if v == "" {
			return nil, fmt.Errorf("inherited_capabilities[%d].%s[%d]: scope is required", row, field, j)
		}
		if strings.ContainsRune(v, '\x00') {
			return nil, fmt.Errorf("inherited_capabilities[%d].%s[%d]: contains NUL", row, field, j)
		}
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out, nil
}

func containsSecretRefEnv(refs []SecretRefGrant, env string) bool {
	for _, ref := range refs {
		if ref.Env == env {
			return true
		}
	}
	return false
}

func secretShapedValue(v string) bool {
	lc := strings.ToLower(strings.TrimSpace(v))
	return strings.HasPrefix(lc, "sk-") ||
		strings.Contains(lc, "api_key=") ||
		strings.Contains(lc, "token=") ||
		strings.Contains(lc, "password=") ||
		strings.Contains(lc, "secret=")
}

func intersectStrings(parent, requested []string) []string {
	if len(parent) == 0 || len(requested) == 0 {
		return nil
	}
	parentSet := map[string]bool{}
	for _, p := range parent {
		parentSet[strings.TrimSpace(p)] = true
	}
	out := make([]string, 0, len(requested))
	for _, r := range requested {
		r = strings.TrimSpace(r)
		if r != "" && parentSet[r] {
			out = append(out, r)
		}
	}
	return out
}

func auditInheritedEnvelope(e InheritedEnvelope) InheritedAudit {
	a := InheritedAudit{}
	if len(e.Env) > 0 {
		a.EnvNames = make([]string, 0, len(e.Env))
		a.EnvValueDigests = map[string]string{}
		for name, value := range e.Env {
			a.EnvNames = append(a.EnvNames, name)
			a.EnvValueDigests[name] = inheritedDigest(value)
		}
		sort.Strings(a.EnvNames)
	}
	if len(e.SecretRefs) > 0 {
		a.SecretRefDigests = map[string]string{}
		for _, ref := range e.SecretRefs {
			a.SecretRefEnvNames = append(a.SecretRefEnvNames, ref.Env)
			a.SecretRefDigests[ref.Env] = inheritedDigest(ref.Ref)
		}
		sort.Strings(a.SecretRefEnvNames)
	}
	if e.CWD != "" {
		a.CWDDigest = inheritedDigest(e.CWD)
	}
	a.WritablePathDigests = digestStrings(e.WritablePaths)
	a.PersistencePathDigests = digestStrings(e.PersistencePaths)
	a.EgressRefDigests = digestStrings(e.EgressRefs)
	return a
}

func digestStrings(vals []string) []string {
	if len(vals) == 0 {
		return nil
	}
	out := make([]string, 0, len(vals))
	for _, v := range vals {
		out = append(out, inheritedDigest(v))
	}
	sort.Strings(out)
	return out
}

func inheritedDigest(v string) string {
	sum := sha256.Sum256([]byte(v))
	return "sha256:" + hex.EncodeToString(sum[:])
}
