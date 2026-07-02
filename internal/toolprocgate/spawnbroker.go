package toolprocgate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	// CapAgentRunSpawn is the negotiated capability token for brokered child
	// process creation under an AgentRun envelope.
	CapAgentRunSpawn abi.Capability = "agentrun.spawn.v1"

	EnvAgentRunID   = "FAK_AGENT_RUN_ID"
	EnvParentRunID  = "FAK_PARENT_RUN_ID"
	EnvToolCallID   = "FAK_TOOL_CALL_ID"
	EnvPolicyDigest = "FAK_POLICY_DIGEST"
	EnvSpawnBackend = "FAK_SPAWN_BACKEND"
	EnvSpawnGrantID = "FAK_SPAWN_GRANT_ID"
)

const (
	SpawnVerdictAllow = "allow"
	SpawnVerdictDeny  = "deny"
)

type CapabilityEnvelope struct {
	Capabilities     []abi.Capability `json:"capabilities,omitempty"`
	DeadlineMS       int64            `json:"deadline_ms,omitempty"`
	HeartbeatEveryMS int64            `json:"heartbeat_every_ms,omitempty"`
}

type EnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type SpawnAttempt struct {
	AgentRunID   string             `json:"agent_run_id"`
	ParentRunID  string             `json:"parent_run_id,omitempty"`
	ToolCallID   string             `json:"tool_call_id"`
	PolicyDigest string             `json:"policy_digest"`
	Argv         []string           `json:"argv"`
	Env          []EnvVar           `json:"env,omitempty"`
	CWD          string             `json:"cwd"`
	Backend      string             `json:"backend"`
	Envelope     CapabilityEnvelope `json:"capability_envelope"`
}

type SpawnGrant struct {
	GrantID      string             `json:"grant_id"`
	AgentRunID   string             `json:"agent_run_id"`
	ParentRunID  string             `json:"parent_run_id,omitempty"`
	ToolCallID   string             `json:"tool_call_id"`
	PolicyDigest string             `json:"policy_digest"`
	Argv         []string           `json:"argv"`
	Env          []EnvVar           `json:"env,omitempty"`
	CWD          string             `json:"cwd"`
	Backend      string             `json:"backend"`
	Envelope     CapabilityEnvelope `json:"capability_envelope"`
	Audit        SpawnAudit         `json:"audit"`
}

type SpawnAudit struct {
	Verdict            string             `json:"verdict"`
	Reason             string             `json:"reason,omitempty"`
	GrantID            string             `json:"grant_id,omitempty"`
	AgentRunID         string             `json:"agent_run_id,omitempty"`
	ParentRunID        string             `json:"parent_run_id,omitempty"`
	ToolCallID         string             `json:"tool_call_id,omitempty"`
	PolicyDigest       string             `json:"policy_digest,omitempty"`
	ArgvDigest         string             `json:"argv_digest,omitempty"`
	Argv0              string             `json:"argv0,omitempty"`
	EnvNames           []string           `json:"env_names,omitempty"`
	EnvDigest          string             `json:"env_digest,omitempty"`
	CWD                string             `json:"cwd,omitempty"`
	Backend            string             `json:"backend,omitempty"`
	CapabilityEnvelope CapabilityEnvelope `json:"capability_envelope"`
}

type SpawnDeniedError struct {
	Audit SpawnAudit
}

func (e SpawnDeniedError) Error() string {
	if e.Audit.Reason == "" {
		return "toolprocgate: spawn denied"
	}
	return "toolprocgate: spawn denied: " + e.Audit.Reason
}

type SpawnBroker struct {
	mu     sync.Mutex
	audits []SpawnAudit
}

func NewSpawnBroker() *SpawnBroker {
	return &SpawnBroker{}
}

func (b *SpawnBroker) Admit(attempt SpawnAttempt) (SpawnGrant, error) {
	audit := auditForAttempt(attempt)
	normalized, err := normalizeSpawnAttempt(attempt)
	if err != nil {
		audit.Verdict = SpawnVerdictDeny
		audit.Reason = err.Error()
		b.appendAudit(audit)
		return SpawnGrant{}, SpawnDeniedError{Audit: audit}
	}

	grant := SpawnGrant{
		AgentRunID:   normalized.AgentRunID,
		ParentRunID:  normalized.ParentRunID,
		ToolCallID:   normalized.ToolCallID,
		PolicyDigest: normalized.PolicyDigest,
		Argv:         cloneStrings(normalized.Argv),
		Env:          cloneEnv(normalized.Env),
		CWD:          normalized.CWD,
		Backend:      normalized.Backend,
		Envelope:     normalized.Envelope,
	}
	grant.Env = withGrantMetadataEnv(grant.Env, grant)
	grant.GrantID = "spawn_" + shortDigest(grantIDMaterial(grant))
	grant.Env = withGrantMetadataEnv(grant.Env, grant)

	audit = auditForGrant(grant, SpawnVerdictAllow, "")
	grant.Audit = audit
	b.appendAudit(audit)
	return grant, nil
}

func (b *SpawnBroker) Audits() []SpawnAudit {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]SpawnAudit, len(b.audits))
	copy(out, b.audits)
	for i := range out {
		out[i].EnvNames = cloneStrings(out[i].EnvNames)
		out[i].CapabilityEnvelope.Capabilities = cloneCaps(out[i].CapabilityEnvelope.Capabilities)
	}
	return out
}

func (b *SpawnBroker) appendAudit(a SpawnAudit) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.audits = append(b.audits, a)
}

func normalizeSpawnAttempt(a SpawnAttempt) (SpawnAttempt, error) {
	a.AgentRunID = strings.TrimSpace(a.AgentRunID)
	a.ParentRunID = strings.TrimSpace(a.ParentRunID)
	a.ToolCallID = strings.TrimSpace(a.ToolCallID)
	a.PolicyDigest = strings.TrimSpace(a.PolicyDigest)
	a.CWD = strings.TrimSpace(a.CWD)
	a.Backend = strings.TrimSpace(a.Backend)

	switch {
	case a.AgentRunID == "":
		return SpawnAttempt{}, fmt.Errorf("MISSING_AGENT_RUN_ID")
	case a.ToolCallID == "":
		return SpawnAttempt{}, fmt.Errorf("MISSING_TOOL_CALL_ID")
	case a.PolicyDigest == "":
		return SpawnAttempt{}, fmt.Errorf("MISSING_POLICY_DIGEST")
	case a.Backend == "":
		return SpawnAttempt{}, fmt.Errorf("MISSING_BACKEND")
	case a.CWD == "":
		return SpawnAttempt{}, fmt.Errorf("MISSING_CWD")
	case len(a.Argv) == 0:
		return SpawnAttempt{}, fmt.Errorf("MISSING_ARGV")
	}
	if strings.ContainsRune(a.AgentRunID+a.ParentRunID+a.ToolCallID+a.PolicyDigest+a.Backend+a.CWD, '\x00') {
		return SpawnAttempt{}, fmt.Errorf("NUL_IN_SPAWN_METADATA")
	}
	a.CWD = filepath.Clean(a.CWD)
	argv := cloneStrings(a.Argv)
	argv[0] = strings.TrimSpace(argv[0])
	for i, arg := range argv {
		if strings.ContainsRune(arg, '\x00') {
			return SpawnAttempt{}, fmt.Errorf("NUL_IN_ARGV")
		}
		if i == 0 && arg == "" {
			return SpawnAttempt{}, fmt.Errorf("MISSING_ARGV0")
		}
	}
	env, err := normalizeEnv(a.Env)
	if err != nil {
		return SpawnAttempt{}, err
	}
	envp := a.Envelope
	if envp.DeadlineMS < 0 || envp.HeartbeatEveryMS < 0 {
		return SpawnAttempt{}, fmt.Errorf("NEGATIVE_CAPABILITY_ENVELOPE")
	}
	envp.Capabilities = normalizeCaps(envp.Capabilities)

	a.Argv = argv
	a.Env = env
	a.Envelope = envp
	return a, nil
}

func normalizeEnv(in []EnvVar) ([]EnvVar, error) {
	out := make([]EnvVar, 0, len(in))
	pos := map[string]int{}
	for _, kv := range in {
		name := strings.TrimSpace(kv.Name)
		if name == "" {
			return nil, fmt.Errorf("EMPTY_ENV_NAME")
		}
		if strings.ContainsAny(name, "=\x00") || strings.ContainsRune(kv.Value, '\x00') {
			return nil, fmt.Errorf("INVALID_ENV")
		}
		if i, ok := pos[name]; ok {
			out[i].Value = kv.Value
			continue
		}
		pos[name] = len(out)
		out = append(out, EnvVar{Name: name, Value: kv.Value})
	}
	return out, nil
}

func normalizeCaps(in []abi.Capability) []abi.Capability {
	out := make([]abi.Capability, 0, len(in))
	seen := map[abi.Capability]bool{}
	for _, c := range in {
		c = abi.Capability(strings.TrimSpace(string(c)))
		if c == "" || seen[c] {
			continue
		}
		seen[c] = true
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func withGrantMetadataEnv(in []EnvVar, g SpawnGrant) []EnvVar {
	out := cloneEnv(in)
	setEnv := func(name, value string) {
		for i := range out {
			if out[i].Name == name {
				out[i].Value = value
				return
			}
		}
		out = append(out, EnvVar{Name: name, Value: value})
	}
	setEnv(EnvAgentRunID, g.AgentRunID)
	setEnv(EnvParentRunID, g.ParentRunID)
	setEnv(EnvToolCallID, g.ToolCallID)
	setEnv(EnvPolicyDigest, g.PolicyDigest)
	setEnv(EnvSpawnBackend, g.Backend)
	setEnv(EnvSpawnGrantID, g.GrantID)
	return out
}

func EnvFromStrings(env []string) ([]EnvVar, error) {
	out := make([]EnvVar, 0, len(env))
	for _, kv := range env {
		i := strings.IndexByte(kv, '=')
		if i < 0 {
			return nil, fmt.Errorf("INVALID_ENV")
		}
		out = append(out, EnvVar{Name: kv[:i], Value: kv[i+1:]})
	}
	return normalizeEnv(out)
}

func EnvStrings(env []EnvVar) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		out = append(out, kv.Name+"="+kv.Value)
	}
	return out
}

func auditForAttempt(a SpawnAttempt) SpawnAudit {
	argv := cloneStrings(a.Argv)
	if len(argv) > 0 {
		argv[0] = strings.TrimSpace(argv[0])
	}
	envNames := envNames(a.Env)
	return SpawnAudit{
		AgentRunID:   strings.TrimSpace(a.AgentRunID),
		ParentRunID:  strings.TrimSpace(a.ParentRunID),
		ToolCallID:   strings.TrimSpace(a.ToolCallID),
		PolicyDigest: strings.TrimSpace(a.PolicyDigest),
		ArgvDigest:   digest(argv),
		Argv0:        argv0(argv),
		EnvNames:     envNames,
		EnvDigest:    digest(a.Env),
		CWD:          cleanOptionalCWD(a.CWD),
		Backend:      strings.TrimSpace(a.Backend),
		CapabilityEnvelope: CapabilityEnvelope{
			Capabilities:     normalizeCaps(a.Envelope.Capabilities),
			DeadlineMS:       a.Envelope.DeadlineMS,
			HeartbeatEveryMS: a.Envelope.HeartbeatEveryMS,
		},
	}
}

func auditForGrant(g SpawnGrant, verdict, reason string) SpawnAudit {
	a := auditForAttempt(SpawnAttempt{
		AgentRunID:   g.AgentRunID,
		ParentRunID:  g.ParentRunID,
		ToolCallID:   g.ToolCallID,
		PolicyDigest: g.PolicyDigest,
		Argv:         g.Argv,
		Env:          g.Env,
		CWD:          g.CWD,
		Backend:      g.Backend,
		Envelope:     g.Envelope,
	})
	a.Verdict = verdict
	a.Reason = reason
	a.GrantID = g.GrantID
	return a
}

func envNames(env []EnvVar) []string {
	set := map[string]bool{}
	for _, kv := range env {
		name := strings.TrimSpace(kv.Name)
		if name == "" || strings.ContainsAny(name, "=\x00") {
			continue
		}
		set[name] = true
	}
	out := make([]string, 0, len(set))
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func grantIDMaterial(g SpawnGrant) any {
	return struct {
		AgentRunID   string
		ParentRunID  string
		ToolCallID   string
		PolicyDigest string
		Argv         []string
		EnvDigest    string
		CWD          string
		Backend      string
		Envelope     CapabilityEnvelope
	}{
		AgentRunID:   g.AgentRunID,
		ParentRunID:  g.ParentRunID,
		ToolCallID:   g.ToolCallID,
		PolicyDigest: g.PolicyDigest,
		Argv:         g.Argv,
		EnvDigest:    digest(g.Env),
		CWD:          g.CWD,
		Backend:      g.Backend,
		Envelope:     g.Envelope,
	}
}

func cleanOptionalCWD(cwd string) string {
	if s := strings.TrimSpace(cwd); s != "" {
		return filepath.Clean(s)
	}
	return ""
}

func argv0(argv []string) string {
	if len(argv) == 0 {
		return ""
	}
	return argv[0]
}

func digest(v any) string {
	b, _ := json.Marshal(v)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortDigest(v any) string {
	d := strings.TrimPrefix(digest(v), "sha256:")
	if len(d) > 16 {
		return d[:16]
	}
	return d
}

func cloneStrings(in []string) []string {
	return append([]string(nil), in...)
}

func cloneEnv(in []EnvVar) []EnvVar {
	return append([]EnvVar(nil), in...)
}

func cloneCaps(in []abi.Capability) []abi.Capability {
	return append([]abi.Capability(nil), in...)
}

func init() {
	abi.RegisterCapability(CapAgentRunSpawn)
}
