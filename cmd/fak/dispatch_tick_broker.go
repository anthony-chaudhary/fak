package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/url"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

const launchBrokerMetadataSchema = "agent-run-spawn-adapter/1"

type launchBrokerMetadata struct {
	Schema         string
	Surface        string
	Backend        string
	AgentRunID     string
	ParentRunID    string
	ToolCallID     string
	PolicyDigest   string
	ArgvDigest     string
	EnvDigest      string
	EnvCount       int
	SecretEnvCount int
	CWD            string
}

type launchBrokerAttempt struct {
	Surface  string
	Backend  string
	Argv     []string
	Env      map[string]string
	CWD      string
	Metadata launchBrokerMetadata
	Spawn    toolprocgate.SpawnAttempt
}

type launchBrokerGrant struct {
	Allow         bool
	Reason        string
	Metadata      launchBrokerMetadata
	Argv          []string
	Env           map[string]string
	CWD           string
	SanitizedArgv []string
	SanitizedEnv  map[string]string
	SanitizedCWD  string
	SpawnGrant    toolprocgate.SpawnGrant
}

var launchSpawnBroker = defaultLaunchSpawnBroker

func defaultLaunchSpawnBroker(a launchBrokerAttempt) launchBrokerGrant {
	grant, err := toolprocgate.NewSpawnBroker().Admit(a.Spawn)
	if err != nil {
		reason := err.Error()
		var denied toolprocgate.SpawnDeniedError
		if errors.As(err, &denied) && denied.Audit.Reason != "" {
			reason = denied.Audit.Reason
		}
		return denyLaunchBrokerGrant(a, strings.TrimSpace(reason))
	}
	return grantLaunchBrokerGrant(a, grant, "allowed")
}

func allowLaunchBrokerGrant(a launchBrokerAttempt, reason string) launchBrokerGrant {
	return grantLaunchBrokerGrant(a, toolprocgate.SpawnGrant{
		AgentRunID:   a.Metadata.AgentRunID,
		ParentRunID:  a.Metadata.ParentRunID,
		ToolCallID:   a.Metadata.ToolCallID,
		PolicyDigest: a.Metadata.PolicyDigest,
		Argv:         append([]string(nil), a.Argv...),
		Env:          launchBrokerEnvVars(a.Env),
		CWD:          strings.TrimSpace(a.CWD),
		Backend:      a.Metadata.Backend,
	}, reason)
}

func grantLaunchBrokerGrant(a launchBrokerAttempt, grant toolprocgate.SpawnGrant, reason string) launchBrokerGrant {
	if strings.TrimSpace(reason) == "" {
		reason = "allowed"
	}
	env := launchBrokerEnvMap(grant.Env)
	return launchBrokerGrant{
		Allow:         true,
		Reason:        reason,
		Metadata:      a.Metadata,
		Argv:          append([]string(nil), grant.Argv...),
		Env:           env,
		CWD:           grant.CWD,
		SanitizedArgv: launchBrokerRedactedArgv(grant.Argv),
		SanitizedEnv:  launchBrokerRedactedEnv(env),
		SanitizedCWD:  launchBrokerRedactedCWD(grant.CWD),
		SpawnGrant:    grant,
	}
}

func denyLaunchBrokerGrant(a launchBrokerAttempt, reason string) launchBrokerGrant {
	if strings.TrimSpace(reason) == "" {
		reason = "denied"
	}
	g := allowLaunchBrokerGrant(a, reason)
	g.Allow = false
	return g
}

func newLaunchBrokerAttempt(surface, backend string, argv []string, env map[string]string, cwd string) launchBrokerAttempt {
	a := launchBrokerAttempt{
		Surface: strings.TrimSpace(surface),
		Backend: strings.TrimSpace(backend),
		Argv:    append([]string(nil), argv...),
		Env:     copyStringMap(env),
		CWD:     strings.TrimSpace(cwd),
	}
	envShape := launchBrokerEnvShape(a.Env)
	a.Metadata = launchBrokerMetadata{
		Schema:         launchBrokerMetadataSchema,
		Surface:        a.Surface,
		Backend:        a.Backend,
		ParentRunID:    firstNonEmpty(a.Env["FAK_AGENT_RUN_ID"], a.Env["AGENT_RUN_ID"], a.Env["CLAUDE_CODE_SESSION_ID"]),
		ToolCallID:     firstNonEmpty(a.Env["TOOL_CALL_ID"], a.Env["CLAUDE_TOOL_CALL_ID"]),
		ArgvDigest:     "argv-sha256:" + launchBrokerDigest(a.Argv),
		EnvDigest:      "env-sha256:" + launchBrokerDigest(envShape),
		EnvCount:       len(a.Env),
		SecretEnvCount: launchBrokerSecretEnvCount(a.Env),
		CWD:            launchBrokerRedactedCWD(a.CWD),
	}
	if a.Metadata.ToolCallID == "" {
		a.Metadata.ToolCallID = "toolcall-" + launchBrokerDigest([]string{a.Surface, a.Backend, a.Metadata.ArgvDigest})
	}
	a.Metadata.PolicyDigest = "policy-sha256:" + launchBrokerDigest([]string{
		a.Metadata.Surface,
		a.Metadata.Backend,
		a.Metadata.ArgvDigest,
		a.Metadata.EnvDigest,
		a.Metadata.CWD,
	})
	a.Metadata.AgentRunID = "agentrun-" + launchBrokerDigest([]string{
		a.Metadata.ParentRunID,
		a.Metadata.ToolCallID,
		a.Metadata.PolicyDigest,
	})
	a.Spawn = toolprocgate.SpawnAttempt{
		AgentRunID:   a.Metadata.AgentRunID,
		ParentRunID:  a.Metadata.ParentRunID,
		ToolCallID:   a.Metadata.ToolCallID,
		PolicyDigest: a.Metadata.PolicyDigest,
		Argv:         append([]string(nil), a.Argv...),
		Env:          launchBrokerEnvVars(a.Env),
		CWD:          strings.TrimSpace(a.CWD),
		Backend:      a.Metadata.Backend,
		Envelope: toolprocgate.CapabilityEnvelope{
			Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn},
		},
	}
	return a
}

func launchBrokerGrantMap(g launchBrokerGrant) map[string]any {
	out := launchBrokerMetadataMap(g.Metadata)
	out["allow"] = g.Allow
	out["reason"] = g.Reason
	if g.SpawnGrant.GrantID != "" {
		out["grant_id"] = g.SpawnGrant.GrantID
	}
	return out
}

func launchBrokerEnvVars(env map[string]string) []toolprocgate.EnvVar {
	out := make([]toolprocgate.EnvVar, 0, len(env))
	for _, key := range sortedMapKeys(env) {
		out = append(out, toolprocgate.EnvVar{Name: key, Value: env[key]})
	}
	return out
}

func launchBrokerEnvMap(env []toolprocgate.EnvVar) map[string]string {
	out := map[string]string{}
	for _, kv := range env {
		out[kv.Name] = kv.Value
	}
	return out
}

func launchBrokerMetadataMap(m launchBrokerMetadata) map[string]any {
	out := map[string]any{
		"schema":        m.Schema,
		"surface":       m.Surface,
		"agent_run_id":  m.AgentRunID,
		"policy_digest": m.PolicyDigest,
		"argv_digest":   m.ArgvDigest,
		"env_digest":    m.EnvDigest,
		"env_count":     m.EnvCount,
		"cwd":           m.CWD,
	}
	if m.Backend != "" {
		out["backend"] = m.Backend
	}
	if m.ParentRunID != "" {
		out["parent_run_id"] = m.ParentRunID
	}
	if m.ToolCallID != "" {
		out["tool_call_id"] = m.ToolCallID
	}
	if m.SecretEnvCount > 0 {
		out["secret_env_count"] = m.SecretEnvCount
	}
	return out
}

func launchBrokerRedactedArgv(argv []string) []string {
	out := make([]string, 0, len(argv))
	redactNext := false
	for _, arg := range argv {
		if redactNext {
			out = append(out, "<redacted>")
			redactNext = false
			continue
		}
		if k, v, ok := strings.Cut(arg, "="); ok {
			if launchBrokerSensitiveKey(strings.TrimLeft(k, "-")) {
				out = append(out, k+"=<redacted>")
				continue
			}
			if shaped := launchBrokerRedactURL(v); shaped != v {
				out = append(out, k+"="+shaped)
				continue
			}
		}
		shaped := launchBrokerRedactURL(arg)
		out = append(out, shaped)
		if strings.HasPrefix(arg, "-") && launchBrokerSensitiveKey(strings.TrimLeft(arg, "-")) && !strings.Contains(arg, "=") {
			redactNext = true
		}
	}
	return out
}

func launchBrokerRedactedEnv(env map[string]string) map[string]string {
	out := map[string]string{}
	for _, key := range sortedMapKeys(env) {
		if launchBrokerSensitiveKey(key) {
			out[key] = "<redacted>"
		} else {
			out[key] = "<set>"
		}
	}
	return out
}

func launchBrokerEnvShape(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for _, key := range sortedMapKeys(env) {
		class := "set"
		if launchBrokerSensitiveKey(key) {
			class = "secret"
		}
		out = append(out, key+"="+class)
	}
	return out
}

func launchBrokerSecretEnvCount(env map[string]string) int {
	n := 0
	for key, value := range env {
		if strings.TrimSpace(value) != "" && launchBrokerSensitiveKey(key) {
			n++
		}
	}
	return n
}

func launchBrokerRedactedCWD(cwd string) string {
	if strings.TrimSpace(cwd) == "" {
		return ""
	}
	return "<cwd>"
}

func launchBrokerRedactURL(s string) string {
	u, err := url.Parse(s)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return s
	}
	u.User = nil
	u.RawQuery = ""
	return u.String()
}

func launchBrokerSensitiveKey(s string) bool {
	low := strings.ToLower(strings.TrimSpace(s))
	for _, needle := range []string{
		"token", "oauth", "api-key", "apikey", "api_key", "authorization",
		"bearer", "secret", "password", "credential", "cookie",
	} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

func launchBrokerDigest(parts []string) string {
	h := sha256.New()
	for _, part := range parts {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func copyStringMap(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sortedMapKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
