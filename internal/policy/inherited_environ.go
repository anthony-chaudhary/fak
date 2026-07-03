package policy

import "strings"

// secretishEnvNameTokens is the conservative NAME heuristic for env variables
// that are treated as secret-bearing by default, so their VALUE is never handed
// to a child even when a tool rule lists a broad env allow-list. The repo has no
// pre-existing secret-NAME matcher — internal/secretload gates the child env by
// an allow-list (deny-by-default by name) and internal/policy's secretShapedValue
// inspects the VALUE — so this list is defined here, conservatively. Value-shape
// stripping (secretShapedValue, applied inside Resolve) is the second line of
// defense for a secret whose NAME does not match any of these tokens.
var secretishEnvNameTokens = []string{
	"KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL",
}

// looksSecretEnvName reports whether an env variable NAME appears secret-bearing
// under the conservative token heuristic (case-insensitive substring match).
func looksSecretEnvName(name string) bool {
	up := strings.ToUpper(strings.TrimSpace(name))
	if up == "" {
		return false
	}
	for _, tok := range secretishEnvNameTokens {
		if strings.Contains(up, tok) {
			return true
		}
	}
	return false
}

// NewInheritedParentFromEnviron builds an InheritedParent from a LIVE parent
// process environment (a []string of "NAME=VALUE" entries, as returned by
// os.Environ()). It parses each entry into InheritedParent.Env and flags every
// secret-NAMED variable (looksSecretEnvName) into InheritedParent.SecretEnv so it
// is stripped by default when Resolve/ResolveLaunch materializes the child
// envelope. CWD, writable/persistence paths, and egress refs are NOT derivable
// from an environ, so they are left zero for the caller (a launcher) to supply.
//
// This is the single call a launcher makes to turn a raw parent environment into
// a sanitized InheritedParent without hand-building the struct (which would invite
// drift and raw-env leaks). All default-deny, secret-shaped-value stripping, and
// bounded-audit behavior is inherited unchanged from Resolve.
func NewInheritedParentFromEnviron(environ []string) InheritedParent {
	p := InheritedParent{
		Env:       make(map[string]string, len(environ)),
		SecretEnv: map[string]bool{},
	}
	for _, kv := range environ {
		i := strings.IndexByte(kv, '=')
		if i <= 0 {
			continue
		}
		name := kv[:i]
		value := kv[i+1:]
		p.Env[name] = value
		if looksSecretEnvName(name) {
			p.SecretEnv[name] = true
		}
	}
	return p
}

// ResolveFromEnviron is the single-call helper for a launcher (dispatch/gateway):
// it sanitizes a LIVE parent environ into an InheritedParent
// (NewInheritedParentFromEnviron), overlays the caller-supplied non-environ scope
// (CWD, writable/persistence paths, egress refs, and any additional Env/SecretEnv
// entries from extra), and feeds the result through the EXISTING ResolveLaunch so
// every default-deny, secret-name, secret-shaped-value, and audit rule applies
// unchanged. A nil table default-denies: the returned envelope is empty.
//
// Precedence: entries in extra.Env / extra.SecretEnv win over the environ-derived
// values, so a caller can pin or re-flag a specific variable. A tool rule's env
// allow-list still gates which names cross at all; a secret-flagged name never
// crosses regardless of the allow-list.
func (t *InheritedTable) ResolveFromEnviron(tool string, environ []string, extra InheritedParent) InheritedEnvelope {
	parent := NewInheritedParentFromEnviron(environ)
	for name, value := range extra.Env {
		parent.Env[name] = value
	}
	for name, isSecret := range extra.SecretEnv {
		if isSecret {
			parent.SecretEnv[name] = true
		}
	}
	if strings.TrimSpace(extra.CWD) != "" {
		parent.CWD = extra.CWD
	}
	parent.WritablePaths = extra.WritablePaths
	parent.PersistencePaths = extra.PersistencePaths
	parent.EgressRefs = extra.EgressRefs
	return t.ResolveLaunch(tool, parent)
}
