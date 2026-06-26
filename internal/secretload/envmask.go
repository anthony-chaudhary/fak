package secretload

import "strings"

// SandboxEnvAllowKey names the config/secretload key that extends the inherited
// child-process environment allow-list. Its value is a comma/semicolon/whitespace
// separated list of variable names, for example:
//
//	FAK_SANDBOX_ENV_ALLOW=HTTP_PROXY,HTTPS_PROXY,SSL_CERT_FILE
const SandboxEnvAllowKey = "FAK_SANDBOX_ENV_ALLOW"

// SandboxEnvInheritKey is the explicit escape hatch for legacy tools that need
// full parent-env inheritance. Any true-ish value ("1", "true", "yes", "all")
// disables inherited-env filtering for that child.
const SandboxEnvInheritKey = "FAK_SANDBOX_INHERIT_ENV"

var defaultSandboxEnvAllow = []string{
	"PATH", "PATHEXT", "COMSPEC", "SystemRoot", "SYSTEMROOT", "WINDIR",
	"HOME", "USER", "USERNAME", "LOGNAME", "SHELL",
	"TEMP", "TMP", "TMPDIR",
	"PWD", "OLDPWD",
	"LANG", "LC_ALL", "LC_CTYPE", "TERM", "COLORTERM", "NO_COLOR", "CI",
	"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
	"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
}

// SandboxEnv returns the child environment for a sandbox/adapter exec boundary:
// parent variables are deny-by-default except for a small platform/runtime
// allow-list plus names supplied through SandboxEnvAllowKey, while explicit
// key=value entries supplied by the caller always cross the boundary.
func SandboxEnv(environ []string, explicit ...string) []string {
	return SandboxEnvWithLoader(Default(), environ, explicit...)
}

// SandboxEnvWithLoader is SandboxEnv with an injected loader for tests and for
// hosts that source policy from an encrypted file or vault-backed SecretSource.
func SandboxEnvWithLoader(loader *Loader, environ []string, explicit ...string) []string {
	if loader == nil {
		loader = Default()
	}
	if inheritAllSandboxEnv(loader) {
		return mergeExplicit(environ, explicit)
	}

	allow := make(map[string]struct{}, len(defaultSandboxEnvAllow)+8)
	for _, k := range defaultSandboxEnvAllow {
		allow[k] = struct{}{}
	}
	if raw, ok := loader.Lookup(SandboxEnvAllowKey); ok {
		for _, k := range parseEnvNameList(raw) {
			allow[k] = struct{}{}
		}
	}

	explicitKeys := envKeySet(explicit)
	out := make([]string, 0, len(allow)+len(explicit))
	seen := map[string]struct{}{}
	for _, kv := range environ {
		k, ok := envKey(kv)
		if !ok {
			continue
		}
		if _, keep := allow[k]; !keep {
			continue
		}
		if _, overridden := explicitKeys[k]; overridden {
			continue
		}
		if _, dup := seen[k]; dup {
			continue
		}
		out = append(out, kv)
		seen[k] = struct{}{}
	}
	return append(out, explicit...)
}

func inheritAllSandboxEnv(loader *Loader) bool {
	v, ok := loader.Lookup(SandboxEnvInheritKey)
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "y", "on", "all", "inherit", "full":
		return true
	default:
		return false
	}
}

func mergeExplicit(environ []string, explicit []string) []string {
	explicitKeys := envKeySet(explicit)
	out := make([]string, 0, len(environ)+len(explicit))
	seen := map[string]struct{}{}
	for _, kv := range environ {
		k, ok := envKey(kv)
		if !ok {
			continue
		}
		if _, overridden := explicitKeys[k]; overridden {
			continue
		}
		if _, dup := seen[k]; dup {
			continue
		}
		out = append(out, kv)
		seen[k] = struct{}{}
	}
	return append(out, explicit...)
}

func envKeySet(env []string) map[string]struct{} {
	out := make(map[string]struct{}, len(env))
	for _, kv := range env {
		if k, ok := envKey(kv); ok {
			out[k] = struct{}{}
		}
	}
	return out
}

func envKey(kv string) (string, bool) {
	i := strings.IndexByte(kv, '=')
	if i <= 0 {
		return "", false
	}
	return kv[:i], true
}

func parseEnvNameList(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == ';' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" || strings.Contains(f, "=") {
			continue
		}
		out = append(out, f)
	}
	return out
}
