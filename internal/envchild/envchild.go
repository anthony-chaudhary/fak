// Package envchild computes the environment a tool child process may inherit from
// its host.
//
// A gateway process carries upstream provider credentials in its own environment:
// the --api-key-env variable, the --require-key-env bearer variable, and every
// --key-principal target variable. A tool child started from os.Environ() inherits
// all of them, so `env` inside any agent tool call prints the operator's secrets
// (#14450). This package closes that path with one policy that both the codetools
// bash engine and the sandbox L1 engine apply to every child they start:
//
//   - the parent environment is reduced to Allowlist, a closed set of non-secret
//     names; everything else is dropped;
//   - per-spec additions (Options.Extra) are admitted on top, except when their
//     name is protected;
//   - protected names — the well-known gateway key variables plus the secret-shaped
//     families those keys live in — never reach a child, even when a spec adds them.
package envchild

import "strings"

// Allowlist is the closed set of host environment variable names a tool child may
// inherit. It is the opt-in passthrough list: to let another non-secret variable
// through, add it here or pass it per-spec via Options.Extra. Names match
// case-insensitively, which is exact on Unix and correct on Windows, where
// environment names are case-insensitive.
var Allowlist = map[string]bool{
	"PATH": true, "PATHEXT": true, "SYSTEMROOT": true, // executable resolution + cmd.exe
	"HOME": true, "USERPROFILE": true, "HOMEDRIVE": true, "HOMEPATH": true,
	"LANG": true, "LC_ALL": true, "LC_CTYPE": true,
	"TMPDIR": true, "TEMP": true, "TMP": true, "GOTMPDIR": true,
	"TERM": true,
}

// protectedExact are well-known gateway key variable names denied by exact match.
var protectedExact = map[string]bool{
	"FAK_GATEWAY_KEY": true, // canonical --require-key-env bearer
	"FAK_API_KEY":     true,
	"GATEWAY_KEY":     true,
	"API_KEY":         true,
}

// protectedSuffixes are name shapes denied wherever they appear: they cover the
// variable families the gateway keys live in (--api-key-env provider keys,
// --require-key-env bearers, --key-principal targets).
var protectedSuffixes = []string{
	"_API_KEY", "_APIKEY", "_API_TOKEN", "_TOKEN", "_SECRET",
	"_PASSWORD", "_PASSWD", "_CREDENTIAL", "_ACCESS_KEY", "_PRIVATE_KEY", "_KEY",
}

func isProtected(name string) bool {
	up := strings.ToUpper(name)
	if protectedExact[up] {
		return true
	}
	for _, suf := range protectedSuffixes {
		if strings.HasSuffix(up, suf) {
			return true
		}
	}
	return false
}

// Options tunes one ChildEnv computation.
type Options struct {
	// Extra holds KEY=VALUE entries a spec explicitly adds for this child. Each is
	// admitted unless its name is protected; on duplicate names Extra wins over the
	// inherited parent value.
	Extra []string
	// Protected lists additional exact variable names (case-insensitive) that must
	// never reach the child — e.g. the gateway's registered --api-key-env,
	// --require-key-env and --key-principal names.
	Protected []string
}

// ChildEnv returns the environment slice a tool child should run with: the parent
// environment reduced to Allowlist, plus Options.Extra, minus every protected name.
// Malformed entries (no '=' or an empty name) are dropped. The result is never nil.
func ChildEnv(parent []string, opts Options) []string {
	protected := make(map[string]bool, len(opts.Protected))
	for _, n := range opts.Protected {
		if n != "" {
			protected[strings.ToUpper(n)] = true
		}
	}
	denied := func(name string) bool {
		up := strings.ToUpper(name)
		return protected[up] || isProtected(up)
	}

	out := make([]string, 0, len(Allowlist)+len(opts.Extra))
	pos := make(map[string]int, len(out))
	add := func(entry string, allowOnly bool) {
		name, _, ok := strings.Cut(entry, "=")
		if !ok || name == "" || denied(name) {
			return
		}
		if allowOnly && !Allowlist[strings.ToUpper(name)] {
			return
		}
		up := strings.ToUpper(name)
		if i, dup := pos[up]; dup {
			out[i] = entry
			return
		}
		pos[up] = len(out)
		out = append(out, entry)
	}
	for _, entry := range parent {
		add(entry, true)
	}
	for _, entry := range opts.Extra {
		add(entry, false)
	}
	return out
}
