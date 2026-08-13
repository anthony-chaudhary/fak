package gitbroker

import "strings"

// gitTerminalPrompt is git's own switch for "may I read a credential off the
// terminal". A fleet worker is headless, so a prompt there is not a slow
// success — nobody will ever answer it — which makes it the INTERACTIVE_HANG
// failure class rather than an error. childEnv pins it to 0 on every child.
const gitTerminalPrompt = "GIT_TERMINAL_PROMPT"

// gitEnvAllow is the closed set of variables inside git's own control namespace
// that a brokered child may still inherit from the parent process.
//
// Each one is here because it steers something the broker does NOT decide and
// git offers no other channel for:
//   - GIT_INDEX_FILE is how a staging operation is deliberately pointed at a
//     throwaway index (internal/patchcommit, cmd/fak's build check, the isolated
//     worker land) — and it is also what git hands a hook, and fak runs as a
//     hook. Dropping it would move those writes onto the shared repository's
//     real index and make a hook answer about a different index than the commit
//     it was called for.
//   - the identity variables are commit attribution, not repository selection.
//   - GIT_OPTIONAL_LOCKS is the broker's own read-side setting.
//   - SSH_AUTH_SOCK names an agent socket, not a program to run, so it carries
//     credentials without carrying code — unlike SSH_ASKPASS or GIT_SSH_COMMAND.
var gitEnvAllow = map[string]bool{
	"GIT_INDEX_FILE":      true,
	"GIT_AUTHOR_NAME":     true,
	"GIT_AUTHOR_EMAIL":    true,
	"GIT_AUTHOR_DATE":     true,
	"GIT_COMMITTER_NAME":  true,
	"GIT_COMMITTER_EMAIL": true,
	"GIT_COMMITTER_DATE":  true,
	"GIT_OPTIONAL_LOCKS":  true,
	"SSH_AUTH_SOCK":       true,
}

// childEnv builds the environment for one brokered git child process.
//
// A brokered call must run against the repository the broker resolved, with the
// configuration the broker chose, and it must never block on a terminal read.
// All three of those are decided by environment variables git reads, so
// inheriting the parent's environment wholesale hands the decision to whatever
// process happened to launch fak. `GIT_DIR` re-aims the operation at another
// repository; the `GIT_CONFIG_*` family injects config — `core.hooksPath`, an
// alias, a `credential.helper` — with no trace at the call site; `GIT_ASKPASS`,
// `SSH_ASKPASS`, `GIT_SSH` and `GIT_SSH_COMMAND` name a program to run.
//
// The rule is an allowlist over git's OWN control surface: every GIT_* and SSH_*
// variable is dropped unless gitEnvAllow names it. Variables outside that
// surface pass through, and that asymmetry is deliberate rather than lazy — a
// brokered `git commit` runs this repository's hooks, which are fak processes
// that read the fleet's FAK_* environment, so scrubbing everything would change
// what those hooks decide, which is a different change than this one.
//
// inv is appended AFTER the filtered parent, because a variable named at the
// call site is a decision the broker's caller made on purpose. What the broker
// refuses is ambient steering nobody chose. gitTerminalPrompt is the one
// exception to that ordering: it is forced last, so no path — ambient or
// per-invocation — can re-enable a prompt a headless worker cannot answer.
func childEnv(parent, inv []string) []string {
	out := make([]string, 0, len(parent)+len(inv)+1)
	for _, kv := range parent {
		name, _, ok := strings.Cut(kv, "=")
		if ok && !inheritableGitEnv(name) {
			continue
		}
		out = append(out, kv)
	}
	for _, kv := range inv {
		if name, _, ok := strings.Cut(kv, "="); ok && envNameIs(name, gitTerminalPrompt) {
			continue
		}
		out = append(out, kv)
	}
	return append(out, gitTerminalPrompt+"=0")
}

// inheritableGitEnv reports whether one parent variable may reach a brokered
// git child.
//
// The comparison is case-insensitive because Windows environment names are:
// `Git_Dir` and `GIT_DIR` are one variable there, and a scrub that only caught
// the shouting spelling would be a scrub in name only.
func inheritableGitEnv(name string) bool {
	u := strings.ToUpper(name)
	// Windows carries per-drive working directories as entries whose name is
	// empty ("=C:=C:\work"). They are not git's, and dropping them would move
	// the child's idea of a drive-relative path.
	if u == "" {
		return true
	}
	if !strings.HasPrefix(u, "GIT_") && !strings.HasPrefix(u, "SSH_") {
		return true
	}
	return gitEnvAllow[u]
}

// envNameIs compares an environment variable name against a canonical spelling
// the same way the host's environment does.
func envNameIs(name, canonical string) bool {
	return strings.EqualFold(name, canonical)
}
