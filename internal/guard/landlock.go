// Package guard holds the agent-spawn containment seam for `fak guard`. Its first
// member is the Linux Landlock HOOK-FLOOR: when an operator opts in, the agent child
// `fak guard` spawns runs under a kernel ruleset that makes the git hook surface
// (.git/hooks and any core.hooksPath dir) READ-ONLY while the rest of the tree stays
// writable, so a laundered write — `cp`, an editor, an os.WriteFile the argv prefilter
// never sees — cannot drop an executable hook that fires on the next commit.
//
// WHY A RE-EXEC TRAMPOLINE, NOT IN-PROCESS. landlock_restrict_self restricts the calling
// thread (and everything it later execs / clones), and the restriction is INHERITED across
// execve. The parent `fak guard` process IS the gateway: it serves the adjudication proxy
// and writes the durable decision journal, so it must stay UNrestricted. So instead of
// jailing the parent (which syscall.AllThreadsSyscall could, but it is the wrong process
// and is cgo-fragile), `fak guard` re-execs ITSELF as a hidden trampoline verb that applies
// Landlock to itself and then execs the real agent — jailing only the agent subtree, with
// the gateway left free by construction.
//
// THE UNION-SEMANTICS RULE (the part that is easy to get wrong). Landlock is allow-list and
// UNIONS accesses across every rule on an accessed file's ancestor path — you CANNOT subtract.
// A read-only rule on .git/hooks does nothing if any ANCESTOR (.git, the repo root, /) was
// granted write. So the deny is achieved structurally: never grant a write-class access on an
// ANCESTOR of .git/hooks. The ruleset grants full access on every sibling at each level down
// to .git, grants full on .git's children EXCEPT hooks, and grants read-only on the hook dirs
// themselves. See buildRuleset (landlock_linux.go).
//
// HONEST SCOPE (Linux-only, opt-in, defense-in-depth): Landlock is a Linux LSM (kernel 5.13+).
// macOS/Windows fleet members get NOTHING from this — the same platform ceiling Codex hits
// with bwrap (Linux) vs Seatbelt (macOS). It is OFF by default, does NOT replace the gitgate
// prefilter or the git hooks, protects only the hook PATH (not every laundering vector), and
// FAILS OPEN with a log line on a kernel without Landlock so an older host is never blocked.
//
// This file is the pure, cross-platform core: the spec the parent serializes for the
// trampoline, the hook-dir resolution policy, and the fail-open decision table — all unit-
// testable on any OS. The raw syscalls live in landlock_linux.go; the no-op twin in
// landlock_other.go.
package guard

import (
	"encoding/base64"
	"encoding/json"
	"path/filepath"
	"strings"
)

// EnvOptIn is the environment variable the parent sets (from the --landlock-hooks flag) to
// turn the hook-floor on for the spawned child. OFF unless set to 1/on/true. The flag writes
// it so buildGuardChild can consult one source without a signature change at both call sites.
const EnvOptIn = "FAK_GUARD_LANDLOCK"

// TrampolineVerb is the hidden `fak` sub-verb the parent re-execs into. It is intentionally
// undocumented (not in usage()): an internal implementation detail of the spawn seam, not a
// user-facing command. Mirrors the repo's existing `__fak_*__` hidden-token convention.
const TrampolineVerb = "__guard_landlock_exec__"

// OptedIn reports whether the hook-floor is enabled, given a getenv function. Pure so a test
// drives it without touching the real environment.
func OptedIn(getenv func(string) string) bool {
	switch strings.ToLower(strings.TrimSpace(getenv(EnvOptIn))) {
	case "1", "on", "true", "yes":
		return true
	}
	return false
}

// RulesetSpec is the serialized hook-floor the parent hands the trampoline. ReadOnlyDirs are
// the absolute hook directories to make read-only (.git/hooks + a resolved core.hooksPath).
// GitDir and RepoRoot anchor the write-grant decomposition the linux builder performs. All
// paths are absolute and already resolved by the parent (which has git available); the
// trampoline does no git itself.
type RulesetSpec struct {
	RepoRoot     string   `json:"repo_root"`      // absolute work-tree root ("" if bare / no work tree)
	GitDir       string   `json:"git_dir"`        // absolute .git dir (real, via --absolute-git-dir)
	ReadOnlyDirs []string `json:"read_only_dirs"` // absolute dirs to make read-only (hook dirs)
}

// Encode serializes a spec for the trampoline argv. base64(JSON) so the spec is one opaque,
// shell-safe token regardless of the paths inside it (spaces, quotes, non-ASCII).
func (s RulesetSpec) Encode() string {
	b, _ := json.Marshal(s)
	return base64.RawStdEncoding.EncodeToString(b)
}

// DecodeSpec parses a token produced by Encode. A malformed token is an error the trampoline
// turns into fail-open (exec the agent unrestricted), never a hard failure.
func DecodeSpec(tok string) (RulesetSpec, error) {
	var s RulesetSpec
	b, err := base64.RawStdEncoding.DecodeString(strings.TrimSpace(tok))
	if err != nil {
		return s, err
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return s, err
	}
	return s, nil
}

// TrampolineArgv builds the argv that re-execs fak as the trampoline: the fak binary, the
// hidden verb, the encoded spec, a "--" separator, then the real agent argv. The trampoline
// applies Landlock to itself and execs everything after "--".
func TrampolineArgv(fakBin string, spec RulesetSpec, agentArgv []string) []string {
	out := make([]string, 0, len(agentArgv)+4)
	out = append(out, fakBin, TrampolineVerb, spec.Encode(), "--")
	out = append(out, agentArgv...)
	return out
}

// SplitTrampolineArgs splits the trampoline's own argv (everything after the verb) into the
// encoded spec and the agent argv at the "--" separator. Returns ok=false if there is no
// separator or no agent command after it — the trampoline then fails open on the raw args.
func SplitTrampolineArgs(args []string) (specTok string, agentArgv []string, ok bool) {
	if len(args) < 1 {
		return "", nil, false
	}
	specTok = args[0]
	rest := args[1:]
	for i, a := range rest {
		if a == "--" {
			agentArgv = rest[i+1:]
			return specTok, agentArgv, len(agentArgv) > 0
		}
	}
	return "", nil, false
}

// ResolveSpec computes the read-only hook directories from git's OWN resolution outputs —
// NEVER by string-concatenating root + "/.git/hooks" (that breaks linked worktrees and
// submodules, where .git is a file). The caller (the parent, which has git) supplies:
//   - repoRoot:  `git rev-parse --show-toplevel`  ("" for a bare repo / no work tree)
//   - gitDir:    `git rev-parse --absolute-git-dir`
//   - hooksPath: `git rev-parse --git-path hooks`  (git's resolved hooks dir, honoring config)
//   - bare:      `git rev-parse --is-bare-repository` == "true"
//
// hooksPath from --git-path is relative to the cwd git ran in when the config value is
// relative, so it is joined against repoRoot (or gitDir, for a bare repo) when not absolute.
// The result is the set of absolute dirs to mark read-only — the canonical .git/hooks plus,
// when core.hooksPath points elsewhere, that directory too. A blank gitDir yields no spec
// (nothing to protect → the trampoline fails open).
func ResolveSpec(repoRoot, gitDir, hooksPath string, bare bool) RulesetSpec {
	gitDir = strings.TrimSpace(gitDir)
	repoRoot = strings.TrimSpace(repoRoot)
	spec := RulesetSpec{RepoRoot: repoRoot, GitDir: gitDir}
	if gitDir == "" {
		return spec // no git dir resolved → empty spec → fail open
	}

	dirs := map[string]bool{}
	add := func(p string) {
		p = strings.TrimSpace(p)
		if p == "" {
			return
		}
		if !filepath.IsAbs(p) {
			anchor := repoRoot
			if anchor == "" || bare {
				anchor = gitDir
			}
			p = filepath.Join(anchor, p)
		}
		dirs[filepath.Clean(p)] = true
	}

	// The canonical hook dir is always under the real git dir.
	add(filepath.Join(gitDir, "hooks"))
	// git's resolved hooks dir (honors core.hooksPath); usually equals the above, but when
	// core.hooksPath is configured it is the (possibly external) override dir.
	add(hooksPath)

	for d := range dirs {
		spec.ReadOnlyDirs = append(spec.ReadOnlyDirs, d)
	}
	// Deterministic order so the spec + any test assertion is stable.
	sortStrings(spec.ReadOnlyDirs)
	return spec
}

// sortStrings is a tiny insertion sort kept local so the pure file imports no extra package
// (sort would be fine, but this keeps the dependency surface minimal and the order obvious).
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// FailOpenReason is the typed outcome of the ABI/probe decision. Apply means "build and
// enforce the ruleset"; any other value means "exec the agent unrestricted" and Log carries
// the one-line stderr explanation.
type FailOpenReason struct {
	Apply bool   // true => proceed to build+enforce; false => fail open (exec unrestricted)
	Log   string // the single stderr line to emit (always set when Apply is false)
}

// errno constants the probe maps. They mirror the Linux errno values without importing the
// build-tagged syscall package into this cross-platform file, so the decision table is
// testable on any OS.
const (
	errnoNone       = 0
	errnoENOSYS     = 38 // function not implemented — no Landlock syscall (pre-5.13)
	errnoEOPNOTSUPP = 95 // not supported — Landlock compiled but disabled at boot (lsm=)
)

// DecideFailOpen maps a landlock_create_ruleset(NULL,0,VERSION) probe result to the action.
// version is the syscall's return value (the ABI version when >= 1) and errno is the errno
// when it returned -1 (0 otherwise). The contract (issue #824): supported (version >= 1) =>
// Apply; ENOSYS/EOPNOTSUPP/version<1/any errno => fail open with a specific reason. The
// caller never blocks the spawn — every non-Apply branch still execs the agent.
func DecideFailOpen(version int, errno int) FailOpenReason {
	switch {
	case errno == errnoENOSYS:
		return FailOpenReason{Apply: false, Log: "fak guard: landlock hook-floor not applied — kernel has no Landlock (ENOSYS); spawning agent unrestricted"}
	case errno == errnoEOPNOTSUPP:
		return FailOpenReason{Apply: false, Log: "fak guard: landlock hook-floor not applied — Landlock disabled at boot (EOPNOTSUPP); spawning agent unrestricted"}
	case errno != errnoNone:
		return FailOpenReason{Apply: false, Log: "fak guard: landlock hook-floor not applied — Landlock probe failed; spawning agent unrestricted"}
	case version < 1:
		return FailOpenReason{Apply: false, Log: "fak guard: landlock hook-floor not applied — Landlock ABI < 1; spawning agent unrestricted"}
	default:
		return FailOpenReason{Apply: true}
	}
}
