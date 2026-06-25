//go:build linux && (amd64 || arm64)

package guard

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"syscall"
	"unsafe"
)

// Landlock syscall numbers. Go 1.26's syscall package does NOT define SYS_LANDLOCK_* on
// amd64/arm64 (only loong64 carries them), so they are declared here. These three values are
// the same on amd64, arm64, and loong64 — the arches this file is built for (the //go:build
// constraint excludes the rest, which fall through to the no-op twin rather than risk a
// mis-numbered syscall).
const (
	sysLandlockCreateRuleset = 444
	sysLandlockAddRule       = 445
	sysLandlockRestrictSelf  = 446
)

// LANDLOCK_CREATE_RULESET_VERSION: pass to create_ruleset(NULL,0,flags) to query the ABI
// version instead of building a ruleset. The return value is the version (>=1) or -1+errno.
const landlockCreateRulesetVersion = 1 << 0

// oPath is O_PATH — Go's syscall package does not export it on linux (it lives in
// golang.org/x/sys/unix, which the stdlib-only invariant forbids). The value is the same on
// every linux arch this file builds for. A parent fd for a path_beneath rule is "preferably
// opened with O_PATH" per the kernel docs.
const oPath = 0x200000

// landlock_rule_type: the only filesystem rule type.
const landlockRulePathBeneath = 1

// LANDLOCK_ACCESS_FS_* bits (stable across ABI versions; the higher-version-only bits are
// gated by abiAccessMask below so a too-new bit never makes create_ruleset return EINVAL).
const (
	fsExecute    = 1 << 0
	fsWriteFile  = 1 << 1
	fsReadFile   = 1 << 2
	fsReadDir    = 1 << 3
	fsRemoveDir  = 1 << 4
	fsRemoveFile = 1 << 5
	fsMakeChar   = 1 << 6
	fsMakeDir    = 1 << 7
	fsMakeReg    = 1 << 8
	fsMakeSock   = 1 << 9
	fsMakeFifo   = 1 << 10
	fsMakeBlock  = 1 << 11
	fsMakeSym    = 1 << 12
	fsRefer      = 1 << 13 // ABI v2+
	fsTruncate   = 1 << 14 // ABI v3+
	fsIoctlDev   = 1 << 15 // ABI v5+
)

// readClass is what a read-only hook dir gets: read the files, list the dir, run the hooks.
const readClass = fsReadFile | fsReadDir | fsExecute

// fileClassBase is the access set valid on a REGULAR-FILE rule target: only the file-content
// bits. Landlock rejects a rule that carries a directory-semantic bit (MAKE_*/REMOVE_*/
// READ_DIR/REFER) on a non-directory fd, so a file grant must be masked to exactly these.
// (TRUNCATE is added by abiAccessMask when the ABI supports it.)
const fileClassBase = fsWriteFile | fsReadFile | fsExecute

// dirClassBase is the full read+write class valid on a DIRECTORY rule target. The
// ABI-version-specific bits (refer/truncate/ioctl_dev) are added by abiAccessMask only when
// the running kernel supports them.
const dirClassBase = readClass |
	fsWriteFile | fsRemoveDir | fsRemoveFile |
	fsMakeChar | fsMakeDir | fsMakeReg | fsMakeSock | fsMakeFifo | fsMakeBlock | fsMakeSym

// systemRoots is the fixed allowlist of top-level directories the agent legitimately writes.
// Band 1 grants the full dir class on each that EXISTS and is NOT an ancestor of a hook dir.
// It is a fixed list, never an enumeration of "/", so a stray file at "/" is never a rule
// target. A path the agent needs under one of these (e.g. $HOME under /home) inherits the
// grant; the repo's own tree is handled by Band 2.
var systemRoots = []string{
	"/tmp", "/home", "/usr", "/etc", "/var", "/dev", "/proc", "/run",
	"/opt", "/bin", "/sbin", "/lib", "/lib64", "/root", "/mnt", "/srv",
}

// rulesetAttr mirrors struct landlock_ruleset_attr (handled_access_fs is the only field we
// set; the net/scoped fields are left zero, which is valid for a v1 attr of size 8).
type rulesetAttr struct {
	handledAccessFS uint64
}

// pathBeneathAttr mirrors struct landlock_path_beneath_attr, whose UAPI definition is
// __attribute__((packed)) and exactly 12 bytes: __u64 allowed_access; __s32 parent_fd. Go
// lays this out with allowed_access at offset 0 and parent_fd at offset 8 (4 trailing pad
// bytes the kernel never reads — add_rule takes no size arg and reads a fixed 12-byte
// layout, so only offsets 0..11 matter and they match the kernel exactly).
type pathBeneathAttr struct {
	allowedAccess uint64
	parentFD      int32
}

// abiAccessMask returns the access bit sets valid for the running ABI version, clamped so a
// too-new bit never makes create_ruleset fail EINVAL. v1: base; v2: +REFER (dir only); v3:
// +TRUNCATE (file + dir); v5: +IOCTL_DEV (dir). handled is what the ruleset governs (the full
// dir class, which supersets the file + read classes); fileClass/dirClass are the per-target
// grant sets. restrict_self DENIES each handled access no rule re-grants.
func abiAccessMask(version int) (handled, fileClass, dirClass uint64) {
	fileClass = fileClassBase
	dirClass = dirClassBase
	if version >= 2 {
		dirClass |= fsRefer
	}
	if version >= 3 {
		fileClass |= fsTruncate
		dirClass |= fsTruncate
	}
	if version >= 5 {
		dirClass |= fsIoctlDev
	}
	return dirClass, fileClass, dirClass
}

// probeABI calls landlock_create_ruleset(NULL, 0, VERSION). Returns the ABI version (>=1) or
// (-1, errno).
func probeABI() (version int, errno int) {
	r, _, e := syscall.Syscall(sysLandlockCreateRuleset, 0, 0, landlockCreateRulesetVersion)
	if int(r) < 0 || e != 0 {
		return -1, int(e)
	}
	return int(r), 0
}

// LandlockTrampoline is the hidden-verb entry point: it decodes the ruleset spec from args,
// applies the Landlock hook-floor to THIS process, then execs the real agent. EVERY failure
// path fails OPEN — it execs the agent unrestricted with a logged reason — because a
// defense-in-depth floor that silently fails CLOSED would break agents on older hosts. It
// returns only if the final exec itself fails (a genuine, non-fail-open error).
func LandlockTrampoline(args []string) error {
	specTok, agentArgv, ok := SplitTrampolineArgs(args)
	if !ok {
		return fmt.Errorf("guard: landlock trampoline: malformed args (need <spec> -- <agent argv>)")
	}
	spec, err := DecodeSpec(specTok)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — bad spec (%v); spawning agent unrestricted\n", err)
		return execAgent(agentArgv)
	}

	applyHookFloor(spec) // logs + fails open internally; never returns an error to abort the spawn
	return execAgent(agentArgv)
}

// applyHookFloor probes Landlock, builds the union-correct ruleset, and restricts this
// process. Fail-open at every rung: on an unsupported kernel, a build error, or a restrict
// failure it logs one line and returns, leaving the subsequent exec unrestricted.
func applyHookFloor(spec RulesetSpec) {
	if len(spec.ReadOnlyDirs) == 0 || spec.GitDir == "" {
		fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not applied — no hook dir to protect; spawning agent unrestricted")
		return
	}

	version, errno := probeABI()
	if d := DecideFailOpen(version, errno); !d.Apply {
		fmt.Fprintln(os.Stderr, d.Log)
		return
	}

	// no_new_privs is required before restrict_self.
	if err := setNoNewPrivs(); err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — PR_SET_NO_NEW_PRIVS failed (%v); spawning agent unrestricted\n", err)
		return
	}

	handled, fileClass, dirClass := abiAccessMask(version)
	rulesetFD, err := createRuleset(handled)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — create_ruleset failed (%v); spawning agent unrestricted\n", err)
		return
	}
	defer syscall.Close(rulesetFD)

	if err := buildRuleset(rulesetFD, spec, fileClass, dirClass); err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — rule construction failed (%v); spawning agent unrestricted\n", err)
		return
	}

	// restrict_self restricts the CALLING thread (and what it execs/clones). Lock this
	// goroutine to its OS thread so the SAME thread that gets restricted is the one that
	// then calls execAgent — otherwise the runtime could exec on an unrestricted thread.
	runtime.LockOSThread()
	if err := restrictSelf(rulesetFD); err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — restrict_self failed (%v); spawning agent unrestricted\n", err)
		runtime.UnlockOSThread()
		return
	}
	// Intentionally do NOT UnlockOSThread: the very next action is execAgent (syscall.Exec)
	// on this same locked thread, carrying the Landlock domain across execve.
}

// createRuleset builds an empty ruleset handling the given access mask and returns its fd.
func createRuleset(handled uint64) (int, error) {
	attr := rulesetAttr{handledAccessFS: handled}
	r, _, e := syscall.Syscall(sysLandlockCreateRuleset, uintptr(unsafe.Pointer(&attr)), unsafe.Sizeof(attr), 0)
	if e != 0 {
		return -1, e
	}
	return int(r), nil
}

// buildRuleset adds the path_beneath rules that, under Landlock's UNION semantics, make every
// dir in spec.ReadOnlyDirs read-only while the rest of the tree stays writable.
//
// The deny is structural: NO write access is ever granted on an ANCESTOR of a hook dir, and
// Landlock cannot subtract, so a hook dir whose every ancestor (.git, repo root, /) carries no
// write grant is read-only. The rest of the tree is re-granted in three bands:
//
//	Band 1 — a FIXED allowlist of system roots (never an enumeration of /): each that exists
//	  and is not an ancestor of a hook dir gets the full dir class.
//	Band 2 — the SHORT deny-chain decomposition (/ → repoRoot → .git → hooks): at each
//	  ancestor level, grant on every child that is neither another ancestor nor a hook dir.
//	  The grant is MASKED to the child's type — a directory gets dirClass, a regular file gets
//	  fileClass — which keeps repo-root files (README/go.mod) AND .git loose files
//	  (index/HEAD/packed-refs) writable while .git/hooks stays denied. (A directory-semantic
//	  access bit on a regular-file fd is rejected by the kernel; masking is the fix.)
//	Band 3 — the hook dirs themselves: read-only (read + list + execute existing hooks).
//
// Every rule is best-effort: a missing path (ENOENT), a perm error, or an incompatible access
// (EINVAL) skips that one rule, never aborts the build — a single odd sibling must not collapse
// the whole floor to fail-open.
func buildRuleset(fd int, spec RulesetSpec, fileClass, dirClass uint64) error {
	hookDirs := map[string]bool{}
	for _, d := range spec.ReadOnlyDirs {
		hookDirs[filepath.Clean(d)] = true
	}

	// The deny chain: every ancestor (inclusive) of every hook dir, up to /. These dirs get NO
	// write grant; their non-chain children are re-granted in Band 2.
	denyChain := map[string]bool{}
	for d := range hookDirs {
		for p := d; ; {
			denyChain[p] = true
			parent := filepath.Dir(p)
			if parent == p {
				break
			}
			p = parent
		}
	}

	// Band 0: every deny-chain directory itself gets READ-ONLY (read files, list, execute). The
	// chain dirs (repo root, .git, the hook dir's parents) must be TRAVERSABLE and LISTABLE —
	// git reads the work-tree root '.' and walks .git — they just must not be WRITE-granted (that
	// is what would re-enable the hook dir under union semantics). Without this, a jailed `git`
	// fails "could not open directory '.'" because the repo root (an ancestor of an out-of-tree
	// hook dir) carried no read grant. The hook dirs are re-affirmed read-only in Band 3.
	chain := make([]string, 0, len(denyChain))
	for d := range denyChain {
		chain = append(chain, d)
	}
	sort.Strings(chain)
	for _, d := range chain {
		grantReadonly(fd, d)
	}

	// Band 1: fixed system roots (whole-dir grants), skipping any on the deny chain.
	for _, r := range systemRoots {
		if denyChain[filepath.Clean(r)] {
			continue
		}
		grantWritable(fd, r, fileClass, dirClass)
	}

	// Band 2: decompose ONLY the deny chain. At each ancestor level, grant on every child not
	// itself on the chain and not a hook dir, masked to the child's type.
	levels := make([]string, 0, len(denyChain))
	for lvl := range denyChain {
		levels = append(levels, lvl)
	}
	sort.Strings(levels)
	for _, lvl := range levels {
		if hookDirs[lvl] {
			continue // a level that IS a hook dir: its children must stay read-only — do NOT
			// enumerate and write-grant them (Band 3 grants the hook dir read-only as a whole).
		}
		entries, err := os.ReadDir(lvl)
		if err != nil {
			continue // unlistable level — skip; nothing to re-grant here
		}
		for _, e := range entries {
			child := filepath.Clean(filepath.Join(lvl, e.Name()))
			if denyChain[child] || hookDirs[child] {
				continue // on the path to a hook dir — must stay un-write-granted
			}
			grantWritable(fd, child, fileClass, dirClass)
		}
	}

	// Band 3: the hook dirs themselves — read-only.
	hooks := make([]string, 0, len(hookDirs))
	for d := range hookDirs {
		hooks = append(hooks, d)
	}
	sort.Strings(hooks)
	for _, d := range hooks {
		grantReadonly(fd, d)
	}
	// FUNDAMENTAL LIMITATION (not a bug): when a hook dir is UNDER .git (the default
	// .git/hooks), .git is on the deny chain, so it is never granted MAKE_REG and a jailed
	// process cannot create a NEW file directly in .git/ — including git's own .git/index.lock.
	// So a jailed agent cannot run `git commit` ITSELF. This is by design: fak guard channels
	// commits through the UNjailed parent (safecommit + the adjudicated path), not raw git in
	// the sandbox. When hooks are OUT-OF-TREE (core.hooksPath points to a sibling of .git),
	// .git is NOT on the deny chain → Band 2 grants it full dirClass → in-jail git also works,
	// and hooks stay protected. Both behaviors fall out of the deny-chain construction above
	// with no special-casing; see the package doc and the guard docs for the honest boundary.
	return nil
}

// grantWritable adds a path_beneath rule re-granting write on path, MASKED to the fd's type: a
// directory gets dirClass, a regular file gets fileClass (the file-only bits — a
// directory-semantic bit on a file fd is rejected by the kernel). Best-effort: a missing path
// or any add error is swallowed (the access simply stays denied — conservative, never a build
// abort). Opened O_PATH|O_NOFOLLOW so a file target opens and a symlink is not followed at
// construct time.
func grantWritable(rulesetFD int, path string, fileClass, dirClass uint64) {
	pfd, err := syscall.Open(path, oPath|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return
	}
	defer syscall.Close(pfd)
	var st syscall.Stat_t
	if err := syscall.Fstat(pfd, &st); err != nil {
		return
	}
	var access uint64
	switch st.Mode & syscall.S_IFMT {
	case syscall.S_IFDIR:
		access = dirClass
	case syscall.S_IFREG:
		access = fileClass
	default:
		return // symlink/dev/socket/fifo — nothing to re-grant
	}
	addRule(rulesetFD, pfd, access)
}

// grantReadonly adds a read-only rule on a hook dir: read files, list the dir, execute hooks.
func grantReadonly(rulesetFD int, path string) {
	pfd, err := syscall.Open(path, oPath|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return // hook dir absent (ENOENT) → nothing to protect; tolerated
	}
	defer syscall.Close(pfd)
	addRule(rulesetFD, pfd, readClass)
}

// addRule issues landlock_add_rule(PATH_BENEATH) for an already-open parent fd. An incompatible
// access/fd-type (EINVAL) or any other errno is swallowed — one bad rule never aborts the build.
func addRule(rulesetFD, parentFD int, access uint64) {
	attr := pathBeneathAttr{allowedAccess: access, parentFD: int32(parentFD)}
	_, _, _ = syscall.Syscall6(sysLandlockAddRule, uintptr(rulesetFD), landlockRulePathBeneath, uintptr(unsafe.Pointer(&attr)), 0, 0, 0)
}

// restrictSelf enforces the ruleset on the calling thread and its future exec/children.
func restrictSelf(rulesetFD int) error {
	_, _, e := syscall.Syscall(sysLandlockRestrictSelf, uintptr(rulesetFD), 0, 0)
	if e != 0 {
		return e
	}
	return nil
}

// setNoNewPrivs sets PR_SET_NO_NEW_PRIVS, a precondition for landlock_restrict_self.
func setNoNewPrivs() error {
	const prSetNoNewPrivs = 38
	_, _, e := syscall.Syscall6(syscall.SYS_PRCTL, prSetNoNewPrivs, 1, 0, 0, 0, 0)
	if e != 0 {
		return e
	}
	return nil
}

// execAgent replaces this process image with the agent argv, inheriting the current env and
// (when applied) the Landlock domain across execve.
func execAgent(argv []string) error {
	if len(argv) == 0 {
		return fmt.Errorf("guard: landlock trampoline: empty agent argv")
	}
	bin, err := exarLookPath(argv[0])
	if err != nil {
		return fmt.Errorf("guard: landlock trampoline: agent %q not found: %w", argv[0], err)
	}
	return syscall.Exec(bin, argv, os.Environ())
}

// exarLookPath resolves argv[0] to an absolute executable path (syscall.Exec needs a path,
// not a bare name). An absolute/relative path with a separator is used as-is; a bare name is
// resolved against PATH.
func exarLookPath(name string) (string, error) {
	if filepath.Base(name) != name {
		return name, nil // already a path
	}
	return findInPath(name)
}

// findInPath resolves a bare program name against $PATH, returning the first executable match.
func findInPath(name string) (string, error) {
	pathEnv := os.Getenv("PATH")
	for _, dir := range filepath.SplitList(pathEnv) {
		if dir == "" {
			dir = "."
		}
		cand := filepath.Join(dir, name)
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() && fi.Mode()&0o111 != 0 {
			return cand, nil
		}
	}
	return "", fmt.Errorf("%q not found in PATH", name)
}
