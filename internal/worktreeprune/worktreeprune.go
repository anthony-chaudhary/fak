package worktreeprune

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

var (
	// ErrForeignRegistration refuses a repository-wide prune because at least one
	// registration's gitdir uses a path syntax foreign to the running OS.
	ErrForeignRegistration = errors.New("worktreeprune: broad prune refused: a registration uses a foreign-OS path")
	// ErrRegistrationsUnreadable means the registration set could not be read, so
	// nothing about it can be proven safe.
	ErrRegistrationsUnreadable = errors.New("worktreeprune: worktree registrations unreadable")
	// ErrCommonDirUnresolved means root does not lead to a git common directory.
	ErrCommonDirUnresolved = errors.New("worktreeprune: git common dir unresolved")
	// ErrOwnRegistrationNotFound means no registration points at the caller's worktree.
	ErrOwnRegistrationNotFound = errors.New("worktreeprune: own worktree registration not found")
	// ErrOwnRegistrationLocked means the caller's registration carries a lock and is kept.
	ErrOwnRegistrationLocked = errors.New("worktreeprune: own worktree registration is locked")
	// ErrOwnRegistrationMismatch means the worktree's .git pointer names an admin
	// entry whose gitdir does not point back at that worktree. A stale or forged
	// pointer can name a peer's live entry, so the entry is kept.
	ErrOwnRegistrationMismatch = errors.New("worktreeprune: .git pointer names a registration that does not point back")
)

// Outcome is the closed result of CleanupOwn.
type Outcome string

const (
	OutcomeRemoved Outcome = "own_registration_removed"
	OutcomeAbsent  Outcome = "own_registration_absent"
	OutcomeKept    Outcome = "own_registration_kept"
)

// Registration is one <common>/worktrees/<Name> admin entry.
type Registration struct {
	Name     string
	AdminDir string
	Gitdir   string
	Locked   bool
}

// ReadRegistrations lists every admin entry under commonDir. A missing worktrees
// directory is an empty set; any other read failure is ErrRegistrationsUnreadable.
func ReadRegistrations(commonDir string) ([]Registration, error) {
	root := filepath.Join(commonDir, "worktrees")
	entries, err := os.ReadDir(root)
	if err != nil {
		// Windows reports ReadDir on a regular file as not-exist; only a truly
		// absent directory is an empty set.
		if _, serr := os.Lstat(root); errors.Is(serr, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("%w: %v", ErrRegistrationsUnreadable, err)
	}
	var regs []Registration
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		reg := Registration{Name: entry.Name(), AdminDir: filepath.Join(root, entry.Name())}
		raw, err := os.ReadFile(filepath.Join(reg.AdminDir, "gitdir"))
		switch {
		case err == nil:
			reg.Gitdir = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(raw)), "gitdir: "))
		case !errors.Is(err, os.ErrNotExist):
			return nil, fmt.Errorf("%w: %s: %v", ErrRegistrationsUnreadable, reg.Name, err)
		}
		// The marker itself locks the entry, even if it is a dangling or
		// looping symlink. Only confirmed absence permits removal.
		if _, err := os.Lstat(filepath.Join(reg.AdminDir, "locked")); err == nil {
			reg.Locked = true
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s locked: %v", ErrRegistrationsUnreadable, reg.Name, err)
		}
		regs = append(regs, reg)
	}
	return regs, nil
}

// IsForeignOSPath reports whether p is a gitdir the running OS cannot prove
// live. On Windows: any rooted POSIX path (/mnt/c/..., /home/..., /root/...), a
// volume-less path, or a forward-slash UNC path; Git for Windows records every
// UNC lane as //host/share/... (even from \\host\share input), resolvable only
// while that host, e.g. a WSL distro, is up. Elsewhere: a drive-letter or
// backslash path, or any leading // (//host/share, //?/, //./), since POSIX git
// records realpath()s, which never start with //.
func IsForeignOSPath(p string) bool { return isForeignOSPathFor(runtime.GOOS, p) }

func isForeignOSPathFor(goos, p string) bool {
	if p == "" {
		return false
	}
	if goos == "windows" {
		return strings.HasPrefix(p, "/") || !HasWindowsVolume(p)
	}
	return IsWindowsDrivePath(p) || strings.Contains(p, `\`) || strings.HasPrefix(p, "//")
}

// IsWindowsDrivePath reports whether p starts with a drive letter and separator.
func IsWindowsDrivePath(p string) bool {
	return len(p) >= 3 && isDriveLetter(p[0]) && p[1] == ':' && (p[2] == '/' || p[2] == '\\')
}

// HasWindowsVolume reports whether p carries a drive or UNC volume.
func HasWindowsVolume(p string) bool {
	if len(p) >= 2 && isDriveLetter(p[0]) && p[1] == ':' {
		return true
	}
	return strings.HasPrefix(p, `\\`) || strings.HasPrefix(p, "//")
}

func isDriveLetter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// ForeignRegistrations returns the registrations whose gitdir is foreign to the
// running OS.
func ForeignRegistrations(regs []Registration) []Registration {
	return foreignFor(runtime.GOOS, regs)
}

func foreignFor(goos string, regs []Registration) []Registration {
	var foreign []Registration
	for _, reg := range regs {
		if isForeignOSPathFor(goos, reg.Gitdir) {
			foreign = append(foreign, reg)
		}
	}
	return foreign
}

// GuardBroadPrune returns nil only when every registration under commonDir uses
// the running OS's path syntax. Lock state is not consulted: a refusal costs a
// lingering entry, a wrong allow strands a lane.
func GuardBroadPrune(commonDir string) error { return guardFor(runtime.GOOS, commonDir) }

func guardFor(goos, commonDir string) error {
	regs, err := ReadRegistrations(commonDir)
	if err != nil {
		return err
	}
	foreign := foreignFor(goos, regs)
	if len(foreign) == 0 {
		return nil
	}
	names := make([]string, len(foreign))
	for i, reg := range foreign {
		names[i] = reg.Name
	}
	sort.Strings(names)
	return fmt.Errorf("%w: %d of %d registrations (%s)", ErrForeignRegistration, len(foreign), len(regs), strings.Join(names, ", "))
}

// CommonDir resolves the git common directory for a checkout root, following a
// linked worktree's .git file and its commondir pointer.
func CommonDir(root string) (string, error) {
	dotgit := filepath.Join(root, ".git")
	fi, err := os.Stat(dotgit)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrCommonDirUnresolved, err)
	}
	if fi.IsDir() {
		return dotgit, nil
	}
	admin, err := readDotGitPointer(dotgit)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrCommonDirUnresolved, err)
	}
	if raw, err := os.ReadFile(filepath.Join(admin, "commondir")); err == nil {
		common := filepath.FromSlash(strings.TrimSpace(string(raw)))
		if !filepath.IsAbs(common) {
			common = filepath.Join(admin, common)
		}
		return filepath.Clean(common), nil
	}
	if filepath.Base(filepath.Dir(admin)) == "worktrees" {
		return filepath.Dir(filepath.Dir(admin)), nil
	}
	return admin, nil
}

func readDotGitPointer(dotgit string) (string, error) {
	raw, err := os.ReadFile(dotgit)
	if err != nil {
		return "", err
	}
	s := strings.TrimSpace(string(raw))
	if !strings.HasPrefix(s, "gitdir: ") {
		return "", fmt.Errorf("%s is not a gitdir pointer", dotgit)
	}
	p := filepath.FromSlash(strings.TrimSpace(strings.TrimPrefix(s, "gitdir: ")))
	if !filepath.IsAbs(p) {
		p = filepath.Join(filepath.Dir(dotgit), p)
	}
	return filepath.Clean(p), nil
}

// RemoveOwnRegistration deletes only the admin entry registered for
// worktreePath. It matches the worktree's own .git pointer first, then the
// entry's gitdir, and never touches any other registration.
func RemoveOwnRegistration(commonDir, worktreePath string) (Registration, error) {
	regs, err := ReadRegistrations(commonDir)
	if err != nil {
		return Registration{}, err
	}
	own, err := findOwn(worktreePath, regs)
	if err != nil {
		return Registration{}, err
	}
	if own.Locked {
		return own, fmt.Errorf("%w: %s", ErrOwnRegistrationLocked, own.Name)
	}
	if err := os.RemoveAll(own.AdminDir); err != nil {
		return own, fmt.Errorf("worktreeprune: remove %s: %w", own.AdminDir, err)
	}
	return own, nil
}

// findOwn applies git's validate_worktree pairing: an entry named by the
// worktree's .git pointer is its own only when the entry's gitdir points back at
// that worktree. Without a usable pointer, the entry whose gitdir names the
// worktree's .git is its own.
func findOwn(worktreePath string, regs []Registration) (Registration, error) {
	dotgit := filepath.Join(worktreePath, ".git")
	if admin, err := readDotGitPointer(dotgit); err == nil {
		for _, reg := range regs {
			if !samePath(reg.AdminDir, admin) {
				continue
			}
			if !pointsToWorktree(reg, worktreePath) {
				return Registration{}, fmt.Errorf("%w: %s names %s, whose gitdir is %q", ErrOwnRegistrationMismatch, dotgit, reg.Name, reg.Gitdir)
			}
			return reg, nil
		}
	}
	for _, reg := range regs {
		if pointsToWorktree(reg, worktreePath) {
			return reg, nil
		}
	}
	return Registration{}, fmt.Errorf("%w: %s", ErrOwnRegistrationNotFound, worktreePath)
}

// pointsToWorktree compares checkout locations, not .git file identities:
// distinct checkouts may have hardlinked or symlinked .git files. Parent
// symlinks/junctions and missing checkout tails still use samePath's matching.
func pointsToWorktree(reg Registration, worktreePath string) bool {
	if reg.Gitdir == "" {
		return false
	}
	p := gitdirPath(reg)
	return equalPath(filepath.Base(p), ".git") && samePath(filepath.Dir(p), worktreePath)
}

func gitdirPath(reg Registration) string {
	p := filepath.FromSlash(reg.Gitdir)
	if !filepath.IsAbs(p) {
		p = filepath.Join(reg.AdminDir, p)
	}
	return filepath.Clean(p)
}

// samePath reports whether a and b name one location. git records resolved real
// paths (symlinks and Windows junctions followed) while a caller may hold an
// unresolved one, and the location may already be deleted. So past a literal
// match it requires equal tails below each side's deepest existing ancestor and
// the same ancestor by resolved real path or by file identity; EvalSymlinks does
// not follow junctions, os.Stat does.
func samePath(a, b string) bool {
	ca, cb := filepath.Clean(a), filepath.Clean(b)
	if equalPath(ca, cb) {
		return true
	}
	baseA, tailA, okA := existingAncestor(ca)
	baseB, tailB, okB := existingAncestor(cb)
	if !okA || !okB || !equalPath(tailA, tailB) {
		return false
	}
	if ra, err := filepath.EvalSymlinks(baseA); err == nil {
		if rb, err := filepath.EvalSymlinks(baseB); err == nil && equalPath(ra, rb) {
			return true
		}
	}
	fa, errA := os.Stat(baseA)
	fb, errB := os.Stat(baseB)
	return errA == nil && errB == nil && os.SameFile(fa, fb)
}

func equalPath(a, b string) bool {
	return a == b || (runtime.GOOS == "windows" && strings.EqualFold(a, b))
}

// existingAncestor splits p into its deepest existing ancestor and the tail
// below it; the tail is empty when p itself exists.
func existingAncestor(p string) (base, tail string, ok bool) {
	for base = p; ; {
		if _, err := os.Stat(base); err == nil {
			return base, tail, true
		}
		parent := filepath.Dir(base)
		if parent == base {
			return "", "", false
		}
		tail = filepath.Join(filepath.Base(base), tail)
		base = parent
	}
}

// CleanupOwn is the fallback for a failed `git worktree remove --force <own>`:
// it removes only worktreePath's admin entry under root's common dir and never
// runs a repository-wide prune. A non-nil error always pairs with OutcomeKept.
func CleanupOwn(root, worktreePath string) (Outcome, error) {
	common, err := CommonDir(root)
	if err != nil {
		return OutcomeKept, err
	}
	if _, err := RemoveOwnRegistration(common, worktreePath); err != nil {
		if errors.Is(err, ErrOwnRegistrationNotFound) {
			return OutcomeAbsent, nil
		}
		return OutcomeKept, err
	}
	return OutcomeRemoved, nil
}
