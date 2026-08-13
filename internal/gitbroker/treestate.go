package gitbroker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// CLASS B: WORKING-TREE STATE, AND THE ONLY CACHE IN THIS PACKAGE THAT NEEDS AN
// INVALIDATION ARGUMENT (#5623).
//
// Class A (gitbroker.go) caches content-addressed reads, which cannot go stale.
// This file caches the actually-hot query — is the tree dirty, and why — which
// is mutable and shared with peers committing into the same checkout. Every line
// below exists to make one claim checkable: a peer's write busts this cache.
//
// THE KEY. (`.git/index` mtime+size, HEAD OID, refs mtime). A peer's commit
// rewrites the index and moves the branch ref, so it changes at least two of the
// four; a bare `git add` rewrites the index. The key is re-derived on EVERY read
// — this cache is validated, never timed out, so there is no TTL to tune and no
// window during which a known-changed tree is served from memory.
//
// THE STALE-READ BUDGET, STATED (the acceptance gate asks for exactly this).
// A Class B reader can observe a state that predates its call by at most:
//
//   - the in-flight execution it joined, if it coalesced (see singleflight.go —
//     one `git status`), plus
//   - zero for the cache itself: an entry is served only when a freshly sampled
//     key equals the stored key, so a served entry describes a tree that has not
//     changed in any way the key can see.
//
// The residual hazard is therefore ONLY "a mutation the key cannot see", and
// there is exactly one: filesystem timestamp granularity. If a peer rewrote
// `.git/index` to the same size within one mtime tick of our sample, mtime would
// not move and the key would not change. settledFor closes that hole rather than
// arguing about it: an entry is stored only if the sample was taken at least one
// granularity window AFTER the last index/refs write, which makes any subsequent
// write land in a strictly later mtime bucket and therefore visible. A tree
// being actively written degrades to always-fresh instead of being served
// stale — the conservative direction.
//
// So the worst case a Class B reader observes is one in-flight `git status`, and
// that is acceptable for every current Class B caller because a Class B caller
// is by definition one that only reports state. Any caller that FEEDS a commit
// gate, a mutation, or a refusal is Class C, is never cached, is never coalesced,
// and observes a snapshot no older than its own call. That line is enforced by
// the source scan in classc_scan_test.go.

// Class names the invalidation class of a broker query. It travels on the wire
// so the DECISION about how an answer may be reused is made by the caller that
// knows what the answer is for, not guessed at by the broker.
type Class string

const (
	// ClassA — content-addressed and immutable. Keyed by a full OID; see IsOID.
	ClassA Class = "a"
	// ClassB — working-tree state consumed for REPORTING. May be served from the
	// keyed cache in this file and may join an in-flight execution.
	ClassB Class = "b"
	// ClassC — decision-bearing: the answer feeds a commit gate, a mutation, or
	// a refusal. Always computed fresh: never cached, never coalesced.
	ClassC Class = "c"
)

// Decisional reports whether a query must be answered by a fresh execution.
//
// The default is the SAFE one on purpose: ClassC and every unrecognized value —
// including the zero value "", which is what an older client or a caller that
// simply forgot to declare a class sends — are decisional. Forgetting to
// classify costs a spawn; it can never buy a stale refusal.
func (c Class) Decisional() bool { return c != ClassA && c != ClassB }

// TreeState is the working-tree answer: whether the tree is dirty and the
// porcelain status that says why.
type TreeState struct {
	Dirty  bool     `json:"dirty"`
	Status string   `json:"status,omitempty"`
	Key    StateKey `json:"key"`
}

// TreeResult is a TreeState plus the provenance of the answer that produced it.
type TreeResult struct {
	TreeState
	Provenance Provenance `json:"provenance"`
}

// StateKey is the Class B invalidation key. It is comparable, and equality is
// the entire cache-validity test.
type StateKey struct {
	IndexMod  int64  `json:"index_mod"` // .git/index mtime, unix nanoseconds
	IndexSize int64  `json:"index_size"`
	HeadOID   string `json:"head_oid"`
	RefsMod   int64  `json:"refs_mod"` // newest of refs/ and packed-refs, unix nanoseconds
}

// usable reports whether the key names enough of the repository to be trusted as
// an invalidation witness. A key we could not fully sample (no index, no
// resolvable HEAD) is not a weak key — it is no key, and an answer under it is
// never cached.
func (k StateKey) usable() bool { return k.IndexMod != 0 && k.HeadOID != "" }

// settledFor reports whether the tree had been quiet for at least window when
// this key was sampled at sampledAt.
//
// This is the filesystem-granularity guard described at the top of the file. If
// the newest write we can see is at least one window in the past, then any write
// that happens from sampledAt onward must record an mtime in a strictly later
// bucket, so it cannot hide behind our sample. If it is not settled, the answer
// is simply not cached.
func (k StateKey) settledFor(sampledAt time.Time, window time.Duration) bool {
	newest := k.IndexMod
	if k.RefsMod > newest {
		newest = k.RefsMod
	}
	return sampledAt.UnixNano()-newest >= int64(window)
}

// DefaultTreeRaceWindow is the assumed worst-case filesystem mtime granularity.
// 2s covers FAT and the coarse-timestamp network mounts; NTFS (100ns) and ext4
// (1ns) are far finer, so an operator who knows the filesystem can lower it via
// Config.TreeRaceWindow and buy back cache hits on a busy tree.
const DefaultTreeRaceWindow = 2 * time.Second

// TreeRunner computes working-tree state. The Key field of what it returns is
// ignored and overwritten by the broker, which owns key sampling — a backend
// cannot be trusted to report the invalidation witness for its own answer.
type TreeRunner interface {
	TreeState(ctx context.Context) (TreeState, error)
}

// SpawnTreeRunner answers with its own `git status --porcelain` against
// RepoRoot. Like SpawnRunner, this is exactly the pre-broker path, which is what
// makes the fail-open guarantee byte-exact.
type SpawnTreeRunner struct{ RepoRoot string }

// TreeState implements TreeRunner by spawning git.
func (r SpawnTreeRunner) TreeState(ctx context.Context) (TreeState, error) {
	cmd := exec.CommandContext(ctx, "git", "-C", r.RepoRoot, "status", "--porcelain")
	// Same reason as SpawnRunner: this runs under background automation on a
	// Windows fleet, and an unhooked spawn flashes a console window
	// (DESKTOP_POPUP_REGRESSION).
	windowgate.ConfigureBackgroundCommand(cmd)
	// An inherited GIT_DIR/GIT_WORK_TREE would report a different tree's
	// dirtiness under this repository's name. See childEnv.
	cmd.Env = childEnv(os.Environ(), nil)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return TreeState{}, fmt.Errorf("gitbroker: git status --porcelain: %w: %s", err, strings.TrimSpace(errb.String()))
	}
	status := out.String()
	return TreeState{Dirty: strings.TrimSpace(status) != "", Status: status}, nil
}

// resolveGitDir finds the real git directory for repoRoot. `.git` is a directory
// in a normal clone and a FILE containing `gitdir: <path>` in a worktree — this
// fleet's sanctioned per-worker worktrees (#1334) are the second shape, so
// getting it wrong would silently key every worker's cache off a path that does
// not exist.
func resolveGitDir(repoRoot string) (string, error) {
	dot := filepath.Join(repoRoot, ".git")
	fi, err := os.Stat(dot)
	if err != nil {
		return "", err
	}
	if fi.IsDir() {
		return dot, nil
	}
	b, err := os.ReadFile(dot)
	if err != nil {
		return "", err
	}
	line := strings.TrimSpace(string(b))
	rest, ok := strings.CutPrefix(line, "gitdir:")
	if !ok {
		return "", fmt.Errorf("gitbroker: %s is not a gitdir pointer", dot)
	}
	dir := strings.TrimSpace(rest)
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(repoRoot, dir)
	}
	return filepath.Clean(dir), nil
}

// commonDir is where refs actually live. In a worktree the per-worktree gitdir
// holds HEAD and index but delegates refs to the main checkout via `commondir`.
func commonDir(gitDir string) string {
	b, err := os.ReadFile(filepath.Join(gitDir, "commondir"))
	if err != nil {
		return gitDir
	}
	dir := strings.TrimSpace(string(b))
	if dir == "" {
		return gitDir
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(gitDir, dir)
	}
	return filepath.Clean(dir)
}

// sampleStateKey reads the invalidation key for repoRoot and reports whether the
// sample is settled enough to be cacheable under window (see settledFor).
func sampleStateKey(repoRoot string, window time.Duration) (StateKey, bool) {
	gitDir, err := resolveGitDir(repoRoot)
	if err != nil {
		return StateKey{}, false
	}
	now := time.Now()
	common := commonDir(gitDir)

	var k StateKey
	if fi, err := os.Stat(filepath.Join(gitDir, "index")); err == nil {
		k.IndexMod = fi.ModTime().UnixNano()
		k.IndexSize = fi.Size()
	}
	k.HeadOID = headOID(gitDir, common)
	k.RefsMod = newestModTime(
		filepath.Join(common, "refs"),
		filepath.Join(common, "packed-refs"),
	)
	if !k.usable() {
		return k, false
	}
	return k, k.settledFor(now, window)
}

// headOID resolves what HEAD points at, without spawning git. A symref is
// followed into the loose ref and then into packed-refs; a detached HEAD holds
// the OID directly.
func headOID(gitDir, common string) string {
	b, err := os.ReadFile(filepath.Join(gitDir, "HEAD"))
	if err != nil {
		return ""
	}
	head := strings.TrimSpace(string(b))
	rest, isSymref := strings.CutPrefix(head, "ref:")
	if !isSymref {
		if IsOID(head) {
			return head
		}
		return ""
	}
	ref := strings.TrimSpace(rest)
	for _, base := range []string{gitDir, common} {
		if rb, err := os.ReadFile(filepath.Join(base, filepath.FromSlash(ref))); err == nil {
			if oid := strings.TrimSpace(string(rb)); IsOID(oid) {
				return oid
			}
		}
	}
	if oid := packedRefOID(filepath.Join(common, "packed-refs"), ref); oid != "" {
		return oid
	}
	// An unborn branch (`git init`, nothing committed) has no OID at all. Say so
	// distinctly rather than returning "": the tree is real and cacheable, there
	// is simply no commit yet, and a later first commit changes this string.
	return "unborn:" + ref
}

func packedRefOID(path, ref string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "^") {
			continue
		}
		oid, name, ok := strings.Cut(line, " ")
		if ok && strings.TrimSpace(name) == ref && IsOID(oid) {
			return oid
		}
	}
	return ""
}

// newestModTime is the newest mtime among paths that exist, in unix nanoseconds.
func newestModTime(paths ...string) int64 {
	var newest int64
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		if m := fi.ModTime().UnixNano(); m > newest {
			newest = m
		}
	}
	return newest
}

// ErrTreeUnkeyable is returned when working-tree state was asked for as Class B
// but the repository could not be keyed (no index, no resolvable HEAD). The
// answer is still computed and returned; only its cacheability is lost.
var ErrTreeUnkeyable = errors.New("gitbroker: working tree has no usable invalidation key")

// treeCache holds at most ONE Class B answer, because a repository has exactly
// one working tree. There is no eviction policy to get wrong and no budget to
// tune: the single entry is either key-valid or it is gone.
//
// Its accessors are deliberately named lookup/store rather than get/put so that
// the source scan in classc_scan_test.go can find every call site in the package
// by name alone, independent of what any receiver happens to be called.
type treeCache struct {
	mu    sync.Mutex
	key   StateKey
	state TreeState
	full  bool
}

// lookup returns the stored state only if key is byte-identical to the key the
// entry was stored under. Any difference at all is a miss.
func (t *treeCache) lookup(key StateKey) (TreeState, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.full || t.key != key {
		return TreeState{}, false
	}
	return t.state, true
}

// store replaces the single entry.
func (t *treeCache) store(key StateKey, state TreeState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.key, t.state, t.full = key, state, true
}

// held reports whether an entry is resident, for Stats.
func (t *treeCache) held() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.full
}
