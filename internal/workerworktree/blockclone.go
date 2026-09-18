package workerworktree

import (
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
)

const (
	gitWorktreeBackendName = "git-worktree"
	blockCloneBackendName  = "block-clone"
)

// ErrBlockCloneUnsupported reports that the host platform or filesystem cannot
// provide copy-on-write block/directory cloning, so callers should fall back.
var ErrBlockCloneUnsupported = errors.New("block clone unsupported")

// treeCloneFastPath counts how many block-clone materializations used the
// whole-tree native clone fast path instead of the per-file fallback loop.
var treeCloneFastPath atomic.Int64

// TreeCloneFastPathCount returns the number of materializations that took the
// whole-tree native clone fast path. It exists so tests can witness the fast
// path was exercised.
func TreeCloneFastPathCount() int64 { return treeCloneFastPath.Load() }

// CloneTree clones the directory tree at src to dst using the host's native
// copy-on-write clone primitive when available.
func CloneTree(src, dst string) error { return cloneTree(src, dst) }

// cloneTreeWalk recursively recreates src's tree at dst, cloning each regular
// file with cloneFileBlocks and degrading to a byte copy when cloning is
// unsupported. Directories and symlinks are recreated; device/fifo/socket
// nodes are skipped. It is the shared fallback for the per-OS cloneTree.
func cloneTreeWalk(src, dst string) error {
	info, err := os.Lstat(src)
	if err != nil {
		return err
	}
	switch {
	case info.IsDir():
		if err := mkdirForClone(dst, info.Mode().Perm()); err != nil {
			return err
		}
		entries, err := os.ReadDir(src)
		if err != nil {
			return err
		}
		for _, entry := range entries {
			if err := cloneTreeWalk(filepath.Join(src, entry.Name()), filepath.Join(dst, entry.Name())); err != nil {
				return err
			}
		}
		return nil
	case info.Mode().IsRegular():
		if err := cloneFileBlocks(src, dst); err == nil {
			return nil
		}
		return copyFileBytes(src, dst, info.Mode().Perm())
	case info.Mode()&os.ModeSymlink != 0:
		target, err := os.Readlink(src)
		if err != nil {
			return err
		}
		return os.Symlink(target, dst)
	default:
		// Device/fifo/socket nodes are not cloned.
		return nil
	}
}

// mkdirForClone creates dst as a directory, or accepts it if it already exists
// as a real directory. It deliberately uses Lstat: a pre-existing SYMLINK (or
// any non-directory) at dst is an error, never followed or silently accepted.
// Following a symlink would let the walk write source files outside the
// destination tree, and accepting a non-directory would report a successful
// clone of an empty source directory that never actually materialized.
func mkdirForClone(dst string, perm os.FileMode) error {
	if err := os.Mkdir(dst, perm); err == nil {
		return nil
	} else if !errors.Is(err, fs.ErrExist) {
		return err
	}
	existing, err := os.Lstat(dst)
	if err != nil {
		return err
	}
	if existing.Mode()&os.ModeSymlink != 0 {
		return &os.PathError{Op: "clone", Path: dst, Err: errors.New("destination is a symlink, refusing to follow it")}
	}
	if !existing.IsDir() {
		return &os.PathError{Op: "clone", Path: dst, Err: errors.New("destination exists and is not a directory")}
	}
	return nil
}

// copyFileBytes is the last-resort fallback when a filesystem cannot clone a
// regular file. It preserves the source permission bits.
func copyFileBytes(src, dst string, perm os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, perm)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

type blockCloneProbe func(targetRoot string) error
type blockCloneFile func(src, dst string) error
type blockCloneTree func(src, dst string) error

type blockClone struct {
	probe     blockCloneProbe
	clone     blockCloneFile
	treeClone blockCloneTree
}

func newBlockCloneBackend() blockClone {
	return blockClone{probe: probeBlockClone, clone: cloneFileBlocks, treeClone: CloneTree}
}

var _ ownedIsolationBackend = blockClone{}

func (b blockClone) Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	return b.MaterializeOwned(root, lane, key, baseSHA, wtRoot, git, defaultOwnerStamp(lane))
}

func (b blockClone) MaterializeOwned(root, lane, key, baseSHA, wtRoot string, git GitRunner, owner OwnerStamp) Result {
	targetRoot := resolveWorktreeRoot(wtRoot)
	if err := b.probe(targetRoot); err != nil {
		res := gitWorktree{}.MaterializeOwned(root, lane, key, baseSHA, wtRoot, git, owner)
		res.Backend = gitWorktreeBackendName
		if res.OK {
			res.Detail = "block-clone unavailable; fell back to git-worktree: " + err.Error()
		}
		return res
	}
	base := baseSHA
	if base == "" {
		base = TrunkHeadSHA(root, git)
	}
	if base == "" {
		return Result{OK: false, Reason: "could not resolve trunk HEAD (git error) — fail open"}
	}
	wt := Path(lane, key, wtRoot)
	if _, err := os.Stat(wt); err == nil {
		rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
		if rc == 0 {
			for _, p := range parseWorktreePaths(out) {
				if samePath(p, wt) {
					if PoolCap() > 0 {
						if meta, err := readPoolMember(wt); err == nil && meta.State == poolStateIdle {
							if res, ok := leaseSpecificPooled(root, wt, base, git, owner); ok {
								return res
							}
							return Result{OK: false, Path: wt, BaseSHA: base,
								Reason: "same-key idle pool member could not be leased — fail open"}
						}
					}
					return Result{OK: true, Path: wt, BaseSHA: base, Reused: true}
				}
			}
		}
	}
	if k := PoolCap(); k > 0 {
		if res, ok := leasePooled(root, lane, base, wtRoot, git, owner); ok {
			return res
		}
	}
	res := b.materialize(root, lane, key, base, wtRoot, git)
	if res.OK {
		res.Backend = blockCloneBackendName
		return res
	}
	fallback := gitWorktree{}.MaterializeOwned(root, lane, key, base, wtRoot, git, owner)
	fallback.Backend = gitWorktreeBackendName
	if fallback.OK {
		fallback.Detail = "block-clone declined during materialization; fell back to git-worktree: " + res.Reason
	}
	return fallback
}

func (b blockClone) materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	base := baseSHA
	if base == "" {
		base = TrunkHeadSHA(root, git)
	}
	if base == "" {
		return Result{OK: false, Reason: "could not resolve trunk HEAD (git error) — fail open"}
	}
	wt := Path(lane, key, wtRoot)
	if _, err := os.Stat(wt); err == nil {
		return Result{OK: false, Path: wt, BaseSHA: base, Reason: "block-clone target already exists"}
	}
	if err := os.MkdirAll(filepath.Dir(wt), 0o755); err != nil {
		return Result{OK: false, Path: wt, BaseSHA: base, Reason: "could not create worktree root: " + err.Error() + " — fail open"}
	}
	rc, out := run(git, root, []string{"-c", "core.longpaths=true", "worktree", "add", "--detach", "--no-checkout", wt, base})
	if rc != 0 {
		return Result{OK: false, Path: wt, BaseSHA: base, Reason: "git worktree add --no-checkout failed — fail open", Detail: tail(out, 500)}
	}
	fail := func(reason string) Result {
		cleanup := ForceReap(root, wt, git)
		if !cleanup.OK {
			reason += "; cleanup failed: " + cleanup.Reason
		}
		return Result{OK: false, Path: wt, BaseSHA: base, Reason: reason}
	}

	// Whole-tree native clone fast path: only valid when the working tree is
	// perfectly clean, because CloneTree copies the working tree verbatim and
	// is equivalent to the committed base only with no dirty tracked files and
	// no untracked files. Any dirt (or any decline) falls through to the
	// per-file loop below, which reconciles each blob against the base tree.
	if res, handled := b.materializeTreeClone(root, base, wt, git, fail); handled {
		return res
	}

	return b.materializePerFile(root, base, wt, git, fail)
}

// treeCloneClean reports whether root's working tree can be cloned verbatim and
// still be byte-equivalent to the committed base tree. It is strictly stronger
// than "status is clean": a plain `git status --porcelain` empty output is NOT
// sufficient because (1) ignored files still clone verbatim, (2) clean/smudge
// and text=auto content filters make working bytes differ from the blob while
// status reads clean, and (3) assume-unchanged/skip-worktree bits hide dirt from
// status. Any git failure declines the fast path (fail-closed to the per-file
// loop, which reconciles each blob against the base).
func treeCloneClean(root string, git GitRunner) bool {
	// (1) No tracked dirt, no untracked paths, and no ignored paths. Adding
	// --ignored closes the hole where a gitignored file inside a tracked
	// top-level tree would be cloned verbatim (status alone does not list it).
	rc, out := run(git, root, []string{"status", "--porcelain=v1", "--untracked-files=all", "--ignored"})
	if rc != 0 || strings.TrimSpace(out) != "" {
		return false
	}

	// (2) No assume-unchanged / skip-worktree entries. `git ls-files -v` tags
	// normal tracked files with an uppercase 'H' followed by a space; any other
	// tag (lowercase h/s/S, etc.) marks assume-valid/skip-worktree, which hides
	// dirt from status and would let the fast path clone the dirty bytes.
	rc, listed := run(git, root, []string{"ls-files", "-v"})
	if rc != 0 {
		return false
	}
	for _, line := range strings.Split(listed, "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "H ") {
			return false
		}
	}

	// (3) No byte-transforming content attribute on any tracked file, and no
	// global eol policy. A clean/smudge filter, an eol/text attribute, or a
	// working-tree-encoding attribute, or a core.autocrlf / core.eol config
	// makes the working-tree bytes differ from the committed blob even though
	// status reads clean; cloning the working bytes would materialize content
	// that is not the base object. Decline any global eol policy first. A
	// `config --get` on an unset key exits non-zero: that is a normal unset
	// (continue), not a git failure.
	rc, autocrlf := run(git, root, []string{"config", "--get", "core.autocrlf"})
	if rc == 0 {
		if v := strings.TrimSpace(autocrlf); v != "" && v != "false" {
			return false
		}
	}
	rc, eol := run(git, root, []string{"config", "--get", "core.eol"})
	if rc == 0 && strings.TrimSpace(eol) != "" {
		return false
	}

	rc, tracked := run(git, root, []string{"ls-files", "-z"})
	if rc != 0 {
		return false
	}
	var paths []string
	for _, p := range strings.Split(tracked, "\x00") {
		if p != "" {
			paths = append(paths, p)
		}
	}
	if len(paths) == 0 {
		return true
	}
	// check-attr's `-z` output is NUL-separated triples:
	// <path>\0<attribute>\0<value>\0. All four byte-transforming attributes are
	// queried in one call per batch; any value other than the literal
	// "unspecified" is active and unsound. Argument lists can be large, so batch
	// the paths to stay well under the OS argv limit.
	const checkAttrBatch = 500
	for start := 0; start < len(paths); start += checkAttrBatch {
		end := start + checkAttrBatch
		if end > len(paths) {
			end = len(paths)
		}
		args := append([]string{"check-attr", "-z", "text", "eol", "filter", "working-tree-encoding", "--"}, paths[start:end]...)
		rc, attrs := run(git, root, args)
		if rc != 0 {
			return false
		}
		fields := strings.Split(attrs, "\x00")
		for i := 0; i+2 < len(fields); i += 3 {
			if fields[i+2] != "unspecified" {
				return false
			}
		}
	}
	return true
}

// materializeTreeClone attempts the whole-tree native clone fast path for a
// clean working tree. The returned handled flag means "final": handled=true
// carries a Result materialize must return directly (success, or an
// unrecoverable partial-materialization failure), and handled=false means a
// clean decline for materialize to run the per-file fallback loop. A decline
// has no side effects on wt: it is rolled back to the state git worktree add
// created (empty except .git) for the fallback to fill.
func (b blockClone) materializeTreeClone(root, base, wt string, git GitRunner, fail func(string) Result) (Result, bool) {
	if !treeCloneClean(root, git) {
		return Result{}, false
	}
	treeClone := b.treeClone
	if treeClone == nil {
		treeClone = CloneTree
	}

	rc, listing := run(git, root, []string{"ls-tree", "-z", "--full-tree", base})
	if rc != 0 {
		return Result{}, false
	}
	type topEntry struct {
		mode string
		typ  string
		name string
	}
	var entries []topEntry
	for _, record := range strings.Split(listing, "\x00") {
		if record == "" {
			continue
		}
		tab := strings.IndexByte(record, '\t')
		if tab < 0 {
			return Result{}, false
		}
		fields := strings.Fields(record[:tab])
		if len(fields) != 3 {
			return Result{}, false
		}
		entries = append(entries, topEntry{mode: fields[0], typ: fields[1], name: record[tab+1:]})
	}

	staging, err := os.MkdirTemp(filepath.Dir(wt), ".fak-treeclone-")
	if err != nil {
		return Result{}, false
	}
	defer os.RemoveAll(staging)

	for _, entry := range entries {
		name := filepath.FromSlash(entry.name)
		src := filepath.Join(root, name)
		dst := filepath.Join(staging, name)
		switch entry.typ {
		case "tree":
			if err := treeClone(src, dst); err != nil {
				return Result{}, false
			}
		case "blob":
			info, err := os.Lstat(src)
			if err != nil || !info.Mode().IsRegular() {
				return Result{}, false
			}
			if info.Size() == 0 {
				f, createErr := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm())
				if createErr == nil {
					createErr = f.Close()
				}
				if createErr != nil {
					return Result{}, false
				}
			} else if err := b.clone(src, dst); err != nil {
				return Result{}, false
			}
			if entry.mode == "100755" {
				_ = os.Chmod(dst, 0o755)
			}
		default:
			// Submodule gitlinks and any unexpected entry type are not cloned
			// correctly by CloneTree; decline to the per-file loop.
			return Result{}, false
		}
	}

	// Move staged top-level entries into wt, which is empty except for .git.
	// Same parent directory, so each move is a rename. A failure mid-loop would
	// leave wt partially populated and break the per-file fallback (whose clone
	// is O_EXCL-style), so roll back every entry already moved before declining.
	staged, err := os.ReadDir(staging)
	if err != nil {
		return Result{}, false
	}
	var moved []string
	rollbackMoved := func() bool {
		ok := true
		for i := len(moved) - 1; i >= 0; i-- {
			name := moved[i]
			if err := os.Rename(filepath.Join(wt, name), filepath.Join(staging, name)); err != nil {
				ok = false
			}
		}
		return ok
	}
	for _, entry := range staged {
		name := entry.Name()
		if err := os.Rename(filepath.Join(staging, name), filepath.Join(wt, name)); err != nil {
			if !rollbackMoved() {
				// wt is partially materialized and cannot be restored; the
				// per-file loop would fail on the already-present entries, so
				// do not decline. Report a truthful final failure and let the
				// fail closure ForceReap the partial worktree, so materialize
				// never falls through to a Reused git-worktree over live wt.
				return fail("tree-clone partial materialization in " + wt + "; rollback failed"), true
			}
			return Result{}, false
		}
		moved = append(moved, name)
	}

	if rc, out := run(git, wt, []string{"reset", "--mixed", base}); rc != 0 {
		// The whole tree is already moved into wt, so this is NOT a clean
		// decline: the per-file fallback would collide with the populated wt.
		// Report a truthful final failure; ForceReap removes the partial wt so
		// materialize returns a failed Result instead of a Reused git-worktree.
		return fail("tree-clone reset after materialization failed: " + tail(out, 200)), true
	}
	treeCloneFastPath.Add(1)
	return Result{OK: true, Path: wt, BaseSHA: base, Detail: "tree-clone fast path"}, true
}

// materializePerFile is the fallback: it walks the base tree blob-by-blob,
// reconciling each working-tree file against its committed object (dirty or
// missing paths are force-checked-out from base) and block-cloning the clean
// remainder. It is the original materialization loop, retained byte-for-byte.
func (b blockClone) materializePerFile(root, base, wt string, git GitRunner, fail func(string) Result) Result {
	rc, listing := run(git, root, []string{"ls-tree", "-r", "-z", "--full-tree", base})
	if rc != 0 {
		return fail("git ls-tree failed")
	}
	for _, record := range strings.Split(listing, "\x00") {
		if record == "" {
			continue
		}
		tab := strings.IndexByte(record, '\t')
		if tab < 0 {
			return fail("unexpected git ls-tree record")
		}
		fields := strings.Fields(record[:tab])
		if len(fields) != 3 || fields[1] != "blob" {
			return fail("unexpected git ls-tree record")
		}
		mode, objectID, rel := fields[0], fields[2], record[tab+1:]
		src, dst := filepath.Join(root, filepath.FromSlash(rel)), filepath.Join(wt, filepath.FromSlash(rel))
		info, err := os.Lstat(src)
		if err != nil || !info.Mode().IsRegular() || gitBlobSHA1(src) != objectID {
			rc, checkoutOut := run(git, wt, []string{"checkout", "--force", base, "--", rel})
			if rc != 0 {
				return fail("git checkout fallback failed for " + rel + ": " + tail(checkoutOut, 200))
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return fail("create clone parent for " + rel + ": " + err.Error())
		}
		if info.Size() == 0 {
			f, createErr := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm())
			if createErr == nil {
				createErr = f.Close()
			}
			if createErr != nil {
				return fail("create empty clone for " + rel + ": " + createErr.Error())
			}
		} else if err := b.clone(src, dst); err != nil {
			return fail("block clone failed for " + rel + ": " + err.Error())
		}
		if mode == "100755" {
			_ = os.Chmod(dst, 0o755)
		}
	}
	if rc, out := run(git, wt, []string{"reset", "--mixed", base}); rc != 0 {
		return fail("git reset after block clone failed: " + tail(out, 200))
	}
	return Result{OK: true, Path: wt, BaseSHA: base}
}

func (blockClone) Release(root, wtPath string, git GitRunner) Result {
	return gitWorktree{}.Release(root, wtPath, git)
}

func gitBlobSHA1(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	h := sha1.New()
	_, _ = fmt.Fprintf(h, "blob %s\x00", strconv.Itoa(len(data)))
	_, _ = h.Write(data)
	return fmt.Sprintf("%x", h.Sum(nil))
}
