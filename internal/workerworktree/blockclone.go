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
)

const (
	gitWorktreeBackendName = "git-worktree"
	blockCloneBackendName  = "block-clone"
)

// ErrBlockCloneUnsupported reports that the host platform or filesystem cannot
// provide copy-on-write block/directory cloning, so callers should fall back.
var ErrBlockCloneUnsupported = errors.New("block clone unsupported")

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

type blockClone struct {
	probe blockCloneProbe
	clone blockCloneFile
}

func newBlockCloneBackend() blockClone {
	return blockClone{probe: probeBlockClone, clone: cloneFileBlocks}
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
