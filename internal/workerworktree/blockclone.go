package workerworktree

import (
	"crypto/sha1"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	gitWorktreeBackendName = "git-worktree"
	blockCloneBackendName  = "block-clone"
)

type blockCloneProbe func(targetRoot string) error
type blockCloneFile func(src, dst string) error

type blockClone struct {
	probe blockCloneProbe
	clone blockCloneFile
}

func newBlockCloneBackend() blockClone {
	return blockClone{probe: probeBlockClone, clone: cloneFileBlocks}
}

func (b blockClone) Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	targetRoot := resolveWorktreeRoot(wtRoot)
	if err := b.probe(targetRoot); err != nil {
		res := gitWorktree{}.Materialize(root, lane, key, baseSHA, wtRoot, git)
		res.Backend = gitWorktreeBackendName
		if res.OK {
			res.Detail = "block-clone unavailable; fell back to git-worktree: " + err.Error()
		}
		return res
	}
	res := b.materialize(root, lane, key, baseSHA, wtRoot, git)
	if res.OK {
		res.Backend = blockCloneBackendName
		return res
	}
	fallback := gitWorktree{}.Materialize(root, lane, key, baseSHA, wtRoot, git)
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
	rc, out := run(git, root, []string{"worktree", "add", "--detach", "--no-checkout", wt, base})
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
		fields := strings.Fields(record[:tab])
		if tab < 0 || len(fields) != 3 || fields[1] != "blob" {
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
