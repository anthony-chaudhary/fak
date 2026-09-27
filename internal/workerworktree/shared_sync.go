package workerworktree

import (
	"os"
	"path"
	"path/filepath"
	"strings"
)

// LandResultLandedSyncIncomplete is a landed result — the target ref moved and
// the commit is durable — whose post-CAS refresh of the shared root checkout did
// not bring every landed path to the new commit. It stays OK/Committed because a
// retry would re-land nothing, but it is deliberately not LandResultSuccess: the
// root index or working tree still disagrees with HEAD for the paths named in
// SharedSync, and every peer building from the root would inherit that silently
// (#13538).
const LandResultLandedSyncIncomplete = "landed-root-sync-incomplete"

const (
	SharedSyncSynced     = "synced"
	SharedSyncIncomplete = "incomplete"
)

// SharedSyncReceipt records the post-CAS refresh of the shared root checkout for
// the landed paths.
type SharedSyncReceipt struct {
	Status string `json:"status"`
	// Deleted are landed deletions dropped from the root index and working tree.
	Deleted []string `json:"deleted,omitempty"`
	// Preserved are landed deletions whose root copy (staged or on disk) carried
	// local changes; it is left in place for an operator to reconcile.
	Preserved []string `json:"preserved,omitempty"`
	// Unsynced are landed paths whose root index still disagrees with the landed
	// commit after the refresh.
	Unsynced []string `json:"unsynced,omitempty"`
	Error    string   `json:"error,omitempty"`
}

// syncSharedCheckout refreshes the shared root checkout's index and working tree
// for paths after the target ref moved from parent to landed. A single
// `git checkout <landed> -- <paths>` cannot do this when the landed commit
// deletes any path: git rejects the unknown pathspec and aborts the WHOLE
// checkout, and the root then shows the inverse of the landed commit as staged
// changes (#13538). Deletions are therefore dropped from the index directly and
// their working-tree copies removed only while they still hold the parent's
// blob, so peer WIP is never clobbered. The index is read back against the
// landed commit afterwards so the receipt reports what the root actually holds.
func syncSharedCheckout(git GitRunner, root, parent, landed string, paths []string) SharedSyncReceipt {
	var r SharedSyncReceipt
	var errs []string
	deleted := landedDeletions(git, root, parent, landed, paths)
	gone := make(map[string]bool, len(deleted))
	for _, p := range deleted {
		gone[p] = true
	}
	present := make([]string, 0, len(paths))
	for _, p := range paths {
		if !gone[syncPathKey(p)] {
			present = append(present, p)
		}
	}
	if len(present) > 0 {
		if rc, out := run(git, root, append([]string{"checkout", landed, "--"}, present...)); rc != 0 {
			errs = append(errs, "checkout: "+tail(out, 200))
		}
	}
	if len(deleted) > 0 {
		if msg := r.removeLandedDeletions(git, root, parent, deleted); msg != "" {
			errs = append(errs, msg)
		}
	}
	preserved := make(map[string]bool, len(r.Preserved))
	for _, p := range r.Preserved {
		preserved[p] = true
	}
	want := make(map[string]bool, len(paths))
	for _, p := range paths {
		want[syncPathKey(p)] = true
	}
	if len(paths) > 0 {
		rc, out := run(git, root, append([]string{"diff-index", "--cached", "--name-only", "-z", landed, "--"}, paths...))
		if rc != 0 {
			errs = append(errs, "readback: "+tail(out, 200))
		} else {
			for _, name := range strings.Split(out, "\x00") {
				if key := syncPathKey(name); want[key] && !preserved[key] {
					r.Unsynced = append(r.Unsynced, key)
				}
			}
		}
	}
	r.Error = strings.Join(errs, "; ")
	r.finalize()
	return r
}

// landedDeletions returns the paths (slash-separated) that the landed commit
// deletes relative to parent. On a read failure it returns none, which keeps the
// plain checkout; the post-sync readback still reports any resulting gap.
func landedDeletions(git GitRunner, root, parent, landed string, paths []string) []string {
	if strings.TrimSpace(parent) == "" || len(paths) == 0 {
		return nil
	}
	want := make(map[string]bool, len(paths))
	for _, p := range paths {
		want[syncPathKey(p)] = true
	}
	args := append([]string{"diff-tree", "-r", "--no-renames", "--name-only", "-z", "--diff-filter=D", parent, landed, "--"}, paths...)
	rc, out := run(git, root, args)
	if rc != 0 {
		return nil
	}
	var deleted []string
	seen := make(map[string]bool)
	for _, name := range strings.Split(out, "\x00") {
		if key := syncPathKey(name); want[key] && !seen[key] {
			seen[key] = true
			deleted = append(deleted, key)
		}
	}
	return deleted
}

// removeLandedDeletions drops each deleted path from the root index when the
// index still holds the parent's blob, and removes the working-tree file only
// when its content is that same blob. Anything else is local work: a differing
// staged entry is left untouched, and a differing file is kept on disk (as an
// untracked file once its stale index entry is gone). It returns a non-empty
// message on a git or filesystem failure.
func (r *SharedSyncReceipt) removeLandedDeletions(git GitRunner, root, parent string, deleted []string) string {
	parentBlobs := syncBlobListing(git, root, []string{"ls-tree", "-r", "-z", parent, "--"}, deleted, 2, -1)
	indexBlobs := syncBlobListing(git, root, []string{"ls-files", "-s", "-z", "--"}, deleted, 1, 2)
	var dropIndex, onDisk, handled []string
	for _, p := range deleted {
		blob, ok := parentBlobs[p]
		if !ok {
			continue // unverifiable; the readback reports the index gap
		}
		if staged, tracked := indexBlobs[p]; tracked {
			if staged != blob {
				r.Preserved = append(r.Preserved, p)
				continue
			}
			dropIndex = append(dropIndex, p)
		}
		info, err := os.Lstat(filepath.Join(root, filepath.FromSlash(p)))
		switch {
		case os.IsNotExist(err):
			handled = append(handled, p)
		case err == nil && info.Mode().IsRegular():
			onDisk = append(onDisk, p)
		default:
			r.Preserved = append(r.Preserved, p)
		}
	}
	if len(dropIndex) > 0 {
		if rc, out := run(git, root, append([]string{"update-index", "--force-remove", "--"}, dropIndex...)); rc != 0 {
			return "update-index: " + tail(out, 200)
		}
	}
	var hashes []string
	if len(onDisk) > 0 {
		if rc, out := run(git, root, append([]string{"hash-object", "--"}, onDisk...)); rc == 0 {
			hashes = strings.Fields(out)
		}
	}
	var failures []string
	for i, p := range onDisk {
		if len(hashes) != len(onDisk) || hashes[i] != parentBlobs[p] {
			r.Preserved = append(r.Preserved, p)
			continue
		}
		abs := filepath.Join(root, filepath.FromSlash(p))
		if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
			failures = append(failures, err.Error())
			r.Preserved = append(r.Preserved, p)
			continue
		}
		pruneEmptySyncDirs(root, filepath.Dir(abs))
		handled = append(handled, p)
	}
	r.Deleted = append(r.Deleted, handled...)
	if len(failures) > 0 {
		return "remove: " + strings.Join(failures, "; ")
	}
	return ""
}

// syncBlobListing runs an ls-tree/ls-files style listing with -z output
// ("<fields> TAB <path>") and maps each listed path to the object id at field
// oidField. With stageField >= 0, an entry at a non-zero (merge-conflict) stage
// maps to "" so it never matches a parent blob.
func syncBlobListing(git GitRunner, root string, prefix, paths []string, oidField, stageField int) map[string]string {
	blobs := make(map[string]string, len(paths))
	rc, out := run(git, root, append(append([]string(nil), prefix...), paths...))
	if rc != 0 {
		return blobs
	}
	for _, entry := range strings.Split(out, "\x00") {
		meta, name, ok := strings.Cut(entry, "\t")
		fields := strings.Fields(meta)
		if !ok || len(fields) <= oidField || len(fields) <= stageField {
			continue
		}
		oid := fields[oidField]
		if stageField >= 0 && fields[stageField] != "0" {
			oid = ""
		}
		blobs[syncPathKey(name)] = oid
	}
	return blobs
}

// pruneEmptySyncDirs removes now-empty directories from dir up to (not
// including) root, mirroring what git does when it deletes a tracked file.
func pruneEmptySyncDirs(root, dir string) {
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return
	}
	for {
		abs, err := filepath.Abs(dir)
		if err != nil || abs == rootAbs || !strings.HasPrefix(abs, rootAbs+string(filepath.Separator)) {
			return
		}
		if os.Remove(abs) != nil {
			return
		}
		dir = filepath.Dir(abs)
	}
}

func syncPathKey(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	if p == "" {
		return ""
	}
	if p = path.Clean(p); p == "." {
		return ""
	}
	return p
}

func (r *SharedSyncReceipt) finalize() {
	r.Deleted = normalizeStrings(r.Deleted)
	r.Preserved = normalizeStrings(r.Preserved)
	r.Unsynced = normalizeStrings(r.Unsynced)
	r.Status = SharedSyncSynced
	if r.Error != "" || len(r.Preserved) > 0 || len(r.Unsynced) > 0 {
		r.Status = SharedSyncIncomplete
	}
}

// withWorkerRetry folds a worker-paths-only retry into the receipt of a combined
// worker+peer refresh that did not complete: the retry is authoritative for the
// worker's own paths, while any peer path the first pass left unsynced (and its
// error) stays reported.
func (r SharedSyncReceipt) withWorkerRetry(retry SharedSyncReceipt, workerPaths []string) SharedSyncReceipt {
	worker := make(map[string]bool, len(workerPaths))
	for _, p := range workerPaths {
		worker[syncPathKey(p)] = true
	}
	merged := retry
	for _, p := range r.Unsynced {
		if !worker[p] {
			merged.Unsynced = append(merged.Unsynced, p)
		}
	}
	merged.Deleted = append(merged.Deleted, r.Deleted...)
	merged.Preserved = append(merged.Preserved, r.Preserved...)
	var errs []string
	for _, e := range []string{r.Error, retry.Error} {
		if e != "" {
			errs = append(errs, e)
		}
	}
	merged.Error = strings.Join(errs, "; ")
	merged.finalize()
	return merged
}

// withSharedSync attaches the shared-root sync receipt to a landed result and,
// when the root does not match the landed commit, turns it into the typed
// LandResultLandedSyncIncomplete outcome with an exact recovery command instead
// of a success carrying a detail suffix.
func withSharedSync(res Result, sync SharedSyncReceipt, landed string) Result {
	res.SharedSync = &sync
	if sync.Status == SharedSyncSynced {
		return res
	}
	var gaps []string
	if len(sync.Unsynced) > 0 {
		gaps = append(gaps, "unsynced: "+strings.Join(sync.Unsynced, ", "))
	}
	if len(sync.Preserved) > 0 {
		gaps = append(gaps, "preserved local copies of deleted paths: "+strings.Join(sync.Preserved, ", "))
	}
	if sync.Error != "" {
		gaps = append(gaps, "error: "+sync.Error)
	}
	summary := strings.Join(gaps, "; ")
	res.Code = LandResultLandedSyncIncomplete
	res.Reason = "landed " + shortSHA(landed) + " but the shared root checkout does NOT match it (" + summary + ")"
	res.Detail += "; shared-root sync incomplete: " + summary
	var actions []string
	if len(sync.Unsynced) > 0 {
		actions = append(actions, "in the shared root run `git restore --source="+landed+" --staged --worktree -- "+strings.Join(sync.Unsynced, " ")+"`")
	}
	if len(sync.Preserved) > 0 {
		actions = append(actions, "reconcile or delete the preserved local copies (the landed commit deletes them): "+strings.Join(sync.Preserved, " "))
	}
	if len(actions) == 0 {
		actions = append(actions, "inspect `git status` in the shared root against "+landed+" before building from it")
	}
	res.RecoveryAction = strings.Join(actions, "; then ")
	return res
}
