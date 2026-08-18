// stageddelete.go adds the NO-OP STAGED DELETION audit to the commit-lane observer
// (#5339). It is the same shape as the rest of this package: a pure decision over
// injected facts, plus a thin gatherer that collects those facts from git. commitlane
// still never mutates the repository — this file OBSERVES the shared index and PRINTS a
// scoped remedy; it never runs `git restore --staged`, `git add` or `git rm`. On a shared
// multi-session trunk an automatic un-stage would silently discard a peer's in-flight
// index work, which is the one failure this audit exists to make impossible.
//
// THE HAZARD. `git rm --cached <path>` (or a peer's aborted index rewrite) leaves a path
// staged as a DELETION while the working-tree file is still on disk and byte-identical to
// HEAD. Such an entry has NO commit effect that anybody intended: a pathspec commit either
// records a deletion the author never asked for, or — with the worktree copy still there —
// clears the deletion and records nothing at all. What it DOES do is pollute `git status`
// for every session sharing the clone, make "what will land" unreadable, and widen the
// input that sweep-guard / BARE_COMMIT_SWEEP reasoning has to defend against. At filing,
// 51 such entries sat in the shared index at once.
//
// WHY BYTE-IDENTITY IS THE ONLY SAFE TEST. A detector that flagged every staged deletion
// would be worse than none, because acting on its output would un-stage a peer's GENUINE
// deletion. Only "the file is on disk AND its blob hash equals HEAD's blob hash for that
// path" proves the staged deletion carries no content decision: restoring the index entry
// puts back exactly the bytes HEAD already has, so the clear is provably a no-op in the
// other direction too. Every case that falls short of that proof — absent from disk,
// present but DIFFERENT, or any fact that could not be read — is reported but excluded
// from the offered clear. See StagedDeletionDiskDiffers for the case that separates a
// correct detector from a plausible one.
//
// This is a HYGIENE audit, not a lane blocker: it never changes Report.Verdict. A tree
// whose only complaint is index churn is still perfectly committable, and promoting churn
// to "blocked" would wedge the very lane the package exists to keep readable.
package commitlane

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/stringlist"
	"path/filepath"
	"sort"
	"strings"
)

// StagedDeletionClass is the closed vocabulary for one staged-deletion entry, so a caller
// (and the tests) can match on the classification without parsing prose.
type StagedDeletionClass string

const (
	// StagedDeletionNoOpChurn: the path is staged for deletion, the working-tree file
	// EXISTS, and its blob hash equals HEAD's blob hash for that path. Pure index churn —
	// the only class offered for a scoped `git restore --staged` clear.
	StagedDeletionNoOpChurn StagedDeletionClass = "noop_churn"
	// StagedDeletionRealDelete: the path is staged for deletion and the working-tree file
	// is genuinely GONE. This is somebody's real, intended deletion on its way to a commit.
	// Never flagged, never offered for clearing — un-staging it would destroy real work.
	StagedDeletionRealDelete StagedDeletionClass = "real_deletion"
	// StagedDeletionDiskDiffers: the path is staged for deletion and the working-tree file
	// EXISTS but its content differs from HEAD.
	//
	// This one is NOT no-op churn and is deliberately excluded from the remedy. Two
	// independent reasons, either sufficient:
	//
	//  1. The staged entry is not content-free. Committing it WOULD change the tree (HEAD
	//     carries a blob at that path; the index says remove it), so calling it a "no-op"
	//     is simply false — the ticket's whole claim is about entries with no commit effect.
	//  2. The disk content is unexplained, so the clear is not reversible-by-inspection.
	//     `git restore --staged <path>` would re-stage HEAD's blob, and the worktree would
	//     then read as MODIFIED against it. That is a different tree state than the author
	//     left, and on a shared clone the difference is most likely a peer's in-flight
	//     rewrite (delete-then-recreate, or a genuine delete whose replacement has already
	//     started landing). Guessing wrong here un-stages a peer's real intent, which is the
	//     exact harm the audit is built to avoid.
	//
	// It is still REPORTED, because an operator looking at index churn wants to see the
	// ambiguous neighbours of the churn; it is just never in the offered path set.
	StagedDeletionDiskDiffers StagedDeletionClass = "disk_differs_from_head"
	// StagedDeletionUnknown: at least one fact could not be read (HEAD has no blob at that
	// path, the on-disk hash was unavailable, or the probe reported an error). Fail CLOSED —
	// an unproven entry is never offered for clearing.
	StagedDeletionUnknown StagedDeletionClass = "unknown"
)

// StagedDeletionFact is ONE injected observation about a path that `git diff --cached
// --diff-filter=D` reports as staged for deletion. Keeping the classifier's whole input in
// this struct is what lets the decision be tested as data: the three-way mixed case needs
// no repository, no commits and no index at all.
//
// DiskHash is the blob hash the working-tree file would hash to (git's own filters
// applied, so it matches what `git add` would store); HeadHash is the blob hash HEAD
// records for the path. Either may be empty when unreadable, which fails closed.
type StagedDeletionFact struct {
	Path     string `json:"path"`
	OnDisk   bool   `json:"on_disk"`
	DiskHash string `json:"disk_hash,omitempty"`
	HeadHash string `json:"head_hash,omitempty"`
	Err      string `json:"err,omitempty"`
}

// StagedDeletionRow is the classified verdict for one staged deletion.
type StagedDeletionRow struct {
	Path   string              `json:"path"`
	Class  StagedDeletionClass `json:"class"`
	Detail string              `json:"detail,omitempty"`
}

// StagedDeletionAudit is the whole read-only finding: every staged deletion classified,
// the subset proven to be no-op churn, and the scoped remedy TEXT for exactly that subset.
//
// Remedy is a string. Nothing in this package, and nothing in the CLI layer that renders
// it, executes it — the operator (or a reviewer who has read the path list) runs it.
type StagedDeletionAudit struct {
	Rows      []StagedDeletionRow `json:"rows,omitempty"`
	NoOpPaths []string            `json:"noop_paths,omitempty"`
	Remedy    string              `json:"remedy,omitempty"`
}

// NoOpCount reports how many staged deletions were proven to be pure index churn.
func (a StagedDeletionAudit) NoOpCount() int { return len(a.NoOpPaths) }

// ClassifyStagedDeletions is the PURE half: it maps injected facts to per-path verdicts
// and builds the scoped remedy for the no-op subset only. It reads no file, runs no
// command and takes no clock — the whole three-way mixed case (no-op / real deletion /
// present-but-different) is expressible as data, which is why it can be tested without a
// fixture repository.
//
// The guard order is deliberate and fails CLOSED at every step: a reported error, then
// absence from disk (a real deletion, decided WITHOUT consulting hashes so a missing HEAD
// blob can never demote it), then any missing hash, and only then the byte-identity
// comparison that is allowed to put a path into the remedy.
func ClassifyStagedDeletions(facts []StagedDeletionFact) StagedDeletionAudit {
	if len(facts) == 0 {
		return StagedDeletionAudit{}
	}
	audit := StagedDeletionAudit{Rows: make([]StagedDeletionRow, 0, len(facts))}
	for _, f := range facts {
		path := strings.TrimSpace(f.Path)
		if path == "" {
			continue
		}
		row := StagedDeletionRow{Path: path}
		switch {
		case strings.TrimSpace(f.Err) != "":
			row.Class = StagedDeletionUnknown
			row.Detail = strings.TrimSpace(f.Err)
		case !f.OnDisk:
			// Genuinely gone from the working tree: somebody's real deletion. Decided before
			// any hash check so an unreadable HEAD blob cannot turn a real deletion into an
			// "unknown" that a future, laxer caller might sweep up.
			row.Class = StagedDeletionRealDelete
			row.Detail = "working-tree file is gone; this is a real staged deletion"
		case f.HeadHash == "" || f.DiskHash == "":
			row.Class = StagedDeletionUnknown
			row.Detail = "could not read both the on-disk and the HEAD blob hash"
		case f.DiskHash == f.HeadHash:
			row.Class = StagedDeletionNoOpChurn
			row.Detail = "working-tree file is byte-identical to HEAD; the staged deletion has no commit effect"
			audit.NoOpPaths = append(audit.NoOpPaths, path)
		default:
			row.Class = StagedDeletionDiskDiffers
			row.Detail = "working-tree file exists but differs from HEAD; the staged deletion is not a no-op"
		}
		audit.Rows = append(audit.Rows, row)
	}
	if len(audit.Rows) == 0 {
		return StagedDeletionAudit{}
	}
	sort.Slice(audit.Rows, func(i, j int) bool { return audit.Rows[i].Path < audit.Rows[j].Path })
	sort.Strings(audit.NoOpPaths)
	audit.Remedy = RestoreStagedCommand(audit.NoOpPaths)
	return audit
}

// RestoreStagedCommand renders the scoped clear for exactly the given paths as TEXT. It
// never runs anything. The `--` separator is mandatory rather than cosmetic: without it a
// path that happens to look like a revision would be parsed as one, and the command would
// act on something other than the audited set.
func RestoreStagedCommand(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(paths))
	for _, p := range paths {
		quoted = append(quoted, quoteGitPathArg(p))
	}
	return "git restore --staged -- " + strings.Join(quoted, " ")
}

// quoteGitPathArg wraps a repo path in double quotes when it contains whitespace, so the
// printed remedy stays copy-pasteable for paths a shell would otherwise split.
func quoteGitPathArg(p string) string {
	if strings.ContainsAny(p, " \t") {
		return `"` + p + `"`
	}
	return p
}

// stagedDeletionChunk bounds how many paths go into one git invocation, so a large churn
// set (51 at filing) cannot overflow the command line.
const stagedDeletionChunk = 100

// ScanStagedDeletions is the THIN impure half: it gathers StagedDeletionFacts from git and
// hands them to the pure classifier. It issues at most one command when the index is clean
// (`diff --cached --diff-filter=D --name-only -z`, which is the common case and returns
// nothing), and only then the batched `ls-tree` / `hash-object` reads. Every command is a
// READ; none writes an object, a ref or the index.
//
// It FAILS OPEN AND SILENT: if any probe cannot run, the audit comes back empty rather
// than half-populated. A hygiene advisory that guesses is worse than one that abstains,
// and the lane report must never grow a spurious warning because a probe misfired.
func ScanStagedDeletions(ctx context.Context, run Runner, stat FileStatFunc, root string) StagedDeletionAudit {
	if run == nil || stat == nil {
		return StagedDeletionAudit{}
	}
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	res := run(ctx, root, "--no-optional-locks", "diff", "--cached", "--diff-filter=D", "--name-only", "-z", "--")
	if res.Err != nil || res.Code != 0 {
		return StagedDeletionAudit{}
	}
	paths := splitNUL(res.Stdout)
	if len(paths) == 0 {
		return StagedDeletionAudit{}
	}

	facts := make([]StagedDeletionFact, 0, len(paths))
	var present []string
	for _, p := range paths {
		f := StagedDeletionFact{Path: p, OnDisk: stat(filepath.Join(root, filepath.FromSlash(p))).Exists}
		if f.OnDisk {
			present = append(present, p)
		}
		facts = append(facts, f)
	}

	// Only a path still ON DISK can possibly be no-op churn, so both blob reads are scoped
	// to that subset: an index carrying nothing but GENUINE deletions costs the one
	// name-only diff and no hashing at all.
	headHashes := headBlobHashes(ctx, run, root, present)
	diskHashes, diskErr := diskBlobHashes(ctx, run, root, present)
	for i := range facts {
		if !facts[i].OnDisk {
			continue
		}
		facts[i].HeadHash = headHashes[facts[i].Path]
		if h, ok := diskHashes[facts[i].Path]; ok {
			facts[i].DiskHash = h
			continue
		}
		if diskErr != "" {
			facts[i].Err = diskErr
		}
	}
	return ClassifyStagedDeletions(facts)
}

// headBlobHashes reads HEAD's blob hash for each path via batched `ls-tree -z`. A path
// HEAD does not carry as a BLOB (absent, or a submodule gitlink) is simply left out, which
// classifies as unknown downstream — the fail-closed posture.
func headBlobHashes(ctx context.Context, run Runner, root string, paths []string) map[string]string {
	out := map[string]string{}
	for _, chunk := range chunkPaths(paths, stagedDeletionChunk) {
		args := append([]string{"--no-optional-locks", "ls-tree", "-z", "HEAD", "--"}, chunk...)
		res := run(ctx, root, args...)
		if res.Err != nil || res.Code != 0 {
			continue
		}
		for _, entry := range splitNUL(res.Stdout) {
			meta, path, ok := strings.Cut(entry, "\t")
			if !ok {
				continue
			}
			fields := strings.Fields(meta)
			if len(fields) < 3 || fields[1] != "blob" {
				continue
			}
			out[path] = fields[2]
		}
	}
	return out
}

// diskBlobHashes hashes the on-disk copies via batched `git hash-object` (no -w: nothing is
// written to the object store). git applies the same filters `git add` would, so the hash
// is directly comparable to HEAD's blob.
//
// Alignment is the whole risk here: hash-object prints one line per argument, in argument
// order, so a chunk whose line count does not match its path count is discarded WHOLESALE
// rather than zipped up wrongly — pairing path N with hash N+1 would be how a real deletion
// gets mislabelled as churn. The returned string is a non-empty error detail when any chunk
// failed, so the affected paths classify as unknown instead of silently vanishing.
func diskBlobHashes(ctx context.Context, run Runner, root string, paths []string) (map[string]string, string) {
	out := map[string]string{}
	detail := ""
	for _, chunk := range chunkPaths(paths, stagedDeletionChunk) {
		args := append([]string{"--no-optional-locks", "hash-object", "--"}, chunk...)
		res := run(ctx, root, args...)
		if res.Err != nil || res.Code != 0 {
			detail = "git hash-object could not read the working-tree copy"
			continue
		}
		lines := stringlist.NonEmptyLines(res.Stdout)
		if len(lines) != len(chunk) {
			detail = "git hash-object returned an unexpected number of hashes"
			continue
		}
		for i, p := range chunk {
			out[p] = lines[i]
		}
	}
	return out, detail
}

func chunkPaths(paths []string, size int) [][]string {
	if size <= 0 || len(paths) == 0 {
		return nil
	}
	var out [][]string
	for start := 0; start < len(paths); start += size {
		end := start + size
		if end > len(paths) {
			end = len(paths)
		}
		out = append(out, paths[start:end])
	}
	return out
}

// splitNUL splits git's -z output. NUL-delimited output is used everywhere here precisely
// so a path containing a space, a quote or a non-ASCII byte survives the round trip
// untouched — the quoting `git status` applies by default is exactly what a path-scoped
// remedy must not inherit.
func splitNUL(s string) []string {
	var out []string
	for _, part := range strings.Split(s, "\x00") {
		part = strings.TrimRight(part, "\r\n")
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}
