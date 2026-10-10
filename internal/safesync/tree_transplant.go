package safesync

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// RenamePath represents a path rename detected across commits.
type RenamePath struct {
	OldPath string
	NewPath string
}

// ParseRenameSummary parses git diff -M --summary (or --name-status) output for path renames.
func ParseRenameSummary(output string) []RenamePath {
	var renames []RenamePath
	lines := strings.Split(output, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		// Handle name-status format if present (e.g. "R100\told\tnew")
		if strings.HasPrefix(line, "R") && strings.Contains(line, "\t") {
			parts := strings.Split(line, "\t")
			if len(parts) >= 3 {
				oldClean := filepath.Clean(filepath.ToSlash(strings.Trim(parts[1], "\"")))
				newClean := filepath.Clean(filepath.ToSlash(strings.Trim(parts[2], "\"")))
				if oldClean != "" && newClean != "" && oldClean != newClean {
					renames = append(renames, RenamePath{OldPath: oldClean, NewPath: newClean})
				}
			}
			continue
		}

		// Handle summary format: "rename <paths> (N%)"
		if !strings.HasPrefix(line, "rename ") {
			continue
		}
		content := strings.TrimPrefix(line, "rename ")
		lastParen := strings.LastIndex(content, " (")
		if lastParen == -1 || !strings.HasSuffix(content, "%)") {
			lastParen = len(content)
		}
		pathPart := strings.TrimSpace(content[:lastParen])
		if !strings.Contains(pathPart, "=>") {
			continue
		}

		var oldPath, newPath string
		openBrace := strings.Index(pathPart, "{")
		closeBrace := strings.LastIndex(pathPart, "}")
		if openBrace != -1 && closeBrace != -1 && openBrace < closeBrace {
			prefix := pathPart[:openBrace]
			suffix := pathPart[closeBrace+1:]
			inner := pathPart[openBrace+1 : closeBrace]
			arrow := strings.Index(inner, "=>")
			if arrow != -1 {
				oldMid := strings.TrimSpace(inner[:arrow])
				newMid := strings.TrimSpace(inner[arrow+2:])
				oldPath = prefix + oldMid + suffix
				newPath = prefix + newMid + suffix
			}
		} else {
			arrow := strings.Index(pathPart, "=>")
			if arrow != -1 {
				oldPath = strings.TrimSpace(pathPart[:arrow])
				newPath = strings.TrimSpace(pathPart[arrow+2:])
			}
		}

		oldClean := filepath.Clean(filepath.ToSlash(strings.Trim(oldPath, "\"")))
		newClean := filepath.Clean(filepath.ToSlash(strings.Trim(newPath, "\"")))
		if oldClean != "" && newClean != "" && oldClean != newClean {
			renames = append(renames, RenamePath{OldPath: oldClean, NewPath: newClean})
		}
	}
	return renames
}

// InspectIncomingRenames inspects incoming commits for renames via `git diff -M --summary headSHA targetSHA`.
// If targetCommit (e.g. newly minted merge commit) is provided, candidate renames are verified
// to ensure targetCommit includes the rename.
func InspectIncomingRenames(ctx context.Context, run Runner, repo, headSHA, targetSHA string, targetCommit ...string) ([]RenamePath, error) {
	if run == nil {
		run = RealRunner
	}
	res := run(ctx, repo, "diff", "-M", "--summary", headSHA, targetSHA)
	if res.Err != nil || res.Code != 0 {
		return nil, fmt.Errorf("git diff -M --summary failed (code %d): %s", res.Code, strings.TrimSpace(string(res.Stderr)))
	}
	candidates := ParseRenameSummary(string(res.Stdout))
	if len(candidates) == 0 {
		return nil, nil
	}

	commitToCheck := targetSHA
	if len(targetCommit) > 0 && strings.TrimSpace(targetCommit[0]) != "" {
		commitToCheck = strings.TrimSpace(targetCommit[0])
	}

	var verified []RenamePath
	for _, c := range candidates {
		// Verify candidates using rev-parse when supported:
		oldHead := run(ctx, repo, "rev-parse", "--verify", "--quiet", headSHA+":"+c.OldPath)
		newCommit := run(ctx, repo, "rev-parse", "--verify", "--quiet", commitToCheck+":"+c.NewPath)
		oldCommit := run(ctx, repo, "rev-parse", "--verify", "--quiet", commitToCheck+":"+c.OldPath)

		revParseWorks := (oldHead.Err == nil && oldHead.Code == 0 && len(strings.TrimSpace(string(oldHead.Stdout))) > 0)
		if revParseWorks {
			if newCommit.Err != nil || newCommit.Code != 0 || len(strings.TrimSpace(string(newCommit.Stdout))) == 0 {
				continue
			}
			if oldCommit.Err == nil && oldCommit.Code == 0 && len(strings.TrimSpace(string(oldCommit.Stdout))) > 0 {
				continue
			}
		}

		verified = append(verified, c)
	}
	return verified, nil
}

// ApplyIncomingRenames removes deleted/renamed source paths from the working directory if clean,
// and checks out target renamed paths from commitSHA.
func ApplyIncomingRenames(ctx context.Context, run Runner, repo, headSHA, commitSHA string, renames []RenamePath) error {
	if run == nil {
		run = RealRunner
	}
	for _, r := range renames {
		fullOld, ok := safeWorktreePath(repo, r.OldPath)
		if ok {
			fi, err := os.Stat(fullOld)
			if err == nil && !fi.IsDir() {
				if cleanEquivalentTo(ctx, run, repo, headSHA, r.OldPath) {
					_ = os.Remove(fullOld)
					removeEmptyParentDirs(repo, fullOld)
					_ = run(ctx, repo, "update-index", "--force-remove", r.OldPath)
				}
			} else if os.IsNotExist(err) {
				_ = run(ctx, repo, "update-index", "--force-remove", r.OldPath)
			}
		}

		_ = run(ctx, repo, "checkout", commitSHA, "--", r.NewPath)
	}
	return nil
}

func removeEmptyParentDirs(repo, fullPath string) {
	dir := filepath.Dir(fullPath)
	repoClean := filepath.Clean(repo)
	for dir != repoClean && strings.HasPrefix(dir, repoClean) {
		if err := os.Remove(dir); err != nil {
			break
		}
		dir = filepath.Dir(dir)
	}
}

// TransplantDisjointTree computes a pure ODB synthetic tree transplantation for two disjoint commits,
// mints a merge commit directly in the Git Object Database, advances the branch reference atomically,
// and synchronizes incoming disjoint paths to the working tree.
func TransplantDisjointTree(ctx context.Context, repo, branch, headSHA, targetSHA, targetRef string) (string, error) {
	return TransplantDisjointTreeWithRunner(ctx, RealRunner, repo, branch, headSHA, targetSHA, targetRef)
}

// TransplantDisjointTreeWithRunner executes synthetic tree transplantation using the supplied Runner.
func TransplantDisjointTreeWithRunner(ctx context.Context, run Runner, repo, branch, headSHA, targetSHA, targetRef string) (string, error) {
	if run == nil {
		run = RealRunner
	}
	headSHA = strings.TrimSpace(headSHA)
	if headSHA == "" {
		return "", errors.New("headSHA cannot be empty")
	}
	targetSHA = strings.TrimSpace(targetSHA)
	if targetSHA == "" {
		return "", errors.New("targetSHA cannot be empty")
	}
	targetRef = strings.TrimSpace(targetRef)
	if targetRef == "" {
		targetRef = targetSHA
	}

	// 1. Compute root merge tree OID directly in Git ODB via git merge-tree --write-tree <headSHA> <targetSHA>
	mtRes := run(ctx, repo, "merge-tree", "--write-tree", headSHA, targetSHA)
	if mtRes.Err != nil {
		return "", fmt.Errorf("git merge-tree execution failed: %w", mtRes.Err)
	}
	if mtRes.Code != 0 {
		return "", fmt.Errorf("git merge-tree exited with code %d: %s", mtRes.Code, strings.TrimSpace(string(mtRes.Stderr)))
	}
	outStr := strings.TrimSpace(string(mtRes.Stdout))
	lines := strings.Split(outStr, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return "", errors.New("git merge-tree returned empty tree OID")
	}
	treeOID := strings.TrimSpace(lines[0])

	// 2. Mint two-parent merge commit object directly in Git ODB
	// Git's effective committer honors environment overrides that user config does not.
	identRes := run(ctx, repo, "var", "GIT_COMMITTER_IDENT")
	if identRes.Err != nil {
		return "", fmt.Errorf("git var GIT_COMMITTER_IDENT execution failed: %w", identRes.Err)
	}
	if identRes.Code != 0 {
		return "", fmt.Errorf("git var GIT_COMMITTER_IDENT exited with code %d: %s", identRes.Code, strings.TrimSpace(string(identRes.Stderr)))
	}
	ident := strings.TrimSpace(string(identRes.Stdout))
	open, close := strings.LastIndexByte(ident, '<'), strings.LastIndexByte(ident, '>')
	if open <= 0 || close <= open+1 || strings.TrimSpace(ident[:open]) == "" || strings.ContainsAny(ident, "\r\n") {
		return "", errors.New("git var GIT_COMMITTER_IDENT returned invalid committer identity")
	}
	// Exclude the timestamp and timezone from the sign-off identity.
	msg := fmt.Sprintf("Merge %s (disjoint integrate) (fak safesync)\n\nSigned-off-by: %s", targetRef, ident[:close+1])
	ctRes := run(ctx, repo, "commit-tree", treeOID, "-p", headSHA, "-p", targetSHA, "-m", msg)
	if ctRes.Err != nil {
		return "", fmt.Errorf("git commit-tree execution failed: %w", ctRes.Err)
	}
	if ctRes.Code != 0 {
		return "", fmt.Errorf("git commit-tree exited with code %d: %s", ctRes.Code, strings.TrimSpace(string(ctRes.Stderr)))
	}
	newCommitSHA := strings.TrimSpace(string(ctRes.Stdout))
	if newCommitSHA == "" {
		return "", errors.New("git commit-tree returned empty commit SHA")
	}

	// 3. Advance branch ref atomically
	trimmedBranch := strings.TrimSpace(branch)
	var fullRef string
	if trimmedBranch == "" {
		symRes := run(ctx, repo, "symbolic-ref", "--quiet", "HEAD")
		if symRes.Err == nil && symRes.Code == 0 {
			fullRef = strings.TrimSpace(string(symRes.Stdout))
		}
	} else if strings.HasPrefix(trimmedBranch, "refs/") {
		fullRef = trimmedBranch
	} else if trimmedBranch == "HEAD" {
		fullRef = "HEAD"
	} else {
		fullRef = "refs/heads/" + trimmedBranch
	}

	if fullRef == "" {
		fullRef = "HEAD"
	}

	urRes := run(ctx, repo, "update-ref", fullRef, newCommitSHA, headSHA)
	if urRes.Err != nil {
		return "", fmt.Errorf("git update-ref %s failed: %w", fullRef, urRes.Err)
	}
	if urRes.Code != 0 {
		return "", fmt.Errorf("git update-ref %s exited with code %d: %s", fullRef, urRes.Code, strings.TrimSpace(string(urRes.Stderr)))
	}

	// If HEAD was pointing to this ref or detached HEAD, ensure HEAD matches newCommitSHA.
	curHeadRes := run(ctx, repo, "rev-parse", "--verify", "HEAD")
	if curHeadRes.Err == nil && curHeadRes.Code == 0 {
		curHead := strings.TrimSpace(string(curHeadRes.Stdout))
		if curHead != newCommitSHA {
			// HEAD does not match newCommitSHA (e.g. detached HEAD pointing to headSHA).
			if curHead == headSHA {
				upHead := run(ctx, repo, "update-ref", "HEAD", newCommitSHA, headSHA)
				if upHead.Err != nil {
					return "", fmt.Errorf("git update-ref HEAD failed: %w", upHead.Err)
				}
				if upHead.Code != 0 {
					return "", fmt.Errorf("git update-ref HEAD exited with code %d: %s", upHead.Code, strings.TrimSpace(string(upHead.Stderr)))
				}
			}
		}
	}

	// 4. Synchronize incoming disjoint paths to working tree.
	// Query non-conflicting incoming paths added/modified between headSHA and targetSHA.
	// In the synthetic merge commit (newCommitSHA), incoming paths are precisely those
	// added or modified relative to headSHA.
	// From here on the ref already names newCommitSHA: a failure must not read as
	// "nothing happened", or a caller's merge fallback becomes an up-to-date no-op
	// over a worktree that is missing the incoming files.
	incomingPaths, err := incomingAMPaths(ctx, run, repo, headSHA, newCommitSHA)
	if err != nil {
		return newCommitSHA, &TransplantWorktreeError{Commit: newCommitSHA, Err: err}
	}
	if err := checkoutPaths(ctx, run, repo, newCommitSHA, incomingPaths); err != nil {
		return newCommitSHA, &TransplantWorktreeError{Commit: newCommitSHA, Err: err}
	}

	// 5. Handle incoming path renames across disjoint integration.
	// Inspect incoming commits for renames (git diff -M --summary headSHA targetSHA).
	// Remove deleted/renamed source path from working directory if clean, and check out target renamed path.
	if renames, err := InspectIncomingRenames(ctx, run, repo, headSHA, targetSHA, newCommitSHA); err == nil && len(renames) > 0 {
		_ = ApplyIncomingRenames(ctx, run, repo, headSHA, newCommitSHA, renames)
	}

	return newCommitSHA, nil
}

// ErrIncomingPathsUnwritten is the closed sentinel for a transplant that
// advanced the branch ref but did not land every incoming path on disk.
var ErrIncomingPathsUnwritten = errors.New("transplant advanced the branch ref but incoming paths are not on disk")

// TransplantWorktreeError reports a transplant whose ref move succeeded and whose
// worktree sync did not. Commit is the merge commit the ref now names.
type TransplantWorktreeError struct {
	Commit  string
	Missing []string
	Err     error
}

func (e *TransplantWorktreeError) Error() string {
	msg := fmt.Sprintf("%v (commit %s", ErrIncomingPathsUnwritten, e.Commit)
	if len(e.Missing) > 0 {
		msg += fmt.Sprintf(", %d path(s) missing or stale: %s", len(e.Missing), strings.Join(e.Missing, ", "))
	}
	msg += ")"
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *TransplantWorktreeError) Is(target error) bool {
	return target == ErrIncomingPathsUnwritten
}

func (e *TransplantWorktreeError) Unwrap() error { return e.Err }

const transplantCheckoutBatch = 100

func incomingAMPaths(ctx context.Context, run Runner, repo, headSHA, commit string) ([]string, error) {
	res := run(ctx, repo, "diff", "-z", "--name-only", "--diff-filter=AM", headSHA, commit)
	if res.Err != nil {
		return nil, fmt.Errorf("git diff incoming paths failed: %w", res.Err)
	}
	if res.Code != 0 {
		return nil, fmt.Errorf("git diff incoming paths exited with code %d: %s", res.Code, strings.TrimSpace(string(res.Stderr)))
	}
	return splitNUL(res.Stdout), nil
}

func checkoutPaths(ctx context.Context, run Runner, repo, commit string, paths []string) error {
	for i := 0; i < len(paths); i += transplantCheckoutBatch {
		end := min(i+transplantCheckoutBatch, len(paths))
		args := append([]string{"checkout", commit, "--"}, paths[i:end]...)
		res := run(ctx, repo, args...)
		if res.Err != nil {
			return fmt.Errorf("git checkout incoming paths failed: %w", res.Err)
		}
		if res.Code != 0 {
			return fmt.Errorf("git checkout incoming paths exited with code %d: %s", res.Code, strings.TrimSpace(string(res.Stderr)))
		}
	}
	return nil
}

// UnsyncedIncomingPaths returns the incoming (added/modified headSHA..commit)
// paths whose on-disk content does not hash to commit's blob, including absent files.
func UnsyncedIncomingPaths(ctx context.Context, run Runner, repo, headSHA, commit string) ([]string, error) {
	if run == nil {
		run = RealRunner
	}
	paths, err := incomingAMPaths(ctx, run, repo, headSHA, commit)
	if err != nil {
		return nil, err
	}
	var unsynced []string
	for i := 0; i < len(paths); i += transplantCheckoutBatch {
		end := min(i+transplantCheckoutBatch, len(paths))
		bad, err := unsyncedBatch(ctx, run, repo, commit, paths[i:end])
		if err != nil {
			return nil, err
		}
		unsynced = append(unsynced, bad...)
	}
	return unsynced, nil
}

func unsyncedBatch(ctx context.Context, run Runner, repo, commit string, paths []string) ([]string, error) {
	args := append([]string{"ls-tree", "-z", "--full-tree", commit, "--"}, paths...)
	res := run(ctx, repo, args...)
	if res.Err != nil || res.Code != 0 {
		return nil, fmt.Errorf("git ls-tree incoming paths exited with code %d: %s", res.Code, runDetail(res))
	}
	want := make(map[string]string, len(paths))
	mode := make(map[string]string, len(paths))
	for _, rec := range strings.Split(string(res.Stdout), "\x00") {
		meta, p, ok := strings.Cut(rec, "\t")
		if !ok {
			continue
		}
		f := strings.Fields(meta)
		if len(f) != 3 {
			continue
		}
		mode[p], want[p] = f[0], f[2]
	}

	var unsynced, hashable []string
	for _, p := range paths {
		full, ok := safeWorktreePath(repo, p)
		if !ok {
			unsynced = append(unsynced, p)
			continue
		}
		fi, err := os.Lstat(full)
		switch {
		case mode[p] == "160000":
		case err != nil:
			unsynced = append(unsynced, p)
		case mode[p] == "120000":
		case !fi.Mode().IsRegular():
			unsynced = append(unsynced, p)
		default:
			hashable = append(hashable, p)
		}
	}
	if len(hashable) == 0 {
		return unsynced, nil
	}
	hres := run(ctx, repo, append([]string{"hash-object", "--"}, hashable...)...)
	if hres.Err != nil || hres.Code != 0 {
		return nil, fmt.Errorf("git hash-object incoming paths exited with code %d: %s", hres.Code, runDetail(hres))
	}
	got := strings.Fields(string(hres.Stdout))
	if len(got) != len(hashable) {
		return nil, fmt.Errorf("git hash-object returned %d hashes for %d paths", len(got), len(hashable))
	}
	for i, p := range hashable {
		if got[i] != want[p] {
			unsynced = append(unsynced, p)
		}
	}
	return unsynced, nil
}

// EnsureIncomingOnDisk proves every incoming headSHA..commit path is on disk at
// commit's blob, re-checking-out whatever is absent or stale a bounded number of
// times. It returns a *TransplantWorktreeError naming the paths that never landed.
func EnsureIncomingOnDisk(ctx context.Context, run Runner, repo, headSHA, commit string) error {
	if run == nil {
		run = RealRunner
	}
	const attempts = 3
	var unsynced []string
	var lastErr error
	for attempt := 0; attempt <= attempts; attempt++ {
		var err error
		unsynced, err = UnsyncedIncomingPaths(ctx, run, repo, headSHA, commit)
		if err != nil {
			return &TransplantWorktreeError{Commit: commit, Err: err}
		}
		if len(unsynced) == 0 {
			_ = run(ctx, repo, "update-index", "-q", "--refresh")
			return nil
		}
		if attempt == attempts || ctx.Err() != nil {
			break
		}
		lastErr = checkoutPaths(ctx, run, repo, commit, unsynced)
	}
	return &TransplantWorktreeError{Commit: commit, Missing: unsynced, Err: lastErr}
}
