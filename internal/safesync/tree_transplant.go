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
	msg := fmt.Sprintf("Merge %s (disjoint integrate) (fak safesync)", targetRef)
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
	diffRes := run(ctx, repo, "diff", "--name-only", "--diff-filter=AM", headSHA, newCommitSHA)
	if diffRes.Err != nil {
		return "", fmt.Errorf("git diff incoming paths failed: %w", diffRes.Err)
	}
	if diffRes.Code != 0 {
		return "", fmt.Errorf("git diff incoming paths exited with code %d: %s", diffRes.Code, strings.TrimSpace(string(diffRes.Stderr)))
	}

	var incomingPaths []string
	for _, line := range strings.Split(strings.TrimSpace(string(diffRes.Stdout)), "\n") {
		p := strings.TrimSpace(line)
		if p != "" {
			incomingPaths = append(incomingPaths, p)
		}
	}

	if len(incomingPaths) > 0 {
		checkoutArgs := append([]string{"checkout", newCommitSHA, "--"}, incomingPaths...)
		coRes := run(ctx, repo, checkoutArgs...)
		if coRes.Err != nil {
			return "", fmt.Errorf("git checkout incoming paths failed: %w", coRes.Err)
		}
		if coRes.Code != 0 {
			return "", fmt.Errorf("git checkout incoming paths exited with code %d: %s", coRes.Code, strings.TrimSpace(string(coRes.Stderr)))
		}
	}

	// 5. Handle incoming path renames across disjoint integration.
	// Inspect incoming commits for renames (git diff -M --summary headSHA targetSHA).
	// Remove deleted/renamed source path from working directory if clean, and check out target renamed path.
	if renames, err := InspectIncomingRenames(ctx, run, repo, headSHA, targetSHA, newCommitSHA); err == nil && len(renames) > 0 {
		_ = ApplyIncomingRenames(ctx, run, repo, headSHA, newCommitSHA, renames)
	}

	return newCommitSHA, nil
}
