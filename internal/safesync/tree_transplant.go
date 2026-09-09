package safesync

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

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

	return newCommitSHA, nil
}
