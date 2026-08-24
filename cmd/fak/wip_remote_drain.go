package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

type wipRemoteDrainResult struct {
	Remote        string                        `json:"remote"`
	Applied       bool                          `json:"applied"`
	DefaultBranch string                        `json:"default_branch"`
	Candidates    []wipref.RemoteDrainCandidate `json:"candidates"`
	Deleted       []string                      `json:"deleted"`
}

func runWipRemoteDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip remote-drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("C", "", "run in this git repo (default cwd)")
	remote := fs.String("remote", "origin", "remote name or URL")
	apply := fs.Bool("apply", false, "delete only remotely-contained checkpoint refs")
	allowPeer := fs.Bool("allow-peer", false, "consider peer session refs (still requires containment)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	res, err := wipRemoteDrain(context.Background(), *repo, *remote, *apply, *allowPeer)
	if code, done := emitResultOrError(stdout, stderr, "fak wip remote-drain", *asJSON, res, err); done {
		return code
	}
	fmt.Fprintf(stdout, "remote checkpoint drain: remote=%s apply=%v branch=%s\n", res.Remote, res.Applied, res.DefaultBranch)
	for _, c := range res.Candidates {
		fmt.Fprintf(stdout, "  %s %s: %s\n", c.State, c.Session, c.Reason)
	}
	return 0
}

func wipRemoteDrain(ctx context.Context, repo, remote string, apply, allowPeer bool) (wipRemoteDrainResult, error) {
	res := wipRemoteDrainResult{Remote: remote, Applied: apply, Candidates: []wipref.RemoteDrainCandidate{}, Deleted: []string{}}
	if !wipref.ValidRemote(remote) {
		return res, fmt.Errorf("invalid remote %q", remote)
	}
	if strings.TrimSpace(repo) == "" {
		repo = "."
	}
	// Clone into OS scratch so report mode changes neither the caller's repository nor
	// the remote. The disposable mirror supplies every object needed for byte-level
	// containment without trusting a stale local tracking ref.
	tmp, err := os.MkdirTemp("", "fak-wip-remote-drain-*")
	if err != nil {
		return res, err
	}
	defer os.RemoveAll(tmp)
	mirror := filepath.Join(tmp, "remote.git")
	remoteURL, err := wipDrainGit(ctx, repo, nil, "remote", "get-url", remote)
	if err != nil {
		remoteURL = remote
	}
	if _, err = wipDrainGit(ctx, repo, nil, "clone", "--mirror", strings.TrimSpace(remoteURL), mirror); err != nil {
		return res, err
	}
	head, err := wipDrainGit(ctx, mirror, nil, "symbolic-ref", "HEAD")
	if err != nil {
		return res, fmt.Errorf("remote default branch is not advertised: %w", err)
	}
	res.DefaultBranch = strings.TrimSpace(head)
	refsText, err := wipDrainGit(ctx, mirror, nil, "for-each-ref", "--format=%(objectname) %(refname)", "refs/fak/wip/")
	if err != nil {
		return res, err
	}
	refs, err := wipref.ParseRemoteRefs(refsText)
	if err != nil {
		return res, err
	}
	owned := map[string]bool{}
	local, _ := wipDrainGit(ctx, repo, nil, "for-each-ref", "--format=%(refname)", "refs/fak/wip/")
	for _, line := range strings.Split(local, "\n") {
		owned[strings.TrimPrefix(strings.TrimSpace(line), "refs/fak/wip/")] = true
	}
	contained := func(r wipref.RemoteRef) (bool, error) {
		patch, err := wipDrainGit(ctx, mirror, nil, "diff", "--binary", r.SHA+"^", r.SHA)
		if err != nil {
			return false, err
		}
		if strings.TrimSpace(patch) == "" {
			return true, nil
		}
		index := filepath.Join(tmp, "containment-index")
		_ = os.Remove(index)
		if _, err = wipDrainGitEnv(ctx, mirror, nil, []string{"GIT_INDEX_FILE=" + index}, "read-tree", res.DefaultBranch); err != nil {
			return false, err
		}
		_, err = wipDrainGitEnv(ctx, mirror, strings.NewReader(patch), []string{"GIT_INDEX_FILE=" + index}, "apply", "--cached", "--reverse", "--check", "-")
		return err == nil, nil
	}
	res.Candidates = wipref.PlanRemoteDrain(refs, owned, contained, allowPeer)
	if !apply {
		return res, nil
	}
	for _, c := range res.Candidates {
		if c.State == wipref.RemoteDrainSafe {
			if _, err = wipDrainGit(ctx, repo, nil, "push", remote, c.DeleteRefspec); err != nil {
				return res, err
			}
			res.Deleted = append(res.Deleted, c.Ref)
		}
	}
	return res, nil
}

func wipDrainGit(ctx context.Context, repo string, stdin io.Reader, args ...string) (string, error) {
	return wipDrainGitEnv(ctx, repo, stdin, nil, args...)
}

func wipDrainGitEnv(ctx context.Context, repo string, stdin io.Reader, extraEnv []string, args ...string) (string, error) {
	var out, detail string
	var code int
	var runErr error
	if stdin != nil {
		out, detail, code, runErr = gitWipStdin(ctx, repo, readDrainInput(stdin), args...)
	} else {
		out, detail, code, runErr = gitWip(ctx, repo, extraEnv, args...)
	}
	if runErr != nil {
		return "", fmt.Errorf("remote checkpoint git operation could not start: %w", runErr)
	}
	if code != 0 {
		return "", fmt.Errorf("remote checkpoint git operation exited %d: %s", code, strings.TrimSpace(detail))
	}
	return out, nil
}

func readDrainInput(r io.Reader) string {
	b, _ := io.ReadAll(r)
	return string(b)
}
