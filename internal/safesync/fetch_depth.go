package safesync

import (
	"context"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

// fetchWithDepth bounds history, not transport bytes. Callers must separately
// admit the complete pinned object closure and index/pack lifetime.
func fetchWithDepth(ctx context.Context, run Runner, repo, remote, branch string, depth int, base, target string) error {
	return fetchWithExpectedBase(ctx, run, repo, remote, branch, depth, base, target, "")
}

// fetchWithExpectedBase permits an explicitly pinned advancement of the common
// ancestor when a previously published local commit is first observed remotely.
// It never permits an unrelated or unpinned post-fetch merge-base.
func fetchWithExpectedBase(ctx context.Context, run Runner, repo, remote, branch string, depth int, base, target, afterBase string) error {
	if depth == 0 && base == "" && target == "" && afterBase == "" {
		_, err := checked(ctx, run, repo, "fetch", remote, branch)
		return err
	}
	validOID := func(s string) bool {
		if len(s) != 40 && len(s) != 64 {
			return false
		}
		_, err := hex.DecodeString(s)
		return err == nil && s == strings.ToLower(s)
	}
	if depth <= 0 || !validOID(base) || !validOID(target) || (afterBase != "" && !validOID(afterBase)) {
		return fmt.Errorf("bounded fetch requires positive depth and full lowercase base/target and optional after-base object IDs")
	}
	ref := "refs/heads/" + branch
	tracking := "refs/remotes/" + remote + "/" + branch
	if _, err := checked(ctx, run, repo, "check-ref-format", ref); err != nil {
		return err
	}
	if _, err := checked(ctx, run, repo, "check-ref-format", tracking); err != nil {
		return err
	}
	checkBase := func(want string) error {
		out, err := checked(ctx, run, repo, "merge-base", "--all", "HEAD", tracking)
		if err != nil {
			return fmt.Errorf("bounded fetch original ancestry unproven: %w", err)
		}
		if strings.TrimSpace(string(out)) != want {
			return fmt.Errorf("bounded fetch original merge-base changed or ambiguous")
		}
		return nil
	}
	checkRemote := func() error {
		out, err := checked(ctx, run, repo, "ls-remote", "--refs", "--", remote, ref)
		if err != nil {
			return err
		}
		if strings.TrimSpace(string(out)) != target+"\t"+ref {
			return fmt.Errorf("bounded fetch remote target changed or unconfirmed")
		}
		return nil
	}
	if err := checkBase(base); err != nil {
		return err
	}
	postBase := base
	if afterBase != "" {
		// Both commits must already be locally proven ancestors. A missing or
		// shallowly hidden relationship refuses before the transport runs.
		if _, err := checked(ctx, run, repo, "merge-base", "--is-ancestor", base, afterBase); err != nil {
			return fmt.Errorf("bounded fetch expected post-base does not provably advance original base: %w", err)
		}
		if _, err := checked(ctx, run, repo, "merge-base", "--is-ancestor", afterBase, "HEAD"); err != nil {
			return fmt.Errorf("bounded fetch expected post-base is not a proven HEAD ancestor: %w", err)
		}
		postBase = afterBase
	}
	if err := checkRemote(); err != nil {
		return err
	}
	// Fetch the immutable advertised target, not the moving branch. A server
	// refusal is terminal; never fall back to an unbounded fetch.
	if _, err := checked(ctx, run, repo, "fetch", "--no-tags", "--no-recurse-submodules", "--depth="+strconv.Itoa(depth), "--", remote, target+":"+tracking); err != nil {
		return fmt.Errorf("bounded fetch transport failed; tracking/shallow/object effects may be partial; reconcile before retry: %w", err)
	}
	postconditionError := func(err error) error {
		return fmt.Errorf("bounded fetch completed; postcondition failed; tracking/shallow/object effects require reconciliation; do not retry blindly: %w", err)
	}
	got, err := rev(ctx, run, repo, tracking)
	if err != nil {
		return postconditionError(err)
	}
	if got != target {
		return postconditionError(fmt.Errorf("bounded fetch tracking target readback mismatch"))
	}
	if err := checkBase(postBase); err != nil {
		return postconditionError(err)
	}
	if err := checkRemote(); err != nil {
		return postconditionError(err)
	}
	return nil
}
