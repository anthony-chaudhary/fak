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
	if depth == 0 && base == "" && target == "" {
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
	if depth <= 0 || !validOID(base) || !validOID(target) {
		return fmt.Errorf("bounded fetch requires positive depth and full lowercase base/target object IDs")
	}
	ref := "refs/heads/" + branch
	tracking := "refs/remotes/" + remote + "/" + branch
	if _, err := checked(ctx, run, repo, "check-ref-format", ref); err != nil {
		return err
	}
	if _, err := checked(ctx, run, repo, "check-ref-format", tracking); err != nil {
		return err
	}
	checkBase := func() error {
		out, err := checked(ctx, run, repo, "merge-base", "--all", "HEAD", tracking)
		if err != nil {
			return fmt.Errorf("bounded fetch original ancestry unproven: %w", err)
		}
		if strings.TrimSpace(string(out)) != base {
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
	if err := checkBase(); err != nil {
		return err
	}
	if err := checkRemote(); err != nil {
		return err
	}
	// Fetch the immutable advertised target, not the moving branch. A server
	// refusal is terminal; never fall back to an unbounded fetch.
	if _, err := checked(ctx, run, repo, "fetch", "--no-tags", "--no-recurse-submodules", "--depth="+strconv.Itoa(depth), "--", remote, target+":"+tracking); err != nil {
		return err
	}
	got, err := rev(ctx, run, repo, tracking)
	if err != nil {
		return err
	}
	if got != target {
		return fmt.Errorf("bounded fetch tracking target readback mismatch")
	}
	if err := checkBase(); err != nil {
		return err
	}
	return checkRemote()
}
