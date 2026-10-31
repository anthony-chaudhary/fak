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
// An absent tracking ref may be seeded only from pinned HEAD while preserving
// every previously visible commit and parent edge.
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
	checkPinnedHead := func() error {
		head, err := rev(ctx, run, repo, "HEAD")
		if err != nil {
			return err
		}
		if head != base {
			return fmt.Errorf("bounded fetch bootstrap requires unchanged HEAD equal to explicit base")
		}
		return nil
	}
	checkBootstrap := func() error {
		// A failed merge-base alone does not prove absence. Enumeration must
		// succeed without warnings; even a conflicting descendant refuses.
		res := run(ctx, repo, "for-each-ref", "--format=%(refname)", "--", tracking)
		if res.Err != nil {
			return res.Err
		}
		if res.Code != 0 || len(res.Stderr) != 0 || len(res.Stdout) != 0 {
			return fmt.Errorf("bounded fetch tracking ref absence unproven")
		}
		return checkPinnedHead()
	}
	readVisibleGraph := func() (map[string]map[string]bool, error) {
		res := run(ctx, repo, "rev-list", "--parents", "--all", "HEAD")
		if res.Err != nil {
			return nil, res.Err
		}
		if res.Code != 0 || len(res.Stderr) != 0 {
			return nil, fmt.Errorf("bounded fetch visible commit graph unproven")
		}
		graph := make(map[string]map[string]bool)
		for _, line := range strings.Split(strings.TrimSpace(string(res.Stdout)), "\n") {
			fields := strings.Fields(line)
			if len(fields) == 0 {
				return nil, fmt.Errorf("bounded fetch visible commit graph is empty or malformed")
			}
			for _, oid := range fields {
				if !validOID(oid) {
					return nil, fmt.Errorf("bounded fetch visible commit graph has an invalid object ID")
				}
			}
			if _, exists := graph[fields[0]]; exists {
				return nil, fmt.Errorf("bounded fetch visible commit graph has a duplicate commit")
			}
			parents := make(map[string]bool)
			for _, parent := range fields[1:] {
				parents[parent] = true
			}
			graph[fields[0]] = parents
		}
		if _, exists := graph[base]; !exists {
			return nil, fmt.Errorf("bounded fetch visible commit graph omits pinned HEAD")
		}
		for _, parents := range graph {
			for parent := range parents {
				if _, exists := graph[parent]; !exists {
					return nil, fmt.Errorf("bounded fetch visible commit graph omits a visible parent")
				}
			}
		}
		return graph, nil
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
	bootstrap := false
	if err := checkBase(base); err != nil {
		if bootstrapErr := checkBootstrap(); bootstrapErr != nil {
			return fmt.Errorf("%w; bootstrap refused: %v", err, bootstrapErr)
		}
		bootstrap = true
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
	var before map[string]map[string]bool
	if bootstrap {
		var err error
		before, err = readVisibleGraph()
		if err != nil {
			return err
		}
		if err := checkBootstrap(); err != nil {
			return err
		}
	}
	// Fetch the immutable advertised target, not the moving branch. A server
	// refusal is terminal; never fall back to an unbounded fetch.
	if _, err := checked(ctx, run, repo, "fetch", "--no-tags", "--no-recurse-submodules", "--depth="+strconv.Itoa(depth), "--", remote, target+":"+tracking); err != nil {
		return fmt.Errorf("bounded fetch transport failed; tracking/shallow/object effects may be partial; reconcile before retry: %w", err)
	}
	postconditionError := func(err error) error {
		return fmt.Errorf("bounded fetch completed; postcondition failed; tracking/shallow/object effects require reconciliation; do not retry blindly: %w", err)
	}
	if bootstrap {
		after, err := readVisibleGraph()
		if err != nil {
			return postconditionError(err)
		}
		// A depth boundary can hide old history without changing merge-base.
		// New ancestry is allowed; every previously visible node/edge stays.
		for commit, parents := range before {
			gotParents, exists := after[commit]
			if !exists {
				return postconditionError(fmt.Errorf("bounded fetch hid previously visible commit %s", commit))
			}
			for parent := range parents {
				if !gotParents[parent] {
					return postconditionError(fmt.Errorf("bounded fetch hid previously visible parent edge %s %s", commit, parent))
				}
			}
		}
		if err := checkPinnedHead(); err != nil {
			return postconditionError(err)
		}
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
