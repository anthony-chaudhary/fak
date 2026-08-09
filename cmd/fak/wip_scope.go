package main

// wip_scope.go — the CAPTURE-scope vs CLAIM-scope seam of `fak wip` (#5539).
//
// A checkpoint is minted by staging the ENTIRE working tree into a throwaway index
// (`read-tree HEAD` + `add -A`, wip.go), so on a shared tree the object filed at
// refs/fak/wip/<session> holds every concurrently-dirty peer's edits too. That width is
// DELIBERATE — it is what makes the snapshot lossless for crash recovery — but it means
// the ref's session key names the CAPTURER, never the AUTHOR.
//
// Capture therefore stays tree-wide and this file changes none of it. What it owns is the
// one verb that turns a snapshot into an IRREVERSIBLE act: `wip land`, which commits. A
// land of an undeclared tree-wide snapshot is exactly the `git add -A` sweep this repo
// forbids, laundered through a session key — so land refuses it (TREE_WIDE_SNAPSHOT, exit
// 3) whenever another session's live checkpoint ref is the repo's own evidence that the
// tree is shared. The remedy is a declaration, in either of two places:
//
//   - `fak wip land --path <p>` — the caller declares, at land time, what it owns. Same
//     discipline `fak commit --path` enforces.
//   - `fak wip checkpoint --path <p>` — the session declares AT CAPTURE, and the claim is
//     recorded in wipref.Stamp.Scope. That is the durable half: a fleet host recovering a
//     crashed session cannot know what the dead session owned, and the stamp is the only
//     place that answer survives the session.
//
// `--all` keeps the honest escape open: an operator who genuinely wants the whole snapshot
// says so and gets it. The point of #5539 is that the sweep must be DECLARED, not that it
// must be impossible.

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// Closed refusal tokens for the land scope gate. They are the machine-checkable half of a
// refusal — a caller keys on the token, the prose only explains it.
const (
	// wipReasonTreeWideSnapshot: the snapshot is tree-wide, a peer session holds a live
	// checkpoint, and the caller declared nothing. Landing would commit that peer's work
	// under this session's name.
	wipReasonTreeWideSnapshot = "TREE_WIDE_SNAPSHOT"
	// wipReasonScopeMatchedNothing: a declaration was made but names no file the snapshot
	// actually captured. Committing the whole snapshot instead would be the sweep the
	// declaration exists to prevent, so this fails closed rather than widening.
	wipReasonScopeMatchedNothing = "SCOPE_MATCHED_NOTHING"
)

// wipLandOptions is the declared shape of a land. Paths and All are the two ways a caller
// takes responsibility for what a tree-wide snapshot commits; Message and Push are the
// pre-existing knobs, carried here so wipLand can be a thin wrapper.
type wipLandOptions struct {
	Message string   // commit subject ("" = the audit-OK default, wipLandSubject)
	Push    bool     // push after a verified commit
	Paths   []string // the repo-relative paths this land claims (repeatable --path)
	All     bool     // land the WHOLE snapshot, peers' edits included — declared, not implicit
}

// wipResolveLandScope decides which of a checkpoint's captured files a land may commit.
//
// Precedence, most explicit first:
//
//  1. All — the operator declared the sweep; every captured file lands.
//  2. opts.Paths — the caller declared at land time.
//  3. the checkpoint's stamped Scope — the session declared at capture, and that claim
//     outlives it, which is what lets a fleet host land a crashed session's work safely.
//  4. otherwise: if any OTHER session holds a live checkpoint ref, the tree is provably
//     shared and the snapshot is unattributable → refuse (wipReasonTreeWideSnapshot).
//  5. otherwise: no peer, nothing to misattribute — the whole snapshot lands, which is the
//     single-session behaviour every pre-#5539 caller relies on.
//
// Returns (files to land, files deliberately excluded, refusal token, error). A non-empty
// refusal token always comes with an explanatory error and means the caller must exit 3
// having committed and materialized nothing; an error with an empty token is a plain
// runtime failure.
func wipResolveLandScope(ctx context.Context, repo, session, ref, obj string, captured []string, opts wipLandOptions) (files, excluded []string, reason string, err error) {
	if opts.All {
		return captured, nil, "", nil
	}

	declared, source := wipNormalizeScope(opts.Paths), "--path"
	if len(declared) == 0 {
		rec, rerr := wipRecordAt(ctx, repo, ref, obj)
		if rerr != nil {
			return nil, nil, "", fmt.Errorf("read checkpoint stamp: %w", rerr)
		}
		declared, source = wipNormalizeScope(rec.Stamp.Scope), "the checkpoint's stamped scope"
	}

	if len(declared) == 0 {
		peers, perr := wipPeerCheckpointSessions(ctx, repo, session)
		if perr != nil {
			return nil, nil, "", perr
		}
		if len(peers) > 0 {
			return nil, nil, wipReasonTreeWideSnapshot, fmt.Errorf(
				"checkpoint %s is a tree-wide snapshot of %d file(s) and %d peer session(s) hold live checkpoints (%s), so the snapshot cannot be attributed to %s — declare what you own with --path, or take the whole snapshot deliberately with --all",
				session, len(captured), len(peers), strings.Join(peers, ", "), session)
		}
		return captured, nil, "", nil // no peer checkpoint: nothing to misattribute
	}

	files, excluded = wipPartitionByScope(captured, declared)
	if len(files) == 0 {
		return nil, nil, wipReasonScopeMatchedNothing, fmt.Errorf(
			"%s names %s, none of which the %s checkpoint captured (it holds %s) — nothing to land",
			source, strings.Join(declared, ", "), session, strings.Join(captured, ", "))
	}
	return files, excluded, "", nil
}

// wipPeerCheckpointSessions lists every session OTHER than self that holds a live
// checkpoint ref. Their existence is the repo's own, git-checkable evidence that this
// working tree is shared — the fact that makes an undeclared tree-wide land unsafe. It
// deliberately reads the checkpoint namespace and not the lease refs: a session that
// crashed still owns its captured hunks, and a lease that expired does not un-author them.
func wipPeerCheckpointSessions(ctx context.Context, repo, self string) ([]string, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []string
	for _, r := range recs {
		s := wipSessionOf(r)
		if s == "" || s == self || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out, nil
}

// wipPartitionByScope splits a checkpoint's captured file list into the files a declared
// scope claims and the files it does not. A declaration matches a file exactly, or as a
// directory prefix (`internal/wipref` claims `internal/wipref/wipref.go`), so a caller can
// declare the leaf it owns without enumerating its files. Both sides preserve the input
// order, which git already sorted.
func wipPartitionByScope(captured, declared []string) (in, out []string) {
	for _, f := range captured {
		if wipScopeClaims(declared, wipNormalizeScopePath(f)) {
			in = append(in, f)
		} else {
			out = append(out, f)
		}
	}
	return in, out
}

// wipScopeClaims reports whether any already-normalized declaration covers file.
func wipScopeClaims(declared []string, file string) bool {
	for _, d := range declared {
		if file == d || strings.HasPrefix(file, d+"/") {
			return true
		}
	}
	return false
}

// wipNormalizeScope canonicalizes a declared scope for storage and comparison: repo paths
// with forward slashes, no "./" prefix, no trailing slash, deduplicated, sorted. It returns
// nil (not an empty slice) for an empty declaration, so an absent claim stays absent
// through the stamp's `omitempty` rather than becoming an empty JSON array.
func wipNormalizeScope(scope []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range scope {
		s = wipNormalizeScopePath(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	if len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}

// wipNormalizeScopePath canonicalizes ONE path: Windows separators folded to git's, a
// leading "./" and any trailing "/" removed.
func wipNormalizeScopePath(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, `\`, "/"))
	for strings.HasPrefix(p, "./") {
		p = p[2:]
	}
	return strings.TrimSuffix(p, "/")
}

// wipSameScope compares two already-normalized scopes. It is what reopens the checkpoint's
// unchanged-tree debounce when only the CLAIM changed: the stamped scope is what a later
// land reads, so re-declaring it over an unchanged tree must rewrite the stamp instead of
// being absorbed as "nothing changed".
func wipSameScope(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
