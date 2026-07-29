package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ReasonStaleUntrackedPath is the refusal token for a path that is untracked in THIS
// checkout yet already present on origin/<trunk> as a file whose content DIFFERS from the
// working copy (issue #5408). In a shared clone whose HEAD has fallen behind the trunk, such
// a path shows up in `git status` as `??` and is therefore indistinguishable, by local index
// state alone, from genuinely new work — so a pathspec commit lands a working-tree copy that
// PREDATES what peers already pushed there (the #1088 revert).
//
// It is deliberately distinct from ReasonStaleBaseDeletion. That one answers "would this
// commit drop a contiguous block a peer added to a file I already track?" and reasons over
// line runs; this one answers "does this path already exist upstream while my index has
// never heard of it?" and reasons over blob identity. Conflating them produces the
// misleading diagnosis #5408 documents: for an untracked path `git diff origin/<trunk> -- P`
// shows trunk's blob as WHOLLY deleted, so the line-run reading reports "would drop N
// lines" where N is nothing but trunk's own line count.
const ReasonStaleUntrackedPath = "STALE_UNTRACKED"

// staleUntrackedNoOpListed bounds how many no-op paths the advisory names before it falls
// back to a count. The field case that motivated #5408 had 40 of them; a refusal-sized wall
// of paths would bury the one line an operator has to act on.
const staleUntrackedNoOpListed = 3

// checkStaleUntrackedPath asks, for each requested path P: is P absent from THIS checkout's
// index while origin/<trunk> already holds a FILE at P, and does the working copy differ from
// that file?
//
// Every read goes through the injected run against the ALREADY-PRESENT-LOCALLY
// origin/<trunk> ref — no fetch, no network — so the check can never stall the commit lane
// on a hiccup, and it is fully drivable from canned evidence in tests.
//
// Three outcomes, and the split between them is the point of #5408:
//
//   - refusal != "": P is untracked here, upstream holds a file at P, and the two blobs
//     DIFFER. Committing P by pathspec would supersede the trunk copy with an older one.
//     The caller refuses (block) or records it (warn).
//   - advisory != "": one or more paths are untracked here and byte-IDENTICAL to what
//     upstream already holds. That is a no-op add, not a clobber — nothing of the trunk is
//     lost — so it must NOT refuse. It is named and the commit proceeds. (40 of the 69
//     untracked paths measured in #5408 were this case; refusing them would have blocked
//     more honest work than the 4 real hazards it caught.)
//   - unclaimed: the paths this check said nothing about, for the line-run check at step (4b)
//     to judge. A path claimed here is withheld from (4b) precisely because (4b)'s reading of
//     an untracked path is the misleading one #5408 documents.
//
// Direction of safety. A refusal is emitted ONLY from a positive reading: the ref resolves,
// HEAD does not already contain the trunk tip, `git ls-files -- P` is empty (P is not in the
// index), `git ls-tree -l <origin> -- P` yields exactly ONE entry and it is a BLOB, and
// `git hash-object -- P` yields an object id that differs from it. Every unknown falls back
// to the prior behavior by leaving the path unclaimed — no remote-tracking ref (fresh clone,
// installer never ran), an unreadable merge base, a HEAD that already contains the trunk tip,
// a non-zero or unreadable read, a path that is tracked here, a path absent upstream, a
// pathspec that matches several upstream entries, a path that resolves upstream to a tree
// rather than a file. A check that refused on an unreadable ref would wedge every lane in a
// shared tree, which is strictly worse than the bug it closes.
func checkStaleUntrackedPath(ctx context.Context, run Runner, dir, trunk string, paths []string) (refusal, advisory string, unclaimed []string) {
	originRef := "origin/" + trunk

	tip, code, err := run(ctx, dir, "rev-parse", "--verify", "--quiet", originRef)
	if err != nil || code != 0 || strings.TrimSpace(tip) == "" {
		return "", "", paths // upstream unknown -> prior behavior
	}
	tip = strings.TrimSpace(tip)

	// HEAD already contains the trunk tip => everything upstream is in this index too, so an
	// untracked path here cannot be a stale copy of a trunk file; it is new work, or an index
	// removal that step (4c) owns. One extra read buys the whole scan being skipped in the
	// common up-to-date case, and it keeps this check off a class it would misname.
	mb, code, err := run(ctx, dir, "merge-base", "HEAD", originRef)
	if err != nil || code != 0 || strings.TrimSpace(mb) == "" || strings.TrimSpace(mb) == tip {
		return "", "", paths
	}

	var noop []string
	for _, p := range paths {
		trunkBlob, trunkSize, localBlob, ok := staleUntrackedBlobs(ctx, run, dir, originRef, p)
		if !ok {
			unclaimed = append(unclaimed, p)
			continue
		}
		if localBlob == trunkBlob {
			noop = append(noop, p)
			continue
		}
		if refusal == "" {
			refusal = staleUntrackedDetail(dir, trunk, originRef, p, tip, trunkBlob, localBlob, trunkSize)
		}
	}
	if len(noop) > 0 {
		advisory = staleUntrackedNoOpNote(trunk, originRef, tip, noop)
	}
	return refusal, advisory, unclaimed
}

// staleUntrackedBlobs reads the pair of object ids that decide this class for one path: what
// origin/<trunk> holds at p, and what p hashes to in the working tree. ok is false — meaning
// "not this class, leave the path alone" — for every state the pair cannot be established
// from: p is in the index, p is absent upstream or is a tree there, the pathspec matches more
// than one upstream entry, or either read exits non-zero.
func staleUntrackedBlobs(ctx context.Context, run Runner, dir, originRef, p string) (trunkBlob string, trunkSize int64, localBlob string, ok bool) {
	indexed, code, err := run(ctx, dir, "ls-files", "--", p)
	if err != nil || code != 0 {
		return "", 0, "", false // cannot read the index for p -> prior behavior for p
	}
	if strings.TrimSpace(indexed) != "" {
		return "", 0, "", false // p (or something under it) is tracked here — not this class
	}

	trunkBlob, trunkSize, ok = upstreamBlobAt(ctx, run, dir, originRef, p)
	if !ok {
		return "", 0, "", false // absent upstream, a tree, ambiguous, or unreadable
	}

	out, code, err := run(ctx, dir, "hash-object", "--", p)
	if err != nil || code != 0 {
		return "", 0, "", false // cannot hash the working copy -> prior behavior for p
	}
	localBlob = strings.TrimSpace(out)
	if localBlob == "" {
		return "", 0, "", false
	}
	return trunkBlob, trunkSize, localBlob, true
}

// upstreamBlobAt reads the object id and size origin/<trunk> holds at p, via one
// `git ls-tree -l` against the local remote-tracking ref. ok is false unless the listing is
// EXACTLY one entry AND that entry is a blob: a pathspec that resolves upstream to a tree (a
// requested directory) or to several files is not a single-file collision, and an empty or
// unparseable listing is an unknown — all of which fall back to the prior behavior.
func upstreamBlobAt(ctx context.Context, run Runner, dir, originRef, p string) (blob string, size int64, ok bool) {
	out, code, err := run(ctx, dir, "ls-tree", "-l", originRef, "--", p)
	if err != nil || code != 0 {
		return "", 0, false
	}
	var entries [][]string
	for _, line := range strings.Split(out, "\n") {
		meta, name, cut := strings.Cut(line, "\t")
		if !cut || strings.TrimSpace(name) == "" {
			continue
		}
		entries = append(entries, strings.Fields(meta))
	}
	if len(entries) != 1 {
		return "", 0, false // nothing there, or an ambiguous pathspec
	}
	f := entries[0]
	if len(f) < 4 || f[1] != "blob" {
		return "", 0, false // a tree entry, or a shape we do not recognize
	}
	n, perr := strconv.ParseInt(f[3], 10, 64)
	if perr != nil {
		n = -1 // keep the id, report the size as unknown
	}
	return f[2], n, true
}

// staleUntrackedDetail composes the operator-facing refusal text: what was read (both object
// ids, the trunk tip, both sizes), how to refresh, which comparison to trust, and the
// documented one-shot escape. Two things it states explicitly because #5408 was written about
// getting them wrong: the sha it names is the TRUNK TIP (not this clone's fork point), and
// `git diff <origin> -- P` is called out as MISLEADING for an untracked path rather than
// offered as the comparison. It also names its own boundary — the local remote-tracking ref
// only, so a checkout that has never fetched is outside its reach.
func staleUntrackedDetail(dir, trunk, originRef, p, tip, trunkBlob, localBlob string, trunkSize int64) string {
	return fmt.Sprintf(
		"%s is untracked in this checkout but ALREADY exists on %s (tip %s) as blob %s%s; your working copy hashes to %s%s — committing it by pathspec would land a copy that PREDATES the trunk (#5408, the #1088 revert). "+
			"Refresh first: git fetch origin %s && fak merge --apply %s, then compare content-to-content with `git show %s:%s` "+
			"(NOT `git diff %s -- %s`: for an untracked path git shows trunk's whole file as deleted, so its line counts are trunk's own, not a delta). "+
			"To supersede the trunk copy deliberately, having seen that comparison, re-run with %s=warn. "+
			"(this check reads the local %s ref only — never the network — so a checkout that has never fetched is outside its reach)",
		p, originRef, shortObject(tip), shortObject(trunkBlob), byteSuffix(trunkSize),
		shortObject(localBlob), byteSuffix(localFileSize(dir, p)),
		trunk, originRef,
		originRef, p, originRef, p,
		staleBaseEnvVar,
		originRef,
	)
}

// staleUntrackedNoOpNote is the ADVISORY for the byte-identical case — the one that must not
// refuse. Nothing of the trunk is at risk (the blobs are equal), so the commit proceeds; the
// note exists because the operator's real problem is upstream of this commit: HEAD is behind,
// and these paths are already shipped work masquerading as `??`.
func staleUntrackedNoOpNote(trunk, originRef, tip string, noop []string) string {
	named := noop
	more := ""
	if len(named) > staleUntrackedNoOpListed {
		named = named[:staleUntrackedNoOpListed]
		more = fmt.Sprintf(" (+%d more)", len(noop)-staleUntrackedNoOpListed)
	}
	return fmt.Sprintf(
		"%d requested path(s) are untracked here but ALREADY on %s (tip %s) with BYTE-IDENTICAL content: %s%s. "+
			"Nothing of the trunk is superseded, so this is not a refusal — but committing them adds nothing the trunk lacks. "+
			"git fetch origin %s && fak merge --apply %s clears them from your untracked list.",
		len(noop), originRef, shortObject(tip), strings.Join(named, ", "), more, trunk, originRef,
	)
}

// shortObject abbreviates an object id for the refusal text, leaving anything already short
// (or empty) untouched.
func shortObject(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return id
}

// byteSuffix formats a known size as " (N bytes)" and an unknown one as the empty string, so
// an unstattable working copy never fabricates a number in the refusal.
func byteSuffix(n int64) string {
	if n < 0 {
		return ""
	}
	return fmt.Sprintf(" (%d bytes)", n)
}

// localFileSize stats the working-tree copy of p under dir. It returns -1 when the size
// cannot be determined (no dir, a vanished file, a directory), so the refusal reports the
// object ids it positively read and stays silent about the rest.
func localFileSize(dir, p string) int64 {
	if dir == "" {
		return -1
	}
	fi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(p)))
	if err != nil || fi.IsDir() {
		return -1
	}
	return fi.Size()
}
