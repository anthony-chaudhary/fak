package modver

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
)

// Ghost is a history-only (tombstone) module: one that appears in the trunk
// history but no longer exists at HEAD — the "what died and when" fact the live
// Snapshot excludes by design (liveModules folds it out). Rev is the FINAL
// revision the module reached before deletion (the count of non-merge commits
// that ever touched it, the same rev semantics as a live Module); DeletedCommit
// and DeletedDate mark its last touch — for a module absent at HEAD, that last
// commit is the one that removed its final file(s), i.e. the deletion commit.
type Ghost struct {
	Name          string `json:"module"`
	Kind          string `json:"kind"`
	Rev           int    `json:"rev"`
	DeletedCommit string `json:"deleted_commit"`
	DeletedDate   string `json:"deleted_date"` // committer date (ISO) of the deletion
}

// Version renders the final derived version the module reached before it died,
// the same r<rev>+g<shortsha> shape a live Module reports — here the short SHA
// is the deletion commit.
func (g Ghost) Version() string {
	return fmt.Sprintf("r%d+g%s", g.Rev, g.DeletedCommit)
}

// Ghosts computes the tombstone report for the repo at dir: the modules that
// appear in the trunk history but are absent at HEAD. It mirrors the two git
// calls Snapshot makes — one `git ls-files` to bound the LIVE module set, one
// `git log --no-merges --name-only` pass over history — then keeps the COMPLEMENT
// of the live set (the history-only modules) instead of the intersection. The
// cost is bounded exactly as Snapshot is: one ls-files, one bounded log walk over
// the same trackedRoots.
//
// A renamed module (e.g. internal/foo -> internal/bar) can still surface here as
// a ghost: a plain --name-only log records a rename as a delete+add, and this
// pass does no rename-following. Distinguishing a rename from a real deletion is
// the continuity rule tracked separately (sibling #2476) and is out of scope for
// this tombstone listing.
func Ghosts(ctx context.Context, dir string, run Runner) ([]Ghost, error) {
	if run == nil {
		run = RealRunner
	}
	lsArgs := append([]string{"ls-files", "-z", "--"}, trackedRoots...)
	lsOut, err := run(ctx, dir, lsArgs...)
	if err != nil {
		return nil, err
	}
	live := liveModules(lsOut)
	// Same --no-merges rev semantics as Snapshot (#2475): a merge commit carries
	// no authored module work, so it must not count toward a ghost's final rev
	// nor be mistaken for its deletion commit.
	logArgs := append([]string{"log", "--no-merges", "--pretty=format:%x1e%h%x09%cI", "--name-only", "--"}, trackedRoots...)
	logOut, err := run(ctx, dir, logArgs...)
	if err != nil {
		return nil, err
	}
	return parseGhosts(logOut, live), nil
}

// parseGhosts folds the same newest-first `git log --name-only` output parseLog
// consumes into the modules that are NOT live — the tombstones. Rev counts the
// distinct non-merge commits that touched the module (its final revision), and
// the first record a module appears in (the log is newest-first) is its deletion:
// the last commit that touched it, which for a module absent at HEAD removed its
// final file(s). Live modules are dropped — they belong in the Snapshot report,
// not the tombstone view — which is exactly parseLog's `!live` guard inverted.
func parseGhosts(logOut []byte, live map[string]bool) []Ghost {
	byName := map[string]*Ghost{}
	for _, rec := range bytes.Split(logOut, []byte{0x1e}) {
		lines := strings.Split(strings.TrimSpace(string(rec)), "\n")
		if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
			continue
		}
		sha, date, ok := strings.Cut(strings.TrimSpace(lines[0]), "\t")
		if !ok {
			continue
		}
		seen := map[string]bool{}
		for _, file := range lines[1:] {
			name, kind, ok := moduleOf(file)
			if !ok || seen[name] || live[name] {
				continue
			}
			seen[name] = true
			g := byName[name]
			if g == nil {
				// Newest-first: the first commit a ghost appears in is its last
				// touch — its deletion.
				g = &Ghost{Name: name, Kind: kind, DeletedCommit: sha, DeletedDate: date}
				byName[name] = g
			}
			g.Rev++
		}
	}
	out := make([]Ghost, 0, len(byName))
	for _, g := range byName {
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
