package hooks

// landstree.go — the `HEAD ⊕ staged` change-set view (#5931, epic #5949).
//
// THE SEAM. filesource.go states it plainly in its own doc: fileProbe is "Exists + Size over the
// WORKING TREE", and both *StagedDiff and *TrackedTree satisfy it. So the staged gates resolve
// links, classify files and size blobs against the live shared checkout — not against the tree
// their commit produces. On this repo that checkout is shared by ~20 agent sessions, which makes
// a PEER'S uncommitted edit a tree-wide input to YOUR gate. A gate on a shared checkout has three
// candidate inputs — the working tree, the index, and HEAD ⊕ index — and only the third is the
// tree the commit actually lands. Reading either of the other two makes the verdict a function of
// peer WIP, and because `fak commit` holds the lane lock 60–126s fleet-wide, a false refusal costs
// the whole fleet, not one author. #5128 is one instance of this class (a peer's staged-but-
// uncommitted file wedging DESKTOP_POPUP_REGRESSION); this file fixes the seam they all share.
//
// ⭐ THE DISCRIMINATOR IS SCOPE, NOT SEVERITY: deny what THIS commit introduces. For a path probe
// that means judging HEAD with the staged paths overlaid — exactly the tree a pathspec commit
// produces — rather than the disk the author happens to be standing on.
//
// ⛔ AND IT IS NOT A HEAD-BASELINE SUBTRACTION. The shape this catches is one that subtracting
// "what HEAD already fails" does NOT reach: a peer's unstaged edit is absent from HEAD (not
// committed) and absent from your staged set (not staged), so a HEAD diff calls it "introduced by
// this commit" and denies. The fix has to be the INPUT the gate reads, not a post-hoc diff of its
// output.
//
// ⭐ THE INDEX *IS* HEAD ⊕ STAGED. That is the whole implementation, and it is why this view costs
// nearly nothing: `git ls-files` is the index path set (HEAD's paths, minus what the commit
// deletes, plus what it adds) and `:<path>` is the index blob (the staged content when a path is
// staged, HEAD's byte-identical blob when it is not). ReadStagedDiff ALREADY reads `git ls-files`
// into StagedDiff.IndexPaths, so building the view spawns no git at all on the hot path — which
// matters, because the whole hook runs inside one wall clock (#5335) and a view that cost a git
// fan-out would be paid for out of the budget that exists to let commits through.
//
// ⛔ WHERE IT STILL READS HEAD. Sizes for a path the commit does NOT touch come from
// `git ls-tree -r --long <base>` — ONE lazy spawn for the whole tree, taken only if a gate ever
// asks the size of an untouched path (FILE_ADMISSION only ever asks about staged paths, so on the
// hot path it never runs). Staged paths are sized with `git cat-file -s :<path>`, because for a
// staged modification HEAD's size is the OLD size and using it would be the same lie in a
// different coat. Content falls back from `:<path>` to `<base>:<path>` for the same reason.
//
// ⛔ WHAT IT DELIBERATELY DOES NOT DO: there is no disk fallback anywhere in this file. That
// fallback is the leak — StagedDiff.FileBytes reads an UNTRACKED path off disk when `git show
// :<path>` misses, so a peer's untracked file is readable content to a gate that asked for a
// tracked one. A view that kept the fallback would carry the defect it exists to remove.
//
// ⛔ FAIL-CLOSED ON CONSTRUCTION. newLandsTree returns ok=false when it cannot see an index, and
// the caller then runs the gate over the WORKING TREE exactly as before (gatescope.go). A
// scope-attribution bug must not become a hole in the gates it scopes: this mechanism may only
// ever turn a denial into a pass when the violation is absent from the committed tree, and it can
// never hide a defect you are landing, because your staged paths ARE the view.

import (
	"context"
	"path"
	"strconv"
	"strings"
	"sync"
)

// The view names that ride on Finding.View, so a false refusal is diagnosable from
// `fak hooks pre-commit -json` alone — without taking the commit lock to reproduce it.
const (
	// ViewLandsTree — the finding was computed over HEAD ⊕ staged: the tree this commit lands.
	// A finding carrying this view is attributable to the commit, by construction.
	ViewLandsTree = "lands-tree"
	// ViewWorktree — the finding was computed over the working tree (the shared checkout). Either
	// the gate is WORKTREE_BY_DESIGN (gatescope.go names why, per gate), or it predates this seam.
	ViewWorktree = "worktree"
	// ViewWorktreeFallback — the gate is classified LANDS_TREE but the view could NOT be built
	// this run (no runner, no readable index), so it ran over the working tree. Named distinctly
	// on purpose: "this gate is scoped" and "this gate was SUPPOSED to be scoped and was not" are
	// different claims, and a reader chasing a false refusal needs to tell them apart in one look.
	ViewWorktreeFallback = "worktree-fallback"
)

// landsTree is the HEAD ⊕ staged path/blob source. It implements fileProbe and fileReader, so a
// gate body adopts it without a single line changing — the gates already take their reads through
// those two interfaces (filesource.go).
type landsTree struct {
	root string
	run  Runner
	ctx  context.Context

	// indexish / base are the two treeish prefixes: ":" for the index (the staged side) and
	// "HEAD" for the committed side. Range mode (ReadRangeDiff) sets Treeish to "<tip>:", and both
	// sides then resolve against that committed tip.
	indexish string
	base     string

	// paths is the change-set's file set; dirs is every ancestor directory of a member, so
	// Exists() keeps os.path.exists' semantics (a directory exists) that the link resolvers rely on.
	paths map[string]bool
	dirs  map[string]bool
	// staged is the subset this commit TOUCHES. A staged path must be sized from the index, never
	// from the base tree — for a staged modification the base size is the pre-commit size.
	staged map[string]bool

	mu sync.Mutex
	// cache/sizes memoize per path. Same discipline as StagedDiff.fileCache and for the same
	// reason: the pre-commit CLI ABANDONS a gate that overruns its budget without cancelling it
	// (#5335), so an abandoned Check can reach this view concurrently with the next gate's.
	cache map[string]fileEntry
	sizes map[string]int64
	// baseSizes is the lazily-filled `ls-tree -r --long <base>` table for untouched paths, and
	// baseSizesDone records that the (single) spawn already happened — including when it FAILED,
	// so one broken read is not retried once per probed path.
	baseSizes     map[string]int64
	baseSizesDone bool
}

// newLandsTree builds the view from a StagedDiff. ok=false means "no verdict from this view is
// possible" and the caller must fall back to the working tree unchanged — never to silence.
func newLandsTree(d *StagedDiff) (*landsTree, bool) {
	if d == nil || d.run == nil || strings.TrimSpace(d.Root) == "" {
		return nil, false
	}
	ctx := d.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	indexish := d.Treeish
	if strings.TrimSpace(indexish) == "" {
		indexish = ":"
	}
	base := "HEAD"
	if indexish != ":" {
		base = strings.TrimSuffix(indexish, ":")
	}

	// IndexPaths is `git ls-files` in staged mode and `ls-tree -r --name-only <tip>` in range
	// mode — both already read by the caller, so the common path spawns nothing. The re-read is
	// the fallback for a hand-built diff (a unit fixture, a caller that skipped the listing).
	paths := d.IndexPaths
	if len(paths) == 0 {
		out, code, err := d.run(ctx, d.Root, "ls-files", "-z")
		if err != nil || code != 0 {
			return nil, false
		}
		for _, p := range strings.Split(out, "\x00") {
			if p = strings.TrimSpace(p); p != "" {
				paths = append(paths, p)
			}
		}
	}
	if len(paths) == 0 {
		// An empty index is indistinguishable here from a read that silently returned nothing,
		// and a view that answers "nothing exists" would turn every link into a BROKEN_LINK.
		// Refuse to be the input rather than manufacture a fleet-wide refusal.
		return nil, false
	}

	t := &landsTree{
		root: d.Root, run: d.run, ctx: ctx,
		indexish: indexish, base: base,
		paths: map[string]bool{}, dirs: map[string]bool{}, staged: map[string]bool{},
		cache: map[string]fileEntry{}, sizes: map[string]int64{},
	}
	for _, p := range paths {
		k := normRel(p)
		if k == "" {
			continue
		}
		t.paths[k] = true
		for dir := path.Dir(k); dir != "." && dir != "/" && dir != ""; dir = path.Dir(dir) {
			t.dirs[dir] = true
		}
	}
	for _, set := range [][]string{d.StagedPaths, d.AddedPaths, d.AddedRenamedPaths} {
		for _, p := range set {
			if k := normRel(p); k != "" {
				t.staged[k] = true
			}
		}
	}
	return t, true
}

// normRel puts a probe argument into the one spelling git speaks: forward slashes, no "./", no
// trailing slash. Gates hand it paths assembled by path.Join and by string concatenation, and a
// map lookup that misses on spelling would read as "absent from the commit" — a finding.
func normRel(rel string) string {
	k := strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
	if k == "" {
		return ""
	}
	k = path.Clean(k)
	k = strings.TrimPrefix(k, "./")
	k = strings.TrimSuffix(k, "/")
	if k == "." || k == "/" {
		return ""
	}
	return k
}

// Exists reports whether the path is IN THE TREE THIS COMMIT LANDS — a file the index carries, or
// a directory one of them lives under. It is the os.path.exists twin the link/index resolvers
// call, with the working tree swapped out for the change set.
func (t *landsTree) Exists(rel string) bool {
	k := normRel(rel)
	if k == "" {
		return true // the repo root, which every relative resolve is anchored at
	}
	return t.paths[k] || t.dirs[k]
}

// Size returns the byte size of the blob this commit lands at rel. A path the commit does not
// carry returns (0,false) — the same answer os.Stat gives for a missing file, which the size cap
// already treats as "no size to judge".
func (t *landsTree) Size(rel string) (int64, bool) {
	k := normRel(rel)
	if k == "" || !t.paths[k] {
		return 0, false
	}
	if n, ok := t.cachedSize(k); ok {
		return n, n >= 0
	}
	n := t.readSize(k)
	t.storeSize(k, n)
	return n, n >= 0
}

// readSize prefers the INDEX blob for a path this commit touches and the base tree's size table
// for one it does not. Getting that precedence backwards is the whole defect in miniature: a
// staged modification would be judged at its pre-commit size.
func (t *landsTree) readSize(k string) int64 {
	if !t.staged[k] {
		if n, ok := t.baseSize(k); ok {
			return n
		}
	}
	for _, spec := range []string{t.indexish + k, t.base + ":" + k} {
		out, code, err := t.run(t.ctx, t.root, "cat-file", "-s", spec)
		if err != nil || code != 0 {
			continue
		}
		if n, perr := strconv.ParseInt(strings.TrimSpace(out), 10, 64); perr == nil && n >= 0 {
			return n
		}
	}
	return -1
}

// baseSize serves the untouched majority out of ONE `ls-tree -r --long <base>` spawn instead of
// one `cat-file -s` per path. Lazy: a run where no gate ever sizes an untouched path — which is
// every ordinary pre-commit run, since FILE_ADMISSION only classifies staged paths — never pays
// for it at all.
func (t *landsTree) baseSize(k string) (int64, bool) {
	t.mu.Lock()
	if !t.baseSizesDone {
		t.baseSizesDone = true
		t.mu.Unlock()
		table := t.readBaseSizes()
		t.mu.Lock()
		t.baseSizes = table
	}
	n, ok := t.baseSizes[k]
	t.mu.Unlock()
	return n, ok
}

// readBaseSizes parses `<mode> blob <sha> <size>\t<path>` rows. A tree/commit row (a submodule, a
// subdirectory) carries "-" where the size goes and is skipped, so it never lands as a size of 0.
func (t *landsTree) readBaseSizes() map[string]int64 {
	table := map[string]int64{}
	out, code, err := t.run(t.ctx, t.root, "ls-tree", "-r", "--long", t.base)
	if err != nil || code != 0 {
		return table
	}
	for _, line := range strings.Split(out, "\n") {
		tab := strings.IndexByte(line, '\t')
		if tab < 0 {
			continue
		}
		fields := strings.Fields(line[:tab])
		if len(fields) < 4 || fields[1] != "blob" {
			continue
		}
		n, perr := strconv.ParseInt(fields[3], 10, 64)
		if perr != nil {
			continue
		}
		if k := normRel(line[tab+1:]); k != "" {
			table[k] = n
		}
	}
	return table
}

// FileBytes reads the blob this commit lands at rel: the staged content when the commit stages it,
// the committed content when it does not, and (nil,false) when the commit carries no such path.
//
// ⛔ There is NO disk fallback, and its absence is the point. StagedDiff.FileBytes falls through
// to os.ReadFile when `git show :<path>` misses, which is exactly how a peer's UNTRACKED file
// becomes readable content to a gate — the gate then judges bytes that no commit in the fleet can
// change and cites them in an author's refusal.
func (t *landsTree) FileBytes(rel string) ([]byte, bool) {
	k := normRel(rel)
	if k == "" {
		return nil, false
	}
	if e, ok := t.cachedFile(k); ok {
		return e.data, e.exists
	}
	e := fileEntry{}
	if t.paths[k] {
		for _, spec := range []string{t.indexish + k, t.base + ":" + k} {
			out, code, err := t.run(t.ctx, t.root, "show", spec)
			if err != nil || code != 0 {
				continue
			}
			e = fileEntry{data: []byte(out), exists: true}
			break
		}
	}
	t.storeFile(k, e)
	return e.data, e.exists
}

// cachedFile / storeFile / cachedSize / storeSize are the ONLY map accessors, and none of them
// holds the mutex across a git spawn — a `git show` can block on the very index contention the
// hook's wall clock exists for, and holding the lock across it would turn one wedged gate into a
// serialized stall for every other gate. Losing the race costs one duplicated read of identical
// bytes. Same argument, verbatim, as StagedDiff.cachedFile.
func (t *landsTree) cachedFile(k string) (fileEntry, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	e, ok := t.cache[k]
	return e, ok
}

func (t *landsTree) storeFile(k string, e fileEntry) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.cache[k] = e
}

func (t *landsTree) cachedSize(k string) (int64, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n, ok := t.sizes[k]
	return n, ok
}

func (t *landsTree) storeSize(k string, n int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.sizes[k] = n
}

// LandsTreeView returns a sibling *StagedDiff whose path probes resolve against HEAD ⊕ staged
// instead of the working tree. ok=false means the view could not be built and the caller must
// keep using the working-tree diff (fail-closed to today's behaviour, never to silence).
//
// ⭐ It is a SIBLING, not a mutation of d, and that is load-bearing rather than stylistic. The
// pre-commit CLI abandons an over-budget gate without cancelling it (#5335), so an abandoned
// Check keeps running against d while the loop hands d to the next gate. Swapping a view field on
// the shared d for the duration of one gate would be a data race with that abandoned goroutine —
// and worse, a SILENT one, since the losing read still returns a plausible answer.
//
// The sibling shares the diff's immutable path lists (they are read-only for a gate's whole life)
// and takes its OWN file cache, because the parent's cache holds working-tree answers: reusing it
// would seed the change-set view with the disk reads it exists to stop making.
func (d *StagedDiff) LandsTreeView() (*StagedDiff, bool) {
	if d == nil {
		return nil, false
	}
	if d.probe != nil {
		return d, true // already a change-set view; building one over it would be a no-op
	}
	if v, done := d.cachedLandsView(); done {
		return v, v != nil
	}
	var v *StagedDiff
	if t, ok := newLandsTree(d); ok {
		v = &StagedDiff{
			Root:              d.Root,
			run:               d.run,
			ctx:               d.ctx,
			AddedByFile:       d.AddedByFile,
			StagedPaths:       d.StagedPaths,
			AddedPaths:        d.AddedPaths,
			AddedRenamedPaths: d.AddedRenamedPaths,
			IndexPaths:        d.IndexPaths,
			Treeish:           d.Treeish,
			fileCache:         map[string]fileEntry{},
			probe:             t,
		}
	}
	return d.storeLandsView(v)
}

func (d *StagedDiff) cachedLandsView() (*StagedDiff, bool) {
	d.viewMu.Lock()
	defer d.viewMu.Unlock()
	return d.landsView, d.landsTried
}

// storeLandsView memoizes the (possibly nil) view. A FAILED build is memoized too: without that,
// a repo whose index cannot be read would re-attempt the construction once per LANDS_TREE gate,
// spending the hook's wall clock on a question already answered.
func (d *StagedDiff) storeLandsView(v *StagedDiff) (*StagedDiff, bool) {
	d.viewMu.Lock()
	defer d.viewMu.Unlock()
	if d.landsTried {
		return d.landsView, d.landsView != nil // a peer built it first; one view per diff
	}
	d.landsTried = true
	d.landsView = v
	return v, v != nil
}

// View names which change-set this diff resolves path probes against, for a caller that wants to
// report or assert it.
func (d *StagedDiff) View() string {
	if d != nil && d.probe != nil {
		return ViewLandsTree
	}
	return ViewWorktree
}

// mergeCandidateNotes copies the candidate denominators a gate recorded on the view back onto the
// diff the runner reports from.
//
// ⛔ Not bookkeeping. #5602's per-gate denominator is recorded by the gate itself, from inside
// Check, on whichever *StagedDiff it was handed — so a gate run over the view records onto the
// view, and cmd/fak's buildGateReport reads the ORIGINAL. Without this the JSON report would
// regress every scoped gate to `"candidates": null` (UNREPORTED), which #5602 exists to
// distinguish from zero. Copying preserves that distinction exactly: a gate that recorded nothing
// contributes nothing here and stays UNREPORTED.
func mergeCandidateNotes(dst, src *StagedDiff) {
	if dst == nil || src == nil || dst == src {
		return
	}
	src.candMu.Lock()
	notes := make(map[string]candidateNote, len(src.candidates))
	for gate, n := range src.candidates {
		notes[gate] = n
	}
	src.candMu.Unlock()
	for gate, n := range notes {
		dst.NoteCandidates(gate, n.n, n.unit)
	}
}
