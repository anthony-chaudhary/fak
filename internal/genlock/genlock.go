// Package genlock is a render-determinism lock: it keys a generator's freshness
// check on a hash of the generator's INPUT, never on a comparison of its output.
//
// # Why the output cannot be the question
//
// A generator whose output is not byte-reproducible cannot answer "did the output
// change?" about itself. fak has one in the tree today: `fak marketing aeo` stamps
// `time.Now().UTC().Format(time.RFC3339)` into all four artifacts it writes
// (internal/marketing/aeo.go — the `Updated:` line in llms-updates.txt and
// llms-terms.txt, `dateModified` in docs/marketing/updates.json and
// docs/marketing/disambiguation-terms.json). Two runs a second apart over an
// unchanged repository produce four different files. The disambiguation-terms
// roster is a hard-coded Go slice, so the input is not merely unchanged, it is
// *literally the same bytes* — and the output still differs. TestNaiveOutputCheckIsWrong
// pins that.
//
// This is the same shape as the scar this package is ported from: Chrome does not
// print a byte-identical PDF twice (it stamps a wall-clock /CreationDate and its
// object numbering drifts), so a docpack that compared PDFs rewrote every PDF on
// every run. Comparing non-reproducible outputs is not merely inefficient; the
// comparison is undefined, and "rebuild" is its only stable answer.
//
// The fix is to ask the question of the input instead. The caller hands genlock the
// clock-free identity of what it is about to render — for `fak marketing aeo` that is
// the same renderers called with the zero time, which their own `when.IsZero()` guards
// already make clock-free — and genlock decides. Unchanged input hash and an artifact
// on disk that still hashes to what was recorded means the file already there is the
// file this run would produce, and the correct number of bytes to write is zero.
//
// Skip means NO WRITE AT ALL, not a write of identical bytes. On a checkout ~20-30
// concurrent sessions share, a rewrite of identical bytes is not free: it moves the
// mtime, so every peer's `git status` re-hashes the file, and it puts the path in
// front of anything that sweeps the worktree looking for work that was meant to be
// committed. `fak commit` holds a fleet-wide lock for 60-126s; churn that says nothing
// is the noise that makes real peer WIP hard to classify.
//
// # Where the lock file lives, and why that is a lane decision
//
// The lock lives at tools/genlock/<tool>.lock.json — beside the tools, not beside the
// artifacts it describes. dos.toml's [lanes.trees] decides which path lease a file
// falls under, and the choice is load-bearing:
//
//   - tools/genlock/... matches `tools = ["tools/**", "scripts/**"]`, a NAMED lane in
//     [lanes].concurrent. Two workers on disjoint trees still run in parallel. This is
//     the chosen home.
//   - The repository root (a loose `genlock.lock.json`) matches NOTHING. fak's docs lane
//     enumerates its root files BY NAME — `["docs/**", "README.md", "INDEX.md",
//     "llms.txt", "llms-full.txt", "llms-updates.txt"]` — so a new loose root file is
//     covered by no named tree and falls through to `global = ["**/*"]`, which dos.toml
//     declares EXCLUSIVE. An exclusive lane runs alone: one worker touching that file
//     would serialize the entire fleet. TestLockPathTakesANamedLaneNotTheGlobalCatchAll
//     asserts both halves against the real dos.toml, so the hazard is stated by a test
//     instead of learned by hitting it.
//   - docs/ would match (`docs/**` is a glob, unlike the root entries), so it is not the
//     mechanical trap it is in the repo this came from — but it is still wrong. Build
//     state is not a document: nobody reads it, INDEX_SYNC and the doc gates have no
//     useful opinion to form about it, and it would land in the same lane as the
//     artifacts it describes, so a lock-only update would take the very `docs` lease
//     that a no-op run exists to avoid taking.
//   - internal/genlock/ would match once the leaf is declared, but a Go package
//     directory is source. The lock describes tree-wide artifacts, not this package.
//
// # Why it is committed
//
// The lock's whole job is to answer a question about the COMMITTED artifacts — "were
// these built from the tree as it stands?" (Lock.Stale). A gitignored lock answers that
// for exactly one person: whoever last ran the tool on that machine. Every other
// checkout, and CI, would have to rebuild to find out — which for a non-reproducible
// generator means rebuilding and then being unable to compare the result. So the lock
// is tracked, and tools/genlock/ is deliberately not under any .gitignore rule (unlike
// tools/_registry/ and tools/_watchdog/, which are machine-local state and are ignored).
package genlock

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Dir is the repo-relative directory holding every generator lock.
//
// See the package doc: this is a `tools/**` path on purpose. Moving it to the repo
// root drops it into the exclusive `global` catch-all lane.
const Dir = "tools/genlock"

// PathFor returns the repo-relative lock path for a named generator. One lock per
// generator rather than one shared file, so two generators never serialize on each
// other's state and a lock diff names the tool that produced it.
func PathFor(tool string) string { return Dir + "/" + tool + ".lock.json" }

const note = "Generated by fak. Per artifact, the SHA-256 of the CANONICAL INPUT it was " +
	"built from, plus the SHA-256 of the artifact itself. The generators keyed on this file " +
	"do not produce byte-identical output twice, so this is how fak knows an artifact is " +
	"already current and skips writing it entirely. Committed on purpose: it answers " +
	"\"were these artifacts built from the tree as it stands?\" for every checkout, not just " +
	"the last person who ran the tool. Delete an entry to force a rebuild."

// Lock records, per artifact, the identity of the input that produced it and the
// identity of the artifact that was written.
//
// Both halves are needed, and each covers a different way of being wrong. The input
// hash catches "the source changed". The output hash catches "the file on disk is not
// the artifact that was recorded" — a truncated write, a bad merge, a hand edit. A
// lock that checked only the input would let a corrupt artifact sit in the tree
// forever, because after the first run nothing else ever looks at it.
type Lock struct {
	Note   string            `json:"note"`
	Tool   string            `json:"tool"`
	Input  map[string]string `json:"input_sha256"`  // repo-relative artifact path -> hex sha256 of its canonical input
	Output map[string]string `json:"output_sha256"` // repo-relative artifact path -> hex sha256 of the artifact itself

	root string // absolute repo root; not serialized
	rel  string // repo-relative lock path; not serialized
}

// Open loads the lock for a named generator from root, creating an empty one if it is
// missing or unreadable.
//
// A missing or corrupt lock is deliberately not an error: it means everything rebuilds,
// which is the safe direction. Guessing "stale" costs a slow run; guessing "current"
// ships an artifact that does not match its source.
func Open(root, tool string) (*Lock, error) {
	if tool == "" || strings.ContainsAny(tool, `/\`) || tool == "." || tool == ".." {
		return nil, fmt.Errorf("genlock: %q is not a usable tool name (one path segment, no separators)", tool)
	}
	return OpenAt(root, PathFor(tool)), nil
}

// OpenAt loads a lock from an explicit repo-relative path, for a caller that has its
// own placement argument to make. Check it against dos.toml's [lanes.trees] first:
// a path no named lane covers falls to the exclusive `global` catch-all.
func OpenAt(root, rel string) *Lock {
	l := &Lock{
		Note:   note,
		Tool:   strings.TrimSuffix(filepath.Base(rel), ".lock.json"),
		Input:  map[string]string{},
		Output: map[string]string{},
		root:   root,
		rel:    rel,
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		return l
	}
	var got Lock
	if err := json.Unmarshal(b, &got); err != nil {
		return l
	}
	if got.Input != nil {
		l.Input = got.Input
	}
	if got.Output != nil {
		l.Output = got.Output
	}
	return l
}

// Rel is the repo-relative path this lock is stored at.
func (l *Lock) Rel() string { return l.rel }

// Save writes the lock, and only when its bytes actually changed — so a run that
// rendered nothing leaves the tree completely untouched, including this file. That is
// the point of having it: a skip that still rewrote the lock would have moved the churn
// rather than removed it.
func (l *Lock) Save() error {
	l.Note = note
	next, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	next = append(next, '\n')
	p := filepath.Join(l.root, filepath.FromSlash(l.rel))
	if prev, err := os.ReadFile(p); err == nil && bytes.Equal(prev, next) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	return os.WriteFile(p, next, 0o644)
}

// Sum is the hash both sides of the lock are keyed on: sha256 over the bytes, with
// CRLF folded to LF first for anything that is not binary.
//
// The normalization is what makes a committed lock mean the same thing in every
// checkout. This repo is edited from Windows and Linux at once and git rewrites line
// endings on checkout, so a raw sha256 of a text artifact would differ per platform and
// every checkout would disagree about freshness for reasons that have nothing to do
// with the source. Bytes containing a NUL are treated as binary and hashed verbatim,
// because folding CR LF inside a binary would corrupt the identity it is meant to
// establish. It is the same tolerance conceptcatalog.generatedBytesEqual already takes
// when it compares generated artifacts.
func Sum(b []byte) string {
	if !bytes.Contains(b, []byte{0}) {
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// Canonical joins several input fragments into one canonical input for hashing, with a
// separator no fragment can forge by concatenation. Use it when an artifact's identity
// is more than one thing — a commit sha plus the clock-free rendering, say — so that
// moving a byte from the end of one fragment to the start of the next changes the hash.
func Canonical(parts ...[]byte) []byte {
	var b bytes.Buffer
	for _, p := range parts {
		fmt.Fprintf(&b, "%d\x00", len(p))
		b.Write(p)
	}
	return b.Bytes()
}

// Current reports whether the artifact on disk is the one this input produces.
//
// Three things have to agree: the recorded input hash (the source did not change), the
// file's presence (nobody deleted it), and the file's own hash (it is there AND it is
// the artifact that was recorded). Any disagreement answers "rebuild".
func (l *Lock) Current(artifact string, input []byte) bool {
	if l.Input[artifact] == "" || l.Input[artifact] != Sum(input) {
		return false
	}
	got, err := os.ReadFile(filepath.Join(l.root, filepath.FromSlash(artifact)))
	if err != nil {
		return false
	}
	return Sum(got) == l.Output[artifact]
}

// Outcome is what Sync decided.
type Outcome int

const (
	// Skipped means the recorded input still matches and the artifact on disk is still
	// the recorded one. Nothing was rendered and nothing was written — no bytes, no
	// mtime move.
	Skipped Outcome = iota
	// Wrote means the artifact was rendered and written.
	Wrote
)

func (o Outcome) String() string {
	if o == Skipped {
		return "skipped"
	}
	return "wrote"
}

// Sync brings one artifact up to date with its input, and is the whole point of the
// package.
//
// When the input hash is unchanged and the artifact on disk is still the recorded one,
// render is NOT CALLED and the file is NOT TOUCHED. That is stronger than "write the
// same bytes back": for a generator that stamps a clock, the bytes would not have been
// the same, and even for a deterministic one a rewrite moves the mtime and re-dirties
// the file for every peer sharing the checkout.
//
// Otherwise render runs, its bytes are written, and both identities are recorded from
// what is actually on disk afterwards — so a short write records nothing rather than
// recording a hash of a lie, and the next run rebuilds.
func (l *Lock) Sync(artifact string, input []byte, render func() ([]byte, error)) (Outcome, error) {
	if l.Current(artifact, input) {
		return Skipped, nil
	}
	b, err := render()
	if err != nil {
		return Skipped, err
	}
	p := filepath.Join(l.root, filepath.FromSlash(artifact))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return Skipped, err
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		return Skipped, err
	}
	if err := l.Record(artifact, input); err != nil {
		return Wrote, err
	}
	return Wrote, nil
}

// Record stores both identities of an artifact that is already on disk, for a caller
// that does its own writing (a subprocess generator, say) and only wants the bookkeeping.
//
// A failure to read back what was just written drops the entry rather than recording a
// hash of nothing, so the next run rebuilds instead of trusting a blank.
func (l *Lock) Record(artifact string, input []byte) error {
	got, err := os.ReadFile(filepath.Join(l.root, filepath.FromSlash(artifact)))
	if err != nil {
		delete(l.Input, artifact)
		delete(l.Output, artifact)
		return err
	}
	l.Input[artifact] = Sum(input)
	l.Output[artifact] = Sum(got)
	return nil
}

// Stale answers the gate's question — "were these artifacts built from the tree as it
// stands?" — for the artifacts in inputs, returning the sorted paths that were not.
//
// This is the reason the lock is committed. It runs the same way in any checkout and in
// CI, because every fact it consults is in the tree: the recorded hashes here and the
// artifacts themselves. It never renders anything, so it costs one hash per artifact
// even for a generator whose render is a multi-second subprocess.
func (l *Lock) Stale(inputs map[string][]byte) []string {
	var out []string
	for artifact, input := range inputs {
		if !l.Current(artifact, input) {
			out = append(out, artifact)
		}
	}
	sort.Strings(out)
	return out
}

// Prune drops entries for artifacts that no longer exist, so deleting a generated file
// does not leave its hashes behind forever. Returns the pruned paths, sorted.
func (l *Lock) Prune() []string {
	var gone []string
	for artifact := range l.Input {
		if _, err := os.Stat(filepath.Join(l.root, filepath.FromSlash(artifact))); err != nil {
			gone = append(gone, artifact)
		}
	}
	for _, artifact := range gone {
		delete(l.Input, artifact)
		delete(l.Output, artifact)
	}
	sort.Strings(gone)
	return gone
}
