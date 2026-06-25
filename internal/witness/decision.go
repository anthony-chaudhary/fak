// decision.go — the append-only decision recorder: every adjudication / refusal
// the kernel makes, written as a git note on a DEDICATED side ref so the record
// is durable, peer-readable, and NEVER touches the trunk's commit objects.
//
// THE PRINCIPLE (the same evidence-over-self-report rung the resolver enforces):
// when a producer — safecommit's pathspec-assertion, gitgate's argv law — ALLOWS
// or REFUSES an operation, that decision is forensic evidence. A note on
// refs/fak/decisions anchors the decision to the commit it concerns (or, for a
// pre-commit refusal with no commit to anchor to, to git's well-known empty-tree
// object), so a later reviewer can read WHY a call was allowed or refused without
// trusting any worker's narration of it.
//
// THE REF: refs/notes/fak/decisions — git notes are confined to refs/notes/*, so
// the dedicated decisions ref lives there (see decisionsRef for why a bare
// refs/fak/decisions cannot be a notes ref).
//
// APPEND-ONLY, SIDE-REF-ONLY (the safety contract):
//   - writes ONLY to refs/notes/fak/decisions via `git notes --ref=...`
//   - NEVER mutates main / HEAD / refs/heads, NEVER force-pushes, NEVER merges a
//     note INTO a commit. The recorder reuses the package's existing Runner seam
//     (no second way to shell out to git) so tests drive it deterministically.
//
// The note payload is a JSON-encoded Decision (one object per line for an append;
// the reader splits the note body on newlines). encoding/json keeps the payload a
// plain, diffable, schema-stable record.
package witness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// decisionsRef is the dedicated, append-only notes ref the recorder writes to. It
// is the ONLY ref the recorder ever targets — never a branch, never HEAD.
//
// It is intentionally under refs/notes/: `git notes` structurally confines every
// note to the refs/notes/* namespace (a --ref that is not already qualified there
// is silently re-rooted under refs/notes/, so --ref=refs/fak/decisions would land
// at the double-prefixed refs/notes/refs/fak/decisions). Naming the ref
// refs/notes/fak/decisions makes the on-disk ref EXACTLY this string and keeps the
// "fak/decisions" identity the issue asked for, inside the only namespace git
// notes can durably write.
const decisionsRef = "refs/notes/fak/decisions"

// EmptyTreeSHA is git's well-known empty-tree object id. A pre-commit refusal has
// no commit to anchor a decision to (the operation that would have produced one
// was refused), so its note is anchored here — a sentinel that exists in every
// git repository without writing anything to history.
const EmptyTreeSHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

// The closed verdict label space for a recorded Decision. These are plain strings
// (not abi.VerdictKind) so the notes payload stays a self-describing, producer-
// agnostic record: safecommit and gitgate emit their own label, and a reader
// needs no kernel enum to interpret it.
const (
	VerdictAllow      = "allow"       // the operation was permitted
	VerdictRefuse     = "refuse"      // the operation was refused (cite ReasonClass)
	VerdictAssertPass = "assert-pass" // a post-commit pathspec assertion held
	VerdictAssertFail = "assert-fail" // a post-commit pathspec assertion failed
)

// Decision is one recorded adjudication / refusal, serialized as the git-note
// payload. It is a plain JSON struct on purpose: the record outlives the process
// that wrote it and is read by tools that do not link the kernel.
type Decision struct {
	// Op is the operation/verb the decision concerns (e.g. "commit", "push",
	// "exec", or a producer-specific verb like "safecommit").
	Op string `json:"op"`
	// Verdict is one of the Verdict* constants (allow / refuse / assert-pass /
	// assert-fail).
	Verdict string `json:"verdict"`
	// ReasonClass is the closed refusal-vocabulary token for a refusal (e.g.
	// "OFF_TRUNK", "PATHSPEC_RACE", "LEASE_HELD"). Empty for a plain allow.
	ReasonClass string `json:"reason_class,omitempty"`
	// Lane is the lease lane the decision was made under (forensics).
	Lane string `json:"lane,omitempty"`
	// Tree is the pathspec the decision scoped to (the requested file tree, or the
	// committed pathspec for an assertion).
	Tree []string `json:"tree,omitempty"`
	// RefusedArgv is the argv that was refused, for a refuse verdict — the exact
	// command the producer declined to run.
	RefusedArgv []string `json:"refused_argv,omitempty"`
	// PathspecAssertion is the result of a post-commit pathspec assertion (the
	// committed-set==requested-set check), for an assert-pass / assert-fail
	// verdict. Empty for a pre-commit decision.
	PathspecAssertion string `json:"pathspec_assertion,omitempty"`
}

// Recorder appends decisions to refs/fak/decisions through the package's existing
// Runner seam. Construct with NewRecorder (real git) or NewRecorderWithRunner
// (injected evidence — tests, or an alternate git host). dir is the repo to write
// in ("" = git's own discovery from the process cwd).
type Recorder struct {
	run Runner
	dir string
}

// NewRecorder is the real-git decision recorder.
func NewRecorder() *Recorder { return &Recorder{run: gitRunner} }

// NewRecorderWithRunner injects a Runner + dir (tests, or an alternate evidence
// source) — the SAME seam the Resolver uses, so there is exactly one way the
// package shells out to git.
func NewRecorderWithRunner(r Runner, dir string) *Recorder { return &Recorder{run: r, dir: dir} }

// anchorFor returns the SHA a decision's note is keyed to. An empty/blank
// commitSHA is a pre-commit refusal (no commit was produced); it anchors to the
// empty-tree sentinel so the append always has a valid, history-free target.
func anchorFor(commitSHA string) string {
	if strings.TrimSpace(commitSHA) == "" {
		return EmptyTreeSHA
	}
	return strings.TrimSpace(commitSHA)
}

// AppendDecision appends d to the note for commitSHA on refs/fak/decisions. For a
// pre-commit refusal pass commitSHA == "" — it anchors to the empty-tree sentinel
// and does NOT error. The decision is encoded as one JSON line and appended with
// `git notes --ref=refs/notes/fak/decisions append -F <file> <sha>` (no force), so
// multiple decisions for the same anchor accumulate (append-only). It NEVER
// touches main / HEAD / refs/heads and NEVER force-pushes.
func (rec *Recorder) AppendDecision(ctx context.Context, commitSHA string, d Decision) error {
	sha := anchorFor(commitSHA)

	line, err := json.Marshal(d)
	if err != nil {
		return fmt.Errorf("witness: marshal decision: %w", err)
	}

	// The Runner seam carries no stdin, so the note body goes through a temp file
	// passed to `-F`. One JSON object per line; `append` concatenates onto any
	// existing note for this anchor (newline-separated), so the reader can split.
	f, err := os.CreateTemp("", "fak-decision-*.json")
	if err != nil {
		return fmt.Errorf("witness: temp note payload: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(append(line, '\n')); err != nil {
		f.Close()
		return fmt.Errorf("witness: write note payload: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("witness: close note payload: %w", err)
	}

	// `git notes append` creates the note if the anchor has none and concatenates
	// (newline-separated) if it does — exactly the append-only semantics we want,
	// with no `-f`/force and no branch ever touched. It writes ONLY to the side
	// ref named by --ref.
	_, code, runErr := rec.run(ctx, rec.dir,
		"notes", "--ref="+decisionsRef, "append", "-F", tmp, sha)
	if runErr != nil {
		return fmt.Errorf("witness: run git notes append: %w", runErr)
	}
	if code != 0 {
		return fmt.Errorf("witness: git notes append exited %d for %s", code, sha)
	}
	return nil
}

// ReadDecisions reads back the decisions recorded for commitSHA on
// refs/fak/decisions. An empty/blank commitSHA reads the pre-commit-refusal anchor
// (the empty-tree sentinel). It returns an empty slice (not an error) when no note
// exists for the anchor — absence of a decision is a valid, non-erroneous answer.
func (rec *Recorder) ReadDecisions(ctx context.Context, commitSHA string) ([]Decision, error) {
	sha := anchorFor(commitSHA)

	out, code, runErr := rec.run(ctx, rec.dir,
		"notes", "--ref="+decisionsRef, "show", sha)
	if runErr != nil {
		return nil, fmt.Errorf("witness: run git notes show: %w", runErr)
	}
	if code != 0 {
		// A non-zero exit here is the "no note for this object" case (git notes
		// show exits non-zero when the object has no note). That is an absence,
		// not a failure: return no decisions.
		return nil, nil
	}

	var decisions []Decision
	for _, raw := range strings.Split(out, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		var d Decision
		if err := json.Unmarshal([]byte(line), &d); err != nil {
			return nil, fmt.Errorf("witness: parse note line %q: %w", line, err)
		}
		decisions = append(decisions, d)
	}
	return decisions, nil
}
