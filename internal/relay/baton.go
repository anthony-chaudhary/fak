// Package relay is the perpetual-session substrate (epic #1860): a goal is run as
// an ordered sequence of bounded legs, each handing a small typed BATON to the next
// instead of compacting a growing transcript. This file is rung C1 (issue #1870):
// the Baton type — the root of the whole track, the thing every later rung (parse,
// resolve, verify, handoff, status) is defined against.
//
// The wire contract is the data-only spec `fak.relay.baton.v1`, pinned in
// docs/notes/RELAY-BATON-SCHEMA-2026-07-01.md; the doctrine is in
// docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md. This file is deliberately
// scoped to the TYPE and its zero-value predicate only: no JSON parse/validate
// (rung C2 #1871), no I/O, no CLI (rungs C6/C7), no reload verification (track D).
// It exists so those rungs share one shape.
//
// Two invariants of the schema are load-bearing and are structural here, not merely
// documented:
//
//   - Pointer-only. A baton carries HANDLES into durable stores (commit SHAs, ledger
//     ids, issue numbers, memory slugs, file globs) — never transcript bytes and
//     never a model-written recap of the work. Every field below is a pointer or a
//     one-line label; none is a payload.
//   - No `claimed` field, by construction. Progress is represented ONLY by the
//     re-verifiable cursor in ProgressCursor (a git anchor + a ledger ref the
//     successor re-reads), never by a number the closing leg asserted. There is no
//     field named `claimed` anywhere in the type tree, and TestBatonHasNoClaimedField
//     asserts that reflectively so a future edit cannot reintroduce one silently.
package relay

import "github.com/anthony-chaudhary/fak/internal/ctxplan"

// Schema is the exact, case-sensitive schema tag every valid baton carries. A reader
// rejects any object whose `schema` is not equal to this (schema doc, Reader Contract
// step 3). It is a constant so the writer (C6) and the parser (C2) cannot drift.
const Schema = "fak.relay.baton.v1"

// ArtifactKind is the closed set of durable-store pointer kinds an Artifact may name.
// The membership set is fixed by the schema doc's `artifacts.kind` row; validation of
// a decoded kind against this set is a later rung's job (C2), so this file only names
// the vocabulary — it does not enforce it.
type ArtifactKind string

const (
	// ArtifactCommit points at a git commit by SHA.
	ArtifactCommit ArtifactKind = "commit"
	// ArtifactIssue points at a GitHub issue by "#1234".
	ArtifactIssue ArtifactKind = "issue"
	// ArtifactMemory points at an agent-memory file by slug.
	ArtifactMemory ArtifactKind = "memory"
	// ArtifactLedger points at an intent/run/DOS ledger row by id.
	ArtifactLedger ArtifactKind = "ledger"
	// ArtifactFile points at a repo-relative path or glob (a pointer, never the bytes).
	ArtifactFile ArtifactKind = "file"
)

// Baton is the `fak.relay.baton.v1` handoff object a relay leg writes at a safe point
// for its successor (schema doc, "Top-level Fields"). It is the least-trusted signal
// in the relay: its progress half is not a claim but a cursor the next leg
// re-verifies, so the type has no place to record asserted progress.
//
// Field order mirrors the schema doc's stable-JSON ordering. The zero Baton is the
// "no baton" state (see IsZero); a well-formed baton always sets Schema to Schema.
type Baton struct {
	// Schema must equal the package Schema constant ("fak.relay.baton.v1"). Emitted
	// first; a reader rejects any other value before looking at any other field.
	Schema string `json:"schema"`
	// RelayID is the stable id for the whole relay — non-empty and unchanged across
	// every leg of the goal. It identifies which relay this baton belongs to.
	RelayID string `json:"relay_id"`
	// Leg is the closing leg's number: >= 0, monotonic across the relay, and equal to
	// session.Generation for the closing leg. It is lineage/audit ordering, not progress.
	Leg int `json:"leg"`
	// ParentTrace is the trace id of the closing leg. The successor records it as
	// lineage (session.ParentTrace), never as trusted progress.
	ParentTrace string `json:"parent_trace"`
	// Objective is the active objective pin carried VERBATIM. It is exactly the JSON
	// shape of ctxplan.ObjectivePin (schema doc, "objective"), reused as the type so a
	// reader runs the same ObjectivePin.Verify digest check; a mismatch is corrupt
	// baton input to fail closed on, not objective drift.
	Objective ctxplan.ObjectivePin `json:"objective"`
	// DoneWhen is a one-line, durable-store predicate the successor evaluates BEFORE
	// doing more work; a satisfied done_when ends the relay with RELAY_GOAL_DONE rather
	// than launching another leg. Not a recap.
	DoneWhen string `json:"done_when"`
	// ProgressCursor holds the re-verifiable progress anchors (a git SHA + optional
	// ledger ref + the lease region). It carries no percentage and no self-report —
	// the successor re-reads it against ground truth. See ProgressCursor.
	ProgressCursor ProgressCursor `json:"progress_cursor"`
	// NextAction is one line naming the next atomic action. It replaces a carried-over
	// transcript tail; it must not be a recap of prior work.
	NextAction string `json:"next_action"`
	// OpenQuestions are durable pointers or short labels for unresolved decisions, not
	// essays. An empty slice is valid; it serializes as [].
	OpenQuestions []string `json:"open_questions"`
	// Artifacts are durable pointers the successor may inspect (commit/issue/memory/
	// ledger/file). Rows carry refs, never bytes. An empty slice is valid only when no
	// artifact exists yet; it serializes as [].
	Artifacts []Artifact `json:"artifacts"`
	// DoNotRederive are durable pointers to closed dead ends the successor should not
	// retry — the anti-poison index that lets a fresh window shed confusion instead of
	// blurring it forward. An empty slice is valid; it serializes as [].
	DoNotRederive []string `json:"do_not_rederive"`
	// Tombstone is the closing leg's typed exit record (a closed reason token + the
	// observed SHA + a display-only note). The baton carries the tombstone as its
	// header. See Tombstone.
	Tombstone Tombstone `json:"tombstone"`
}

// ProgressCursor is the re-verified half of the baton: a set of anchors the successor
// re-reads against git and the ledger to reconstruct verified progress. It never
// states HOW MUCH work is done — that is the whole point of the no-`claimed`-field
// invariant. If a cursor no longer matches ground truth the read outcome is
// RELAY_BATON_STALE and the successor re-derives from durable state (track D).
type ProgressCursor struct {
	// StartSHA is the git commit the closing leg used as its progress anchor — the
	// ground-truth base a reader verifies exists and is an ancestor of (or is) the
	// configured base for dos_status/dos_verify. Required.
	StartSHA string `json:"start_sha"`
	// LedgerRef optionally names an intent-ledger / run-ledger / DOS row to re-read for
	// verified progress. If present it must resolve; omitted when unknown.
	LedgerRef string `json:"ledger_ref,omitempty"`
	// HeldRegion is the lane/path region (globs) the successor must re-acquire before
	// writing, so leg N+1 takes the SAME lease and does not collide with peers on the
	// shared tree. Empty is invalid for a write-capable relay but is a representable
	// value; it serializes as [].
	HeldRegion []string `json:"held_region"`
}

// Artifact is one pointer into a durable store: a kind drawn from the closed
// ArtifactKind vocabulary and a store-native ref (a commit SHA, "#1234", a memory
// slug, a ledger id, or a repo-relative path/glob). An artifact never carries bytes;
// a file artifact points at a path/glob, it does not embed content.
type Artifact struct {
	// Kind is one of the ArtifactKind constants. Stored as a string so a decoded value
	// outside the vocabulary is representable and can be rejected by the parser (C2)
	// rather than losing information at unmarshal time.
	Kind string `json:"kind"`
	// Ref is the store-native reference for this kind.
	Ref string `json:"ref"`
}

// Tombstone is the closing leg's typed death note (schema doc, "tombstone"): why the
// leg ended, the SHA it observed when it wrote the baton, and a short operator note.
// The note can explain why the reason fired but must never be consumed as progress —
// it cannot substitute for ProgressCursor, Artifacts, or DoneWhen.
type Tombstone struct {
	// Reason is one of the relay reason tokens (docs/notes/RELAY-REASON-VOCABULARY-
	// 2026-07-01.md): RELAY_ROTATED, RELAY_GOAL_DONE, RELAY_PARKED_UNSAFE, etc. Stored
	// as a string; checking membership against the closed vocabulary is a later rung.
	Reason string `json:"reason"`
	// AtSHA is the git commit observed when the closing leg wrote the baton; a reader
	// verifies it exists. Required.
	AtSHA string `json:"at_sha"`
	// Note is a short, display-only operator note — never consumed as progress. Omitted
	// when empty.
	Note string `json:"note,omitempty"`
}

// IsZero reports whether b is the unset baton — the "no baton" state a caller checks
// before treating a value as a real handoff. It keys off the two fields a well-formed
// baton always sets: the Schema tag and the RelayID. This mirrors
// ctxplan.ObjectivePin.IsZero (identity-plus-content emptiness) rather than requiring
// every field to be empty, so a partially-built baton is still recognized as "not the
// zero value" and does not slip through as absent.
func (b Baton) IsZero() bool {
	return b.Schema == "" && b.RelayID == ""
}
