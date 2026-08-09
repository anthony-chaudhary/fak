// Package memoryindex reconciles an agent-memory INDEX against the memory files
// it claims to describe.
//
// # The defect this exists to catch
//
// An agent-memory index (the `MEMORY.md` pointer file, plus any tier it spills
// into) is a DERIVED ARTEFACT WITH NO GATE, maintained by the same agent whose
// memory it indexes — which is the worst possible custodian, because the agent
// reads the index to decide what it already knows. So the drift is
// self-concealing:
//
//   - a memory file with no pointer row is a memory the next cold session will
//     never recall, and nothing reports it;
//   - a pointer row whose file was deleted is a dead recall target;
//   - a file whose frontmatter `name:` disagrees with its filename breaks every
//     [[wikilink]] aimed at it, in a way that reads as "no such note yet";
//   - two files claiming one slug make every [[wikilink]] to that slug ambiguous.
//
// Nothing in fak measured any of that until this package. internal/memvaluescore
// audits the committed mirror for ROT and folds it into a score — a pressure
// gauge, not a gate: it has no writer, no exit status, and it quantifies over a
// single MEMORY.md. This package is the completeness reconciler, and it is the
// precondition #2618 / #2619 (the [[wikilink]] graph and its materialized
// backlink index) want underneath them: a backlink index built over an
// incomplete file census is confidently wrong.
//
// # Two properties, deliberately
//
// (1) It is GO, UNDER THE GREEN GATE. The custom the port refuses is putting
// this logic in a shell hook: logic in shell is logic `go build ./... && go vet
// ./... && go test ./...` never sees, so the repo's own green gate cannot catch a
// regression in it. Wired as `fak memory index`.
//
// (2) It is a CHECKER FIRST, WRITER SECOND. Check mode is the default: it reports
// the drift and exits non-zero, touching nothing, so the drift is visible in CI
// even where nothing is allowed to rewrite the file. [Apply] is opt-in.
//
// # What the writer may touch
//
// [Apply] reconciles the INDEX toward the FILES and NEVER the reverse. It adds a
// pointer row for an unindexed memory and drops a row whose file is gone; it
// will not rename a memory, rewrite a frontmatter `name:`, invent a description
// or pick a `metadata.type`. Those findings survive `-write` and keep the exit
// status non-zero — a reconciler that could silence every finding by writing is
// a laundering machine, not a gate. [Report.Fixable] is the honest count.
//
// # Layering
//
// [Reconcile] is PURE: it takes a [Store] of already-parsed files and index rows
// and returns findings. No filesystem, no git, no clock, no map-iteration order
// in the output. [Load] is the only reader and [Apply] the only writer, so the
// decision logic unit-tests without a fixture on disk.
//
// # An unresolved [[wikilink]] is not drift
//
// Unresolvable [[wikilink]] targets are resolved against the FILE census and
// reported as their own finding kind, separately from missing index rows,
// because a dangling link and a missing pointer row are different defects and
// #2619 needs them distinguished. They do NOT gate by default: the memory-writing
// contract blesses `[[name]]` for a note not yet written, so a forward reference
// is sanctioned, and internal/memvaluescore already classes broken_wikilink as
// SOFT rot rather than debt. Pass Options.StrictLinks to gate on them anyway.
package memoryindex
