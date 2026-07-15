# Query, not chat: when managed context appends and when it reseeds

A managed fak session is one continuing query against a shared task model, not an
append-only conversation. The query may run for many turns and workers, but every turn
must still sharpen the same task or reconstruct it from durable evidence. This page is
doctrine: it states the rule that gateway and resume work should implement; it does not
claim that every reseed decision is already enforced automatically.

## 1. Doctrine: one query, not an ever-growing chat

Chat treats the transcript as the product: each new message inherits every earlier
message simply because it came later. Managed context treats the task as the product.
History is useful only while it improves the current task model.

The operational question for every turn is therefore not "can this message be appended?"
but "does this information sharpen the query we are still answering?" Append when it
does. Reseed when a clean reconstruction from durable state is more faithful or cheaper
than carrying the conversational residue forward.

## 2. Pin the originating task

Each managed session has one **originating-task pin**: a stable, content-addressable
statement of the task whose outcome the session is pursuing. The pin is the identity of
the query, not merely the oldest user message and not a summary of the latest failure.
Workers may refine an execution plan, but they must be able to trace that plan back to
the same pin.

A turn that changes or replaces the originating task starts a new query and therefore
requires a reseed. Compaction and restart must preserve or restore the pin before work
continues. If the pin cannot be recovered, the session is not entitled to infer it from
stale conversational fragments.

## 3. Keep stable state; swap transient state

Stable state survives append, compaction, reseed, and restart:

- the originating-task pin;
- accepted constraints and decisions that still govern the task;
- durable witnesses, checkpoints, and externally readable effects; and
- the next action when it is justified by those witnesses.

Swappable state is useful only while it changes the next decision:

- the current error, tool result, or provider residue;
- transient hypotheses and failed approaches;
- superseded plans; and
- narration whose effect already exists as a durable witness.

Swappable does not mean unimportant. It means replaceable without changing the query's
identity. A reseed reconstructs the current query from stable state and includes only the
swappable state still needed for the next action.

## 4. Append only task-relevant witness

**Append only when the new turn adds task-relevant witness that sharpens the same
originating-task pin.** A valid append must answer all of these questions positively:

- Is the originating task unchanged?
- Does the new information alter or justify the next action?
- Is it still current rather than superseded by a later witness?
- Is retaining it cheaper and clearer than reconstructing the current state?
- Can another worker explain why this turn belongs to this task without relying on chat
  chronology alone?

Examples include a fresh test result, an adjudicated decision, a newly observed external
effect, or a constraint that narrows the same task. Repetition, status narration, and an
error already captured in a checkpoint are not reasons to append.

## 5. Reseed from durable positive state

**Reseed when conversational history no longer sharpens the pin.** Any one of these
conditions is sufficient:

- the turn changes or replaces the originating task;
- stale errors, results, or abandoned approaches dominate the useful state;
- context pressure makes retained history costlier than reconstructing from durable
  state;
- compaction, restart, or amnesia threatens the originating-task pin; or
- the next action can be reconstructed from the task plus durable witnesses without the
  conversational residue.

Reseed means constructing a new, positive current-state query: originating task,
applicable constraints, confirmed effects, current decision, and next checkable action.
It does **not** mean merely deleting old turns, summarizing their chronology, or adding a
new instruction that says to ignore them.

## 6. Checkable reseed-versus-append procedure

At each managed-context boundary, evaluate this list in order:

1. Read and identify the originating-task pin. If it is missing or changed, **reseed**.
2. Read durable witnesses and checkpoints independently of the transcript.
3. Remove superseded errors, results, hypotheses, plans, and narration from the candidate
   state.
4. Ask whether the proposed turn adds current witness that changes or justifies the next
   action for the same pin.
5. If yes, and retaining it costs less than reconstruction, **append** it.
6. Otherwise, reconstruct task + constraints + confirmed effects + next action and
   **reseed**.
7. Verify that a worker receiving only the reconstructed query can name the same pin and
   take the same justified next action. If not, the reseed is incomplete.

This procedure yields an observable decision: the boundary can name the pin, the durable
witnesses retained, the residue dropped, and the reason for APPEND or RESEED.

## 7. Shared workspace, positive state, and existing seams

The managed query is a global-workspace surface: workers read and update one shared model
of the task rather than maintaining independent chat stories. Durable effects and
checkpoints are the shared facts; a worker's local transcript is only a temporary view.
This lets a replacement worker resume from the same pin without inheriting another
worker's stale reasoning.

The positive-state rule matters because model-visible text is broadcast state. Repeated
negations such as "ignore the old error" keep the old error resident and require the
model to invert it again on every turn. Reseeding asserts what is true now and removes the
obsolete negative operand. Appending remains valuable when it contributes new evidence,
but append-only drift accumulates broadcast contamination and negation tax.

Existing fak seams provide parts of this contract:

- [`internal/gateway/ctxrestore.go`](../internal/gateway/ctxrestore.go) restores a compacted
  originating task by content-addressed identity;
- [`internal/agent/anthropic_compact.go`](../internal/agent/anthropic_compact.go) records
  and restores the origin-task tombstone across provider compaction;
- [`internal/sessionsteer/sessionsteer.go`](../internal/sessionsteer/sessionsteer.go)
  exposes checkpoint and rebuild directives;
- [`internal/gateway/ctxvalue.go`](../internal/gateway/ctxvalue.go) carries managed-context
  step advice; and
- [`internal/negframe`](../internal/negframe/) supplies the positive-state reframe
  boundary.

These are implementation footholds, not a claim that automatic reseed-versus-append
enforcement is complete. Future enforcement should make the procedure above mechanical
while preserving this doctrine's one invariant: every surviving turn must sharpen the
same originating-task pin.
