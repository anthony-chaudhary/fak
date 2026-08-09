package sessionledger

// rewind.go — the two INDEPENDENT rewind axes against a checkpoint (#2426, part of the
// harness-native program #2387 / epic #2394): move the conversation back, re-materialize
// the workspace back, or both — and never destroy anything doing it.
//
// # Two axes, because recovery is two questions
//
// "Undo" is not one operation. A session that talked itself into a corner wants the
// TRANSCRIPT back but keeps the code it wrote; a session whose edits went wrong wants the
// WORKSPACE back but keeps the reasoning that diagnosed it. Those are opposite requests,
// so they are opposite verbs over the same checkpoint (#2425's {ledger_head, tree_witness}
// pair):
//
//	RewindConversation — moves the trace head back to the checkpoint. Tree untouched.
//	RestoreTree        — re-materializes the workspace to the checkpoint's witness.
//	                     Transcript head is never moved backwards.
//
// Calling both is the "rewind everything" cell; calling neither leaves the session alone.
// Nothing in either verb reads the other's state, which is what makes the 2x2 matrix in
// TestRewindAxesIndependent hold rather than merely happening to.
//
// # Why a rewind is an APPEND, never a truncation
//
// The obvious implementation — drop the entries after the checkpoint and re-point the head
// — is wrong twice over. It destroys the evidence of what the session actually did (the
// abandoned turns are the most interesting part of a bad run), and it re-parents a hash
// chain, which is indistinguishable from tampering to any peer verifying it later
// (VerifyCheckpoint reports exactly that as an AxisTranscript failure).
//
// So a rewind APPENDS. RewindConversation writes one new entry whose PARENT is the
// checkpoint record, and points the trace head at that new entry:
//
//	... → turn₃ → checkpoint → turn₄ → turn₅        (the abandoned suffix, still on disk)
//	                    ↳ rewind{abandoned: turn₅}  ← the trace head now
//
// Walking back from the new head no longer traverses turn₄/turn₅ — which is the whole
// point, since the next window rebuild follows exactly that walk — but both entries are
// still in the node table, still on the append-only log, and still reachable by hash via
// Node/ChainFrom. The rewind entry records the abandoned head, so the suffix is not merely
// retained but ADDRESSABLE: that hash is the handle a re-fork (#1200) forks from and an
// auditor replays. No ledger entry is ever deleted by either verb.
//
// # Fail-closed
//
// Every uncertainty refuses instead of rewinding to a place we cannot prove: a checkpoint
// that is no longer held (the MaxNodes bound legitimately forgets old history), a record
// that is not a checkpoint or no longer points at the head it was minted over, a checkpoint
// that is not in this trace's current history. A workspace restore that did not actually
// LAND — the materializer came back with a witness that is not the checkpoint's — refuses
// and appends NOTHING, because a TREE_RESTORE record whose "after" digest disagrees with
// the tree on disk is worse than no record at all.
//
// Like checkpoint.go this file is pure and stdlib-only. It never runs git: re-materializing
// the tree is the caller's TreeMaterializer (cmd/fak drives git plumbing), and this package
// only checks the witness it hands back and records the before/after pair.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Entry kinds for the two axes. Both are part of the digest, so a record cannot be
// re-labelled after the fact without breaking its own hash.
const (
	// RewindKind marks a conversation rewind: head moved back to a checkpoint.
	RewindKind = "rewind"
	// TreeRestoreKind marks a workspace restore: tree re-materialized to a checkpoint.
	TreeRestoreKind = "tree_restore"
)

// RewindRecord is the content of a REWIND entry: which checkpoint the head was moved to,
// and — the load-bearing half — the head that was abandoned. That hash is what keeps the
// discarded suffix addressable for audit and for a re-fork (#1200); without it the entries
// would still exist but nobody could name them.
type RewindRecord struct {
	Checkpoint    Hash `json:"checkpoint"`
	AbandonedHead Hash `json:"abandoned_head"`
}

// TreeRestoreRecord is the content of a TREE_RESTORE entry: the workspace digests on
// either side of the restore. Before is what the operator lost, After is what they got —
// and After is only ever written after this package has checked it equals the checkpoint's
// own witness, so the record cannot claim a restore that did not land.
type TreeRestoreRecord struct {
	Checkpoint Hash        `json:"checkpoint"`
	Before     TreeWitness `json:"before"`
	After      TreeWitness `json:"after"`
}

// Rewind is the receipt a conversation rewind returns.
type Rewind struct {
	Trace string `json:"trace"`
	// Hash is the new trace head: the REWIND entry itself.
	Hash Hash `json:"hash"`
	// Checkpoint is the record the head was moved back to — the new head's parent.
	Checkpoint Hash `json:"checkpoint"`
	// AbandonedHead is the pre-rewind head. Still reachable by hash; fork from it to
	// recover the discarded suffix.
	AbandonedHead Hash `json:"abandoned_head"`
}

// TreeRestore is the receipt a workspace restore returns.
type TreeRestore struct {
	Trace string `json:"trace"`
	// Hash is the TREE_RESTORE entry, appended at the trace's tip.
	Hash       Hash        `json:"hash"`
	Checkpoint Hash        `json:"checkpoint"`
	Before     TreeWitness `json:"before"`
	After      TreeWitness `json:"after"`
}

// TreeMaterializer performs the actual workspace re-materialization — the only part of a
// restore that touches a filesystem, and therefore the only part this package delegates.
// Materialize re-creates target and returns the witness of the tree it ACTUALLY produced
// (re-measured, not echoed back): RestoreTree compares that against the checkpoint and
// refuses to record a restore that did not reach it.
type TreeMaterializer interface {
	Materialize(target TreeWitness) (TreeWitness, error)
}

// RewindConversation moves trace's head back to checkpoint cp by APPENDING a REWIND entry
// parented on the checkpoint record. The workspace is not touched — that is the other axis.
//
// The abandoned head is recorded in the new entry and stays reachable by hash: nothing is
// truncated, and the returned receipt names the suffix so a caller can fork it back.
//
// Refuses (with a typed *CheckpointMismatch on the transcript axis) when the checkpoint is
// no longer held, is not a checkpoint record, no longer points at the head it was minted
// over, or is not in this trace's current history.
func (l *Ledger) RewindConversation(trace string, cp Checkpoint) (Rewind, error) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return Rewind{}, errors.New("sessionledger: conversation rewind needs a trace")
	}
	if cp.Hash == "" {
		return Rewind{}, mismatch(AxisTranscript, "receipt carries no checkpoint hash")
	}
	if cp.Trace != "" && cp.Trace != trace {
		return Rewind{}, mismatch(AxisTranscript,
			"checkpoint %s was minted on trace %q, not %q", cp.Hash, cp.Trace, trace)
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	head, ok := l.heads[trace]
	if !ok {
		return Rewind{}, mismatch(AxisTranscript, "trace %q has no head to rewind", trace)
	}
	rec, ok := l.nodes[cp.Hash]
	if !ok {
		return Rewind{}, mismatch(AxisTranscript,
			"checkpoint %s is no longer held (evicted, or never appended here)", cp.Hash)
	}
	if rec.Kind != CheckpointKind {
		return Rewind{}, mismatch(AxisTranscript, "entry %s is kind %q, not a checkpoint", cp.Hash, rec.Kind)
	}
	if rec.Parent != cp.LedgerHead {
		return Rewind{}, mismatch(AxisTranscript,
			"checkpoint %s was minted over ledger head %s but the record now points at %s",
			cp.Hash, cp.LedgerHead, rec.Parent)
	}
	// A rewind target must be somewhere we have actually BEEN on this trace. Without this
	// the verb would happily re-parent a trace onto an unrelated chain, which is a fork
	// wearing a rewind's name (#1200 is the verb for that).
	if !l.isAncestorLocked(cp.Hash, head) {
		return Rewind{}, mismatch(AxisTranscript,
			"checkpoint %s is not in trace %q's current history (head %s)", cp.Hash, trace, head)
	}

	content, err := json.Marshal(RewindRecord{Checkpoint: cp.Hash, AbandonedHead: head})
	if err != nil { // unreachable for this struct
		return Rewind{}, fmt.Errorf("sessionledger: marshal rewind record: %w", err)
	}
	e, err := l.appendAtLocked(trace, cp.Hash, RewindKind, content)
	if err != nil {
		return Rewind{}, err
	}
	return Rewind{Trace: trace, Hash: e.Hash, Checkpoint: cp.Hash, AbandonedHead: head}, nil
}

// RestoreTree re-materializes the workspace to checkpoint cp's tree witness and APPENDS a
// TREE_RESTORE entry recording the before/after digests. The transcript head is never moved
// backwards — the record lands at the trace's tip like any other turn — so a workspace
// restore leaves the conversation exactly where it was. That is the other axis.
//
// before is the workspace witness as it stands now (the caller measures it; this package
// never runs git). The materializer's returned witness must equal the checkpoint's, or the
// restore refuses on the tree axis and appends NOTHING: an unlanded restore must not leave
// a record claiming it landed.
func (l *Ledger) RestoreTree(trace string, cp Checkpoint, before TreeWitness, m TreeMaterializer) (TreeRestore, error) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return TreeRestore{}, errors.New("sessionledger: workspace restore needs a trace")
	}
	if m == nil {
		return TreeRestore{}, errors.New("sessionledger: workspace restore needs a TreeMaterializer")
	}
	if cp.Hash == "" {
		return TreeRestore{}, mismatch(AxisTranscript, "receipt carries no checkpoint hash")
	}
	if err := cp.Tree.validate(); err != nil {
		return TreeRestore{}, mismatch(AxisTree, "checkpoint tree witness: %v", err)
	}
	if err := before.validate(); err != nil {
		return TreeRestore{}, mismatch(AxisTree, "pre-restore tree witness: %v", err)
	}

	after, err := m.Materialize(cp.Tree)
	if err != nil {
		return TreeRestore{}, fmt.Errorf("sessionledger: workspace restore to checkpoint %s: %w", cp.Hash, err)
	}
	if err := after.validate(); err != nil {
		return TreeRestore{}, mismatch(AxisTree, "post-restore tree witness: %v", err)
	}
	if after != cp.Tree {
		return TreeRestore{}, mismatch(AxisTree,
			"workspace restored to %s/%s but checkpoint %s describes %s/%s — nothing recorded",
			after.HeadSHA, after.DirtySHA256, cp.Hash, cp.Tree.HeadSHA, cp.Tree.DirtySHA256)
	}

	content, err := json.Marshal(TreeRestoreRecord{Checkpoint: cp.Hash, Before: before, After: after})
	if err != nil { // unreachable for this struct
		return TreeRestore{}, fmt.Errorf("sessionledger: marshal tree-restore record: %w", err)
	}
	e, err := l.Append(trace, TreeRestoreKind, content)
	if err != nil {
		return TreeRestore{}, err
	}
	return TreeRestore{Trace: trace, Hash: e.Hash, Checkpoint: cp.Hash, Before: before, After: after}, nil
}

// Node returns one entry by hash, whether or not it is still on any trace's chain. This is
// what "the abandoned suffix stays reachable" MEANS operationally: after a rewind the
// discarded entries are off the head's walk but still addressable here.
func (l *Ledger) Node(h Hash) (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	e, ok := l.nodes[h]
	return e, ok
}

// ChainFrom walks any hash back to its root, oldest-first — the trace-free twin of Chain.
// A rewound suffix is replayed (and re-forked, #1200) by walking from the abandoned head
// the REWIND entry recorded, which no longer belongs to any trace. Eviction is handled the
// same way Chain handles it: the surviving suffix is returned rather than an error.
func (l *Ledger) ChainFrom(h Hash) ([]Entry, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.chainFromLocked(h)
}

// chainFromLocked is the shared walk behind Chain and ChainFrom. Caller holds the lock.
func (l *Ledger) chainFromLocked(h Hash) ([]Entry, error) {
	var rev []Entry
	for h != "" {
		e, ok := l.nodes[h]
		if !ok {
			if len(rev) == 0 {
				return nil, fmt.Errorf("sessionledger: missing node %s", h)
			}
			break // evicted ancestor: return the suffix we still hold
		}
		rev = append(rev, e)
		h = e.Parent
	}
	out := make([]Entry, len(rev))
	for i := range rev {
		out[len(rev)-1-i] = rev[i]
	}
	return out, nil
}

// isAncestorLocked reports whether target is on from's parent walk (inclusive). Caller
// holds the lock. An evicted ancestor stops the walk, so the answer is "no" — fail-closed:
// we refuse to rewind to a point we can no longer prove we came through.
func (l *Ledger) isAncestorLocked(target, from Hash) bool {
	for h := from; h != ""; {
		if h == target {
			return true
		}
		e, ok := l.nodes[h]
		if !ok {
			return false
		}
		h = e.Parent
	}
	return false
}

// appendAtLocked is Append with an EXPLICIT parent instead of the trace's current head —
// the one primitive a rewind needs and nothing else should have, which is why it is
// unexported: an arbitrary-parent append is how you forge a chain. Caller holds the write
// lock. Bounds (MaxContentBytes elision, MaxNodes eviction, rotation) apply identically.
func (l *Ledger) appendAtLocked(trace string, parent Hash, kind string, content []byte) (Entry, error) {
	c := bytes.Clone(content)
	if len(c) > MaxContentBytes {
		c = Elide(content)
	}
	e := Entry{Parent: parent, Kind: kind, Content: c}
	e.Hash = digest(parent, kind, c)
	l.putNode(e)
	l.heads[trace] = e.Hash
	return e, l.appendRecord(record{
		Trace: trace, Hash: e.Hash, Parent: e.Parent, Kind: e.Kind, Content: e.Content,
	})
}
