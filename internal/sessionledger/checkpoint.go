package sessionledger

// checkpoint.go — the two-axis CHECKPOINT witness (#2425, part of the harness-native
// program #2387 / epic #2394): one ledger record that binds "this conversation state"
// to "this workspace state" so a PEER can check the pair without trusting the session
// that minted it.
//
// # Why the binding is the record's own hash
//
// A checkpoint is the pair {ledger_head_hash, tree_witness}. Both halves are already
// hashes, so the cheapest possible binding is the one the chain performs anyway: this
// package's digest() folds parent ‖ kind ‖ content into the entry hash, so appending
// the tree witness AS the content of a "checkpoint" record produces
//
//	checkpoint_id = sha256(ledger_head ‖ "checkpoint" ‖ tree_witness)
//
// The transcript half is the parent pointer; the tree half is the content; the entry
// hash is the signature over both. Nothing is copied — a checkpoint is three hashes and
// a path list, which is what makes it cheap enough to mint automatically (the property
// the ticket requires and a snapshot-copy implementation would destroy; the copy-shaped
// sibling is sessionimage.Checkpoint, a different animal for a different job).
//
// # Why verification names an AXIS
//
// "The checkpoint no longer holds" is useless to an operator: the interesting question
// is WHICH half moved. A tree-axis failure says the workspace drifted from the
// conversation that described it (an edit landed, a commit moved HEAD). A
// transcript-axis failure says the LEDGER itself no longer matches the receipt — the
// record was rewritten, truncated, or re-parented, which is a tampering signal, not
// ordinary drift. They call for opposite responses, so VerifyCheckpoint returns a typed
// *CheckpointMismatch carrying the axis rather than an opaque error string.
//
// # Fail-closed
//
// Every uncertainty resolves to a mismatch, never to a pass: a missing entry, an
// evicted ancestor (the MaxNodes bound legitimately forgets old history), a chain that
// no longer re-digests, a content blob that no longer decodes. A checkpoint that cannot
// be re-derived is not a checkpoint that holds.
//
// This file is pure and stdlib-only. It never runs git: the caller supplies the HEAD
// SHA and the dirty set (cmd/fak reads them with git plumbing), which keeps the ledger
// off any subprocess boundary and keeps the witness hermetically testable.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// CheckpointKind is the entry Kind a checkpoint record is appended under. It is part of
// the digest, so a record cannot be re-labelled after the fact without breaking its own
// hash.
const CheckpointKind = "checkpoint"

// DirtyEntry is one path in the workspace's dirty working set: the porcelain status
// code git reported for it, and the sha256 of its bytes as they are ON DISK now (empty
// for a deletion, a directory, or anything unreadable — see TreeWitness's fail-closed
// note). Origin carries the source path of a rename/copy record, which would otherwise
// vanish from the witness even though the tree changed.
type DirtyEntry struct {
	Path   string `json:"path"`
	Status string `json:"status"`
	SHA256 string `json:"sha256,omitempty"`
	Origin string `json:"origin,omitempty"`
}

// TreeWitness is the git half of a checkpoint: the committed anchor (HEAD SHA) plus one
// digest over everything the workspace has done since. It is deliberately a fixed-size
// value rather than the path list itself — a checkpoint stays three hashes wide however
// dirty the tree is.
//
// DirtySHA256 is always populated (an empty dirty set digests to the sha256 of nothing,
// a constant), so a zero DirtySHA256 means "never built by NewTreeWitness" and is
// refused rather than silently read as clean. DirtyCount is carried for legibility only;
// it is inside the digest, so it cannot disagree with what was hashed.
type TreeWitness struct {
	HeadSHA     string `json:"head_sha"`
	DirtySHA256 string `json:"dirty_sha256"`
	DirtyCount  int    `json:"dirty_count"`
}

// NewTreeWitness folds a HEAD SHA and a dirty set into the witness. The dirty slice is
// sorted by (path, origin) in a COPY — never mutating the caller's slice — so the same
// workspace state digests identically no matter what order git listed it in, which is
// what makes two independent peers able to re-derive the same witness.
func NewTreeWitness(headSHA string, dirty []DirtyEntry) (TreeWitness, error) {
	headSHA = strings.TrimSpace(headSHA)
	if headSHA == "" {
		return TreeWitness{}, errors.New("sessionledger: a tree witness needs a HEAD SHA")
	}
	sorted := make([]DirtyEntry, len(dirty))
	copy(sorted, dirty)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Path != sorted[j].Path {
			return sorted[i].Path < sorted[j].Path
		}
		return sorted[i].Origin < sorted[j].Origin
	})
	h := sha256.New()
	for _, d := range sorted {
		// NUL-separated fields, NUL-terminated records: no field value can forge a
		// record boundary, so two different dirty sets cannot digest alike.
		for _, field := range []string{d.Path, d.Status, d.SHA256, d.Origin} {
			h.Write([]byte(field))
			h.Write([]byte{0})
		}
	}
	return TreeWitness{
		HeadSHA:     headSHA,
		DirtySHA256: hex.EncodeToString(h.Sum(nil)),
		DirtyCount:  len(sorted),
	}, nil
}

// validate refuses a witness that was hand-built rather than produced by NewTreeWitness.
// Minting a checkpoint over a half-filled witness would bind the transcript to nothing.
func (t TreeWitness) validate() error {
	if strings.TrimSpace(t.HeadSHA) == "" {
		return errors.New("sessionledger: tree witness has no HEAD SHA")
	}
	if strings.TrimSpace(t.DirtySHA256) == "" {
		return errors.New("sessionledger: tree witness has no dirty-set digest (build it with NewTreeWitness)")
	}
	return nil
}

// Checkpoint is the receipt a minting returns and a verifier presents back: the two
// bound hashes plus the entry hash that binds them. It is plain data — a caller may
// persist it, ship it to a peer, or reconstruct it from the ledger with
// LatestCheckpoint.
type Checkpoint struct {
	Trace string `json:"trace"`
	// Hash is the checkpoint id: the ledger entry hash, i.e. the signature over both axes.
	Hash Hash `json:"hash"`
	// LedgerHead is the transcript axis: the head this checkpoint was minted over, which
	// is exactly the record's parent pointer.
	LedgerHead Hash `json:"ledger_head"`
	// Tree is the tree axis as it stood at mint time.
	Tree TreeWitness `json:"tree"`
}

// CheckpointAxis names which half of the binding a verification found broken.
type CheckpointAxis string

const (
	// AxisTranscript — the LEDGER no longer matches the receipt (rewritten, truncated,
	// re-parented, or forgotten). A tampering/consistency signal.
	AxisTranscript CheckpointAxis = "transcript"
	// AxisTree — the WORKSPACE drifted from the state the checkpoint described.
	AxisTree CheckpointAxis = "tree"
)

// CheckpointMismatch is a typed verification failure carrying the axis. Callers should
// switch on Axis (via errors.As), never parse Detail.
type CheckpointMismatch struct {
	Axis   CheckpointAxis `json:"axis"`
	Detail string         `json:"detail"`
}

func (m *CheckpointMismatch) Error() string {
	return fmt.Sprintf("checkpoint %s axis: %s", m.Axis, m.Detail)
}

func mismatch(axis CheckpointAxis, format string, args ...any) *CheckpointMismatch {
	return &CheckpointMismatch{Axis: axis, Detail: fmt.Sprintf(format, args...)}
}

// Checkpoint mints a checkpoint on trace: it appends one record whose content is the
// tree witness and whose parent is the trace's current head, so the returned Hash is the
// digest over both axes. The append is the ledger's ordinary one-line write, so a
// checkpoint costs a checkpoint, not a copy of anything.
func (l *Ledger) Checkpoint(trace string, tree TreeWitness) (Checkpoint, error) {
	if strings.TrimSpace(trace) == "" {
		return Checkpoint{}, errors.New("sessionledger: checkpoint needs a trace")
	}
	if err := tree.validate(); err != nil {
		return Checkpoint{}, err
	}
	content, err := json.Marshal(tree)
	if err != nil { // unreachable for this struct
		return Checkpoint{}, fmt.Errorf("sessionledger: marshal tree witness: %w", err)
	}
	e, err := l.Append(trace, CheckpointKind, content)
	if err != nil {
		return Checkpoint{}, err
	}
	// e.Parent is the head the record was digested against — read INSIDE Append under
	// the ledger lock, so a concurrent append on the same trace cannot make the receipt
	// disagree with the record.
	return Checkpoint{Trace: trace, Hash: e.Hash, LedgerHead: e.Parent, Tree: tree}, nil
}

// VerifyCheckpoint re-derives the checkpoint from the ledger as it is NOW and the tree
// witness as it is NOW, and reports which axis (if either) no longer holds.
//
// The transcript axis is checked first and in depth: the record must still be in the
// trace's chain, that chain must still re-digest end to end (so rewriting ANY ancestor
// is caught, not just the checkpoint record itself), the record's parent must still be
// the head it was minted over, and its content must still decode to the witness the
// receipt carries. Only then is the tree axis compared. Both checks are fail-closed.
func (l *Ledger) VerifyCheckpoint(cp Checkpoint, now TreeWitness) error {
	if cp.Hash == "" {
		return mismatch(AxisTranscript, "receipt carries no checkpoint hash")
	}
	if err := now.validate(); err != nil {
		return mismatch(AxisTree, "%v", err)
	}

	chain, err := l.Chain(cp.Trace)
	if err != nil {
		return mismatch(AxisTranscript, "trace %q is no longer readable: %v", cp.Trace, err)
	}
	if err := Verify(chain); err != nil {
		return mismatch(AxisTranscript, "ledger chain for trace %q no longer re-digests: %v", cp.Trace, err)
	}
	var rec *Entry
	for i := range chain {
		if chain[i].Hash == cp.Hash {
			rec = &chain[i]
			break
		}
	}
	if rec == nil {
		return mismatch(AxisTranscript,
			"checkpoint %s is not in trace %q's chain (rewritten, truncated, or evicted)", cp.Hash, cp.Trace)
	}
	if rec.Kind != CheckpointKind {
		return mismatch(AxisTranscript, "entry %s is kind %q, not a checkpoint", cp.Hash, rec.Kind)
	}
	if rec.Parent != cp.LedgerHead {
		return mismatch(AxisTranscript,
			"checkpoint %s was minted over ledger head %s but the record now points at %s",
			cp.Hash, cp.LedgerHead, rec.Parent)
	}
	var recorded TreeWitness
	if err := json.Unmarshal(rec.Content, &recorded); err != nil {
		return mismatch(AxisTranscript, "checkpoint %s content no longer decodes: %v", cp.Hash, err)
	}
	if recorded != cp.Tree {
		return mismatch(AxisTranscript,
			"checkpoint %s records tree witness %s/%s but the receipt claims %s/%s",
			cp.Hash, recorded.HeadSHA, recorded.DirtySHA256, cp.Tree.HeadSHA, cp.Tree.DirtySHA256)
	}

	if now.HeadSHA != cp.Tree.HeadSHA {
		return mismatch(AxisTree, "HEAD moved: checkpoint %s, workspace %s", cp.Tree.HeadSHA, now.HeadSHA)
	}
	if now.DirtySHA256 != cp.Tree.DirtySHA256 {
		return mismatch(AxisTree,
			"working set changed: checkpoint %s (%d paths), workspace %s (%d paths)",
			cp.Tree.DirtySHA256, cp.Tree.DirtyCount, now.DirtySHA256, now.DirtyCount)
	}
	return nil
}

// LatestCheckpoint returns the most recent checkpoint on trace, so a verifier does not
// have to have kept the receipt: the ledger IS the receipt store. Errors when the trace
// is unreadable or carries no checkpoint record.
func (l *Ledger) LatestCheckpoint(trace string) (Checkpoint, error) {
	chain, err := l.Chain(trace)
	if err != nil {
		return Checkpoint{}, err
	}
	for i := len(chain) - 1; i >= 0; i-- {
		e := chain[i]
		if e.Kind != CheckpointKind {
			continue
		}
		var tree TreeWitness
		if err := json.Unmarshal(e.Content, &tree); err != nil {
			return Checkpoint{}, fmt.Errorf("sessionledger: checkpoint %s content does not decode: %w", e.Hash, err)
		}
		return Checkpoint{Trace: trace, Hash: e.Hash, LedgerHead: e.Parent, Tree: tree}, nil
	}
	return Checkpoint{}, fmt.Errorf("sessionledger: trace %q has no checkpoint", trace)
}
