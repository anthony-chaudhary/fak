package sessionledger

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

// fakeWorkspace stands in for the git tree. RestoreTree is the only seam that touches a
// filesystem in production, and it does so through TreeMaterializer precisely so the axes
// stay hermetically testable: here the "tree" is one witness value we can read back.
type fakeWorkspace struct {
	now    TreeWitness
	calls  int
	fail   error
	giveUp bool // materialize but come back with a DIFFERENT tree (a restore that did not land)
}

func (w *fakeWorkspace) Materialize(target TreeWitness) (TreeWitness, error) {
	w.calls++
	if w.fail != nil {
		return TreeWitness{}, w.fail
	}
	if w.giveUp {
		return w.now, nil // unchanged: the restore silently did nothing
	}
	w.now = target
	return w.now, nil
}

func witness(t *testing.T, head string, paths ...string) TreeWitness {
	t.Helper()
	dirty := make([]DirtyEntry, 0, len(paths))
	for _, p := range paths {
		dirty = append(dirty, DirtyEntry{Path: p, Status: " M", SHA256: "sha-" + p})
	}
	w, err := NewTreeWitness(head, dirty)
	if err != nil {
		t.Fatalf("tree witness: %v", err)
	}
	return w
}

const (
	headAtCheckpoint = "1111111111111111111111111111111111111111"
	headAfterDrift   = "2222222222222222222222222222222222222222"
)

// rewindFixture builds the state every 2x2 cell starts from: a trace with two turns, a
// checkpoint minted over them binding a clean tree, then two MORE turns and a workspace
// that has drifted. That suffix is what a conversation rewind abandons and that drift is
// what a workspace restore undoes.
type rewindFixture struct {
	l         *Ledger
	dir       string
	trace     string
	cp        Checkpoint
	suffix    []Hash // the turns appended after the checkpoint
	preHead   Hash   // the head before any rewind
	ws        *fakeWorkspace
	driftTree TreeWitness
}

func newRewindFixture(t *testing.T) *rewindFixture {
	t.Helper()
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	const trace = "trace-2426"
	for _, n := range []string{`{"turn":1}`, `{"turn":2}`} {
		if _, err := l.Append(trace, "turn", []byte(n)); err != nil {
			t.Fatalf("seed turn: %v", err)
		}
	}

	clean := witness(t, headAtCheckpoint, "internal/sessionledger/rewind.go")
	cp, err := l.Checkpoint(trace, clean)
	if err != nil {
		t.Fatalf("mint checkpoint: %v", err)
	}

	var suffix []Hash
	for _, n := range []string{`{"turn":3}`, `{"turn":4}`} {
		e, err := l.Append(trace, "turn", []byte(n))
		if err != nil {
			t.Fatalf("suffix turn: %v", err)
		}
		suffix = append(suffix, e.Hash)
	}

	drift := witness(t, headAfterDrift, "internal/sessionledger/rewind.go", "cmd/fak/oops.go")
	return &rewindFixture{
		l: l, dir: dir, trace: trace, cp: cp, suffix: suffix,
		preHead:   l.Head(trace),
		ws:        &fakeWorkspace{now: drift},
		driftTree: drift,
	}
}

// chainHas reports whether the trace's CURRENT head walks through hash h. This is the
// question that actually matters for a conversation rewind: the next window is rebuilt
// from the head's walk, so an entry off that walk is out of the conversation even though
// it is still on disk.
func chainHas(t *testing.T, l *Ledger, trace string, h Hash) bool {
	t.Helper()
	chain, err := l.Chain(trace)
	if err != nil {
		t.Fatalf("chain %q: %v", trace, err)
	}
	for _, e := range chain {
		if e.Hash == h {
			return true
		}
	}
	return false
}

func kindsOf(t *testing.T, l *Ledger, trace string) []string {
	t.Helper()
	chain, err := l.Chain(trace)
	if err != nil {
		t.Fatalf("chain %q: %v", trace, err)
	}
	out := make([]string, 0, len(chain))
	for _, e := range chain {
		out = append(out, e.Kind)
	}
	return out
}

func countKind(kinds []string, want string) int {
	n := 0
	for _, k := range kinds {
		if k == want {
			n++
		}
	}
	return n
}

// TestRewindAxesIndependent is #2426's named witness: the 2x2 matrix of
// (conversation rewound|kept) x (tree rewound|kept). Each cell is asserted by LEDGER HEAD
// HASH and TREE DIGEST — the two observable outputs — so a verb that leaked into the other
// axis fails here rather than being caught later by a confused operator.
func TestRewindAxesIndependent(t *testing.T) {
	cases := []struct {
		name    string
		conv    bool
		tree    bool
		wantErr bool
	}{
		{name: "neither axis", conv: false, tree: false},
		{name: "conversation only", conv: true, tree: false},
		{name: "tree only", conv: false, tree: true},
		{name: "both axes", conv: true, tree: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newRewindFixture(t)

			var rw Rewind
			var tr TreeRestore
			// Conversation first when both are asked for: the workspace record then lands
			// on the rewound head, which is the composition an operator means by "rewind
			// everything". Neither verb reads the other's state.
			if tc.conv {
				var err error
				rw, err = f.l.RewindConversation(f.trace, f.cp)
				if err != nil {
					t.Fatalf("rewind conversation: %v", err)
				}
			}
			if tc.tree {
				var err error
				tr, err = f.l.RestoreTree(f.trace, f.cp, f.ws.now, f.ws)
				if err != nil {
					t.Fatalf("restore tree: %v", err)
				}
			}

			head := f.l.Head(f.trace)

			// ---- transcript axis, asserted by ledger head hash ----
			// Each verb that ran appended exactly one entry, in the order they were
			// called, so the head is the LAST one to run. A verb that ran must own the
			// head; a verb that did not must leave it exactly where it was.
			wantHead := f.preHead
			if tc.conv {
				wantHead = rw.Hash
			}
			if tc.tree {
				wantHead = tr.Hash
			}
			if head != wantHead {
				t.Fatalf("head is %s, want %s (conv=%v tree=%v)", head, wantHead, tc.conv, tc.tree)
			}

			if tc.conv {
				if rw.AbandonedHead != f.preHead {
					t.Fatalf("rewind abandoned %s, want the pre-rewind head %s", rw.AbandonedHead, f.preHead)
				}
				if rw.Checkpoint != f.cp.Hash {
					t.Fatalf("rewind targeted %s, want checkpoint %s", rw.Checkpoint, f.cp.Hash)
				}
				// The head's walk must reach the checkpoint and must NOT reach the suffix.
				if !chainHas(t, f.l, f.trace, f.cp.Hash) {
					t.Fatal("rewound head does not walk through the checkpoint it rewound to")
				}
				for i, h := range f.suffix {
					if chainHas(t, f.l, f.trace, h) {
						t.Fatalf("abandoned suffix entry %d (%s) is still on the rewound chain", i, h)
					}
				}
			} else {
				// Conversation KEPT: whatever the tree axis did, the transcript is whole.
				// A restore appends at the tip, so it may move the head FORWARD, never back.
				if !chainHas(t, f.l, f.trace, f.preHead) {
					t.Fatal("the conversation axis was not asked for but the head moved backwards")
				}
				for i, h := range f.suffix {
					if !chainHas(t, f.l, f.trace, h) {
						t.Fatalf("suffix entry %d (%s) fell off a chain nobody rewound", i, h)
					}
				}
			}

			// A conversation rewind must never write a workspace record, and vice versa.
			kinds := kindsOf(t, f.l, f.trace)
			if got := countKind(kinds, RewindKind); got != boolToInt(tc.conv) {
				t.Fatalf("chain has %d REWIND entries, want %d (kinds=%v)", got, boolToInt(tc.conv), kinds)
			}
			if got := countKind(kinds, TreeRestoreKind); got != boolToInt(tc.tree) {
				t.Fatalf("chain has %d TREE_RESTORE entries, want %d (kinds=%v)", got, boolToInt(tc.tree), kinds)
			}

			// ---- tree axis, asserted by tree digest ----
			if tc.tree {
				if f.ws.now != f.cp.Tree {
					t.Fatalf("workspace digest %s, want the checkpoint's %s", f.ws.now.DirtySHA256, f.cp.Tree.DirtySHA256)
				}
				if tr.Before != f.driftTree {
					t.Fatalf("TREE_RESTORE before = %+v, want the drifted tree %+v", tr.Before, f.driftTree)
				}
				if tr.After != f.cp.Tree {
					t.Fatalf("TREE_RESTORE after = %+v, want the checkpoint tree %+v", tr.After, f.cp.Tree)
				}
			} else {
				if f.ws.now != f.driftTree {
					t.Fatalf("tree was not asked for but the workspace digest moved to %s", f.ws.now.DirtySHA256)
				}
				if f.ws.calls != 0 {
					t.Fatalf("tree was not asked for but the materializer ran %d time(s)", f.ws.calls)
				}
			}
		})
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// TestRewindIsAppendOnly is #2426's second named witness: a rewind adds and never
// subtracts. The abandoned suffix stays reachable BY HASH (so #1200 can re-fork it), the
// REWIND event shows up in what `fak session log` renders (cmd/fak/session_log.go walks
// exactly l.Chain), and the on-disk log only ever grows.
func TestRewindIsAppendOnly(t *testing.T) {
	f := newRewindFixture(t)

	beforeNodes := f.l.NodeCount()
	beforeLines := logLines(t, f.dir)
	preChain, err := f.l.Chain(f.trace)
	if err != nil {
		t.Fatalf("pre-rewind chain: %v", err)
	}

	rw, err := f.l.RewindConversation(f.trace, f.cp)
	if err != nil {
		t.Fatalf("rewind conversation: %v", err)
	}

	t.Run("no entry is deleted", func(t *testing.T) {
		if got := f.l.NodeCount(); got != beforeNodes+1 {
			t.Fatalf("node count %d after a rewind, want %d (before %d, +1 REWIND entry)",
				got, beforeNodes+1, beforeNodes)
		}
		if got := logLines(t, f.dir); got != beforeLines+1 {
			t.Fatalf("log has %d lines after a rewind, want %d — a rewind must only append",
				got, beforeLines+1)
		}
	})

	t.Run("abandoned suffix stays reachable by hash", func(t *testing.T) {
		for i, h := range f.suffix {
			if chainHas(t, f.l, f.trace, h) {
				t.Fatalf("suffix entry %d is still on the live chain; the rewind did not take", i)
			}
			if _, ok := f.l.Node(h); !ok {
				t.Fatalf("suffix entry %d (%s) is no longer reachable by hash — it was truncated", i, h)
			}
		}
		// The recorded abandoned head is the handle: walking from it re-derives the whole
		// pre-rewind history, and that history still re-digests end to end.
		if rw.AbandonedHead != f.preHead {
			t.Fatalf("rewind recorded abandoned head %s, want %s", rw.AbandonedHead, f.preHead)
		}
		back, err := f.l.ChainFrom(rw.AbandonedHead)
		if err != nil {
			t.Fatalf("chain from the abandoned head: %v", err)
		}
		if len(back) != len(preChain) {
			t.Fatalf("re-forked suffix has %d entries, want the %d the trace had", len(back), len(preChain))
		}
		for i := range back {
			if back[i].Hash != preChain[i].Hash {
				t.Fatalf("re-forked entry %d is %s, want %s", i, back[i].Hash, preChain[i].Hash)
			}
		}
		if err := Verify(back); err != nil {
			t.Fatalf("the abandoned suffix no longer re-digests: %v", err)
		}
	})

	t.Run("fak session log shows the REWIND event", func(t *testing.T) {
		// `fak session log <trace>` renders l.Chain(trace) entry by entry, so an entry on
		// the chain is an entry an operator sees.
		chain, err := f.l.Chain(f.trace)
		if err != nil {
			t.Fatalf("chain: %v", err)
		}
		last := chain[len(chain)-1]
		if last.Kind != RewindKind {
			t.Fatalf("session log's last entry is kind %q, want %q", last.Kind, RewindKind)
		}
		var rec RewindRecord
		if err := json.Unmarshal(last.Content, &rec); err != nil {
			t.Fatalf("REWIND entry content does not decode: %v", err)
		}
		if rec.Checkpoint != f.cp.Hash || rec.AbandonedHead != f.preHead {
			t.Fatalf("REWIND entry records %+v, want checkpoint %s / abandoned %s",
				rec, f.cp.Hash, f.preHead)
		}
	})

	t.Run("survives a reopen", func(t *testing.T) {
		// The bound that matters most: a rewind is durable and still non-destructive after
		// the process restarts and replays the append-only log.
		reopened, err := Open(f.dir)
		if err != nil {
			t.Fatalf("reopen: %v", err)
		}
		if got := reopened.Head(f.trace); got != rw.Hash {
			t.Fatalf("reopened head is %s, want the REWIND entry %s", got, rw.Hash)
		}
		for i, h := range f.suffix {
			if _, ok := reopened.Node(h); !ok {
				t.Fatalf("suffix entry %d (%s) did not survive the reopen", i, h)
			}
		}
	})

	t.Run("the checkpoint still verifies after the rewind", func(t *testing.T) {
		// A rewind must not look like tampering to a peer: the checkpoint it rewound to is
		// still on the chain and the chain still re-digests, so the transcript axis holds.
		if err := f.l.VerifyCheckpoint(f.cp, f.cp.Tree); err != nil {
			t.Fatalf("checkpoint no longer verifies after a rewind: %v", err)
		}
	})
}

// TestRewindRefusesWhatItCannotProve pins the fail-closed edges: neither axis may leave a
// record it cannot stand behind.
func TestRewindRefusesWhatItCannotProve(t *testing.T) {
	t.Run("unlanded restore records nothing", func(t *testing.T) {
		f := newRewindFixture(t)
		f.ws.giveUp = true
		before := f.l.NodeCount()

		_, err := f.l.RestoreTree(f.trace, f.cp, f.ws.now, f.ws)
		var mm *CheckpointMismatch
		if !errors.As(err, &mm) || mm.Axis != AxisTree {
			t.Fatalf("restore that did not land: got %v, want a tree-axis mismatch", err)
		}
		if got := f.l.NodeCount(); got != before {
			t.Fatalf("an unlanded restore appended a record (%d -> %d)", before, got)
		}
	})

	t.Run("materializer failure is not recorded as a restore", func(t *testing.T) {
		f := newRewindFixture(t)
		f.ws.fail = errors.New("git checkout refused")
		before := f.l.NodeCount()

		if _, err := f.l.RestoreTree(f.trace, f.cp, f.ws.now, f.ws); err == nil {
			t.Fatal("a failing materializer should refuse the restore")
		}
		if got := f.l.NodeCount(); got != before {
			t.Fatalf("a failed restore appended a record (%d -> %d)", before, got)
		}
	})

	t.Run("rewind to a foreign checkpoint refuses", func(t *testing.T) {
		f := newRewindFixture(t)
		other := f.cp
		other.Trace = "some-other-trace"
		_, err := f.l.RewindConversation(f.trace, other)
		var mm *CheckpointMismatch
		if !errors.As(err, &mm) || mm.Axis != AxisTranscript {
			t.Fatalf("foreign checkpoint: got %v, want a transcript-axis mismatch", err)
		}
	})

	t.Run("rewind to a non-checkpoint entry refuses", func(t *testing.T) {
		f := newRewindFixture(t)
		bogus := f.cp
		bogus.Hash = f.suffix[0] // a plain turn, not a checkpoint
		_, err := f.l.RewindConversation(f.trace, bogus)
		var mm *CheckpointMismatch
		if !errors.As(err, &mm) || mm.Axis != AxisTranscript {
			t.Fatalf("non-checkpoint target: got %v, want a transcript-axis mismatch", err)
		}
	})

	t.Run("rewind to a checkpoint from another chain refuses", func(t *testing.T) {
		f := newRewindFixture(t)
		// Mint a checkpoint on a DIFFERENT trace, then relabel its receipt as ours. The
		// record exists and is a real checkpoint, but it is not in our history — rewinding
		// there would be a fork wearing a rewind's name.
		if _, err := f.l.Append("elsewhere", "turn", []byte(`{"turn":9}`)); err != nil {
			t.Fatalf("seed elsewhere: %v", err)
		}
		foreign, err := f.l.Checkpoint("elsewhere", witness(t, headAtCheckpoint, "README.md"))
		if err != nil {
			t.Fatalf("mint foreign checkpoint: %v", err)
		}
		foreign.Trace = f.trace
		_, err = f.l.RewindConversation(f.trace, foreign)
		var mm *CheckpointMismatch
		if !errors.As(err, &mm) || mm.Axis != AxisTranscript {
			t.Fatalf("off-chain checkpoint: got %v, want a transcript-axis mismatch", err)
		}
		if !strings.Contains(mm.Detail, "current history") {
			t.Fatalf("refusal should name the ancestry problem, got %q", mm.Detail)
		}
	})
}

func logLines(t *testing.T, dir string) int {
	t.Helper()
	b, err := os.ReadFile(dir + string(os.PathSeparator) + LogName)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	n := 0
	for _, line := range strings.Split(string(b), "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}
