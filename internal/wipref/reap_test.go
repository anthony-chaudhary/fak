package wipref

import (
	"sync"
	"testing"
)

func rec(session, object string, at int64) RefRecord {
	return RefRecord{
		Ref:    SessionRef(session),
		Object: object,
		Stamp:  Stamp{SessionID: session, CheckpointedAt: at},
	}
}

func TestReconcile(t *testing.T) {
	cases := []struct {
		name        string
		current     RefRecord
		candidate   RefRecord
		wantChanged bool
		wantObject  string // object of the returned winner
	}{
		{"empty incumbent yields", RefRecord{}, rec("s", "aaa", 5), true, "aaa"},
		{"newer wins", rec("s", "aaa", 5), rec("s", "bbb", 9), true, "bbb"},
		{"older backs off", rec("s", "bbb", 9), rec("s", "aaa", 5), false, "bbb"},
		{"same object is a no-op", rec("s", "aaa", 5), rec("s", "aaa", 5), false, "aaa"},
		{"tie breaks to greater object", rec("s", "aaa", 7), rec("s", "bbb", 7), true, "bbb"},
		{"tie loses to lesser object", rec("s", "bbb", 7), rec("s", "aaa", 7), false, "bbb"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, changed := Reconcile(c.current, c.candidate)
			if changed != c.wantChanged {
				t.Errorf("Reconcile changed = %v, want %v", changed, c.wantChanged)
			}
			if got.Object != c.wantObject {
				t.Errorf("Reconcile winner object = %q, want %q", got.Object, c.wantObject)
			}
		})
	}
}

// casCell models git's `update-ref <ref> <new> <old>` OLD-VALUE compare-and-swap:
// a swap succeeds only if the cell still holds the expected old object, and the
// mutex makes every read/swap atomic (git holds a per-ref lock for the same effect).
// The concurrency invariant under test lives in Reconcile, not the cell.
type casCell struct {
	mu  sync.Mutex
	rec RefRecord
}

func (c *casCell) load() RefRecord {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.rec
}

func (c *casCell) swap(expectOld string, next RefRecord) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.rec.Object != expectOld {
		return false
	}
	c.rec = next
	return true
}

// writeLoop is the exact shape of the cmd shell's checkpoint CAS loop: read the
// current ref, let Reconcile decide, attempt the compare-and-swap, retry on a lost
// CAS, and stop as soon as the candidate is either landed or superseded.
func writeLoop(cell *casCell, cand RefRecord) {
	for {
		cur := cell.load()
		_, changed := Reconcile(cur, cand)
		if !changed {
			return // superseded by an equal-or-newer checkpoint already anchored
		}
		if cell.swap(cur.Object, cand) {
			return // landed
		}
		// CAS lost: the ref advanced under us — re-read and re-decide.
	}
}

// TestReconcileConcurrentWriters is the #3873 concurrent-writer table test: many
// goroutines checkpoint the SAME session at once, and the ref must converge to one
// valid record — the last writer (max CheckpointedAt, tie-broken by object id) —
// with no torn intermediate ever observed. Run under -race, it is the acceptance
// gate's proof that the checkpoint CAS loop is concurrency-safe.
func TestReconcileConcurrentWriters(t *testing.T) {
	scenarios := []struct {
		name string
		ts   []int64 // one candidate per writer; index i -> object id fmt "obj-%02d"
	}{
		{"ascending", []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		{"descending", []int64{8, 7, 6, 5, 4, 3, 2, 1}},
		{"shuffled", []int64{4, 8, 1, 6, 3, 7, 2, 5}},
		{"ties on timestamp", []int64{5, 5, 5, 5, 5, 5, 5, 5}},
		{"mixed ties and gaps", []int64{2, 9, 9, 4, 9, 1, 4, 2}},
	}
	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			cands := make([]RefRecord, len(sc.ts))
			for i, at := range sc.ts {
				cands[i] = rec("shared", objID(i), at)
			}
			// The expected winner is the strict max under Reconcile's total order.
			want := cands[0]
			for _, c := range cands[1:] {
				if _, changed := Reconcile(want, c); changed {
					want = c
				}
			}

			const rounds = 50 // repeat to widen the interleaving window under -race
			for r := 0; r < rounds; r++ {
				cell := &casCell{}
				var wg sync.WaitGroup
				for _, c := range cands {
					wg.Add(1)
					go func(cand RefRecord) {
						defer wg.Done()
						writeLoop(cell, cand)
					}(c)
				}
				wg.Wait()

				got := cell.load()
				if got.Object != want.Object {
					t.Fatalf("round %d: converged to object %q (ts=%d), want %q (ts=%d)",
						r, got.Object, got.Stamp.CheckpointedAt, want.Object, want.Stamp.CheckpointedAt)
				}
				// No-torn-ref: the final value must be exactly one of the inputs.
				if !isInput(got, cands) {
					t.Fatalf("round %d: converged to a value that was never written: %+v", r, got)
				}
			}
		})
	}
}

func objID(i int) string {
	const hex = "0123456789abcdef"
	// A distinct, well-ordered id per writer so tie-breaks are deterministic.
	return "obj-" + string([]byte{hex[(i/16)%16], hex[i%16]})
}

func isInput(got RefRecord, cands []RefRecord) bool {
	for _, c := range cands {
		if got.Object == c.Object && got.Stamp.CheckpointedAt == c.Stamp.CheckpointedAt {
			return true
		}
	}
	return false
}

func TestReapFold(t *testing.T) {
	recs := []RefRecord{
		rec("landed", "o-landed", 100),
		rec("live", "o-live", 100),
		rec("closed-clean", "o-cc", 100),
		rec("closed-dirty", "o-cd", 100),
		rec("no-owner", "o-none", 100),
	}
	owners := map[string]OwnerState{
		"landed":       OwnerLanded,
		"live":         OwnerLive,
		"closed-clean": OwnerClosedClean,
		"closed-dirty": OwnerClosedDirty,
		// "no-owner" deliberately absent -> OwnerUnknown -> keep.
	}
	want := map[string]ReapAction{
		"landed":       ReapDelete,
		"live":         ReapKeep,
		"closed-clean": ReapDelete,
		"closed-dirty": ReapKeep,
		"no-owner":     ReapKeep,
	}

	got := Reap(recs, owners)
	if len(got) != len(recs) {
		t.Fatalf("Reap returned %d verdicts, want %d", len(got), len(recs))
	}
	// Deterministic session sort.
	for i := 1; i < len(got); i++ {
		if got[i-1].Session > got[i].Session {
			t.Fatalf("Reap verdicts not session-sorted: %q before %q", got[i-1].Session, got[i].Session)
		}
	}
	for _, v := range got {
		if v.Action != want[v.Session] {
			t.Errorf("session %q: action = %s (%s), want %s", v.Session, v.Action, v.Reason, want[v.Session])
		}
		if v.Action == ReapKeep && v.Reason == "" {
			t.Errorf("session %q: kept with no reason", v.Session)
		}
	}
}

func TestReapDecisionUnknownOwnerIsFailSafe(t *testing.T) {
	// An OwnerState the fold does not recognise must collapse to keep/UNKNOWN, never
	// delete — the fail-safe contract that a bad owner label cannot destroy WIP.
	v := ReapDecision(rec("x", "o-x", 1), OwnerState("SOMETHING_NEW"))
	if v.Action != ReapKeep {
		t.Errorf("unrecognised owner state -> action %s, want KEEP", v.Action)
	}
	if v.Owner != OwnerUnknown {
		t.Errorf("unrecognised owner state normalised to %s, want UNKNOWN", v.Owner)
	}
}
