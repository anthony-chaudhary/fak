package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2MeteredLazySession is a fake session reporting both HeldBytes and
// lazy PrefillArenaBytes for TICKET-02 byte-ledger tests.
type inkling2MeteredLazySession struct {
	*inkling2FakeSession
	totalBytes   int64
	prefillBytes int64
}

func (s *inkling2MeteredLazySession) HeldBytes() int64         { return s.totalBytes }
func (s *inkling2MeteredLazySession) PrefillArenaBytes() int64 { return s.prefillBytes }

// fak-test:runtime fast est=1ms
func TestInkling2ByteLedgerTracksOpenGrowTrimAndClose(t *testing.T) {
	const capBytes = 350
	var opened []*inkling2MeteredLazySession
	eng := &inkling2LoopEngine{
		slots:    4,
		capBytes: capBytes,
		open: func() (ggufload.Inkling2LoopSession, error) {
			fs := &inkling2FakeSession{canned: []int{'a', 'b'}}
			s := &inkling2MeteredLazySession{inkling2FakeSession: fs, totalBytes: 100, prefillBytes: 25}
			opened = append(opened, s)
			return s, nil
		},
	}
	st := eng.prefixState()

	// Register cohort arena and prefix store entries alongside slots.
	st.ledger.Set("cohort_arena", inkling2LedgerCohortArena, 80)
	st.ledger.Set("prefix_store", inkling2LedgerPrefixStore, 40)

	// Open first slot (100 bytes: 75 slot + 25 lazy_prefill). Total = 220 <= 350.
	inkling2Gen(t, eng, []int{'A', '1', '2'}, 1)
	snap := st.ledger.Snapshot()
	if snap.Total != 220 || snap.SlotBytes != 75 || snap.LazyPrefillBytes != 25 || snap.CohortArenaBytes != 80 || snap.PrefixStoreBytes != 40 {
		t.Fatalf("after open 1: snapshot=%+v, want total=220 (slot=75 lazy=25 cohort=80 prefix=40)", snap)
	}

	// Open second slot (100 bytes). Total = 320 <= 350.
	inkling2Gen(t, eng, []int{'B', '1', '2'}, 1)
	snap = st.ledger.Snapshot()
	if len(opened) != 2 || snap.Total != 320 || snap.SlotBytes != 150 || snap.LazyPrefillBytes != 50 {
		t.Fatalf("after open 2: opened=%d snapshot=%+v, want 2 slots / total=320", len(opened), snap)
	}

	// Grow slot 1's lazy prefill arena by 60 bytes -> total would be 380 > 350.
	// Next turn on slot 1 triggers prefixTrim, which must evict LRU slot 0.
	opened[1].totalBytes = 160
	opened[1].prefillBytes = 85
	inkling2Gen(t, eng, []int{'B', '1', '2', 'a', '3'}, 1)
	snap = st.ledger.Snapshot()
	if !opened[0].closed || opened[1].closed {
		t.Fatalf("after grow+trim: closed[0]=%v closed[1]=%v, want slot 0 evicted", opened[0].closed, opened[1].closed)
	}
	if snap.Total != 280 || snap.SlotBytes != 75 || snap.LazyPrefillBytes != 85 {
		t.Fatalf("after trim: snapshot=%+v, want total=280 (slot=75 lazy=85 cohort=80 prefix=40)", snap)
	}
	if st.evictions != 1 {
		t.Fatalf("evictions=%d, want 1", st.evictions)
	}

	// Close engine: slot entries removed; remaining total is cohort+prefix only.
	if err := eng.Close(); err != nil {
		t.Fatal(err)
	}
	if got := st.ledger.Total(); got != 120 {
		t.Fatalf("after Close: ledger total=%d, want 120 (cohort 80 + prefix 40)", got)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2EffectiveSlotsStartupWarning(t *testing.T) {
	// 2200 MiB cap with 1600 MiB slot -> fits only 1 slot.
	capBytes := int64(2200) << 20
	slotBytes := int64(1600) << 20
	note := inkling2PrefixEffectiveNote(2, capBytes, slotBytes)
	if !strings.Contains(note, "effective slots=1") || !strings.Contains(note, "cannot hold two slots") {
		t.Fatalf("effective note = %q, want 'effective slots=1' and warning reason", note)
	}

	// First-slot open logs the truthful effective count and trace line.
	var buf bytes.Buffer
	eng := &inkling2LoopEngine{
		slots:    2,
		capBytes: capBytes,
		open: func() (ggufload.Inkling2LoopSession, error) {
			return &inkling2FakeSession{canned: []int{'x'}, held: slotBytes}, nil
		},
	}
	st := eng.prefixState()
	st.trace = true
	st.traceOut = &buf
	defer eng.Close()

	inkling2Gen(t, eng, []int{'P', 'Q'}, 1)
	out := buf.String()
	if !strings.Contains(out, "effective slots=1") || !strings.Contains(out, "cannot hold two slots") {
		t.Fatalf("trace output missing effective slots=1 warning:\n%s", out)
	}
	if !strings.Contains(out, "effective_slots=1") || !strings.Contains(out, "misses=1") || !strings.Contains(out, "held_bytes=") {
		t.Fatalf("trace output missing trace counters:\n%s", out)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2CohortAdmissionEvictsUnpinnedSlotsBeforeReleasingArenas(t *testing.T) {
	const capBytes = 300
	var opened []*inkling2FakeSession
	scalar := &inkling2LoopEngine{
		slots:    3,
		capBytes: capBytes,
		open: func() (ggufload.Inkling2LoopSession, error) {
			s := &inkling2FakeSession{canned: []int{'a'}, held: 100}
			opened = append(opened, s)
			return s, nil
		},
	}
	defer scalar.Close()
	st := scalar.prefixState()

	// Populate two idle scalar slots (200 bytes total).
	inkling2Gen(t, scalar, []int{'A', '1'}, 1)
	inkling2Gen(t, scalar, []int{'B', '1'}, 1)
	// Clear cur to simulate both slots idle (not in-flight).
	scalar.cur = nil

	var order []string
	batch := &inkling2LoopBatchEngine{
		ledger:       st.ledger,
		arenasCached: true,
		evictIdleSlots: func(need int64) int {
			n := scalar.evictIdleSlotsForCohort(need)
			if n > 0 {
				order = append(order, "evict_slots")
			}
			return n
		},
		releaseArenas: func() {
			order = append(order, "release_arenas")
		},
	}
	st.ledger.Set("cohort_arena", inkling2LedgerCohortArena, 100)

	// Cohort needs 250 bytes: total is 300 (200 slots + 100 arena), cap is 300.
	// Step 1: evict both idle slots (frees 200 -> total 100, still 100+250 > 300).
	// Step 2: release cached arenas (frees 100 -> total 0 <= 300).
	batch.admitCohortMemory(250)
	if len(order) != 2 || order[0] != "evict_slots" || order[1] != "release_arenas" {
		t.Fatalf("admission order = %v, want [evict_slots release_arenas]", order)
	}
	if !opened[0].closed || !opened[1].closed {
		t.Fatalf("expected both idle slots evicted, closed=(%v,%v)", opened[0].closed, opened[1].closed)
	}
	if got := st.ledger.Total(); got != 0 {
		t.Fatalf("ledger total after eviction+release = %d, want 0", got)
	}
}
