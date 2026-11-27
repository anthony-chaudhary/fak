package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// A checkpoint keeps only what the next horizon tokens would overwrite and relies
// on its own session's earlier cache rows, so one session holds one conversation's
// prefix: each prefix-cache slot is a whole resident session.
const (
	inkling2PrefixSlotsMax        = 8
	inkling2PrefixCacheMiBDefault = 4096
)

// inkling2PrefixCacheFromEnv reads FAK_INKLING2_PREFIX_SLOTS (resident sessions,
// default 1) and FAK_INKLING2_PREFIX_CACHE_MIB (what more than one slot may hold
// together, default 4096); an unparsable or out-of-range value keeps its default.
func inkling2PrefixCacheFromEnv(lookup func(string) (string, bool)) (slots int, capBytes int64) {
	slots, capBytes = 1, int64(inkling2PrefixCacheMiBDefault)<<20
	if v, ok := lookup("FAK_INKLING2_PREFIX_SLOTS"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 1 {
			slots = min(n, inkling2PrefixSlotsMax)
		}
	}
	if v, ok := lookup("FAK_INKLING2_PREFIX_CACHE_MIB"); ok {
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil && n > 0 && n <= 1<<30 {
			capBytes = n << 20
		}
	}
	return slots, capBytes
}

// inkling2HeldBytes mirrors the resident session's meter of the device memory it holds.
type inkling2HeldBytes interface{ HeldBytes() int64 }

// inkling2PrefillArenaBytes is optionally implemented by resident sessions that
// report the subset of HeldBytes currently allocated for lazy prefill arenas.
type inkling2PrefillArenaBytes interface{ PrefillArenaBytes() int64 }

// inkling2LedgerKind classifies every entry tracked by inkling2ByteLedger.
type inkling2LedgerKind string

const (
	inkling2LedgerSlot        inkling2LedgerKind = "slot"
	inkling2LedgerLazyPrefill inkling2LedgerKind = "lazy_prefill"
	inkling2LedgerCohortArena inkling2LedgerKind = "cohort_arena"
	inkling2LedgerPrefixStore inkling2LedgerKind = "prefix_store"
)

type inkling2LedgerEntry struct {
	kind  inkling2LedgerKind
	bytes int64
}

// inkling2LedgerSnapshot is a point-in-time breakdown of inkling2ByteLedger.
type inkling2LedgerSnapshot struct {
	Total            int64
	Cap              int64
	SlotBytes        int64
	LazyPrefillBytes int64
	CohortArenaBytes int64
	PrefixStoreBytes int64
	Entries          int
}

// inkling2ByteLedger is the single in-process memory watermark for Inkling-2
// serving: scalar slots, lazy prefill arenas, cohort arenas, and the cohort
// prefix store all register their held bytes here under one cap.
//
// Design note (TICKET-02): we use an in-process mutex ledger with the same
// typed-entry semantics as internal/localadmission.ReservationStore rather than
// calling ReservationStore per request/chunk/trim, because ReservationStore
// acquires a host file lock and rewrites JSON on disk on every mutation, which
// is too costly for the per-request serving hot path while all four classes
// live inside this single serve process.
type inkling2ByteLedger struct {
	mu       sync.Mutex
	capBytes int64
	entries  map[string]inkling2LedgerEntry
	total    int64
}

func newInkling2ByteLedger(capBytes int64) *inkling2ByteLedger {
	return &inkling2ByteLedger{
		capBytes: capBytes,
		entries:  make(map[string]inkling2LedgerEntry),
	}
}

// Set registers or updates key with kind and bytes. A non-positive byte count
// deletes the entry so zero-sized entries never linger.
func (l *inkling2ByteLedger) Set(key string, kind inkling2LedgerKind, bytes int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if prev, ok := l.entries[key]; ok {
		l.total -= prev.bytes
		delete(l.entries, key)
	}
	if bytes <= 0 {
		return
	}
	if l.entries == nil {
		l.entries = make(map[string]inkling2LedgerEntry)
	}
	l.entries[key] = inkling2LedgerEntry{kind: kind, bytes: bytes}
	l.total += bytes
}

// Delete removes key from the ledger if present.
func (l *inkling2ByteLedger) Delete(key string) {
	l.Set(key, "", 0)
}

// Total returns the exact sum of all registered entries.
func (l *inkling2ByteLedger) Total() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.total
}

// Cap returns the configured watermark in bytes.
func (l *inkling2ByteLedger) Cap() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.capBytes
}

// SetCap updates the watermark in bytes.
func (l *inkling2ByteLedger) SetCap(capBytes int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.capBytes = capBytes
}

// Fits reports whether adding delta bytes stays within the ledger watermark.
func (l *inkling2ByteLedger) Fits(delta int64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.total+delta <= l.capBytes
}

// Snapshot returns a consistent breakdown of all registered entries.
func (l *inkling2ByteLedger) Snapshot() inkling2LedgerSnapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	snap := inkling2LedgerSnapshot{
		Total:   l.total,
		Cap:     l.capBytes,
		Entries: len(l.entries),
	}
	for _, e := range l.entries {
		switch e.kind {
		case inkling2LedgerSlot:
			snap.SlotBytes += e.bytes
		case inkling2LedgerLazyPrefill:
			snap.LazyPrefillBytes += e.bytes
		case inkling2LedgerCohortArena:
			snap.CohortArenaBytes += e.bytes
		case inkling2LedgerPrefixStore:
			snap.PrefixStoreBytes += e.bytes
		}
	}
	return snap
}

// inkling2PrefixState holds per-engine ledger and trace state outside the
// inkling2LoopEngine struct definition in serve_inkling2.go.
type inkling2PrefixState struct {
	ledger          *inkling2ByteLedger
	slotKeys        map[*inkling2PrefixSlot]string
	nextSlotID      uint64
	hits            uint64
	misses          uint64
	evictions       uint64
	effectiveSlots  int
	effectiveLogged bool
	trace           bool
	traceOut        io.Writer
}

var inkling2PrefixStates sync.Map // *inkling2LoopEngine -> *inkling2PrefixState

func (e *inkling2LoopEngine) prefixState() *inkling2PrefixState {
	if v, ok := inkling2PrefixStates.Load(e); ok {
		st := v.(*inkling2PrefixState)
		if e.capBytes > 0 && st.ledger.Cap() != e.capBytes {
			st.ledger.SetCap(e.capBytes)
		}
		return st
	}
	st := &inkling2PrefixState{
		ledger:         newInkling2ByteLedger(e.capBytes),
		slotKeys:       make(map[*inkling2PrefixSlot]string),
		effectiveSlots: max(e.slots, 1),
		trace:          os.Getenv("FAK_INKLING2_TRACE") == "1",
		traceOut:       os.Stderr,
	}
	actual, _ := inkling2PrefixStates.LoadOrStore(e, st)
	return actual.(*inkling2PrefixState)
}

func (st *inkling2PrefixState) slotKey(sl *inkling2PrefixSlot) string {
	if k, ok := st.slotKeys[sl]; ok {
		return k
	}
	st.nextSlotID++
	k := fmt.Sprintf("slot:%d", st.nextSlotID)
	st.slotKeys[sl] = k
	return k
}

func (st *inkling2PrefixState) writer() io.Writer {
	if st.traceOut != nil {
		return st.traceOut
	}
	return os.Stderr
}

// inkling2PrefixEffectiveSlots computes how many slots of slotBytes actually
// fit under capBytes, clamped to [1, max(configured, 1)].
func inkling2PrefixEffectiveSlots(configured int, capBytes, slotBytes int64) int {
	want := max(configured, 1)
	if slotBytes <= 0 || capBytes <= 0 {
		return want
	}
	fit := int(capBytes / slotBytes)
	if fit < 1 {
		return 1
	}
	if fit < want {
		return fit
	}
	return want
}

// inkling2PrefixEffectiveNote formats the truthful effective slot-count line
// once the first slot's measured byte footprint is known, warning explicitly
// when capBytes cannot hold two slots.
func inkling2PrefixEffectiveNote(configured int, capBytes, slotBytes int64) string {
	want := max(configured, 1)
	eff := inkling2PrefixEffectiveSlots(want, capBytes, slotBytes)
	capMiB := capBytes >> 20
	slotMiB := slotBytes >> 20
	if want > 1 && eff < 2 {
		return fmt.Sprintf("effective slots=%d (configured=%d cap=%dMiB slot=%dMiB: cap cannot hold two slots)", eff, want, capMiB, slotMiB)
	}
	if eff < want {
		return fmt.Sprintf("effective slots=%d (configured=%d cap=%dMiB slot=%dMiB: capped by memory watermark)", eff, want, capMiB, slotMiB)
	}
	return fmt.Sprintf("effective slots=%d (configured=%d cap=%dMiB slot=%dMiB)", eff, want, capMiB, slotMiB)
}

// inkling2PrefixSlot is one resident session and the prompt its checkpoint covers.
type inkling2PrefixSlot struct {
	sess    ggufload.Inkling2LoopSession
	hasCkpt bool
	ckpt    []int
	used    uint64 // engine clock at the slot's last request
}

// prefixHit is the slot whose checkpoint is the longest proper prefix of prompt.
func (e *inkling2LoopEngine) prefixHit(prompt []int) *inkling2PrefixSlot {
	var best *inkling2PrefixSlot
	for _, sl := range e.cache {
		if sl.hasCkpt && len(sl.ckpt) < len(prompt) && (best == nil || len(sl.ckpt) > len(best.ckpt)) && inkling2HasPrefix(prompt, sl.ckpt) {
			best = sl
		}
	}
	st := e.prefixState()
	if best != nil {
		st.hits++
	} else {
		st.misses++
	}
	return best
}

// prefixMeter syncs every resident slot's held bytes (separating lazy prefill
// arenas when reported) into the unified byte ledger and returns the ledger's
// total held bytes and largest single-slot footprint; ok is false when a slot
// cannot report its held bytes.
func (e *inkling2LoopEngine) prefixMeter() (held, largest int64, ok bool) {
	st := e.prefixState()
	for _, sl := range e.cache {
		m, metered := sl.sess.(inkling2HeldBytes)
		if !metered {
			return 0, 0, false
		}
		b := m.HeldBytes()
		var lazy int64
		if pm, hasLazy := sl.sess.(inkling2PrefillArenaBytes); hasLazy {
			lazy = min(max(pm.PrefillArenaBytes(), 0), b)
		}
		key := st.slotKey(sl)
		st.ledger.Set(key, inkling2LedgerSlot, b-lazy)
		st.ledger.Set(key+":prefill", inkling2LedgerLazyPrefill, lazy)
		largest = max(largest, b)
	}
	if len(e.cache) > 0 && largest > 0 {
		eff := inkling2PrefixEffectiveSlots(e.slots, e.capBytes, largest)
		st.effectiveSlots = eff
		if !st.effectiveLogged {
			st.effectiveLogged = true
			if e.slots > 1 || st.trace {
				fmt.Fprintf(st.writer(), "inkling2.prefix %s\n", inkling2PrefixEffectiveNote(e.slots, e.capBytes, largest))
			}
		}
	}
	return st.ledger.Total(), largest, true
}

// prefixCanGrow reports whether a miss may open another slot: always into an empty
// cache, else only under the slot limit with every slot metered and room under the
// unified ledger watermark for one more slot the size of the largest.
func (e *inkling2LoopEngine) prefixCanGrow() bool {
	if len(e.cache) == 0 {
		return true
	}
	if e.prefixReuse.disabled() || len(e.cache) >= max(e.slots, 1) {
		return false
	}
	held, largest, ok := e.prefixMeter()
	return ok && held+largest <= e.capBytes
}

// prefixVictim is the slot a miss runs on when the cache cannot grow: one holding
// no checkpoint, else the least recently used.
func (e *inkling2LoopEngine) prefixVictim() *inkling2PrefixSlot {
	var lru *inkling2PrefixSlot
	for _, sl := range e.cache {
		if !sl.hasCkpt {
			return sl
		}
		if lru == nil || sl.used < lru.used {
			lru = sl
		}
	}
	return lru
}

// prefixTrim closes least recently used slots, never the current one, while more
// than one slot pushes the unified ledger past the cap; a session grows after it
// opens (lazy prefill arenas, checkpoint rows), so the admission projection alone
// is not enough.
func (e *inkling2LoopEngine) prefixTrim() {
	st := e.prefixState()
	for len(e.cache) > 1 {
		if held, _, ok := e.prefixMeter(); ok && held <= e.capBytes {
			break
		}
		var lru *inkling2PrefixSlot
		for _, sl := range e.cache {
			if sl != e.cur && (lru == nil || sl.used < lru.used) {
				lru = sl
			}
		}
		if lru == nil {
			break
		}
		st.evictions++
		_ = e.dropSlot(lru)
	}
	// Sync final post-trim state for single-slot and trace accounting.
	_, _, _ = e.prefixMeter()
	if st.trace {
		snap := st.ledger.Snapshot()
		fmt.Fprintf(st.writer(), "inkling2.prefix effective_slots=%d slots=%d hits=%d misses=%d evictions=%d held_bytes=%d (slot=%d lazy_prefill=%d cohort_arena=%d prefix_store=%d cap=%d)\n",
			st.effectiveSlots, len(e.cache), st.hits, st.misses, st.evictions,
			snap.Total, snap.SlotBytes, snap.LazyPrefillBytes, snap.CohortArenaBytes, snap.PrefixStoreBytes, snap.Cap)
	}
}

// evictIdleSlotsForCohort evicts unpinned (not currently in-flight) scalar slots
// in LRU order until adding needBytes fits under the unified ledger watermark
// (or no unpinned slots remain). Used on cohort arrival before releasing arenas.
func (e *inkling2LoopEngine) evictIdleSlotsForCohort(needBytes int64) int {
	st := e.prefixState()
	evicted := 0
	for len(e.cache) > 0 {
		if _, _, ok := e.prefixMeter(); ok && st.ledger.Fits(needBytes) {
			break
		}
		var lru *inkling2PrefixSlot
		for _, sl := range e.cache {
			if sl != e.cur && (lru == nil || sl.used < lru.used) {
				lru = sl
			}
		}
		if lru == nil {
			break
		}
		st.evictions++
		evicted++
		_ = e.dropSlot(lru)
	}
	return evicted
}

func (e *inkling2LoopEngine) dropSlot(sl *inkling2PrefixSlot) error {
	if sl == nil {
		return nil
	}
	if e.cur == sl {
		e.cur = nil
	}
	for i, c := range e.cache {
		if c == sl {
			e.cache = append(e.cache[:i], e.cache[i+1:]...)
			break
		}
	}
	if v, ok := inkling2PrefixStates.Load(e); ok {
		st := v.(*inkling2PrefixState)
		if key, exists := st.slotKeys[sl]; exists {
			st.ledger.Delete(key)
			st.ledger.Delete(key + ":prefill")
			delete(st.slotKeys, sl)
		}
	}
	return sl.sess.Close()
}

// dropCurrent closes the slot of the request in flight, so a fault or a cancelled
// prefill never leaves a checkpoint claiming a prompt it did not finish.
func (e *inkling2LoopEngine) dropCurrent() error { return e.dropSlot(e.cur) }

func (e *inkling2LoopEngine) dropAllSlots() error {
	var errs []error
	for len(e.cache) > 0 {
		errs = append(errs, e.dropSlot(e.cache[0]))
	}
	e.cur = nil
	inkling2PrefixStates.Delete(e)
	return errors.Join(errs...)
}

func (e *inkling2LoopEngine) prefixHeldBytes() int64 {
	held, _, _ := e.prefixMeter()
	return held
}

func (e *inkling2LoopEngine) prefixSlotCount() int { return len(e.cache) }

// inkling2PrefixSlotsNote names a slot count above the default for the startup message.
func inkling2PrefixSlotsNote(e *inkling2LoopEngine) string {
	if e.prefixReuse.disabled() || e.slots <= 1 {
		return ""
	}
	return fmt.Sprintf("; prefix slots=%d cap=%dMiB", e.slots, e.capBytes>>20)
}

// inkling2ScalarBehindCohorts runs every lone request the cohort engine routes to
// scalar on a fresh session and names the slots that ignores. Kept slots would be
// unsafe there: a cohort captures into and resets the same process-wide graph exec
// without taking the resident graph owner, so a kept session could replay a
// cohort's exec, and nothing frees the slots before a cohort sizes its arenas.
func inkling2ScalarBehindCohorts(scalar *inkling2LoopEngine) string {
	scalar.prefixReuse = inkling2PrefixReuseOffCohort
	if scalar.slots <= 1 {
		return ""
	}
	return fmt.Sprintf("; FAK_INKLING2_PREFIX_SLOTS=%d ignored: lone requests run on a fresh session beside cohorts", scalar.slots)
}

// inkling2BatchPrefixNote names the cohorts' cross-request prefix store for the
// startup message, and refuses at startup a FAK_INKLING2_BATCH_PREFIX_CACHE_MIB that is
// malformed or that the cohorts on be could not use, rather than serving without the
// store the operator asked for.
func inkling2BatchPrefixNote(be any, lookup func(string) (string, bool)) (string, error) {
	mib, on, err := ggufload.Inkling2BatchPrefixCacheMiB(be, lookup)
	if err != nil || !on {
		return "", err
	}
	return fmt.Sprintf("; batch prefix cache=%dMiB (lone requests run as one-row cohorts)", mib), nil
}
