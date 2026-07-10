// Package ctxmmu — the memory-write adjudicator (issue #2874).
//
// MMU.Admit gates what enters the CONTEXT; nothing in this package gated what
// enters durable MEMORY. Hermes curates memory by asking the model to honor a
// prose "Do NOT capture" list — hope-the-model-complies, and #3006 is the
// recorded failure: a 74 KiB skill body copied verbatim into a single memory
// entry. memq.AdjudicateMemoryWrite (#2912/#2836) is the lexical sibling of
// this file: it refuses the junk CLASSES the Hermes prose names (transient
// errors, failed-invocation narratives, run-bound one-offs) with a memq-local
// string vocabulary. This file is the ctxmmu structural floor under the same
// write boundary, judging three shapes that are junk BY CONSTRUCTION and
// refusing them with a WITNESSED reason from the abi closed refusal vocabulary
// (abi/reasons.go), so a memory-write deny is auditable the same way every
// other kernel deny is:
//
//   - an oversize verbatim blob   -> abi.ReasonOversize (the #3006 shape)
//   - secret-shaped content       -> abi.ReasonSecretExfil (the same
//     secretPattern floor MMU.Admit quarantines on — a credential must not
//     enter durable memory any more than it may enter context)
//   - a near-duplicate of an existing entry -> ReasonMemoryNearDuplicate
//     (registered out-of-tree, the egressfloor/toolproc idiom)
//
// Secret detection and oversize detection are two INDEPENDENT structural
// checks — a small secret-bearing entry and a large benign blob each trip
// exactly one. When both hold, the secret verdict is cited (the same rung
// order as MMU.Admit, where ScreenBytes runs before the oversize branch):
// security-relevant evidence must not be masked by a size refusal.
//
// The adjudicator is deterministic and I/O-free — the same "cheap prior, fail
// closed" posture as classifyDurability and GateDisposition — and, like those,
// it never silently drops: every call returns a typed verdict whose Witness
// string names the measured structure (size, detector, duplicate id +
// similarity) without ever echoing secret bytes. It is a pure library today:
// nothing wires it into a live write path yet, which keeps it gated by
// construction until promotion evidence lands (the wiring is the promotion
// step, tracked on #2874).
package ctxmmu

import (
	"fmt"
	"strings"
	"sync"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ReasonMemoryNearDuplicate is the refusal code a near-duplicate memory write
// cites. It is an out-of-tree ADDITIVE label-space entry (ARCHITECTURE.md:
// RegisterReason), not a core reason: the closed core set stays frozen and a
// model trained on it degrades gracefully. Allocation ledger for the open
// range so far: 1024 egressfloor, 1040-1044 toolproc, 1050-1052 taskgraph,
// 1060 here.
const ReasonMemoryNearDuplicate abi.ReasonCode = 1060

// ReasonMemoryNearDuplicateName is the stable name registered for
// ReasonMemoryNearDuplicate.
const ReasonMemoryNearDuplicateName = "MEMORY_NEAR_DUPLICATE"

// MemoryWriteMaxBytes is the single-entry size bound. One memory entry is one
// distilled fact (a sentence to a short paragraph); anything past this is a
// document-sized verbatim copy that belongs in docs/, not a memory cell. The
// value matches memq.MaxDurableFactBytes (16 KiB) so the two write-boundary
// arms refuse the same oversize shape — #3006's 74 KiB write is >4x over —
// while sitting well above any real distilled fact.
const MemoryWriteMaxBytes = 16 << 10

// NearDuplicateJaccard is the similarity at or above which a candidate write
// is refused as a near-duplicate: Jaccard over the word-bigram shingle sets of
// the candidate and a noted entry. Bigrams keep word ORDER load-bearing (a
// word-set overlap would call "A calls B" and "B calls A" identical) while
// staying edit-tolerant: a single-word substitution in a ~30-word fact changes
// 2 of ~30 shingles and scores ~0.87 — refused — where a same-topic but
// genuinely different fact shares only isolated bigrams and scores far below.
// The threshold errs strict (under-refusing), the repo's preferred error
// direction at this boundary: a missed near-duplicate costs one redundant
// entry; a false refusal silently loses a new fact.
const NearDuplicateJaccard = 0.85

// memShingleWords is the shingle width (word bigrams — see NearDuplicateJaccard).
const memShingleWords = 2

// DefaultMaxMemoryEntries bounds the adjudicator's noted-entry ledger, for the
// same reason DefaultMaxHeld bounds the quarantine ledger: a process-lifetime
// gate fed a long stream must not grow without bound. Oldest entries are
// dropped first; a dropped entry simply stops participating in dedup — the
// cheap degradation, never a wrong refusal.
const DefaultMaxMemoryEntries = DefaultMaxHeld

// MemoryWriteVerdict is the typed ruling on one candidate memory write. Reason
// is from the abi closed refusal vocabulary and is abi.ReasonNone exactly when
// Admit is true. Witness is the auditable claim naming WHAT was measured (the
// byte count, the detector, the duplicate id + similarity) — it never carries
// the matched secret bytes. DuplicateOf names the noted entry a near-duplicate
// refusal matched, empty otherwise.
type MemoryWriteVerdict struct {
	Admit       bool
	Reason      abi.ReasonCode
	Witness     string
	DuplicateOf string
}

// memEntry is one noted existing memory entry: its caller-supplied id and the
// word-bigram shingle set of its body (the body bytes themselves are not kept).
type memEntry struct {
	id       string
	shingles map[string]struct{}
}

// MemoryWriteAdjudicator judges candidate durable memory writes by structure
// (#2874). Construct with NewMemoryWriteAdjudicator, Note the entries already
// in the store, then AdmitWrite each candidate. Safe for concurrent use.
type MemoryWriteAdjudicator struct {
	mu         sync.Mutex
	entries    []memEntry // FIFO; oldest evicted past maxEntries
	maxEntries int
}

// NewMemoryWriteAdjudicator builds an adjudicator with the standard
// noted-entry bound (DefaultMaxMemoryEntries).
func NewMemoryWriteAdjudicator() *MemoryWriteAdjudicator {
	return NewMemoryWriteAdjudicatorWithLimit(DefaultMaxMemoryEntries)
}

// NewMemoryWriteAdjudicatorWithLimit builds an adjudicator whose noted-entry
// ledger holds at most maxEntries (oldest dropped first). A non-positive
// maxEntries falls back to DefaultMaxMemoryEntries. This is the seam the
// eviction test uses, mirroring NewWithLimit on the MMU.
func NewMemoryWriteAdjudicatorWithLimit(maxEntries int) *MemoryWriteAdjudicator {
	if maxEntries < 1 {
		maxEntries = DefaultMaxMemoryEntries
	}
	return &MemoryWriteAdjudicator{maxEntries: maxEntries}
}

// Note records an EXISTING memory entry (id + body) as dedup ground truth.
// The caller notes the store's entries at load, and notes an admitted write
// only once it actually lands — AdmitWrite never notes its own candidate, so
// an admitted-but-unwritten candidate cannot block a retry of itself. A body
// that yields no shingles (empty / no words) is skipped.
func (a *MemoryWriteAdjudicator) Note(id string, body []byte) {
	sh := memShingles(body)
	if len(sh) == 0 {
		return
	}
	a.mu.Lock()
	a.entries = append(a.entries, memEntry{id: id, shingles: sh})
	if drop := len(a.entries) - a.maxEntries; drop > 0 {
		// Copy the survivors forward so the evicted prefix does not pin the
		// backing array (the evictExcessLocked concern, in miniature).
		a.entries = append(a.entries[:0:0], a.entries[drop:]...)
	}
	a.mu.Unlock()
}

// NotedLen reports the current number of noted entries (≤ the construction
// bound) — the observability hook for the ledger bound, peer of HeldLen.
func (a *MemoryWriteAdjudicator) NotedLen() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.entries)
}

// AdmitWrite is the pure decision function: it judges one candidate memory
// write by structure alone and returns a typed, witnessed verdict. Rung order
// (secret before oversize before duplicate) is documented on the package
// comment; each rung is an independent check and a body tripping none of them
// admits with abi.ReasonNone.
func (a *MemoryWriteAdjudicator) AdmitWrite(body []byte) MemoryWriteVerdict {
	if secretPattern.Match(body) {
		// The witness names the detector, never the matched bytes: a refusal
		// record must not itself become the secret's second home.
		return refuseMemWrite(abi.ReasonSecretExfil, "secret_pattern",
			"body matches a secret shape (matched bytes withheld)", "")
	}
	if n := len(body); n > MemoryWriteMaxBytes {
		return refuseMemWrite(abi.ReasonOversize, "size_bound",
			fmt.Sprintf("%d bytes exceeds the %d-byte single-entry bound", n, MemoryWriteMaxBytes), "")
	}
	if id, sim, dup := a.nearestDuplicate(body); dup {
		return refuseMemWrite(ReasonMemoryNearDuplicate, "shingle_jaccard",
			fmt.Sprintf("near-duplicate of %s (similarity %.2f >= %.2f)", id, sim, NearDuplicateJaccard), id)
	}
	return MemoryWriteVerdict{Admit: true, Reason: abi.ReasonNone,
		Witness: "ctxmmu.memwrite NONE admitted"}
}

// refuseMemWrite builds a refusal verdict whose Witness follows the package's
// quarantineMeta claim shape: producer, closed-vocabulary reason name,
// detector, then the measured detail.
func refuseMemWrite(reason abi.ReasonCode, detector, detail, duplicateOf string) MemoryWriteVerdict {
	return MemoryWriteVerdict{
		Reason:      reason,
		Witness:     "ctxmmu.memwrite " + abi.ReasonName(reason) + " " + detector + " " + detail,
		DuplicateOf: duplicateOf,
	}
}

// nearestDuplicate scans the noted entries for the highest bigram-Jaccard
// similarity to body and reports it when it clears NearDuplicateJaccard.
func (a *MemoryWriteAdjudicator) nearestDuplicate(body []byte) (id string, sim float64, dup bool) {
	sh := memShingles(body)
	if len(sh) == 0 {
		return "", 0, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, e := range a.entries {
		if s := jaccard(sh, e.shingles); s > sim {
			sim, id = s, e.id
		}
	}
	return id, sim, sim >= NearDuplicateJaccard
}

// memShingles normalizes body (lowercase, letters/digits only) and returns its
// word-bigram shingle set. A body with exactly one word yields that word as
// its single shingle so two one-word entries still compare; no words yields an
// empty set (never compared).
func memShingles(body []byte) map[string]struct{} {
	words := strings.FieldsFunc(strings.ToLower(string(body)), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := make(map[string]struct{}, len(words))
	if len(words) == 0 {
		return out
	}
	if len(words) < memShingleWords {
		out[strings.Join(words, " ")] = struct{}{}
		return out
	}
	for i := 0; i+memShingleWords <= len(words); i++ {
		out[strings.Join(words[i:i+memShingleWords], " ")] = struct{}{}
	}
	return out
}

// jaccard is |a∩b| / |a∪b| over shingle sets; 0 when either is empty.
func jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	small, large := a, b
	if len(small) > len(large) {
		small, large = large, small
	}
	inter := 0
	for s := range small {
		if _, ok := large[s]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

func init() {
	abi.RegisterReason(ReasonMemoryNearDuplicate, ReasonMemoryNearDuplicateName)
	abi.RegisterCapability("ctxmmu.memwrite.v1")
}
