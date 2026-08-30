package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// compaction_contract.go — the CONTINUATION CONTRACT a compaction boundary emits (#2422).
//
// # The prose convention this replaces
//
// A harness that shortens its own transcript has to tell the model so, or the model reads the
// shortened history as evidence the work is nearly over and defensively wraps up. The usual fix
// is a sentence in the system prompt: "I summarize and continue." That sentence is unverifiable
// in both directions — nothing checks that it is true of any particular boundary, and nothing
// downstream can read it without parsing prose.
//
// So the statement becomes an artifact. At every boundary the survival-class gate (#2421) crosses,
// the plan it actually ran is projected into ONE record (CompactionContract) that goes out on both
// channels fak already owns:
//
//	the MODEL      an in-band `[fak]` note, prepended as the turn's first text block
//	an ORCHESTRATOR the same record, typed, on the response's `fak.compaction` extension
//
// One source, two renderings, so they cannot disagree about what the boundary did.
//
// # Once per boundary, and the reason it is take-once
//
// The dedup here is deliberately NOT the once-per-session shape of ctxExpenseNoteOnce. A session
// crosses many boundaries and each one is a distinct fact about a distinct loss — suppressing the
// second would leave the model wrong about the transcript it is currently holding. So the pending
// contract is CONSUMED by the turn that reports it (takeCompactionContract): every boundary emits
// exactly one note, and a turn that crossed no boundary emits none. A boundary whose turn never
// completes (an upstream error) is simply superseded by the next boundary's record — latest wins,
// because the latest is the one that describes the body actually on the wire.
//
// # Derived from the plan, never re-derived
//
// Every field is a projection of the ctxplan.EvictionPlan the compaction ran under. Nothing here
// re-computes what "should" have survived: a second derivation could disagree with the first, and a
// contract that can be wrong about its own boundary is worth less than the prose it replaces.

// compactionContractVersion is the schema version stamped on every emitted contract. Bump it when
// a field's MEANING changes (a new field alone does not: absent-is-nil is already readable).
const compactionContractVersion = 1

// compactionContractInstruction is the closed continuation token. It is a fixed string rather than
// free text so a reader can branch on it: the boundary is a summarize-and-continue, and the turn
// after it must not read its shortened transcript as a reason to stop.
const compactionContractInstruction = "continue-do-not-wrap-up"

// maxReplayablePageDigests caps the digest list. A shed large enough to evict hundreds of
// replayable pages would otherwise put a multi-kilobyte digest array on every boundary response —
// growing the wire at exactly the moment the kernel is shrinking it. The cap is disclosed in the
// note's own wording (it says how many pages are replayable, not how many digests it listed), so a
// truncated list never reads as a complete one.
const maxReplayablePageDigests = 32

// maxRetentionEffects bounds provenance metadata independently of transcript size.
const maxRetentionEffects = 32

// compactionContractFrom projects one eviction plan onto the wire contract, or returns nil when the
// plan evicted nothing — a boundary that dropped no page is not a boundary worth announcing, and an
// empty contract would train both readers to ignore the field.
//
// els is the messages[] element bytes in wire order, index-aligned with pages (both come from the
// same body), so the byte count and the digests describe the pages the plan actually named rather
// than an estimate of them. A plan ID that does not resolve to an element is skipped rather than
// guessed at.
func compactionContractFrom(pages []ctxplan.Page, els [][]byte, plan ctxplan.EvictionPlan) *CompactionContract {
	if len(plan.Evict) == 0 {
		return nil
	}
	byID := make(map[string]int, len(pages))
	for i, p := range pages {
		byID[p.ID] = i
	}
	kept := map[ctxplan.SurvivalClass]int{}
	evicted := map[ctxplan.SurvivalClass]int{}
	contract := &CompactionContract{
		ContractVersion: compactionContractVersion,
		Instruction:     compactionContractInstruction,
	}
	for _, id := range plan.Keep {
		if i, ok := byID[id]; ok {
			kept[pages[i].Class()]++
		}
	}
	for _, id := range plan.Evict {
		i, ok := byID[id]
		if !ok {
			continue
		}
		class := pages[i].Class()
		evicted[class]++
		if i >= len(els) {
			continue
		}
		contract.EvictedByteCount += len(els[i])
		// Only a REPLAYABLE eviction earns a digest: it names bytes that can be paged back.
		// An EVICTABLE page is genuinely gone, and a handle to nothing is a false promise.
		if class == ctxplan.ClassReplayable && len(contract.ReplayablePageDigests) < maxReplayablePageDigests {
			sum := sha256.Sum256(els[i])
			contract.ReplayablePageDigests = append(contract.ReplayablePageDigests, hex.EncodeToString(sum[:]))
		}
	}
	evictedIDs := make(map[string]bool, len(plan.Evict))
	for _, id := range plan.Evict {
		evictedIDs[id] = true
	}
	for _, p := range pages {
		outcome := "kept"
		if evictedIDs[p.ID] {
			outcome = "evicted"
		}
		for _, annotation := range p.Retention {
			if annotation.Intent == ctxplan.RetentionNeutral || len(contract.RetentionEffects) >= maxRetentionEffects {
				continue
			}
			contract.RetentionEffects = append(contract.RetentionEffects, RetentionEffect{
				PageID: p.ID, Intent: string(annotation.Intent), Source: annotation.Source, Outcome: outcome,
			})
		}
	}
	// Resistance order, and only classes the page set actually held — an all-zero row for a class
	// no page carried would read as a loss that never happened.
	for _, class := range []ctxplan.SurvivalClass{ctxplan.ClassPinned, ctxplan.ClassReplayable, ctxplan.ClassEvictable} {
		if kept[class] == 0 && evicted[class] == 0 {
			continue
		}
		contract.PreservedClasses = append(contract.PreservedClasses, PreservedClassSplit{
			Class: class.String(), Kept: kept[class], Evicted: evicted[class],
		})
	}
	return contract
}

// anthropicMessageElements splits an outbound Anthropic body's messages[] into its element bytes,
// index-aligned with anthropicSurvivalPages' page slice (both walk the same array in wire order).
// nil when the body has no messages[], which leaves the contract un-emittable rather than wrong.
//
// This is a SHALLOW split — each element stays raw — and it runs only on a turn that actually
// compacted, so it costs one un-decoded pass over a body the compactor has already rewritten.
func anthropicMessageElements(raw []byte) [][]byte {
	var doc struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil
	}
	els := make([][]byte, 0, len(doc.Messages))
	for _, el := range doc.Messages {
		els = append(els, el)
	}
	return els
}

// noteCompactionContract records the contract this boundary emitted, pending the next completed
// turn on the same trace. A no-op for the empty trace (tests and non-session callers have no
// session to report into) and for a nil contract, so the default path stays untouched.
//
// Bounded by the same maxResetHealthSessions reaper convention as the other per-trace tables: a
// gateway serving unbounded distinct traces must not grow a map forever.
func (s *Server) noteCompactionContract(trace string, c *CompactionContract) {
	if s == nil || c == nil || strings.TrimSpace(trace) == "" {
		return
	}
	s.compactionContractMu.Lock()
	defer s.compactionContractMu.Unlock()
	if s.compactionContract == nil {
		s.compactionContract = map[string]*CompactionContract{}
	}
	if _, seen := s.compactionContract[trace]; !seen && len(s.compactionContract) >= maxResetHealthSessions {
		for k := range s.compactionContract {
			delete(s.compactionContract, k)
			break
		}
	}
	s.compactionContract[trace] = c
}

// takeCompactionContract CONSUMES this trace's pending boundary contract, returning nil when the
// trace crossed no boundary since its last completed turn. The take is what makes the emission
// exactly-once-per-boundary: the reporting turn removes the record, so a later turn on the same
// trace cannot re-announce a boundary it did not cross.
func (s *Server) takeCompactionContract(trace string) *CompactionContract {
	if s == nil || strings.TrimSpace(trace) == "" {
		return nil
	}
	s.compactionContractMu.Lock()
	defer s.compactionContractMu.Unlock()
	c, ok := s.compactionContract[trace]
	if !ok {
		return nil
	}
	delete(s.compactionContract, trace)
	return c
}

// compactionContractNote renders the contract as the model-facing in-band `[fak]` line: the same
// record the `fak.compaction` extension carries, in the one form a client that reads only content
// blocks (Claude Code) can actually see. "" for a nil contract, so the caller prepends nothing.
//
// It names the loss AND the recovery AND the instruction, in that order, because a note that
// reported only the loss is precisely the input that makes a model wrap up early.
func compactionContractNote(c *CompactionContract) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("[fak] context compaction boundary (contract v")
	b.WriteString(strconv.Itoa(c.ContractVersion))
	b.WriteString("): kept/evicted by survival class — ")
	for i, split := range c.PreservedClasses {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(split.Class)
		b.WriteString(" ")
		b.WriteString(strconv.Itoa(split.Kept))
		b.WriteString("/")
		b.WriteString(strconv.Itoa(split.Kept + split.Evicted))
	}
	b.WriteString("; ")
	b.WriteString(strconv.Itoa(c.EvictedByteCount))
	b.WriteString(" bytes evicted")
	if n := replayableEvictedCount(c); n > 0 {
		b.WriteString("; ")
		b.WriteString(strconv.Itoa(n))
		b.WriteString(" evicted page(s) are REPLAYABLE — their full bytes are recoverable by digest")
	}
	b.WriteString(". Older turns were shed to fit the window; the task is unchanged and unfinished. ")
	b.WriteString(compactionContractInstruction)
	b.WriteString(": keep working, do not summarize and stop.")
	return b.String()
}

// replayableEvictedCount reports how many evicted pages classed REPLAYABLE, reading the plan's own
// per-class split rather than len(ReplayablePageDigests) — the digest list is capped, and a note
// that counted the capped list would understate the recoverable set on a large shed.
func replayableEvictedCount(c *CompactionContract) int {
	for _, split := range c.PreservedClasses {
		if split.Class == ctxplan.ClassReplayable.String() {
			return split.Evicted
		}
	}
	return 0
}
