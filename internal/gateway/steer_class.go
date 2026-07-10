package gateway

// steer_class.go — the classified, non-querying-capable generalization of the steer
// splice bus (#2402), part of harness-native program #2387 / epic #2388.
//
// The historical steer verb (#760/#850, http.go steerSession/SteerRequest) splices ONE
// unclassified block of text into the next turn via the loop's drainSteer — every datum
// costs a model turn and arrives with no priority hint. This file generalizes the bus so
// an append carries:
//
//   - a CLASS (SteerNow / SteerNext / SteerLater) — WHEN it reaches the loop, and
//   - a QUERY bit — WHETHER it forces a model turn at all.
//
// Decoupling "context arrived" from "model turn spent" is what makes cheap continuous
// observation of a running loop affordable: an observer feed can append context (Query
// false) without paying a turn per datum. Every append is screened (the same
// ctxmmu.ScreenBytes the result-admission path uses) and taint-stamped BEFORE it is
// staged, so an observer feed can never inject unscreened bytes; a poisonous append is
// held as a QUARANTINE STUB, witnessed by a hash-chained journal, and never reaches the
// loop as prose.
//
// GENERATION posture (gen/next): this is the gated bus mechanism plus its contract test
// (TestSteerClassScheduling). The live consumer — the per-trace agent-loop drain in
// internal/agent that folds a class-ordered, taint-labeled block into the served turn —
// is the named promotion follow-on (#748 is the cross-loop scheduler these classes feed).
// The /steer route already validates the class, computes the query bit, and screens the
// append at ingress, so the wire and the screen are live today; the staging queue is what
// stays gated until the drain lands.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// SteerClass is the CLOSED scheduling class an append on the steer bus carries (#2402).
// It decides WHEN a staged append reaches the loop, separately from the query bit (which
// decides WHETHER it forces a model turn). The zero value is SteerNext, so an append with
// no explicit class behaves exactly like a legacy unclassified steer.
type SteerClass uint8

const (
	// SteerNext folds the append into the next querying turn's input — the default and the
	// byte-for-byte legacy steer behavior.
	SteerNext SteerClass = iota
	// SteerNow interrupts at the next safe boundary: a now append lands BEFORE the next tool
	// dispatch rather than waiting for the turn's folded input.
	SteerNow
	// SteerLater holds the append until the loop would otherwise idle; it never forces a turn.
	SteerLater
)

// String renders the wire spelling of a class.
func (c SteerClass) String() string {
	switch c {
	case SteerNow:
		return "now"
	case SteerLater:
		return "later"
	default:
		return "next"
	}
}

// ParseSteerClass maps the wire string to a class. Empty ⇒ SteerNext (the legacy default).
// ok is false for an unrecognized non-empty value, so the route can 400 a bad class rather
// than silently coerce it into a scheduling decision the caller did not ask for.
func ParseSteerClass(s string) (class SteerClass, ok bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "next":
		return SteerNext, true
	case "now":
		return SteerNow, true
	case "later":
		return SteerLater, true
	default:
		return SteerNext, false
	}
}

// steerClass returns the request's parsed scheduling class and whether it was recognized.
func (r SteerRequest) steerClass() (SteerClass, bool) { return ParseSteerClass(r.Class) }

// querying reports the request's query bit: a nil Query pointer or an explicit true means
// the append forces a model turn (legacy steer semantics); an explicit false means
// "context arrived" — the append is screened, taint-stamped, and staged for the next
// querying turn WITHOUT scheduling a planner call of its own.
func (r SteerRequest) querying() bool { return r.Query == nil || *r.Query }

// steerAppend is one classified datum staged on the bus before the loop drains it. Text is
// the model-visible bytes — a QUARANTINE STUB (never the raw bytes) when the screen held
// the append. Taint is the abi label stamped on ingress: TaintTainted for an admitted
// append (untrusted-by-default operator/observer input), TaintQuarantined for a held one.
type steerAppend struct {
	Text        string
	Class       SteerClass
	Query       bool
	Taint       abi.TaintLabel
	Quarantined bool
	Reason      abi.ReasonCode // set iff Quarantined — why the screen held the bytes
}

// steerScreenFunc screens an append's bytes before it is staged. It returns the reason the
// bytes were HELD and held=true when the append is poisonous (must be quarantined),
// mirroring ctxmmu.ScreenBytes — the default screen. Injectable so a test can force a
// deterministic held verdict without crafting a real poison payload.
type steerScreenFunc func(body []byte) (abi.ReasonCode, bool)

const (
	steerQuarantineJournalSchema = "fak.steer.quarantine.v1"
	steerQuarantineJournalCap    = 1024
	// steerBusPendingCap bounds the staged queue so a stalled consumer cannot grow it without
	// limit (the same unbounded-accumulation class the a2a task store guards). The loop drains
	// every turn, so a healthy bus never approaches it; at the cap the OLDEST append is dropped
	// (FIFO backpressure — the budget-exhaustion sibling tracked by #2021).
	steerBusPendingCap = 4096
)

// steerClassBus stages classified appends between the /steer producer and the loop's turn
// boundary. An append is screened + taint-stamped + (on poison) journaled as a quarantine
// stub BEFORE it is staged. Scheduling is by class + query bit: a non-querying append and a
// SteerLater append schedule ZERO planner calls; a querying SteerNow/SteerNext append
// schedules exactly one. Drains are ordered so a SteerNow interrupt leads. Concurrency-safe.
type steerClassBus struct {
	mu           sync.Mutex
	screen       steerScreenFunc
	pending      []steerAppend
	plannerCalls int
	journal      *steerQuarantineJournal
}

// newSteerClassBus builds a bus with the given screen; a nil screen defaults to the live
// ctxmmu.ScreenBytes, so production wiring gets the real context screen for free.
func newSteerClassBus(screen steerScreenFunc) *steerClassBus {
	if screen == nil {
		screen = ctxmmu.ScreenBytes
	}
	return &steerClassBus{screen: screen, journal: &steerQuarantineJournal{}}
}

// Append screens, taint-stamps, and stages one classified datum, journaling a quarantine
// stub when the screen holds it. It returns the staged append (its Text is a stub when
// held). Scheduling: an admitted querying SteerNow/SteerNext append schedules one planner
// call; a non-querying append, a SteerLater append, and a held append schedule none — this
// is the decoupling of "context arrived" from "model turn spent".
func (b *steerClassBus) Append(text string, class SteerClass, query bool) steerAppend {
	b.mu.Lock()
	defer b.mu.Unlock()
	ap := steerAppend{Class: class, Query: query, Taint: abi.TaintTainted}
	if reason, held := b.screen([]byte(text)); held {
		ap.Quarantined = true
		ap.Taint = abi.TaintQuarantined
		ap.Reason = reason
		ap.Text = steerQuarantineStub(abi.ReasonName(reason))
		b.journal.record(text, reason)
	} else {
		ap.Text = text
	}
	// A planner (model turn) is scheduled only for an admitted, querying, now/next append.
	// A non-querying append is pure context arrival; a later append waits for idle; a held
	// append never buys a turn.
	if query && !ap.Quarantined && (class == SteerNow || class == SteerNext) {
		b.plannerCalls++
	}
	b.pending = append(b.pending, ap)
	if len(b.pending) > steerBusPendingCap {
		b.pending = b.pending[len(b.pending)-steerBusPendingCap:]
	}
	return ap
}

// PlannerCalls reports how many model turns the appends staged so far have scheduled. A
// stream of non-querying / later appends leaves this at zero — the affordability witness.
func (b *steerClassBus) PlannerCalls() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.plannerCalls
}

// TakeInterrupts drains and returns the SteerNow appends staged so far, in arrival order,
// leaving non-now appends staged. The loop calls this at a pre-tool-dispatch checkpoint so
// a now interrupt lands BEFORE the next tool dispatch rather than waiting for the turn's
// folded input.
func (b *steerClassBus) TakeInterrupts() []steerAppend {
	b.mu.Lock()
	defer b.mu.Unlock()
	var now, rest []steerAppend
	for _, ap := range b.pending {
		if ap.Class == SteerNow {
			now = append(now, ap)
		} else {
			rest = append(rest, ap)
		}
	}
	b.pending = rest
	return now
}

// DrainQueryingTurn drains ALL staged appends and returns them folded into the next
// querying turn's input, ordered now → next → later so an un-taken interrupt still leads.
// Each returned append carries its taint label. A non-querying append that was staged still
// appears here verbatim — it just never forced the turn on its own.
func (b *steerClassBus) DrainQueryingTurn() []steerAppend {
	b.mu.Lock()
	defer b.mu.Unlock()
	drained := b.pending
	b.pending = nil
	sort.SliceStable(drained, func(i, j int) bool {
		return steerClassRank(drained[i].Class) < steerClassRank(drained[j].Class)
	})
	return drained
}

// steerClassRank orders classes for a folded drain: now leads, later trails.
func steerClassRank(c SteerClass) int {
	switch c {
	case SteerNow:
		return 0
	case SteerLater:
		return 2
	default:
		return 1
	}
}

// steerQuarantineStub is the placeholder that stands in for a held append's bytes. The raw
// poisonous bytes never enter the model-visible transcript or the journal; only this stub
// and the reason name do.
func steerQuarantineStub(reason string) string {
	return "[steer quarantined: " + reason + "]"
}

// steerQuarantineRecord is one hash-chained journal row witnessing a held append. It stores
// the reason name, the stub, and the LENGTH of the held input — never the raw poisonous
// bytes — so the witness is leak-safe.
type steerQuarantineRecord struct {
	Schema   string `json:"schema"`
	Seq      uint64 `json:"seq"`
	Reason   string `json:"reason"`
	StubText string `json:"stub_text"`
	Bytes    int    `json:"bytes"`
	PrevHash string `json:"prev_hash"`
	Hash     string `json:"hash"`
}

// steerQuarantineJournal is the append-only, hash-chained witness that a poisonous steer
// append was held out of the loop. Bounded at steerQuarantineJournalCap (drop-oldest).
type steerQuarantineJournal struct {
	mu       sync.Mutex
	records  []steerQuarantineRecord
	seq      uint64
	lastHash string
}

// record appends one witness row for a held append and returns it. rawText is used ONLY for
// its length; its bytes are never stored.
func (j *steerQuarantineJournal) record(rawText string, code abi.ReasonCode) steerQuarantineRecord {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.seq++
	reason := abi.ReasonName(code)
	rec := steerQuarantineRecord{
		Schema:   steerQuarantineJournalSchema,
		Seq:      j.seq,
		Reason:   reason,
		StubText: steerQuarantineStub(reason),
		Bytes:    len(rawText),
		PrevHash: j.lastHash,
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s|%d|%s|%d|%s", rec.Schema, rec.Seq, rec.Reason, rec.Bytes, rec.PrevHash)))
	rec.Hash = hex.EncodeToString(sum[:])
	j.lastHash = rec.Hash
	j.records = append(j.records, rec)
	if len(j.records) > steerQuarantineJournalCap {
		j.records = j.records[len(j.records)-steerQuarantineJournalCap:]
	}
	return rec
}

// snapshot returns a copy of the journal rows for witnessing.
func (j *steerQuarantineJournal) snapshot() []steerQuarantineRecord {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]steerQuarantineRecord, len(j.records))
	copy(out, j.records)
	return out
}

// len reports the number of journal rows.
func (j *steerQuarantineJournal) len() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	return len(j.records)
}

// screenSteerText runs an inbound steer append through the SAME context screen the
// result-admission path uses (ctxmmu.ScreenBytes) and returns the stable refusal name and
// held=true when the bytes are poisonous. The /steer route refuses a held append at ingress
// so an observer feed can never inject unscreened bytes into the loop (#2402).
func screenSteerText(text string) (reason string, held bool) {
	code, held := ctxmmu.ScreenBytes([]byte(text))
	if !held {
		return "", false
	}
	return abi.ReasonName(code), true
}
