package gateway

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// debezium.go — a Debezium-compatible envelope projection over fak's two change
// feeds (#3171): the live coherence bus (CoherenceEvent, GET /v1/fak/changes)
// and the durable hash-chained audit journal (journal.Row, GET /v1/fak/events).
// Each event projects onto the canonical Debezium change-event value shape
// (op / ts_ms / before / after / source), so an existing Debezium sink can read
// fak's changelog with no fak-native decoder. This is the "expose as a view"
// answer: a PURE mapper, no route wiring — see docs/explainers/change-data-
// capture-for-agents.md.
//
// Honesty is the load-bearing constraint (docs/explainers/what-fak-is-not.md):
//
//   - No relational row image. fak captures semantic facts, not tuples, so a
//     payload is the typed event's own fields — never a fabricated before/after
//     row. Following Debezium's own semantics, that typed payload rides `after`
//     for a create/update and `before` for a delete; the unused side is null.
//   - No insert/update distinction. A coherence mutation is an invalidation of
//     existing derived state, so it maps to op="u" (the honest default), never a
//     guessed op="c". A revocation is a tombstone: op="d". A durable journal row
//     is an append to an immutable ledger — a genuine insert — so op="c".
//   - ts_ms is sourced, never invented. The live coherence bus orders by Seq and
//     does NOT wall-clock stamp its events, so ToDebezium emits ts_ms=0. The
//     journal DOES stamp (Row.TSUnixNano), so RowToDebezium carries a real ts_ms.

// Debezium `op` codes (the change operation). fak uses a deliberately small
// subset: create (a new immutable ledger row), update (a cache invalidation),
// and delete (a revocation/tombstone). Debezium's "r" (snapshot read) is not
// emitted — fak has no snapshot phase; a lapsed consumer re-syncs from the feed.
const (
	debeziumConnector = "fak"

	dbzOpCreate = "c"
	dbzOpUpdate = "u"
	dbzOpDelete = "d"

	// Logical source names — the feed an event was drained from, carried in
	// source.name so a sink can route the two feeds to separate topics.
	dbzSourceChanges = "fak.changes" // the live coherence bus
	dbzSourceEvents  = "fak.events"  // the durable audit journal
)

// DebeziumSource is the Debezium `source` metadata block. It carries the change's
// log position (as Debezium's lsn) and fak's dual clocks, mirroring the fields a
// Debezium sink expects while remaining honest about what fak actually knows.
type DebeziumSource struct {
	Connector  string `json:"connector"`           // always "fak"
	Name       string `json:"name"`                // logical source: the feed the event came from
	TSMs       int64  `json:"ts_ms"`               // wall-clock at the change; 0 when the feed does not stamp
	LSN        uint64 `json:"lsn"`                 // the drain cursor (Seq) as the log position
	WorldVer   uint64 `json:"world_ver"`           // fak consistency clock at the event
	TrustEpoch uint64 `json:"trust_epoch"`         // fak integrity clock at the event
	Principal  string `json:"principal,omitempty"` // isolation principal that produced a mutation
}

// DebeziumEnvelope is the canonical Debezium change-event value. `before` and
// `after` are raw JSON so an absent side is a literal `null` (not `{}`), exactly
// as a Debezium sink expects; the typed fak payload occupies whichever side the
// op implies (after for c/u, before for d).
type DebeziumEnvelope struct {
	Op     string          `json:"op"`
	TSMs   int64           `json:"ts_ms"`
	Before json.RawMessage `json:"before"`
	After  json.RawMessage `json:"after"`
	Source DebeziumSource  `json:"source"`
}

// dbzChangePayload is the typed coherence-event "row": the semantic content of a
// mutation or revocation. Log position and clocks live in DebeziumSource, not
// here, so the payload is the change itself with no metadata duplication.
type dbzChangePayload struct {
	Kind    string   `json:"kind"`              // "mutation" | "revocation"
	Tool    string   `json:"tool,omitempty"`    // mutation: the write-shaped tool
	Tags    []string `json:"tags,omitempty"`    // mutation: the invalidation scope
	Witness string   `json:"witness,omitempty"` // revocation: the refuted witness (the delete key)
	Evicted int      `json:"evicted,omitempty"` // revocation: pooled entries stranded
}

// ToDebezium projects a coherence-bus event onto a Debezium change envelope.
// A mutation is op="u" with the payload in `after`; a revocation is op="d"
// (tombstone) with the payload — the refuted witness, the real delete key — in
// `before`, and `after` null. ts_ms is 0: the coherence bus is Seq-ordered only.
func ToDebezium(ev CoherenceEvent) DebeziumEnvelope {
	payload := mustMarshal(dbzChangePayload{
		Kind:    ev.Kind,
		Tool:    ev.Tool,
		Tags:    ev.Tags,
		Witness: ev.Witness,
		Evicted: ev.Evicted,
	})

	env := DebeziumEnvelope{
		TSMs: 0, // the live coherence bus does not wall-clock stamp; sourced only from the journal
		Source: DebeziumSource{
			Connector:  debeziumConnector,
			Name:       dbzSourceChanges,
			TSMs:       0,
			LSN:        ev.Seq,
			WorldVer:   ev.WorldVer,
			TrustEpoch: ev.TrustEpoch,
			Principal:  ev.principal,
		},
	}

	if ev.Kind == "revocation" {
		// A tombstone: the revoked entry (keyed by its witness) is what existed and
		// is now gone — it rides `before`, per Debezium delete semantics. We hold no
		// prior row image, so `after` is null, never fabricated.
		env.Op = dbzOpDelete
		env.Before = payload
		env.After = jsonNull
		return env
	}

	// A mutation invalidates existing derived state: op="u", payload in `after`.
	// `before` is null — fak has no prior tuple, only the new invalidation scope.
	env.Op = dbzOpUpdate
	env.Before = jsonNull
	env.After = payload
	return env
}

// dbzRowPayload is the typed journal "row": the durable audit record's decision
// fields. The hash + prev_hash carry the tamper-evident chain identity so a sink
// can verify the ledger it replicates.
type dbzRowPayload struct {
	Kind     string `json:"kind"`               // DECIDE | DENY | QUARANTINE | ...
	Tool     string `json:"tool,omitempty"`     //
	TraceID  string `json:"trace_id,omitempty"` // session correlation
	Verdict  string `json:"verdict,omitempty"`  //
	Reason   string `json:"reason,omitempty"`   // closed-vocabulary refusal class
	Witness  string `json:"witness,omitempty"`  // bounded-disclosure claim the verdict surfaced
	Hash     string `json:"hash"`               // chainHash of this row
	PrevHash string `json:"prev_hash"`          // chain link to the previous row
}

// RowToDebezium projects a durable audit-journal row onto a Debezium change
// envelope. Every row is an append to an immutable hash-chained ledger — a
// genuine insert — so op="c" with the record in `after` and `before` null. ts_ms
// is real here (Row.TSUnixNano, ns → ms); lsn is the row's chain Seq.
func RowToDebezium(r journal.Row) DebeziumEnvelope {
	payload := mustMarshal(dbzRowPayload{
		Kind:     r.Kind,
		Tool:     r.Tool,
		TraceID:  r.TraceID,
		Verdict:  r.Verdict,
		Reason:   r.Reason,
		Witness:  r.Witness,
		Hash:     r.Hash,
		PrevHash: r.PrevHash,
	})
	return DebeziumEnvelope{
		Op:     dbzOpCreate,
		TSMs:   r.TSUnixNano / 1_000_000, // ns → ms, the Debezium ts_ms unit
		Before: jsonNull,
		After:  payload,
		Source: DebeziumSource{
			Connector:  debeziumConnector,
			Name:       dbzSourceEvents,
			TSMs:       r.TSUnixNano / 1_000_000,
			LSN:        r.Seq,
			WorldVer:   0, // the journal row carries no world clock; 0 is honest, not inferred
			TrustEpoch: 0,
		},
	}
}

// jsonNull is the shared literal `null` for the unused before/after side.
var jsonNull = json.RawMessage("null")

// mustMarshal serializes a payload struct whose fields are all JSON-safe scalars
// and string slices; such a value cannot fail to marshal, so an error here is a
// programmer bug and we fall back to `null` rather than panicking on the wire.
func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return jsonNull
	}
	return b
}
