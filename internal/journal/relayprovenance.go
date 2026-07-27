package journal

// Relayed-message provenance — the chain-anchored witness for a message that
// crossed a PLATFORM BOUNDARY (#2851, Track D of #2834).
//
// WHAT THIS CLOSES. A hosted relay of the Hermes shape fronts many platforms over a
// per-turn bidirectional WebSocket and authenticates each connector with an HMAC.
// That HMAC proves WHO OPENED THE SOCKET; it says nothing verifiable about an
// individual MESSAGE. Which platform carried it, which end-user sent it, which turn
// of which session it belongs to, and whether the delivery floor allowed it are all
// transport metadata — a self-report of the connector, discarded when the socket
// closes. So a conversation that spans Telegram -> Slack -> CLI has an authenticated
// pipe but no audit trail: nothing an operator can re-derive after the fact, and
// nothing that would notice if a row were edited.
//
// The fix is to spend the journal that already exists. Every relayed message — in
// BOTH directions — becomes one genuine chained row (KindRelayMsg) in the same
// tamper-evident ledger that carries DECIDE / DENY / RESTART_HOP, so the record
// inherits the hash chain rather than inventing a second one. The load-bearing
// identity rides the CHAINED decision fields (see chainHash's explicit field list):
//
//	Kind       KindRelayMsg
//	Tool       the platform token ("telegram", "slack", "cli", ...)
//	TraceID    the SESSION KEY — the cross-platform correlation anchor
//	Verdict    RelayAllow | RelayDeny — the adjudication verdict, not a claim
//	Reason     the refusal class on a deny (closed vocabulary at the call site)
//	By         "relay-inbound" | "relay-outbound" — the direction
//	ArgsDigest the body digest — content-addressed, never the body itself
//
// Everything else (user id, turn id, destination, redaction count, and the
// per-session predecessor link) rides the non-chained Relay payload field, the same
// layering RestartHop uses. That split is deliberate and it is what makes the
// per-session trail verifiable rather than merely recorded: the payload's PrevSeq /
// PrevHash NAME a predecessor row, and VerifyRelayTrail re-checks that name against
// the actual chained Hash of the row at that Seq. A forged correlation link
// therefore has to break the journal chain to survive verification — the payload
// cannot lie about its own history unaided.
//
// WHY A SEPARATE HEAD INDEX. RelayChain, not Journal, tracks the per-session-key
// head. The journal interleaves every session and every row kind, so "the previous
// message of THIS conversation" is not "the previous row"; it needs its own index.
// Keeping that index beside the journal rather than inside it means a journal that
// never relays anything pays nothing, and this rung adds no field to Journal.
//
// This leaf stays platform-agnostic on purpose: Platform and Reason are plain
// strings, so the delivery seam (internal/egressfloor's adjudicated
// platform-delivery floor, #2850) supplies its own vocabulary and internal/journal
// never imports upward. It changes no relay wire format — it only records what
// crossed.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"
)

// KindRelayMsg marks a relayed-message provenance row. It is a genuine chained row
// (it consumes the next Seq and chains onto the prior head) that carries an
// adjudication verdict but is not a per-call kernel decision, so decision-folding
// consumers that key on DECIDE/DENY skip it; readers that key on Kind (the
// RelayProvenanceFor read-path) fold it into a session's cross-platform trail.
const KindRelayMsg = "RELAY_MSG"

// RelayProvenanceSchema names the correlated per-message record carried on a
// RELAY_MSG row's Relay field. Versioned like every fak wire schema: additive-only;
// never edit a shipped /vN in place.
const RelayProvenanceSchema = "fak.gateway.relay_provenance.v1"

// Relay directions — the CLOSED vocabulary for RelayProvenance.Direction, mirrored
// onto the chained By field as "relay-<direction>". Both halves are recorded: an
// audit trail that witnessed only what the agent SENT would miss what it was TOLD.
const (
	// RelayInbound is a message that arrived from a platform and entered the session.
	RelayInbound = "inbound"
	// RelayOutbound is a message the session emitted toward a platform.
	RelayOutbound = "outbound"
)

// Relay adjudication verdicts — the CLOSED vocabulary for RelayProvenance.Verdict,
// mirrored onto the chained Verdict field. A denied message is still witnessed:
// "the floor refused this" is exactly the fact an audit needs most.
const (
	RelayAllow = "allow"
	RelayDeny  = "deny"
)

// RelayProvenance is the correlated per-message record (RelayProvenanceSchema):
// every axis of one relayed message tied together in a single value, instead of
// scattered across socket frames that vanish with the connection.
type RelayProvenance struct {
	Schema    string `json:"schema"`    // RelayProvenanceSchema
	Direction string `json:"direction"` // RelayInbound | RelayOutbound
	Platform  string `json:"platform"`  // "telegram" | "slack" | "cli" | ... (caller's vocabulary)

	// UserID is the end-user identity WITHIN the platform (a platform user id), not a
	// fak identity — it is what ties a message to a person across a platform hop.
	UserID string `json:"user_id,omitempty"`
	// SessionKey is the cross-platform correlation anchor: the one value that stays
	// constant while the conversation moves Telegram -> Slack -> CLI. It is mirrored
	// onto the chained TraceID field.
	SessionKey string `json:"session_key"`
	// TurnID identifies the conversational turn this message belongs to, so an audit
	// can tell a reply from a fresh turn without re-deriving it from timestamps.
	TurnID string `json:"turn_id,omitempty"`
	// Destination is the target WITHIN the platform for an outbound message (a
	// chat/channel/user id); empty for inbound.
	Destination string `json:"destination,omitempty"`

	// Verdict is the adjudication outcome the delivery floor returned (RelayAllow |
	// RelayDeny); Reason is its refusal class on a deny (the caller's closed
	// vocabulary, e.g. egressfloor's DELIVERY_BLOCK). Both are mirrored onto chained
	// fields so the verdict is tamper-evident, not a payload claim.
	Verdict string `json:"verdict"`
	Reason  string `json:"reason,omitempty"`

	// BodyDigest is the content address of the message body (RelayBodyDigest), never
	// the body: the journal witnesses THAT a specific message crossed without
	// becoming a transcript of everything anyone ever typed. Mirrored onto the chained
	// ArgsDigest field.
	BodyDigest string `json:"body_digest,omitempty"`
	// Redactions is how many secret spans the pre-send redactor replaced, so an
	// auditor can see a message left redacted without seeing what was redacted.
	Redactions int `json:"redactions,omitempty"`

	// PrevSeq / PrevHash link this message to the previous relayed message of the SAME
	// SessionKey — the per-session chain that survives a platform hop. Both are zero /
	// empty on the first message of a session (genesis). They NAME a predecessor;
	// VerifyRelayTrail re-checks the name against the row actually committed at that
	// Seq, so the link cannot be forged without breaking the journal's own chain.
	PrevSeq  uint64 `json:"prev_seq,omitempty"`
	PrevHash string `json:"prev_hash,omitempty"`
}

// RelayBodyDigest content-addresses a message body as "sha256:<hex>". An empty body
// digests to "" rather than to the digest of the empty string, so an absent body is
// distinguishable from a present-but-empty one in the witness.
func RelayBodyDigest(body string) string {
	if body == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(body))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// RelayChain is the per-session-key head index that turns a stream of interleaved
// journal rows into one linked trail per conversation. It is safe for concurrent use
// (a relay fronting several platforms services them from different goroutines).
//
// The zero value is not usable; construct with NewRelayChain.
type RelayChain struct {
	mu   sync.Mutex
	head map[string]relayLink
}

// relayLink is one session's current head: the Seq and chained Hash of the most
// recent RELAY_MSG row recorded for that session key.
type relayLink struct {
	seq  uint64
	hash string
}

// NewRelayChain returns an empty per-session head index.
func NewRelayChain() *RelayChain {
	return &RelayChain{head: map[string]relayLink{}}
}

// Append records one relayed message as a durable, chained RELAY_MSG row and returns
// the committed row (with its stamped Seq/hash). It fills the per-session
// PrevSeq/PrevHash from this chain's head for p.SessionKey, mirrors the record's
// load-bearing identity onto the chained decision fields, commits the row, and
// advances the head.
//
// It is a no-op returning the zero Row on a nil receiver or a nil journal, so a
// caller that guarded the journal on may call it unconditionally. p.Schema defaults
// to RelayProvenanceSchema when unset.
//
// Like AppendRestartHop, the row is written directly through the chain (not the ABI
// fan-out): relaying a message is transport, not a kernel decision, and routing a
// synthetic event through the frozen ABI would fan it out to every decision-stream
// folder that assumes an event IS an adjudication. The write is flushed per row by
// append, so the record survives the process that wrote it.
func (c *RelayChain) Append(j *Journal, p RelayProvenance) Row {
	if c == nil || j == nil {
		return Row{}
	}
	if p.Schema == "" {
		p.Schema = RelayProvenanceSchema
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	prev := c.head[p.SessionKey]
	p.PrevSeq, p.PrevHash = prev.seq, prev.hash

	row := Row{
		Kind:       KindRelayMsg,
		Tool:       p.Platform,
		TraceID:    p.SessionKey,
		Verdict:    p.Verdict,
		Reason:     p.Reason,
		By:         "relay-" + p.Direction,
		ArgsDigest: p.BodyDigest,
		Relay:      &p,
	}

	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()

	c.head[p.SessionKey] = relayLink{seq: committed.Seq, hash: committed.Hash}
	return committed
}

// Head returns the Seq and chained Hash of the most recent relayed message recorded
// for sessionKey, and whether any exists. It is the cheap "where is this
// conversation now" probe, for a caller that does not want to re-fold the journal.
func (c *RelayChain) Head(sessionKey string) (seq uint64, hash string, ok bool) {
	if c == nil {
		return 0, "", false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	link, ok := c.head[sessionKey]
	return link.seq, link.hash, ok
}

// RelayEntry is one relayed message as the read-path returns it: the correlated
// record together with the chain coordinates of the row that carries it. Those
// coordinates are what make the entry checkable — Hash is the journal's own chained
// hash, not something the record asserted about itself.
type RelayEntry struct {
	Seq        uint64          // the row's order anchor in the journal
	TSUnixNano int64           // the row's wall-clock anchor
	PrevHash   string          // the JOURNAL-chain predecessor (any kind of row)
	Hash       string          // this row's chained hash
	Record     RelayProvenance // the correlated per-message record
}

// RelayProvenanceFor is the `gateway provenance` read-path: it folds a journal's rows
// into the ordered relayed-message trail for ONE session key. Rows of any other kind,
// rows for other sessions, and RELAY_MSG rows with no payload are skipped; the result
// is ordered by Seq, which is the journal's own commit order, so a trail that hopped
// Telegram -> Slack -> CLI reads back in the order it actually happened.
//
// It returns nil (not an error) for a session with no relayed messages: "this
// conversation never crossed a platform boundary" is a valid, empty answer.
func RelayProvenanceFor(rows []Row, sessionKey string) []RelayEntry {
	var out []RelayEntry
	for _, r := range rows {
		if r.Kind != KindRelayMsg || r.Relay == nil {
			continue
		}
		// Trust the CHAINED TraceID for session identity, not the payload's copy: the
		// chained field is what the hash covers.
		if r.TraceID != sessionKey {
			continue
		}
		out = append(out, RelayEntry{
			Seq:        r.Seq,
			TSUnixNano: r.TSUnixNano,
			PrevHash:   r.PrevHash,
			Hash:       r.Hash,
			Record:     *r.Relay,
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out
}

// RelayPlatforms returns the DISTINCT platforms a trail crossed, in first-appearance
// order — the one-line answer to "did this conversation actually span platforms?"
// (a trail whose result has length > 1 is a cross-platform conversation).
func RelayPlatforms(trail []RelayEntry) []string {
	seen := map[string]bool{}
	var out []string
	for _, e := range trail {
		if p := e.Record.Platform; p != "" && !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

// VerifyRelayTrail checks that a trail returned by RelayProvenanceFor is genuinely
// chain-linked, against rows as the authority. It is the audit half of the read-path:
// reading a trail proves nothing on its own, because the per-session PrevSeq/PrevHash
// ride the non-chained payload. This closes that gap by re-deriving every link from
// the journal itself:
//
//  1. every entry's own Hash equals chainHash(its PrevHash, its row) — the row was
//     not edited after commit;
//  2. the first entry is genesis for this session (PrevSeq 0, PrevHash "");
//  3. every later entry's PrevSeq/PrevHash equal the PRECEDING entry's Seq/Hash; and
//  4. that named predecessor hash equals the hash the journal actually committed at
//     that Seq — so a rewritten link is caught even if the two payloads agree.
//
// It returns nil for an empty trail (nothing to contradict) and a descriptive error
// naming the first broken link otherwise.
func VerifyRelayTrail(rows []Row, trail []RelayEntry) error {
	if len(trail) == 0 {
		return nil
	}
	// The journal is the authority for what was committed at a given Seq.
	bySeq := make(map[uint64]Row, len(rows))
	for _, r := range rows {
		bySeq[r.Seq] = r
	}

	for i, e := range trail {
		row, ok := bySeq[e.Seq]
		if !ok {
			return fmt.Errorf("relay trail: entry %d cites seq %d, absent from the journal", i, e.Seq)
		}
		if want := chainHash(row.PrevHash, row); want != row.Hash {
			return fmt.Errorf("relay trail: entry %d (seq %d): row hash %s does not re-derive (want %s)",
				i, e.Seq, row.Hash, want)
		}
		if row.Hash != e.Hash {
			return fmt.Errorf("relay trail: entry %d (seq %d): trail hash %s disagrees with the journal's %s",
				i, e.Seq, e.Hash, row.Hash)
		}

		if i == 0 {
			if e.Record.PrevSeq != 0 || e.Record.PrevHash != "" {
				return fmt.Errorf("relay trail: first entry (seq %d) is not genesis: prev_seq=%d prev_hash=%q",
					e.Seq, e.Record.PrevSeq, e.Record.PrevHash)
			}
			continue
		}

		prev := trail[i-1]
		if e.Record.PrevSeq != prev.Seq {
			return fmt.Errorf("relay trail: entry %d (seq %d): prev_seq=%d, want %d",
				i, e.Seq, e.Record.PrevSeq, prev.Seq)
		}
		if e.Record.PrevHash != prev.Hash {
			return fmt.Errorf("relay trail: entry %d (seq %d): prev_hash=%s, want %s",
				i, e.Seq, e.Record.PrevHash, prev.Hash)
		}
		// The link must also match what the JOURNAL committed at that seq, so a trail
		// whose two payloads were rewritten in agreement is still caught.
		if committed, ok := bySeq[e.Record.PrevSeq]; !ok {
			return fmt.Errorf("relay trail: entry %d (seq %d): prev_seq %d absent from the journal",
				i, e.Seq, e.Record.PrevSeq)
		} else if committed.Hash != e.Record.PrevHash {
			return fmt.Errorf("relay trail: entry %d (seq %d): prev_hash=%s, journal committed %s at seq %d",
				i, e.Seq, e.Record.PrevHash, committed.Hash, e.Record.PrevSeq)
		}
	}
	return nil
}
