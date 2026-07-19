package escalation

// ledger.go — the durable half of fak.escalation.v1: one JSONL file carrying
// packet rows and ack rows (distinguished by their schema tag alone), plus the
// pure Fold every consumer reads through. Emit and Ack validate fail-closed
// before anything touches disk; the fold is where idempotency lives — rows are
// append-only facts, and duplicates collapse deterministically at read time,
// so a re-fired producer or a re-delivered ack can never double-open or
// double-close a packet.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Row is one decoded ledger line: exactly one of Packet / Ack is set.
type Row struct {
	Packet *Packet
	Ack    *Ack
}

// Ledger is the append-only fak.escalation.v1 store: JSONL at Path, one
// bare-object row per line. Append-only is the whole write API — there is no
// update or delete; closure is a NEW ack row, never a mutation.
type Ledger struct {
	Path string
}

// Emit validates and appends one packet row, assigning the deterministic id
// (and the schema tag) when the producer left them empty, and returns the
// packet as landed — the witnessed artifact the #4479 EmitEscalation verb
// hands back. A packet that fails Validate never reaches disk.
func (l Ledger) Emit(p Packet) (Packet, error) {
	if p.Schema == "" {
		p.Schema = Schema
	}
	if p.ID == "" {
		p.ID = p.deriveID()
	}
	if err := p.Validate(); err != nil {
		return Packet{}, err
	}
	if err := l.appendJSON(p); err != nil {
		return Packet{}, err
	}
	return p, nil
}

// Ack validates and appends one ack row. Closure semantics stay idempotent
// per (PacketID, Rev): appending the same ack again lands another row, but the
// fold collapses them to the first — an ack can never close twice.
func (l Ledger) Ack(a Ack) error {
	if a.Schema == "" {
		a.Schema = AckSchema
	}
	if err := a.Validate(); err != nil {
		return err
	}
	return l.appendJSON(a)
}

// Load reads every row. A ledger that does not exist yet is an empty ledger,
// not an error; a line that decodes to neither schema fails closed with its
// line number — a corrupt ledger is surfaced, never skipped over.
func (l Ledger) Load() ([]Row, error) {
	raw, err := os.ReadFile(l.Path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var rows []Row
	for i, line := range bytes.Split(raw, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		row, err := decodeRow(line)
		if err != nil {
			return nil, fmt.Errorf("escalation: ledger %s line %d: %w", l.Path, i+1, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// appendJSON marshals v and appends it as one line, creating the parent
// directory and the file on first use.
func (l Ledger) appendJSON(v any) error {
	body, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(l.Path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(l.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(body, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// decodeRow decodes one bare-object line by its schema tag.
func decodeRow(line []byte) (Row, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(line, &probe); err != nil {
		return Row{}, err
	}
	switch probe.Schema {
	case Schema:
		var p Packet
		if err := json.Unmarshal(line, &p); err != nil {
			return Row{}, err
		}
		return Row{Packet: &p}, nil
	case AckSchema:
		var a Ack
		if err := json.Unmarshal(line, &a); err != nil {
			return Row{}, err
		}
		return Row{Ack: &a}, nil
	default:
		return Row{}, fmt.Errorf("%w: row schema %q", ErrSchema, probe.Schema)
	}
}

// Pair is one closed escalation: the packet, the ack row that closed it, and
// the handling time between them — the exact pair R1's
// escalation_handling_p50 is computed from.
type Pair struct {
	Packet          Packet  `json:"packet"`
	Ack             Ack     `json:"ack"`
	HandlingSeconds float64 `json:"handling_seconds"`
}

// Report is the consumer projection of the ledger: what is open, what expired
// unhandled (safe default prescribed), what closed and how fast. Pure — built
// by Fold from rows the caller loaded, no I/O and no clock.
type Report struct {
	Schema       string `json:"schema"` // the packet schema this report projects
	AsOfUnixNano int64  `json:"as_of_unix_nano"`

	Open    []Packet `json:"open"`    // emitted, unacked, expiry not reached — oldest first
	Expired []Packet `json:"expired"` // emitted, unacked, PAST expiry: SafeDefault is the prescribed action — oldest first
	Acked   []Pair   `json:"acked"`   // closed by an ack row, ack-time order

	// HandlingSeconds is Acked's handling times sorted ascending — the R1
	// consumer takes the median directly from this slice.
	HandlingSeconds []float64 `json:"handling_seconds,omitempty"`

	// OrphanAcks are ack rows naming no known packet — surfaced, never dropped:
	// an ack that binds nothing is a producer/sink bug someone must see.
	OrphanAcks []Ack `json:"orphan_acks,omitempty"`
}

// Fold projects ledger rows into the Report as of asOf (zero asOf falls back
// to the newest row timestamp, keeping the fold deterministic). Idempotency
// lands here: packets collapse by ID (first row wins — the id embeds Rev, so
// a producer re-fire of the same rev is the same packet), acks collapse per
// (PacketID, Rev) and a packet closes on its EARLIEST surviving ack.
func Fold(rows []Row, asOf time.Time) Report {
	if asOf.IsZero() {
		var maxTS int64
		for _, r := range rows {
			if r.Packet != nil && r.Packet.EmittedAtUnixNano > maxTS {
				maxTS = r.Packet.EmittedAtUnixNano
			}
			if r.Ack != nil && r.Ack.AckedAtUnixNano > maxTS {
				maxTS = r.Ack.AckedAtUnixNano
			}
		}
		asOf = time.Unix(0, maxTS).UTC()
	}

	packets := map[string]Packet{}
	var order []string
	ackSeen := map[string]bool{} // packet_id "#" rev — the idempotency key
	firstAck := map[string]Ack{} // packet id -> earliest surviving ack
	var strayAcks []Ack

	for _, r := range rows {
		switch {
		case r.Packet != nil:
			p := *r.Packet
			if _, dup := packets[p.ID]; dup {
				continue // re-emit of the same (anchor, rev): one packet
			}
			packets[p.ID] = p
			order = append(order, p.ID)
		case r.Ack != nil:
			a := *r.Ack
			key := a.PacketID + "#" + utoa(a.Rev)
			if ackSeen[key] {
				continue // re-delivered ack: idempotent per Rev, first row wins
			}
			ackSeen[key] = true
			if _, known := packets[a.PacketID]; !known {
				strayAcks = append(strayAcks, a)
				continue
			}
			if prev, ok := firstAck[a.PacketID]; !ok || a.AckedAtUnixNano < prev.AckedAtUnixNano {
				firstAck[a.PacketID] = a
			}
		}
	}

	rep := Report{Schema: Schema, AsOfUnixNano: asOf.UnixNano()}

	// An ack row may precede its packet row in a concatenated ledger; re-bind
	// the strays now that every packet is known before calling one an orphan.
	for _, a := range strayAcks {
		if _, known := packets[a.PacketID]; known {
			if prev, ok := firstAck[a.PacketID]; !ok || a.AckedAtUnixNano < prev.AckedAtUnixNano {
				firstAck[a.PacketID] = a
			}
			continue
		}
		rep.OrphanAcks = append(rep.OrphanAcks, a)
	}

	for _, id := range order {
		p := packets[id]
		if a, ok := firstAck[id]; ok {
			handling := time.Duration(a.AckedAtUnixNano - p.EmittedAtUnixNano).Seconds()
			if handling < 0 {
				handling = 0 // clock skew never mints a negative handling time
			}
			rep.Acked = append(rep.Acked, Pair{Packet: p, Ack: a, HandlingSeconds: handling})
			rep.HandlingSeconds = append(rep.HandlingSeconds, handling)
			continue
		}
		if asOf.UnixNano() >= p.ExpiresAtUnixNano {
			rep.Expired = append(rep.Expired, p)
			continue
		}
		rep.Open = append(rep.Open, p)
	}

	sort.SliceStable(rep.Open, func(i, j int) bool {
		return rep.Open[i].EmittedAtUnixNano < rep.Open[j].EmittedAtUnixNano
	})
	sort.SliceStable(rep.Expired, func(i, j int) bool {
		return rep.Expired[i].EmittedAtUnixNano < rep.Expired[j].EmittedAtUnixNano
	})
	sort.SliceStable(rep.Acked, func(i, j int) bool {
		return rep.Acked[i].Ack.AckedAtUnixNano < rep.Acked[j].Ack.AckedAtUnixNano
	})
	sort.Float64s(rep.HandlingSeconds)
	return rep
}

// utoa is a stdlib-fmt-free uint64 -> string (mirrors internal/waiting).
func utoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
