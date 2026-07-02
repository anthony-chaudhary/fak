package sessiondesc

import (
	"fmt"
	"sort"
)

// SchemaVersion is the wire schema tag every Descriptor carries. Version bumps
// are additive-only within v1; a reader seeing an unknown schema string must
// treat the record as forward-incompatible and skip it (the same
// skip-don't-blind rule the leaseref readers apply to unparseable blobs).
const SchemaVersion = "fak.session.descriptor.v1"

// Presence says whether one key space is bound on a descriptor, and when it is
// not, WHY — a closed vocabulary, so absence is a typed answer a surface can
// render (and a test can pin) rather than a nil a reader guesses about.
type Presence string

const (
	// Bound: the key space is populated from an observed source row.
	Bound Presence = "BOUND"
	// AbsentNotObserved: the caller never consulted this source (e.g. a fold
	// built only from a gateway snapshot, with no leaseref scan requested).
	AbsentNotObserved Presence = "ABSENT_NOT_OBSERVED"
	// AbsentSourceUnavailable: the source was consulted and FAILED (gateway
	// down, git not executable). Distinct from a clean miss so an outage is
	// never read as "no such session".
	AbsentSourceUnavailable Presence = "ABSENT_SOURCE_UNAVAILABLE"
	// AbsentNoBinding: the source answered and held no row for this id — a
	// clean, observed miss.
	AbsentNoBinding Presence = "ABSENT_NO_BINDING"
)

// SourceStatus is the caller's statement about one source feeding a Fold:
// observed cleanly, consulted but unavailable, or never consulted. The fold
// maps it onto the per-descriptor Presence for every id that source did not
// bind.
type SourceStatus string

const (
	SourceObserved     SourceStatus = "OBSERVED"
	SourceUnavailable  SourceStatus = "UNAVAILABLE"
	SourceNotConsulted SourceStatus = "NOT_CONSULTED"
)

// presenceForMiss translates a source's status into the Presence a descriptor
// carries when that source bound nothing for its id.
func presenceForMiss(s SourceStatus) Presence {
	switch s {
	case SourceObserved:
		return AbsentNoBinding
	case SourceUnavailable:
		return AbsentSourceUnavailable
	default:
		return AbsentNotObserved
	}
}

// DriveRow mirrors the identity slice of the gateway's per-session DRIVE state
// (gateway.SessionState): the trace id that keys /v1/fak/sessions and the JSON
// --log sink, the run state, the lineage fields, and the optimistic-concurrency
// rev. A mirror type, not an import — see the package doc's layering rule.
type DriveRow struct {
	TraceID        string `json:"trace_id"`
	Run            string `json:"run,omitempty"`
	ContinuationID string `json:"continuation_id,omitempty"`
	ParentTrace    string `json:"parent_trace,omitempty"`
	Generation     int    `json:"generation,omitempty"`
	Rev            uint64 `json:"rev,omitempty"`
}

// RefRow mirrors the identity slice of the cross-host leaseref session
// descriptor (leaseref.SessionDescriptor at refs/fak/locks/session-<id>).
type RefRow struct {
	ID        string `json:"id"`
	Host      string `json:"host,omitempty"`
	PCBState  string `json:"pcb_state,omitempty"`
	UpdatedAt int64  `json:"updated_at,omitempty"`
	TTLSecs   int64  `json:"ttl_seconds,omitempty"`
}

// HarnessRow binds a session id to the agent + account identity that runs it
// (harnessprofile Name + fleetaccounts rotation identity). The binding itself
// is produced by the caller (the guard knows which child it launched; the
// census derives it from the transcript namespace) — this package only joins.
type HarnessRow struct {
	SessionID string `json:"session_id"`
	Agent     string `json:"agent,omitempty"`    // harnessprofile name: claude / codex / opencode / aider / hermes
	Identity  string `json:"identity,omitempty"` // account / rotation identity bucket
}

// DriveKey is the gateway key space as carried on a Descriptor.
type DriveKey struct {
	Presence       Presence `json:"presence"`
	TraceID        string   `json:"trace_id,omitempty"`
	Run            string   `json:"run,omitempty"`
	ContinuationID string   `json:"continuation_id,omitempty"`
	ParentTrace    string   `json:"parent_trace,omitempty"`
	Generation     int      `json:"generation,omitempty"`
	Rev            uint64   `json:"rev,omitempty"`
}

// RefKey is the leaseref key space as carried on a Descriptor.
type RefKey struct {
	Presence  Presence `json:"presence"`
	ID        string   `json:"id,omitempty"`
	Host      string   `json:"host,omitempty"`
	PCBState  string   `json:"pcb_state,omitempty"`
	UpdatedAt int64    `json:"updated_at,omitempty"`
	TTLSecs   int64    `json:"ttl_seconds,omitempty"`
}

// HarnessKey is the harness-identity key space as carried on a Descriptor.
type HarnessKey struct {
	Presence Presence `json:"presence"`
	Agent    string   `json:"agent,omitempty"`
	Identity string   `json:"identity,omitempty"`
}

// CensusKey is the transcript/census key space (#2213). Reserved in v1: the
// fold never binds it, but the slot exists so the census lands as an additive
// change, not a schema bump.
type CensusKey struct {
	Presence Presence `json:"presence"`
}

// Descriptor is one session across all four key spaces. ID is the canonical
// session id every bound space agrees on (the exact-join key). A Descriptor
// deliberately carries NO progress/claim fields — identity and observation
// pointers only (package doc, rule 4).
type Descriptor struct {
	Schema  string     `json:"schema"`
	ID      string     `json:"id"`
	Drive   DriveKey   `json:"drive"`
	Ref     RefKey     `json:"ref"`
	Harness HarnessKey `json:"harness"`
	Census  CensusKey  `json:"census"`
}

// Sources is everything a caller hands Fold: per-source rows plus the honest
// status of each source. Rows are already parsed — the caller owns I/O.
type Sources struct {
	DriveStatus   SourceStatus
	Drive         []DriveRow
	RefStatus     SourceStatus
	Refs          []RefRow
	HarnessStatus SourceStatus
	Harness       []HarnessRow
}

// Fold joins the sources into one Descriptor per distinct session id, sorted
// by id for a stable view. The join is EXACT: a drive row, a ref row, and a
// harness row meet on one descriptor iff their ids are byte-equal. Rows with
// an empty id are rejected (an unidentifiable row folded into "" would merge
// unrelated sessions — the exact failure this schema exists to prevent).
func Fold(src Sources) ([]Descriptor, error) {
	byID := map[string]*Descriptor{}
	get := func(id string) *Descriptor {
		d, ok := byID[id]
		if !ok {
			d = &Descriptor{
				Schema:  SchemaVersion,
				ID:      id,
				Drive:   DriveKey{Presence: presenceForMiss(src.DriveStatus)},
				Ref:     RefKey{Presence: presenceForMiss(src.RefStatus)},
				Harness: HarnessKey{Presence: presenceForMiss(src.HarnessStatus)},
				Census:  CensusKey{Presence: AbsentNotObserved},
			}
			byID[id] = d
		}
		return d
	}

	for i, r := range src.Drive {
		if r.TraceID == "" {
			return nil, fmt.Errorf("sessiondesc: drive row %d has empty trace_id", i)
		}
		d := get(r.TraceID)
		if d.Drive.Presence == Bound {
			return nil, fmt.Errorf("sessiondesc: duplicate drive row for id %q", r.TraceID)
		}
		d.Drive = DriveKey{
			Presence:       Bound,
			TraceID:        r.TraceID,
			Run:            r.Run,
			ContinuationID: r.ContinuationID,
			ParentTrace:    r.ParentTrace,
			Generation:     r.Generation,
			Rev:            r.Rev,
		}
	}

	for i, r := range src.Refs {
		if r.ID == "" {
			return nil, fmt.Errorf("sessiondesc: ref row %d has empty id", i)
		}
		d := get(r.ID)
		if d.Ref.Presence == Bound {
			return nil, fmt.Errorf("sessiondesc: duplicate ref row for id %q", r.ID)
		}
		d.Ref = RefKey{
			Presence:  Bound,
			ID:        r.ID,
			Host:      r.Host,
			PCBState:  r.PCBState,
			UpdatedAt: r.UpdatedAt,
			TTLSecs:   r.TTLSecs,
		}
	}

	for i, r := range src.Harness {
		if r.SessionID == "" {
			return nil, fmt.Errorf("sessiondesc: harness row %d has empty session_id", i)
		}
		d := get(r.SessionID)
		if d.Harness.Presence == Bound {
			return nil, fmt.Errorf("sessiondesc: duplicate harness row for id %q", r.SessionID)
		}
		d.Harness = HarnessKey{Presence: Bound, Agent: r.Agent, Identity: r.Identity}
	}

	out := make([]Descriptor, 0, len(byID))
	for _, d := range byID {
		out = append(out, *d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// BoundCount reports how many of the descriptor's key spaces are bound —
// the one-glance "how well is this session joined" number a pane sorts by.
func (d Descriptor) BoundCount() int {
	n := 0
	for _, p := range []Presence{d.Drive.Presence, d.Ref.Presence, d.Harness.Presence, d.Census.Presence} {
		if p == Bound {
			n++
		}
	}
	return n
}
