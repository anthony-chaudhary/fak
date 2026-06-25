package snapshot

// codecs.go — the kind REGISTRY (the ladder, named) and the typed convenience codecs
// built on the generic envelope. The registry lets a tool answer "what can I dump?" and
// lets a restore reject an unknown kind; the typed codecs (trace, fleet) turn a live fak
// value into a Snapshot and back. The session and rsi levels are reachable through the
// same seam — session via the richer internal/sessionimage bundle, rsi via a generic
// Marshal of its journal row — so the whole ladder (turn -> tool -> session -> fleet ->
// rsi) is dumpable, with this package shipping the typed codecs for the levels whose
// state is a single JSON value.

import (
	"fmt"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Canonical kind names — the rungs of the loops ladder. An application may register its
// own kinds; these are the ones fak ships.
const (
	KindTurn    = "turn"    // a trajectory's per-turn rows (the turn level)
	KindTool    = "tool"    // a tool's definition / facts (the syscall level)
	KindSession = "session" // a full session (drive + content + trajectory) — see internal/sessionimage
	KindFleet   = "fleet"   // a set of sessions' drive states (the fleet level)
	KindRSI     = "rsi"     // an RSI loop's keep/revert ladder (the rsi level)
)

// Kind describes a registered primitive kind: its name, its level on the loops ladder
// (1 turn, 2 session, 3 fleet, 4 rsi; tool sits at the syscall sub-level, 1), a one-line
// description, and whether THIS package ships a typed codec for it (Typed=false means the
// generic Marshal/Parse seam carries it, or another package owns the rich format).
type Kind struct {
	Name  string `json:"name"`
	Level int    `json:"level"`
	Desc  string `json:"desc"`
	Typed bool   `json:"typed"`
}

var (
	kindsMu sync.RWMutex
	kinds   = map[string]Kind{}
)

// Register adds (or replaces) a kind in the registry. It is safe for concurrent use; fak
// registers its canonical ladder at init, and an application registers its own kinds the
// same way before enumerating or restoring.
func Register(k Kind) {
	kindsMu.Lock()
	kinds[k.Name] = k
	kindsMu.Unlock()
}

// Known reports whether a kind name is registered, returning its descriptor.
func Known(name string) (Kind, bool) {
	kindsMu.RLock()
	defer kindsMu.RUnlock()
	k, ok := kinds[name]
	return k, ok
}

// Kinds returns every registered kind, sorted by ladder Level then Name — the "what can
// I dump?" enumeration.
func Kinds() []Kind {
	kindsMu.RLock()
	out := make([]Kind, 0, len(kinds))
	for _, k := range kinds {
		out = append(out, k)
	}
	kindsMu.RUnlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].Level != out[j].Level {
			return out[i].Level < out[j].Level
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func init() {
	Register(Kind{Name: KindTool, Level: 1, Desc: "a tool's definition / facts (the syscall level)", Typed: false})
	Register(Kind{Name: KindTurn, Level: 1, Desc: "a trajectory's per-turn rows (the turn level)", Typed: true})
	Register(Kind{Name: KindSession, Level: 2, Desc: "a full session: drive + recall core image + trajectory (rich multi-part bundle: internal/sessionimage)", Typed: false})
	Register(Kind{Name: KindFleet, Level: 3, Desc: "a set of sessions' live drive states (the fleet level)", Typed: true})
	Register(Kind{Name: KindRSI, Level: 4, Desc: "an RSI loop's keep/revert ladder rows (the rsi level)", Typed: false})
}

// ---------------------------------------------------------------------------
// turn level — a trajectory's Turn rows
// ---------------------------------------------------------------------------

// TraceBody is the body of a turn-level snapshot: a trace id and its ordered Turn rows
// (the same stable schema trajectory.Recorder exports), so dumping/restoring a trace is
// lossless.
type TraceBody struct {
	TraceID string            `json:"trace_id"`
	Turns   []trajectory.Turn `json:"turns"`
}

// DumpTrace freezes a trace's Turn rows into a turn-level snapshot.
func DumpTrace(traceID string, turns []trajectory.Turn, now int64) (Snapshot, error) {
	return Marshal(KindTurn, traceID, TraceBody{TraceID: traceID, Turns: turns}, nil, now)
}

// RestoreTrace thaws a turn-level snapshot back to its Turn rows, refusing a snapshot of
// the wrong kind.
func (s Snapshot) RestoreTrace() (TraceBody, error) {
	if s.Kind != KindTurn {
		return TraceBody{}, fmt.Errorf("snapshot: RestoreTrace on kind %q (want %q)", s.Kind, KindTurn)
	}
	var b TraceBody
	if err := s.Into(&b); err != nil {
		return TraceBody{}, err
	}
	return b, nil
}

// ---------------------------------------------------------------------------
// fleet level — a set of sessions' drive states
// ---------------------------------------------------------------------------

// FleetBody is the body of a fleet-level snapshot: every session's drive State, in the
// scheduler order session.Table.Snapshot returns.
type FleetBody struct {
	Sessions []session.State `json:"sessions"`
}

// DumpFleet freezes a drive table's whole-fleet snapshot into a fleet-level snapshot —
// every live session's run-state / budget / priority / pace at this instant.
func DumpFleet(fleetID string, tbl *session.Table, now int64) (Snapshot, error) {
	if tbl == nil {
		return Snapshot{}, fmt.Errorf("snapshot: DumpFleet: nil table")
	}
	return Marshal(KindFleet, fleetID, FleetBody{Sessions: tbl.Snapshot()}, nil, now)
}

// RestoreFleet re-attaches every session's drive into tbl VERBATIM (via the §5
// session.Table.Restore — Rev preserved, a stopped session restored stopped), and
// returns the number of sessions restored. It refuses a snapshot of the wrong kind.
func (s Snapshot) RestoreFleet(tbl *session.Table) (int, error) {
	if s.Kind != KindFleet {
		return 0, fmt.Errorf("snapshot: RestoreFleet on kind %q (want %q)", s.Kind, KindFleet)
	}
	if tbl == nil {
		return 0, fmt.Errorf("snapshot: RestoreFleet: nil table")
	}
	var b FleetBody
	if err := s.Into(&b); err != nil {
		return 0, err
	}
	for _, st := range b.Sessions {
		tbl.Restore(st.TraceID, st)
	}
	return len(b.Sessions), nil
}
