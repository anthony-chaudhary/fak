// Package epochbridge is the explicit converter between the two epoch/generation
// lineages of the ONE agent lineage family (epic #912, child #914): the served
// session's continuation lineage (internal/session — continuationID + State.Generation,
// a parent that on budget exhaustion re-continues into a fresh-budget child) and the
// kernel's speculation lineage (internal/abi — SpeculationContext{Epoch,ParentEpoch} +
// Outcome, a parent that spawns provisional children that commit or get discarded).
//
// They are the same idea — a parent minting children under one id space — kept in two
// id spaces with no converter. This package maps a session generation onto an abi epoch
// so a Recontinue mints a KV epoch and (per #809) a speculative branch is a generation
// under one id family. It owns NEITHER type: it sits above both, pivoting through the
// session continuation id's uint64 epoch (internal/session.ContinuationEpoch), so the
// frozen abi types are untouched (SEAM 4) and session need not import abi.
//
// HONEST about the commit-semantics difference. The two lineages share the epoch ID
// SPACE, not the commit semantics. A session continuation is a DURABLE generation —
// it maps to Speculative=false and abi.OutcomeCommitted (the default), never a
// provisional branch that can squash. A generation-0 session (an original trace that
// never re-continued) and a session with no parent both read as epoch 0, abi's
// ordinary-committed zero value — never a guessed non-zero epoch.
package epochbridge

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// SpecContextFor projects a session generation onto the shared abi epoch lineage.
// Epoch is this generation's lineage point (the uint64 its continuation id encodes,
// or 0 for an original generation-0 trace); ParentEpoch is the generation it was
// re-continued FROM (0 when there is no parent). Speculative is always false: a
// session continuation is a committed generation, not a provisional speculative call.
func SpecContextFor(st session.State) abi.SpeculationContext {
	epoch, _ := session.ContinuationEpoch(st.TraceID)
	parent, _ := session.ContinuationEpoch(st.ParentTrace)
	return abi.SpeculationContext{
		Speculative: false,
		Epoch:       epoch,
		ParentEpoch: parent,
	}
}

// GenerationOutcome is the resolution a completed session generation carries in the
// shared family: always abi.OutcomeCommitted. A re-continuation is a durable fresh
// window — it never squashes or rolls back the way a speculative branch can. This is
// the constant that makes "a Recontinue is the committed end of the Outcome closed
// set" explicit rather than implied.
func GenerationOutcome() abi.Outcome { return abi.OutcomeCommitted }
