package beam

import "github.com/anthony-chaudhary/fak/internal/model"

// StepPlan describes one decode step across the batched session: which token each beam
// proposes next (IDs, indexed by logical beam) and which beams are active this step
// (Active). A nil/empty Active means "every beam is active".
//
// NOTE: wiring actual beam-search node expansion (survivor selection, scoring,
// top-k/top-p over the returned logits) is DEFERRED to #12407, which owns the beam
// engine. This type is deliberately only the transport between a caller's per-beam
// choice and the real model API.
type StepPlan struct {
	IDs    []int
	Active []bool
}

// BatchDriver ties a BeamIndirectionTable to a REAL model.BatchSession. It performs no
// beam-search logic itself: it forwards the chosen token ids to the existing
// StepBatch / StepBatchActive decode entry points and exposes the exact work receipt
// (LastStepMACs) so a beam driver can attribute cost per step. The indirection table is
// exposed read-only by slot so a caller can map logical beam positions to physical KV
// slots without this package ever touching a float tensor.
type BatchDriver struct {
	bs    *model.BatchSession
	table *BeamIndirectionTable
}

// NewBatchDriver wires a batch session to an indirection table. Both are required; a nil
// either argument panics, since a driver without a session or table has no meaning.
func NewBatchDriver(bs *model.BatchSession, table *BeamIndirectionTable) *BatchDriver {
	if bs == nil {
		panic("beam: NewBatchDriver requires a non-nil *model.BatchSession")
	}
	if table == nil {
		panic("beam: NewBatchDriver requires a non-nil *BeamIndirectionTable")
	}
	return &BatchDriver{bs: bs, table: table}
}

// Step runs one decode step for the plan and returns the per-beam next-token logits.
// When every beam is active it takes the full-batch path (bit-identical to a direct
// StepBatch call); when some beams are idle it takes the ragged path, which compacts
// the batch dimension and reports proportionally less work via LastStepMACs.
func (d *BatchDriver) Step(plan StepPlan) [][]float32 {
	if plan.Active == nil || allActive(plan.Active) {
		return d.bs.StepBatch(plan.IDs)
	}
	return d.bs.StepBatchActive(plan.IDs, plan.Active)
}

// LastStepMACs returns the exact projection MAC count of the most recent Step, straight
// from the model session. It is an execution receipt, not admission evidence.
func (d *BatchDriver) LastStepMACs() int64 { return d.bs.LastStepMACs() }

// Slot delegates to the indirection table: logical (beam, pos) -> physical KV slot.
func (d *BatchDriver) Slot(beam, pos int) int32 { return d.table.Slot(beam, pos) }

// Table returns the wired indirection table.
func (d *BatchDriver) Table() *BeamIndirectionTable { return d.table }

func allActive(active []bool) bool {
	for _, a := range active {
		if !a {
			return false
		}
	}
	return true
}
