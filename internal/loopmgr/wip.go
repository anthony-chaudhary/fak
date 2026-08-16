package loopmgr

// wip.go -- the flow-limit SIGNAL: how much started-and-unfinished work the loops in
// this ledger currently own, and how much they have merely queued.
//
// The ledger already records the two transitions a WIP measurement needs. EventStart
// marks a unit BEGUN and EventEnd marks it FINISHED, so started-minus-ended is, by
// construction, the count of units that are in hand right now. Nothing else has to be
// probed, dated, or self-reported: this is the same "a durable event IS the stamp"
// contract the rest of the ledger rests on, and it cannot be back-dated by a worker.
//
// WHY THIS IS NOT res.Live. Dispatch admission measures live worker PROCESSES. A process
// is not a unit of work: when a session ends, crashes, or is reaped, it leaves res.Live
// but the unit it started keeps its branch, its commits, and its unclosed issue. Started
// minus ended still counts that unit, which is the whole point -- it is precisely the
// work a process count cannot see, and precisely the work a new start would be piled on
// top of.
//
// INVENTORY IS KEPT SEPARATE AND IS NOT WIP. EventAdmit means a unit was let through the
// door; EventStart means someone actually began it. Admitted-but-never-started units are
// queued inventory: they cost nothing to hold and must not be charged against a flow
// limit, or a repo with a large backlog would refuse every spawn forever. The two counts
// are returned separately so the consumer cannot accidentally add them.

// WIP folds the ledger status into the flow-limit census: the number of units that were
// BEGUN and have not FINISHED (wip), and the number that were ADMITTED but never begun
// (inventory).
//
// Both counts are clamped per loop rather than in aggregate. A rotated or truncated
// ledger can show a loop whose ends outnumber its starts (the starts were sealed into an
// earlier segment); clamping each loop at zero keeps that loop from lending a negative
// balance that would mask a genuinely over-limit sibling and silently inflate the
// allowance. Clamping is the safe direction to be wrong in: it can only UNDER-report a
// deficit, never invent headroom.
func (s Status) WIP() (wip, inventory int) {
	for _, loop := range s.Loops {
		if loop.Started > loop.Ended {
			wip += int(loop.Started - loop.Ended)
		}
		if loop.Admitted > loop.Started {
			inventory += int(loop.Admitted - loop.Started)
		}
	}
	return wip, inventory
}
