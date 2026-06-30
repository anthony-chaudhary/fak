package leaseref

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// RehydrateFenceFiresAt is the first dormancy band where a restored holder must re-read
// its lease generation before writing. The lease TTL may be much shorter than the Frozen
// planning band, so the concrete lease rung fires from Cold upward (#1182) while the
// generic orchestrator still leaves reason staging injectable.
const RehydrateFenceFiresAt = dormancy.Cold

// NewRehydrateFenceRung builds the #1182 lease-fence rung for a dormant holder re-entering
// a session. The caller passes the lease record it believes it still holds (ID, Holder, and
// Generation are load-bearing). On Cold/Frozen/Ancient admission the rung re-reads the live
// lease via Store.Fence; any non-OK fence verdict refuses rehydration with STALE_LEASE so the
// caller halts and reacquires (or stands down) before the first post-wake write.
func NewRehydrateFenceRung(store *Store, presented Record, now func() time.Time) rehydrate.Rung {
	return rehydrate.NewRungAt(rehydrate.StaleLease, RehydrateFenceFiresAt, func(ctx context.Context) rehydrate.Verdict {
		if store == nil {
			return rehydrate.Refuse(rehydrate.StaleLease, "lease fence has no store; halt and reacquire before writing")
		}
		if presented.ID == "" {
			return rehydrate.Refuse(rehydrate.StaleLease, "lease fence has no lease id; halt and reacquire before writing")
		}
		at := time.Now()
		if now != nil {
			at = now()
		}
		v, err := store.Fence(ctx, presented, at)
		if err != nil {
			return rehydrate.Refuse(rehydrate.StaleLease, "lease fence check failed: "+err.Error())
		}
		if v.OK {
			return rehydrate.Clear()
		}
		return rehydrate.Refuse(rehydrate.StaleLease, rehydrateFenceDetail(presented.ID, v))
	})
}

func rehydrateFenceDetail(id string, v FenceVerdict) string {
	detail := v.Detail
	if detail == "" {
		detail = "lease is not current"
	}
	if v.Reason == "" {
		return fmt.Sprintf("lease %s fence refused: %s; halt and reacquire before writing", id, detail)
	}
	return fmt.Sprintf("lease %s fence refused %s (presented generation %d, current generation %d, holder %q): %s; halt and reacquire before writing",
		id, v.Reason, v.Presented, v.Current, v.Holder, detail)
}
