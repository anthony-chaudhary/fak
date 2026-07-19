package scmbridge

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/servicelease"
)

// LaunchFence fences the three Windows launchers — the SCM service, the S4U
// watchdog task, and the InteractiveToken broker — behind ONE durable
// generation/lease per workload (#4756 "one durable generation/lease").
//
// The servicelease.Table already guarantees at most one valid OWNER across
// nodes/boots; what it cannot see is that three different launch paths on the
// SAME incarnation race each other after a crash (SCM recovery fires, the
// watchdog fires, and the logon broker drains its queue — all for one
// workload). The fence records WHICH role holds the live launch grant under
// the lease's fencing token; a second role is refused until the grant is
// released, expires, or is fenced by a generation bump.
//
// Pure state: the caller persists the struct (plain JSON) and supplies the
// logical clock, exactly like servicelease.
type LaunchFence struct {
	Table  *servicelease.Table      `json:"table"`
	Active map[string]*ActiveLaunch `json:"active"`
}

// ActiveLaunch records the launcher that currently holds the workload's
// launch grant, bound to the ONE fencing token that made it valid.
type ActiveLaunch struct {
	Role      Role                      `json:"role"`
	Token     servicelease.FencingToken `json:"token"`
	RequestID string                    `json:"request_id"`
	AtMS      int64                     `json:"at_ms"`
}

// Grant is a successful launch admission: the fencing token to present on
// every effectful claim, and the durable request ID that pairs with the
// resume receipt in the ledger.
type Grant struct {
	Workload  string                    `json:"workload"`
	Role      Role                      `json:"role"`
	Token     servicelease.FencingToken `json:"token"`
	RequestID string                    `json:"request_id"`
	ExpiresMS int64                     `json:"expires_ms"`
}

// ErrDuplicateLaunch — another launcher role already holds the live launch
// grant for this workload under the current generation/lease.
var ErrDuplicateLaunch = errors.New("scmbridge: duplicate launch refused; another launcher holds the live grant")

// NewLaunchFence builds an empty fence with the given lease TTL (0 = the
// servicelease default).
func NewLaunchFence(ttlMS int64) *LaunchFence {
	return &LaunchFence{Table: servicelease.NewTable(ttlMS), Active: map[string]*ActiveLaunch{}}
}

// Admit grants the workload launch to one role of one incarnation. Refusals:
//
//   - another ROLE holds a still-live grant (ErrDuplicateLaunch) — the SCM /
//     watchdog / broker race, collapsed to one launcher;
//   - another INCARNATION holds a valid lease (servicelease.ErrLeaseHeld) —
//     the cross-node duplicate-owner case;
//   - a stale incarnation (servicelease.ErrStaleIncarnation).
//
// The same role re-admitting refreshes its grant (heartbeat) under a fresh
// token; every prior token is stale from that call on.
func (f *LaunchFence) Admit(workload string, role Role, by servicelease.Incarnation, nowMS int64) (Grant, error) {
	if f.Active == nil {
		f.Active = map[string]*ActiveLaunch{}
	}
	if a := f.Active[workload]; a != nil && a.Role != role && f.grantLive(workload, a, nowMS) {
		return Grant{}, fmt.Errorf("%w: %s held by %s (request %s)", ErrDuplicateLaunch, workload, a.Role, a.RequestID)
	}
	l, err := f.Table.Acquire(workload, by, nowMS)
	if err != nil {
		return Grant{}, err
	}
	g := Grant{
		Workload:  workload,
		Role:      role,
		Token:     l.Token,
		RequestID: fmt.Sprintf("launch-%s-g%d-s%d", workload, l.Token.Generation, l.Token.LeaseSeq),
		ExpiresMS: l.ExpiresMS,
	}
	f.Active[workload] = &ActiveLaunch{Role: role, Token: l.Token, RequestID: g.RequestID, AtMS: nowMS}
	return g, nil
}

// grantLive reports whether the recorded active launch is still fenced-valid:
// its token must still be the workload's newest under a currently valid
// lease. A generation bump or lease expiry kills it without any explicit
// release.
func (f *LaunchFence) grantLive(workload string, a *ActiveLaunch, nowMS int64) bool {
	l, ok := f.Table.Leases[workload]
	return ok && l.Token == a.Token && !f.Table.ValidOwner(workload, nowMS).Zero()
}

// Release hands the launch grant back after a classified stop. Only the
// holding role presenting its own token releases; anything else is a no-op
// (reports false) so a stale launcher cannot free a successor's grant.
func (f *LaunchFence) Release(workload string, role Role, tok servicelease.FencingToken) bool {
	a := f.Active[workload]
	if a == nil || a.Role != role || a.Token != tok {
		return false
	}
	delete(f.Active, workload)
	return true
}

// Supersede bumps the workload's durable generation: every outstanding grant
// and token is fenced from this call on (desired-state change, operator
// takeover). The next Admit wins the fresh generation.
func (f *LaunchFence) Supersede(workload string) servicelease.Generation {
	return f.Table.BumpGeneration(workload)
}

// Holder reports which role currently holds a live launch grant, or "" when
// none does.
func (f *LaunchFence) Holder(workload string, nowMS int64) Role {
	if a := f.Active[workload]; a != nil && f.grantLive(workload, a, nowMS) {
		return a.Role
	}
	return ""
}
