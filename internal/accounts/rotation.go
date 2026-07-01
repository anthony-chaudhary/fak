package accounts

// rotation.go — the live ROTATION READ that fak's own launch/guard path can consult,
// instead of only emitting a `rotation:` config block for an external switcher to obey.
//
// WHY THIS EXISTS. The registry already records which seats are enabled, reserved, and
// which dir each is logged into — and Reconcile already collapses seats that are really
// ONE account (one rate-limit bucket). What was missing is the engine that turns those
// facts into an ORDER: a round-robin over the DISTINCT account buckets, so a launcher can
// rotate off a walled account onto a fresh one instead of always re-launching the single
// active seat. The rotation policy (order/near_cap_util/avoid_reserved) used to live ONLY
// as data in the generated dos/job roster views, consumed by an out-of-product Python
// switcher; this brings the read into the product as a pure, typed, tested function.
//
// HONESTY. The plan is witnessed end to end: every seat's inclusion or exclusion is
// derived from registry fields (Active/Enabled/Reserved/Identity), and two seats sharing
// one account bucket are deduped so the pool never double-counts a single rate-limit
// window as independent capacity. It deliberately does NOT claim a `by_reset` / near-cap
// ordering: the registry carries no per-account usage or reset-time telemetry, so the
// applied order is a deterministic stable-by-name round-robin, and the requested policy
// is surfaced alongside it (RotationResult.OrderApplied vs Policy.Order) rather than
// pretended. A near-cap-aware ordering is a follow-on that first needs a usage signal.
//
// Pure over the homes' disk-derived Identity, exactly like Reconcile — Refresh the
// registry first for a live answer.

import "sort"

// RotationStatus says whether a seat is in the rotation pool, and if not, why. A CLOSED
// set: every home in the registry maps to exactly one of these.
type RotationStatus string

const (
	// RotationIncluded — eligible and carrying its account bucket in the pool.
	RotationIncluded RotationStatus = "included"
	// RotationReserved — held OUT of routine rotation (the --reserved last-resort flag),
	// excluded only when the policy's avoid_reserved is set (the default).
	RotationReserved RotationStatus = "reserved"
	// RotationDisabled — carries an explicit enabled:false.
	RotationDisabled RotationStatus = "disabled"
	// RotationTombstoned — a retired seat (it rehomes via Serve; never a rotation target).
	RotationTombstoned RotationStatus = "tombstoned"
	// RotationUnservable — active+enabled but cannot serve right now (dir missing / no live
	// credentials). Excluded so rotation never lands on a seat that would drop into /login.
	RotationUnservable RotationStatus = "unservable"
	// RotationDuplicate — shares an INCLUDED seat's account bucket (one rate-limit window);
	// collapsed onto that seat so the pool counts each bucket once.
	RotationDuplicate RotationStatus = "duplicate"
)

// RotationSeat is one seat's place in the rotation decision.
type RotationSeat struct {
	Name string `json:"name"`
	Dir  string `json:"dir,omitempty"`
	// Account is the rate-limit bucket key (uuid:… or tok:…); "" when the seat carries no
	// derivable identity (an unidentifiable serveable seat is its own singleton bucket).
	Account   string         `json:"account,omitempty"`
	Status    RotationStatus `json:"status"`
	Login     LoginStatus    `json:"login_status,omitempty"`
	CanServe  bool           `json:"can_serve"`
	Email     string         `json:"email,omitempty"`
	Canonical string         `json:"canonical,omitempty"` // for a duplicate, the pool seat it collapses onto
	// Headroom is the injected per-bucket headroom score this seat carried into the pool
	// ordering (higher == more room; nil when the plan ran in stable-by-name mode with no
	// signal). Surfaced so a launcher can show WHY one bucket sorted ahead of another.
	Headroom *float64 `json:"headroom,omitempty"`
}

// RotationHeadroom is an OPTIONAL per-account-bucket headroom signal a caller supplies so the
// rotation pool is ordered MOST-HEADROOM-FIRST instead of the default stable-by-name
// round-robin. The key is the account bucket key (Identity.AccountKey(), e.g. "uuid:…"); the
// value is a score where HIGHER == more headroom (launch this bucket sooner). A bucket absent
// from the map scores 0. An empty/nil map means "no signal" — the plan falls back to
// stable-by-name exactly as before (OrderApplied stays "stable-by-name"). A non-empty map
// switches the pool order to "headroom-desc", with the seat name as a deterministic tiebreak
// so two equal-headroom buckets still order the same way every time.
//
// The config registry itself carries NO usage/quota telemetry (RotationPlan is pure over
// disk-derived identity), so this signal is derived and passed in by the caller from the live
// runtime layer (internal/fleetaccounts: usage-throttle + offerability). Keeping it an
// injected map is what lets RotationPlan stay a pure function AND become headroom-aware
// without the config plane reaching across into the runtime plane — the "usage signal" the
// package doc named as the prerequisite for a near-cap-aware ordering.
type RotationHeadroom map[string]float64

// RotationPolicy is the configured rotation policy, read from the registry's generated
// view blocks (the `rotation:` block the dos/job rosters carry). AvoidReserved defaults to
// TRUE — a reserved seat is the last-resort fallback, held out of routine rotation, per the
// `fak accounts add --reserved` contract. Order and NearCapUtil are CARRIED for surfacing
// but are not yet applied (no per-account usage/reset telemetry exists to apply them).
type RotationPolicy struct {
	Order         string  `json:"order,omitempty"`
	NearCapUtil   float64 `json:"near_cap_util,omitempty"`
	AvoidReserved bool    `json:"avoid_reserved"`
}

// RotationResult is the full, witnessed rotation decision.
type RotationResult struct {
	Policy RotationPolicy `json:"policy"`
	// OrderApplied is the ordering this plan ACTUALLY used — honest about the gap between
	// the requested Policy.Order and what witnessed data supports. Today always
	// "stable-by-name" (a deterministic round-robin key); never the unwitnessed "by_reset".
	OrderApplied string `json:"order_applied"`
	// Pool is the included buckets in rotation order — one seat per distinct account.
	Pool []RotationSeat `json:"pool"`
	// Excluded lists every non-pool seat with its reason, sorted by name.
	Excluded []RotationSeat `json:"excluded,omitempty"`
}

// RotationPlan classifies every home and returns the rotation pool: the eligible seats,
// one per DISTINCT account bucket, in a deterministic round-robin order, plus every
// excluded seat with the reason it is out. It is pure over the homes' disk-derived
// Identity (Refresh first for a live answer) and reads the configured RotationPolicy to
// decide whether reserved seats are held out. It is the stable-by-name form —
// RotationPlanWithHeadroom(nil).
func (r Registry) RotationPlan() RotationResult { return r.RotationPlanWithHeadroom(nil) }

// RotationPlanWithHeadroom is RotationPlan with an OPTIONAL per-bucket headroom signal (see
// RotationHeadroom). When the signal is non-empty the pool is ordered most-headroom-first
// (name breaks ties) and OrderApplied becomes "headroom-desc"; each pool seat carries the
// score it sorted on. An empty/nil signal is byte-for-byte the historical stable-by-name plan.
func (r Registry) RotationPlanWithHeadroom(hr RotationHeadroom) RotationResult {
	pol := r.RotationPolicy()
	res := RotationResult{Policy: pol, OrderApplied: "stable-by-name"}

	var eligible []Home
	for _, h := range r.Homes {
		switch st := rotationStatusFor(h, pol); st {
		case RotationIncluded:
			eligible = append(eligible, h)
		case RotationReserved:
			res.Excluded = append(res.Excluded, seatStatus(h, RotationReserved))
		default:
			res.Excluded = append(res.Excluded, seatStatus(h, st))
		}
	}

	// Dedup the eligible seats by account bucket: elect ONE canonical seat per bucket (so a
	// twin never presents one rate-limit window as two rotatable accounts), and record the
	// rest as duplicates collapsed onto it. A seat with no derivable account key cannot be
	// collapsed, so it stands as its own singleton bucket.
	byAcct := map[string][]Home{}
	var pool []RotationSeat
	for _, h := range eligible {
		k := h.Identity.AccountKey()
		if k == "" {
			pool = append(pool, seatStatus(h, RotationIncluded))
			continue
		}
		byAcct[k] = append(byAcct[k], h)
	}
	for k, group := range byAcct {
		winner := canonicalSeat(group)
		ws := seatStatus(winner, RotationIncluded)
		ws.Account = k
		pool = append(pool, ws)
		for _, h := range group {
			if h.Name == winner.Name {
				continue
			}
			ds := seatStatus(h, RotationDuplicate)
			ds.Account = k
			ds.Canonical = winner.Name
			res.Excluded = append(res.Excluded, ds)
		}
	}

	// Order the pool. With a headroom signal, stamp each seat's score and sort most-headroom
	// first, using the name as a deterministic tiebreak so equal-headroom buckets stay stable;
	// without one, keep the historical stable-by-name round-robin exactly.
	useHeadroom := len(hr) > 0
	if useHeadroom {
		res.OrderApplied = "headroom-desc"
		for i := range pool {
			v := hr[pool[i].Account] // absent bucket -> 0
			pool[i].Headroom = &v
		}
	}
	sort.Slice(pool, func(i, j int) bool {
		if useHeadroom {
			if hi, hj := derefFloat(pool[i].Headroom), derefFloat(pool[j].Headroom); hi != hj {
				return hi > hj
			}
		}
		return pool[i].Name < pool[j].Name
	})
	sort.Slice(res.Excluded, func(i, j int) bool { return res.Excluded[i].Name < res.Excluded[j].Name })
	res.Pool = pool
	return res
}

// derefFloat reads a *float64 as 0 when nil, so the pool comparator can treat an unstamped
// seat (stable-by-name mode) as neutral without a nil check at every call site.
func derefFloat(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// rotationStatusFor is the single primary-status bridge from login readiness into
// rotation. LoginStatus owns whether a config home can serve; rotation layers only
// rotation-specific policy on top, such as holding reserved seats out of the routine pool.
func rotationStatusFor(h Home, pol RotationPolicy) RotationStatus {
	switch h.LoginStatus() {
	case LoginReady:
		if pol.AvoidReserved && h.Reserved {
			return RotationReserved
		}
		return RotationIncluded
	case LoginTombstoned:
		return RotationTombstoned
	case LoginDisabled:
		return RotationDisabled
	case LoginMissingDir, LoginNeedsLogin:
		return RotationUnservable
	default:
		return RotationUnservable
	}
}

// NextInRotation returns the next pool seat AFTER the account bucket that `after` belongs
// to, wrapping around — so a caller rotates onto a DIFFERENT account than the one it is on.
// With `after` empty or naming a seat outside the pool, the FIRST pool seat is returned (a
// fresh rotation start). ok is false when the pool is empty (nothing to rotate to), or when
// `after` is the pool's ONLY bucket (nowhere else to go) — so a caller can fail loud instead
// of silently re-handing the same walled account. It is the stable-by-name form —
// NextInRotationWithHeadroom(after, nil).
func (r Registry) NextInRotation(after string) (RotationSeat, bool) {
	return r.NextInRotationWithHeadroom(after, nil)
}

// NextInRotationWithHeadroom is NextInRotation with an OPTIONAL per-bucket headroom signal.
// With a signal the pool is ordered most-headroom-first, so "next" is the BEST-headroom
// bucket that is not the anchor's own bucket — a rotate never re-hands the walled/capped seat
// the caller is leaving, and it prefers the account with the most room rather than the mere
// next name in the ring. Without a signal it is the historical stable round-robin. ok is false
// on an empty pool, or when the anchor's bucket is the pool's only one.
func (r Registry) NextInRotationWithHeadroom(after string, hr RotationHeadroom) (RotationSeat, bool) {
	res := r.RotationPlanWithHeadroom(hr)
	if len(res.Pool) == 0 {
		return RotationSeat{}, false
	}
	afterKey := r.bucketKey(after)
	if len(hr) > 0 {
		// Headroom mode: the pool is already ordered most-headroom-first, so the first seat
		// that is NOT the anchor's bucket is the best account to rotate onto. A fresh start
		// (empty anchor) returns pool[0], the highest-headroom bucket.
		for _, s := range res.Pool {
			if after != "" && s.Name == after {
				continue
			}
			if afterKey != "" && s.Account == afterKey {
				continue
			}
			return s, true
		}
		return RotationSeat{}, false // every pool seat is the anchor's bucket; nowhere else to go
	}
	idx := -1
	for i, s := range res.Pool {
		if (after != "" && s.Name == after) || (afterKey != "" && s.Account == afterKey) {
			idx = i
			break
		}
	}
	if idx < 0 {
		return res.Pool[0], true // unknown / not-in-pool seat -> start of rotation
	}
	if len(res.Pool) == 1 {
		return RotationSeat{}, false // the only bucket; nowhere else to rotate
	}
	return res.Pool[(idx+1)%len(res.Pool)], true
}

// bucketKey returns the account-bucket key of the seat named `name`, or "" when the name is
// empty or absent. It lets NextInRotation resolve a requested seat (which may itself be a
// duplicate or reserved, hence not in the pool) to the bucket it shares with a pool seat.
func (r Registry) bucketKey(name string) string {
	if name == "" {
		return ""
	}
	if h, ok := r.home(name); ok {
		return h.Identity.AccountKey()
	}
	return ""
}

// seatStatus builds a RotationSeat snapshot of a home with the given status.
func seatStatus(h Home, st RotationStatus) RotationSeat {
	return RotationSeat{
		Name:     h.Name,
		Dir:      h.Dir,
		Account:  h.Identity.AccountKey(),
		Status:   st,
		Login:    h.LoginStatus(),
		CanServe: h.CanServe(),
		Email:    h.Identity.Email,
	}
}

// RotationPolicy reads the rotation policy from the registry's generated view blocks. It
// prefers the job view's `rotation:` block (the switcher's own config), then the dos view,
// and falls back to AvoidReserved=true when no view carries one — so a registry with no
// rotation config still rotates sanely (reserved seats held out by default).
func (r Registry) RotationPolicy() RotationPolicy {
	pol := RotationPolicy{AvoidReserved: true}
	block := r.rotationBlock()
	if block == nil {
		return pol
	}
	if v, ok := block["order"].(string); ok {
		pol.Order = v
	}
	if v, ok := numFromAny(block["near_cap_util"]); ok {
		pol.NearCapUtil = v
	}
	if v, ok := block["avoid_reserved"].(bool); ok {
		pol.AvoidReserved = v
	}
	return pol
}

// rotationBlock returns the first `rotation:` view block found (job before dos), or nil.
func (r Registry) rotationBlock() map[string]any {
	for _, view := range []string{"job", "dos"} {
		vc, ok := r.Views[view]
		if !ok {
			continue
		}
		if b, ok := vc.Blocks["rotation"].(map[string]any); ok {
			return b
		}
	}
	return nil
}

// numFromAny coerces a JSON-decoded numeric value (float64 from encoding/json, or an int
// from a hand-built map in a test) to float64. ok is false for a non-numeric value.
func numFromAny(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
