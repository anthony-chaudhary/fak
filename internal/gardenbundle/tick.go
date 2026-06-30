package gardenbundle

// tick.go — the ACT side of the garden bundle. The default `fak garden` fold is
// READ-ONLY (see doc.go): it runs each member, reads its control-pane envelope,
// and reports. This file adds the LIVE, RECURRING TICK that turns a surfaced
// condition into a documented, idempotent, safe remediation instead of a report
// nobody reads (#1386).
//
// The decision of WHAT to act on is PURE and TESTABLE here; the side effects
// themselves (run `fak leaseref reap`, append a witnessed run-end to the loop
// ledger) live in the cmd verb, which calls PlanTick to learn its worklist.
//
// SAFETY (the load-bearing invariants, all enforced by PlanTick, all tested):
//
//   - Act ONLY on a member whose folded state is "action" or "red" — i.e. it
//     surfaced a real, past-threshold condition. An "ok" member is left
//     untouched. The staleness threshold itself is the member's own (loop
//     recover's --stale-min, leaseref's TTL/Expired): the tick never re-decides
//     freshness, it acts on the member's already-rendered ACTION verdict, so a
//     fresh/live run or lease the member reported "ok" is structurally
//     un-acted-on.
//   - --dry-run yields a plan whose every decision is Performed=false /
//     Mode="dry-run": it acts on NOTHING, preserving today's report-only
//     behavior behind the flag.
//   - Only members with a registered, idempotent remediation get a Perform
//     decision. Re-running the tick is safe: reap deletes only already-expired
//     leases (a no-op once gone); the orphan surface appends one bounded
//     witness event.
//   - release_staleness is advisory-only here (acting needs the release path,
//     tracked by #1367): it is reported, never auto-acted.

// ActKind is the closed set of remediations the garden tick can take. It is the
// per-member action policy from the ticket made into a typed value.
type ActKind string

const (
	// ActNone: the member is ok, or has no registered remediation — nothing to do.
	ActNone ActKind = "none"
	// ActReap: an expired cross-machine lease lingers — delete the reapable
	// records (the `fak leaseref reap` remediation). Idempotent: a second reap of
	// an already-gone lease is a no-op.
	ActReap ActKind = "reap"
	// ActSurface: orphaned/unwitnessed runs exist — surface the recovery worklist
	// as a witnessed tick event so the operator can re-dispatch/re-verify
	// (re-dispatch stays gated, never automatic). Idempotent: one bounded event.
	ActSurface ActKind = "surface"
	// ActAdvisory: the member surfaced a condition whose remediation is owned
	// elsewhere (release_staleness -> #1367). Reported, never auto-acted by the tick.
	ActAdvisory ActKind = "advisory"
)

// memberActs binds each member key to the remediation the tick takes when that
// member surfaces a condition. A key absent from this map is reported but never
// acted on — a new advisory member can join the bundle without the tick acting
// on it until a remediation is registered here on purpose.
var memberActs = map[string]ActKind{
	"orphaned_runs":     ActSurface,
	"stale_leases":      ActReap,
	"release_staleness": ActAdvisory,
}

// ActDecision is one member's tick decision: which remediation applies, whether
// the tick will Perform it (false under --dry-run, or when the member is ok), and
// a human Reason. The cmd verb reads Perform+Act to do the side effect; Detail
// carries the member's own surfaced reason for the tick's witness record.
type ActDecision struct {
	Key     string  `json:"key"`
	Label   string  `json:"label"`
	State   string  `json:"state"`
	Act     ActKind `json:"act"`
	Perform bool    `json:"perform"`
	Mode    string  `json:"mode"` // "act" or "dry-run"
	Reason  string  `json:"reason"`
	Detail  string  `json:"detail"`
}

// TickPlan is the folded act-pass plan: the per-member decisions plus the
// summary counts the verb and its witness event report.
type TickPlan struct {
	DryRun    bool          `json:"dry_run"`
	Decisions []ActDecision `json:"decisions"`
	// ToReap / ToSurface count the members the tick WILL act on (0 under dry-run).
	ToReap    int `json:"to_reap"`
	ToSurface int `json:"to_surface"`
	// Advisory counts members surfacing a condition acted on elsewhere.
	Advisory int `json:"advisory"`
}

// PlanTick folds member results into the act-pass plan. It is pure: same results
// + dryRun in, same plan out, no I/O. dryRun=true forces every decision to
// Perform=false / Mode="dry-run" (acts on nothing).
func PlanTick(results []MemberResult, dryRun bool) TickPlan {
	plan := TickPlan{DryRun: dryRun}
	for _, r := range results {
		act := memberActs[r.Key]
		if act == "" {
			act = ActNone
		}
		// Only a member that surfaced a real condition (action/red) is a candidate
		// to act on. An "ok" or "errored" member is never acted on by the tick: an
		// ok member has nothing to remediate, and an errored member couldn't measure
		// — acting on an unmeasured condition would be unsafe.
		surfaced := r.State == "action" || r.State == "red"

		d := ActDecision{
			Key:    r.Key,
			Label:  r.Label,
			State:  r.State,
			Act:    act,
			Detail: r.Detail,
			Mode:   "act",
		}
		if dryRun {
			d.Mode = "dry-run"
		}

		switch {
		case !surfaced || act == ActNone:
			d.Act = ActNone
			d.Perform = false
			d.Reason = reasonNoOp(r, act)
		case act == ActAdvisory:
			d.Perform = false
			d.Reason = "advisory: remediation owned elsewhere (release path, #1367); reported, not auto-acted"
			plan.Advisory++
		case dryRun:
			d.Perform = false
			d.Reason = "dry-run: would " + verb(act) + " — " + r.Detail
		default:
			d.Perform = true
			d.Reason = verb(act) + ": " + r.Detail
			switch act {
			case ActReap:
				plan.ToReap++
			case ActSurface:
				plan.ToSurface++
			}
		}
		plan.Decisions = append(plan.Decisions, d)
	}
	return plan
}

// Acted reports whether the plan performed any real side effect (false under
// dry-run, or when every member was ok). The verb uses it to decide whether to
// stamp a "tick acted" vs "tick clean" witness summary.
func (p TickPlan) Acted() bool { return p.ToReap > 0 || p.ToSurface > 0 }

// reasonNoOp explains why a member was not acted on, so the plan is legible even
// when it does nothing (the common, healthy case).
func reasonNoOp(r MemberResult, act ActKind) string {
	switch {
	case r.State == "ok":
		return "ok: nothing to remediate"
	case r.State == "errored":
		return "errored: member could not measure — not acting on an unmeasured condition"
	case act == ActNone:
		return "surfaced a condition but no remediation is registered for this member — reported only"
	default:
		return "no action"
	}
}

// verb renders an ActKind as a present-tense verb for the decision reason.
func verb(a ActKind) string {
	switch a {
	case ActReap:
		return "reap expired lease(s)"
	case ActSurface:
		return "surface recovery worklist"
	case ActAdvisory:
		return "defer to the release path"
	default:
		return "no action"
	}
}
