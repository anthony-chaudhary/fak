package dispatchtick

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

const (
	PreflightSchema = "fleet-dispatch-preflight/1"

	PreflightOKVerdict        = "SPAWN_OK"
	PreflightRefuseHost       = "REFUSE_HOST"
	PreflightRefuseNoAccount  = "REFUSE_NO_ACCOUNT"
	PreflightRefuseNoSeat     = "REFUSE_NO_SEAT"
	PreflightRefuseAtCap      = "REFUSE_AT_CAP"
	PreflightRefuseInspect    = "REFUSE_INSPECT"
	PreflightObserveSettling  = "OBSERVE_SCALING_IN_PROGRESS"
	PreflightRefuseTreePoison = "TREE_POISONED"
	// PreflightRefuseLandSaturated throttles new spawns while the serialized-land
	// funnel is saturated (#3574). It is only ever emitted when PreflightInput's
	// LandContention term is Enabled; with the knob off the fold never reaches it.
	PreflightRefuseLandSaturated = "LAND_SATURATED"
)

const (
	HostCoresPerWorker   = 2
	HostRAMMBPerWorker   = 1500
	HostThreadsPerCore   = 400
	HostThreadsPerWorker = 200
	HostCapFloor         = 1
)

// HostBudgets are the per-worker charge constants the host-capacity gradient
// folds (#1337). The Host* consts above are the built-in conservative guesses;
// DefaultHostBudgets overlays the FAK_HOST_* env knobs so a shared or measured
// box recalibrates the gradient without a rebuild -- the same overrides the
// Python preflight honors, closing the drift where FAK_HOST_THREADS_PER_CORE
// raised the Python threads budget while this Go mirror stayed pinned at 400.
// The zero value means "built-in defaults", so a caller that sets nothing gets
// the old behavior and the pure fold stays hermetically testable.
type HostBudgets struct {
	CoresPerWorker   int `json:"cores_per_worker"`
	RAMMBPerWorker   int `json:"ram_mb_per_worker"`
	ThreadsPerCore   int `json:"threads_per_core"`
	ThreadsPerWorker int `json:"threads_per_worker"`
}

// DefaultHostBudgets resolves the per-worker budgets with the FAK_HOST_* env
// overrides applied (FAK_HOST_CORES_PER_WORKER, FAK_HOST_RAM_MB_PER_WORKER,
// FAK_HOST_THREADS_PER_CORE, FAK_HOST_THREADS_PER_WORKER). The impure shell
// passes this into PreflightInput; the pure fold itself never reads env.
func DefaultHostBudgets() HostBudgets {
	return HostBudgets{
		CoresPerWorker:   envPosInt("FAK_HOST_CORES_PER_WORKER", HostCoresPerWorker),
		RAMMBPerWorker:   envPosInt("FAK_HOST_RAM_MB_PER_WORKER", HostRAMMBPerWorker),
		ThreadsPerCore:   envPosInt("FAK_HOST_THREADS_PER_CORE", HostThreadsPerCore),
		ThreadsPerWorker: envPosInt("FAK_HOST_THREADS_PER_WORKER", HostThreadsPerWorker),
	}
}

func (b HostBudgets) normalized() HostBudgets {
	if b.CoresPerWorker <= 0 {
		b.CoresPerWorker = HostCoresPerWorker
	}
	if b.RAMMBPerWorker <= 0 {
		b.RAMMBPerWorker = HostRAMMBPerWorker
	}
	if b.ThreadsPerCore <= 0 {
		b.ThreadsPerCore = HostThreadsPerCore
	}
	if b.ThreadsPerWorker <= 0 {
		b.ThreadsPerWorker = HostThreadsPerWorker
	}
	return b
}

type HostResources struct {
	Cores        *int `json:"cores"`
	FreeRAMMB    *int `json:"free_ram_mb"`
	TotalThreads *int `json:"total_threads"`
}

type HostCapacityInfo struct {
	Cores        *int           `json:"cores"`
	FreeRAMMB    *int           `json:"free_ram_mb"`
	TotalThreads *int           `json:"total_threads"`
	Components   map[string]int `json:"components"`
	HostCap      *int           `json:"host_cap"`
	Binding      string         `json:"binding"`
}

type HostCheck struct {
	Safe         bool     `json:"safe"`
	Error        string   `json:"error,omitempty"`
	Flagged      int      `json:"flagged"`
	FlaggedNames []string `json:"flagged_names,omitempty"`
}

type AccountCheck struct {
	Available bool   `json:"available"`
	Tag       string `json:"tag,omitempty"`
	Dir       string `json:"dir,omitempty"`
	Tier      any    `json:"tier,omitempty"`
	Model     string `json:"model,omitempty"`
	Reason    string `json:"reason,omitempty"`
	// Error is set ONLY when the account probe itself could not complete (the roster
	// was unreadable / the scan timed out) -- as distinct from a scan that ran and
	// found no free account. classifyPreflight routes a non-empty Error to
	// REFUSE_INSPECT, never REFUSE_NO_ACCOUNT, so a failed probe is never misread as
	// genuine pool saturation.
	Error string `json:"error,omitempty"`
	// Blocked is the flat list of blocked account tags (kept for back-compat).
	Blocked []string `json:"blocked,omitempty"`
	// BlockedAccounts carries the per-account block REASON (throttled / needs-login /
	// at session cap), not just the tag -- so a REFUSE_NO_ACCOUNT verdict can name why
	// each seat in the target tier was refused instead of only that it was.
	BlockedAccounts []BlockedAccount `json:"blocked_accounts,omitempty"`
	LoginStatus     string           `json:"login_status,omitempty"`
	CanServe        *bool            `json:"can_serve,omitempty"`
}

type KernelCheck struct {
	Alive   *int   `json:"alive"`
	Target  *int   `json:"target"`
	Error   string `json:"error,omitempty"`
	Verdict string `json:"verdict,omitempty"`
}

type SeatCheck struct {
	Total            *int   `json:"total"`
	Free             *int   `json:"free,omitempty"`
	Leased           *int   `json:"leased,omitempty"`
	Depleted         bool   `json:"depleted,omitempty"`
	UnattributedLive int    `json:"unattributed_live,omitempty"`
	Skipped          string `json:"skipped,omitempty"`
	Error            string `json:"error,omitempty"`
}

type TreeCheck struct {
	Poisoned bool   `json:"poisoned"`
	Package  string `json:"package,omitempty"`
	Error    string `json:"error,omitempty"`
}

// LandContention is the OPTIONAL land-saturation backpressure term (#3574). When
// Enabled, classifyPreflight refuses with PreflightRefuseLandSaturated while the
// serialized-land contention Signal sits at or above HighWater, holds the Prior
// tick's decision between the marks (a Schmitt-trigger hysteresis band, so a signal
// oscillating around one threshold does not flap the fleet), and admits once Signal
// falls below LowWater. Enabled defaults false, so the zero value is a pure no-op and
// the preflight fold stays byte-identical to before this term existed.
type LandContention struct {
	Enabled   bool    `json:"enabled"`
	Signal    float64 `json:"signal"`
	HighWater float64 `json:"high_water"`
	LowWater  float64 `json:"low_water"`
	// Prior carries the previous tick's land-contention decision so the hysteresis band
	// can hold it: PreflightRefuseLandSaturated when the last tick refused for saturation,
	// "" otherwise. Only consulted while Signal is inside the [LowWater, HighWater) band.
	Prior string `json:"prior,omitempty"`
}

type PreflightInput struct {
	Workspace  string
	MaxWorkers int
	Host       HostCheck
	Tree       TreeCheck
	Account    AccountCheck
	Kernel     KernelCheck
	Seat       SeatCheck
	Resources  HostResources
	// Budgets are the per-worker host charges the capacity gradient folds; the
	// zero value means the built-in Host* consts. The shell passes
	// DefaultHostBudgets() so the FAK_HOST_* env knobs reach the fold.
	Budgets       HostBudgets
	OSWorkerProcs int
	// WorkerFloor is the SLOW predictive loop's forecast floor (#3368, two-timescale
	// scaling). Every cap above is a lowering min(); a reactive dip (a low kernel/lease
	// target) can therefore leave the fleet under-provisioned for a load ramp the
	// forecast already sees coming. The fast reactive tick clamps UP to this floor via
	// max, so capacity is pre-warmed ahead of demand -- but the floor is itself bounded
	// by the HARD physical/config ceiling (config max, host cap, seat inventory), never
	// the reactive lease target, so pre-warming can override a soft reactive dip WITHOUT
	// ever overbooking the box or the seat pool. Zero (the default) means "no forecast
	// floor" and is byte-identical to before this term existed.
	WorkerFloor int
	// ContractionTarget is the post-contraction worker count when a scale-down / drain
	// is in flight (#4038). While the fleet is draining N toward this target T, admits
	// are capped at T so no new worker is placed onto capacity that is about to be
	// reclaimed -- kvcached's in_shrink guard, where available_size() floors to the
	// pending target so an admit never races the contraction. It is applied AFTER the
	// predictive floor (WorkerFloor), so it also bounds a pre-warm: admitting onto
	// reclaimed capacity via the forecast floor is exactly the race being closed. Pure
	// lowering term -- it can only tighten the cap, never raise it. Zero (the default)
	// means "no contraction in flight" and is byte-identical to before this term existed.
	ContractionTarget int
	// LandContention is the OPTIONAL serialized-land backpressure term (#3574). Its zero
	// value (Enabled=false) is a no-op: classifyPreflight only consults it when Enabled,
	// so the fold is byte-identical to before this term existed unless the knob is set.
	LandContention LandContention
	// WorktreeIsolation is the OPTIONAL evidence-gated cap raise (#3185): with per-worker
	// worktree isolation ON (#3168) AND a #3180 A/B report witnessing zero build-poison at
	// a measured peak concurrency, the artificial ~6 shared-trunk ceiling (#1333) is lifted
	// toward that PROVEN number. The zero value earns nothing, so with isolation off -- or
	// on but unproven -- the cap is byte-identical to before this term existed.
	WorktreeIsolation WorktreeIsolation
	// SettleBeforeRedecide is the OPTIONAL settle-before-redecide gate (#3370). When true,
	// classifyPreflight returns PreflightObserveSettling while the kernel is mid-scale
	// (Alive != Target) -- holding a new scaling decision until the prior one converges so
	// decisions do not stack while the fleet is still moving. It defaults false, so the zero
	// value is a no-op and the preflight fold stays byte-identical to before this term
	// existed: it only shadows the downstream cap/account/seat classification when the knob
	// is explicitly set. Gated off by default to match the #3574 LandContention convention --
	// #3370 originally landed this verdict UNgated, which red the trunk on ~40 integration
	// tests that assert the ordinary mid-scale SPAWN path (alive < target still admits toward
	// target). The behavior can be enabled deliberately alongside its integration-test flip.
	SettleBeforeRedecide bool
	// WIP is the OPTIONAL flow-limit census (wiplimit.go): how many started-and-unfinished
	// units the fleet already owns, and the most it may own. Every other term in this input
	// is a RESOURCE limit; this is the only one that charges a spawn against WORK IN HAND.
	// Its zero value (Measured=false, Limit=0) abstains, so the fold is byte-identical to
	// before this term existed unless a producer supplies a measured census.
	WIP WIPCensus
}

type PreflightResult struct {
	Schema        string           `json:"schema"`
	OK            bool             `json:"ok"`
	Verdict       string           `json:"verdict"`
	Reason        string           `json:"reason"`
	Workspace     string           `json:"workspace"`
	Cap           int              `json:"cap"`
	Live          int              `json:"live"`
	Headroom      int              `json:"headroom"`
	CapTerms      CapTerms         `json:"cap_terms"`
	MaxWorkers    int              `json:"max_workers"`
	HostCap       *int             `json:"host_cap"`
	HostCapacity  HostCapacityInfo `json:"host_capacity"`
	Seat          SeatCheck        `json:"seat"`
	Host          HostCheck        `json:"host"`
	Tree          TreeCheck        `json:"tree"`
	Account       AccountCheck     `json:"account"`
	Kernel        KernelCheck      `json:"kernel"`
	OSWorkerProcs int              `json:"os_worker_procs"`
}

type CapTerms struct {
	ConfiguredCap int  `json:"configured_cap"`
	LeaseCap      *int `json:"lease_cap"`
	HostCap       *int `json:"host_cap"`
	SeatCap       *int `json:"seat_cap"`
	// WorkerFloor is the ceiling-bounded predictive floor (#3368) that raised the
	// effective cap, or 0 when no forecast floor applied. When it is what set the
	// effective cap (it lifted capacity above the tightest lowering cap), Limiting
	// reads "floor" so an operator can see the forecast, not a min() cap, is binding.
	WorkerFloor int `json:"worker_floor"`
	// ContractionCap is the pending-contraction target (#4038) that capped admits while
	// a scale-down was in flight, or nil when no contraction was pending. When it is the
	// tightest term, Limiting reads "contraction" so an operator sees the drain -- not a
	// physical/config cap -- is what is holding admits down.
	ContractionCap *int `json:"contraction_cap,omitempty"`
	// WorktreeCap is the evidence-earned isolation raise (#3185) that lifted the effective
	// cap, or 0 when no proven raise applied. When it is what set the effective cap,
	// Limiting reads "worktree" so an operator can see the A/B poison-free evidence -- not
	// a min() cap -- is what permits the fleet to run above the old shared-trunk ceiling.
	WorktreeCap int `json:"worktree_cap"`
	// WIPStarted and WIPLimit record the flow limit (wiplimit.go) when it is the binding
	// term: the started-and-unfinished units the fleet owns, and the most it may own.
	// Every other term here is a RESOURCE limit -- what the box and the seat pool can
	// carry -- so when Limiting reads "wip_limit" an operator can see the cap came from
	// work in hand, not from capacity, and that the remedy is to FINISH rather than to
	// provision. Both stay 0 unless the term binds, so the JSON is unchanged otherwise.
	WIPStarted   int    `json:"wip_started,omitempty"`
	WIPLimit     int    `json:"wip_limit,omitempty"`
	EffectiveCap int    `json:"effective_cap"`
	Limiting     string `json:"limiting"`
}

func IntPtr(n int) *int { return &n }

func HostCapacity(res HostResources) HostCapacityInfo {
	return HostCapacityWith(res, HostBudgets{})
}

// HostCapacityWith folds the host resources against explicit per-worker budgets
// (zero fields fall back to the built-in consts). HostCapacity is the
// built-in-budget shorthand; the shell reaches this via PreflightInput.Budgets.
func HostCapacityWith(res HostResources, budgets HostBudgets) HostCapacityInfo {
	b := budgets.normalized()
	info := HostCapacityInfo{
		Cores:        res.Cores,
		FreeRAMMB:    res.FreeRAMMB,
		TotalThreads: res.TotalThreads,
		Components:   map[string]int{},
	}
	if res.Cores != nil && *res.Cores > 0 {
		info.Components["cores"] = *res.Cores / b.CoresPerWorker
	}
	if res.FreeRAMMB != nil && *res.FreeRAMMB >= 0 {
		info.Components["ram"] = *res.FreeRAMMB / b.RAMMBPerWorker
	}
	if res.TotalThreads != nil && *res.TotalThreads >= 0 && res.Cores != nil && *res.Cores > 0 {
		freeThreads := *res.Cores*b.ThreadsPerCore - *res.TotalThreads
		if freeThreads < 0 {
			freeThreads = 0
		}
		info.Components["threads"] = freeThreads / b.ThreadsPerWorker
	}
	if len(info.Components) == 0 {
		return info
	}
	minComponent := 0
	// Iterate a fixed priority order (cores, ram, threads) rather than ranging the
	// Components map: Go randomizes map iteration, so on a TIE for the minimum the
	// binding -- and thus cap_terms.Limiting -- would flip between equally-tight
	// components across identical inputs. A canonical order makes the reported
	// limiter deterministic, which the #3368 forecast-floor observability (Limiting
	// "floor") relies on to be stable. These three are the only keys ever set above.
	for _, name := range []string{"cores", "ram", "threads"} {
		value, ok := info.Components[name]
		if !ok {
			continue
		}
		if info.Binding == "" || value < minComponent {
			info.Binding = name
			minComponent = value
		}
	}
	if minComponent < HostCapFloor {
		minComponent = HostCapFloor
	}
	info.HostCap = IntPtr(minComponent)
	return info
}

func EvaluatePreflight(in PreflightInput) PreflightResult {
	hostCapInfo := HostCapacityWith(in.Resources, in.Budgets)
	capacity := in.MaxWorkers
	if in.Kernel.Target != nil && *in.Kernel.Target > 0 {
		capacity = minInt(capacity, *in.Kernel.Target)
	}
	if hostCapInfo.HostCap != nil {
		capacity = minInt(capacity, *hostCapInfo.HostCap)
	}
	if capacity < 0 {
		capacity = 0
	}
	capPreSeat := capacity
	foldSeats := in.Seat.Total != nil && *in.Seat.Total > 0
	if foldSeats {
		capacity = minInt(capacity, *in.Seat.Total)
	}
	if capacity < 0 {
		capacity = 0
	}

	// Two-timescale scaling (#3368): the slow predictive loop only sets a FLOOR,
	// never a direct count; the fast reactive tick (every min() cap above) decides
	// freely and clamps UP to that floor here via max. The floor lifts the effective
	// cap back over a soft reactive dip (a low kernel/lease target) so capacity is
	// pre-warmed for an incoming ramp -- but it is bounded by the HARD ceiling (the
	// physical/config caps only: config max, host cap, seat inventory; NOT the reactive
	// lease target it is meant to override), so it can never overbook the box or the
	// seat pool. floorApplied carries the ceiling-bounded value into cap_terms.
	floorApplied := 0
	if in.WorkerFloor > 0 {
		ceiling := hardCeiling(in.MaxWorkers, hostCapInfo.HostCap, foldSeats, in.Seat.Total)
		floorApplied = minInt(in.WorkerFloor, ceiling)
		if floorApplied > capacity {
			capacity = floorApplied
		}
	}

	// Evidence-gated worktree cap raise (#3185, the payoff of epic #3165): the ~6 safe
	// ceiling (#1333) exists because a shared trunk build-poisons past ~4 workers. Once
	// per-worker worktree isolation (#3168) removes the shared build surface AND a #3180
	// A/B report witnesses ZERO poison at a measured peak, that ceiling is artificial and
	// the fleet may rise toward the PROVEN concurrency. Gated on the evidence, never the
	// flag: WorktreeProvenCap returns 0 for isolation-on-but-unproven, so this cannot
	// repeat #1334's raise-on-belief. The raise lifts capacity over the reactive LEASE
	// target (the dos [supervise].target that the poison-avoidance policy pins low) --
	// that is the artificial term -- but stays bounded by the HARD ceiling: the operator's
	// explicit config max, host capacity (#1337), and seat inventory. Bounding by
	// MaxWorkers matches the #3368 forecast-floor convention (a raising term never crosses
	// an explicit operator ceiling), so proving isolation never silently overshoots a
	// deliberately small --max-workers; an operator opts into the higher number once, and
	// the evidence then lets the fleet actually reach it instead of being held at the lease
	// target. Applied BEFORE the contraction clamp so a pending drain still bounds it.
	worktreeApplied := 0
	if proven := WorktreeProvenCap(in.WorktreeIsolation); proven > 0 {
		worktreeApplied = hardCeiling(minInt(proven, in.MaxWorkers), hostCapInfo.HostCap, foldSeats, in.Seat.Total)
		if worktreeApplied > capacity {
			capacity = worktreeApplied
		}
	}

	// Pending-contraction floor (#4038): while a scale-down / drain is in flight toward
	// ContractionTarget T, cap admits at T so no worker is admitted onto capacity being
	// reclaimed (kvcached's in_shrink guard). Applied AFTER the predictive floor so it
	// bounds a pre-warm too -- admitting onto reclaimed capacity via the forecast floor
	// is the same race. Pure lowering term; T <= 0 (the default) is a no-op.
	if in.ContractionTarget > 0 && in.ContractionTarget < capacity {
		capacity = in.ContractionTarget
	}

	aliveKernelForCap := 0
	if in.Kernel.Target != nil && *in.Kernel.Target > 0 && in.Kernel.Alive != nil {
		aliveKernelForCap = *in.Kernel.Alive
	}
	live := mathx.MaxInt(aliveKernelForCap, in.OSWorkerProcs)
	seat := accountUnattributedLiveSlots(in.Seat, live)
	headroom := capacity - live

	seatsDepleted := false
	if foldSeats && seat.Depleted && *seat.Total <= capPreSeat && seat.Leased != nil && *seat.Leased > 0 {
		seatsDepleted = true
	}

	classifyInput := in
	classifyInput.Seat = seat
	verdict, reason := classifyPreflight(classifyInput, capacity, live, seatsDepleted, hostCapInfo.HostCap)
	ok := verdict == PreflightOKVerdict
	// The flow-limit term (wiplimit.go) is applied LAST, over the fully classified
	// verdict, for the same reason the other backpressure folds are: it must only speak
	// when it is the SOLE binding term. A preflight that already refused for host, seat,
	// account, or at-cap is not growing anyway, and its verdict names a more actionable
	// cause. An unmeasured census (the zero value every existing caller passes) makes
	// this an identity function.
	return ApplyWIPLimit(PreflightResult{
		Schema:        PreflightSchema,
		OK:            ok,
		Verdict:       verdict,
		Reason:        reason,
		Workspace:     in.Workspace,
		Cap:           capacity,
		Live:          live,
		Headroom:      headroom,
		CapTerms:      capTerms(in, hostCapInfo.HostCap, capacity, floorApplied, worktreeApplied),
		MaxWorkers:    in.MaxWorkers,
		HostCap:       hostCapInfo.HostCap,
		HostCapacity:  hostCapInfo,
		Seat:          seat,
		Host:          in.Host,
		Tree:          in.Tree,
		Account:       publicAccount(in.Account),
		Kernel:        in.Kernel,
		OSWorkerProcs: in.OSWorkerProcs,
	}, in.WIP)
}

func accountUnattributedLiveSlots(seat SeatCheck, live int) SeatCheck {
	if seat.Total == nil || *seat.Total <= 0 || live <= 0 {
		return seat
	}
	leased := intValue(seat.Leased)
	missing := live - leased
	if missing <= 0 {
		return seat
	}
	total := *seat.Total
	free := intValue(seat.Free)
	if seat.Free != nil {
		adjusted := free - missing
		if adjusted < 0 {
			adjusted = 0
		}
		seat.Free = IntPtr(adjusted)
	} else {
		occupied := minInt(live, total)
		seat.Free = IntPtr(mathx.MaxInt(0, total-occupied))
	}
	seat.Leased = IntPtr(mathx.MaxInt(leased, minInt(live, total)))
	seat.Depleted = seat.Depleted || intValue(seat.Free) == 0
	seat.UnattributedLive = missing
	return seat
}

func capTerms(in PreflightInput, hostCap *int, effective, floor, worktree int) CapTerms {
	terms := CapTerms{
		ConfiguredCap: in.MaxWorkers,
		LeaseCap:      positivePtr(in.Kernel.Target),
		HostCap:       copyIntPtr(hostCap),
		SeatCap:       positivePtr(in.Seat.Total),
		WorkerFloor:   floor,
		WorktreeCap:   worktree,
		EffectiveCap:  effective,
		Limiting:      "configured",
	}
	if in.ContractionTarget > 0 {
		terms.ContractionCap = IntPtr(in.ContractionTarget)
	}
	best := in.MaxWorkers
	for _, candidate := range []struct {
		name  string
		value *int
	}{
		{name: "lease", value: terms.LeaseCap},
		{name: "host", value: terms.HostCap},
		{name: "seat", value: terms.SeatCap},
		{name: "contraction", value: terms.ContractionCap},
	} {
		if candidate.value != nil && *candidate.value < best {
			best = *candidate.value
			terms.Limiting = candidate.name
		}
	}
	if best < 0 {
		terms.Limiting = "configured"
	}
	// A RAISING term lifted the effective cap above the tightest lowering cap, so name
	// which one: the forecast floor (#3368) or the isolation evidence (#3185), not a min()
	// cap, now sets what the fleet may run. (When a raising term only matches an existing
	// lowering cap it did not raise anything, so the lowering cap stays named as the
	// limiter.) The worktree raise wins a tie because the poison-free measurement is the
	// stronger claim about what the fleet may safely run; with no raise earned
	// (worktree == 0) this reduces exactly to the prior floor-only behavior.
	if effective > best {
		switch {
		case worktree > 0 && worktree >= effective && worktree >= floor:
			terms.Limiting = "worktree"
		case floor > 0:
			terms.Limiting = "floor"
		}
	}
	return terms
}

// landSaturated is the #3574 land-contention Schmitt trigger. With the knob enabled it
// latches ON at or above HighWater, latches OFF below LowWater, and inside the
// [LowWater, HighWater) band holds whatever the prior tick decided (c.Prior). Disabled
// (the zero value) it always returns false, so classifyPreflight is byte-identical to
// before the term existed. A degenerate HighWater<=LowWater collapses to a single
// threshold at HighWater (the band is empty).
func landSaturated(c LandContention) bool {
	if !c.Enabled {
		return false
	}
	switch {
	case c.Signal >= c.HighWater:
		return true
	case c.Signal < c.LowWater:
		return false
	default:
		return c.Prior == PreflightRefuseLandSaturated
	}
}

func classifyPreflight(in PreflightInput, capacity, live int, seatsDepleted bool, hostCap *int) (string, string) {
	switch {
	case strings.TrimSpace(in.Host.Error) != "" || strings.TrimSpace(in.Kernel.Error) != "":
		reason := strmatch.FirstTrimmed(in.Host.Error, in.Kernel.Error, "a preflight safety check could not run")
		return PreflightRefuseInspect, reason
	case in.Tree.Poisoned:
		return PreflightRefuseTreePoison, fmt.Sprintf("shared tree build is red: %s", strmatch.FirstTrimmed(in.Tree.Package, in.Tree.Error, "failing package unknown"))
	case !in.Host.Safe:
		names := strings.Join(in.Host.FlaggedNames, ", ")
		if strings.TrimSpace(names) == "" {
			names = "see proc_resource_guard"
		}
		return PreflightRefuseHost, fmt.Sprintf("host resource guard flagged %d process(es): %s - reap/inspect before growing the fleet", in.Host.Flagged, names)
	case seatsDepleted:
		total, leased := intValue(in.Seat.Total), intValue(in.Seat.Leased)
		return PreflightRefuseNoSeat, fmt.Sprintf("seat pool depleted: 0 of %d session slot(s) free (%d leased to live worker(s), live=%d); a slot frees when a worker exits - refusing rather than overbook a busy account", total, leased, live)
	case landSaturated(in.LandContention):
		lc := in.LandContention
		return PreflightRefuseLandSaturated, fmt.Sprintf("land contention saturated: signal %.3f (high-water %.3f, low-water %.3f) - throttling new spawns until the serialized-land funnel drains", lc.Signal, lc.HighWater, lc.LowWater)
	case in.SettleBeforeRedecide && in.Kernel.Alive != nil && in.Kernel.Target != nil && *in.Kernel.Alive != *in.Kernel.Target:
		return PreflightObserveSettling, fmt.Sprintf("observing scaling in progress: %d alive, target %d", *in.Kernel.Alive, *in.Kernel.Target)
	case live >= capacity:
		return PreflightRefuseAtCap, fmt.Sprintf("live workers %d >= cap %d (kernel alive=%s, os procs=%d, dos target=%s, host_cap=%s, max-workers=%d)",
			live, capacity, ptrString(in.Kernel.Alive), in.OSWorkerProcs, ptrString(in.Kernel.Target), ptrString(hostCap), in.MaxWorkers)
	case !in.Account.Available:
		// A failed account PROBE (roster unreadable / scan timed out) is not the same as
		// a probe that ran and found the pool saturated. Refuse to INSPECT so a timed-out
		// scan is never misread as "every account is busy" -- the operator needs to fix a
		// broken probe, not wait for capacity that was never actually measured.
		if detail := strings.TrimSpace(in.Account.Error); detail != "" {
			return PreflightRefuseInspect, "account availability scan could not complete: " + detail + " - refusing on an unmeasured account pool (a probe failure, not saturation)"
		}
		// Genuine no-account: the scan ran, there is spawn headroom (live<cap, checked
		// above), but no account is free at the requested tier. Name the live/cap so this
		// reads as "no free seat", not "at cap", and cite each blocked seat's REASON.
		reason := fmt.Sprintf("switcher has no available worker account at the requested tier (live=%d, cap=%d)", live, capacity)
		if blocked := blockedAccountsSummary(in.Account); blocked != "" {
			reason += " [blocked: " + blocked + "]"
		}
		if detail := strings.TrimSpace(in.Account.Reason); detail != "" {
			reason += ": " + detail
		}
		return PreflightRefuseNoAccount, reason
	default:
		return PreflightOKVerdict, fmt.Sprintf("safe to spawn: host clean, account '%s' (t%v) free, %d/%d live (headroom %d)",
			in.Account.Tag, in.Account.Tier, live, capacity, capacity-live)
	}
}

func (r PreflightResult) Map() map[string]any {
	return map[string]any{
		"schema":          r.Schema,
		"ok":              r.OK,
		"verdict":         r.Verdict,
		"reason":          r.Reason,
		"workspace":       r.Workspace,
		"cap":             r.Cap,
		"live":            r.Live,
		"headroom":        r.Headroom,
		"cap_terms":       r.CapTerms.Map(),
		"max_workers":     r.MaxWorkers,
		"host_cap":        ptrAny(r.HostCap),
		"host_capacity":   r.HostCapacity.Map(),
		"seat":            r.Seat.Map(),
		"host":            r.Host.Map(),
		"account":         r.Account.Map(),
		"kernel":          r.Kernel.Map(),
		"os_worker_procs": r.OSWorkerProcs,
	}
}

func (c CapTerms) Map() map[string]any {
	return map[string]any{
		"configured_cap":  c.ConfiguredCap,
		"contraction_cap": ptrAny(c.ContractionCap),
		"lease_cap":       ptrAny(c.LeaseCap),
		"host_cap":        ptrAny(c.HostCap),
		"seat_cap":        ptrAny(c.SeatCap),
		"worker_floor":    c.WorkerFloor,
		"worktree_cap":    c.WorktreeCap,
		"wip_started":     c.WIPStarted,
		"wip_limit":       c.WIPLimit,
		"effective_cap":   c.EffectiveCap,
		"limiting":        c.Limiting,
	}
}

func (h HostCapacityInfo) Map() map[string]any {
	return map[string]any{
		"cores":         ptrAny(h.Cores),
		"free_ram_mb":   ptrAny(h.FreeRAMMB),
		"total_threads": ptrAny(h.TotalThreads),
		"components":    h.Components,
		"host_cap":      ptrAny(h.HostCap),
		"binding":       h.Binding,
	}
}

func (h HostCheck) Map() map[string]any {
	return map[string]any{"safe": h.Safe, "error": h.Error, "flagged": h.Flagged, "flagged_names": h.FlaggedNames}
}

func (a AccountCheck) Map() map[string]any {
	return map[string]any{
		"available":        a.Available,
		"tag":              a.Tag,
		"dir":              a.Dir,
		"tier":             a.Tier,
		"model":            a.Model,
		"reason":           a.Reason,
		"error":            a.Error,
		"blocked":          a.Blocked,
		"blocked_accounts": a.BlockedAccounts,
		"login_status":     a.LoginStatus,
		"can_serve":        a.CanServe,
	}
}

func (k KernelCheck) Map() map[string]any {
	return map[string]any{"alive": ptrAny(k.Alive), "target": ptrAny(k.Target), "error": k.Error, "verdict": k.Verdict}
}

func (s SeatCheck) Map() map[string]any {
	return map[string]any{
		"total":             ptrAny(s.Total),
		"free":              ptrAny(s.Free),
		"leased":            ptrAny(s.Leased),
		"depleted":          s.Depleted,
		"unattributed_live": s.UnattributedLive,
		"skipped":           s.Skipped,
		"error":             s.Error,
	}
}

func publicAccount(a AccountCheck) AccountCheck {
	return AccountCheck{
		Available: a.Available, Tag: a.Tag, Dir: a.Dir, Tier: a.Tier, Model: a.Model,
		Reason: a.Reason, Error: a.Error, Blocked: append([]string(nil), a.Blocked...),
		BlockedAccounts: append([]BlockedAccount(nil), a.BlockedAccounts...),
		LoginStatus:     a.LoginStatus, CanServe: a.CanServe,
	}
}

func ptrAny(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func copyIntPtr(p *int) *int {
	if p == nil {
		return nil
	}
	return IntPtr(*p)
}

func positivePtr(p *int) *int {
	if p == nil || *p <= 0 {
		return nil
	}
	return IntPtr(*p)
}

func intValue(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func ptrString(p *int) string {
	if p == nil {
		return "<nil>"
	}
	return fmt.Sprint(*p)
}

// blockedAccountsSummary renders the per-account block reasons for a NO_ACCOUNT verdict
// as "tag=reason" pairs (throttled / needs-login / at session cap), so the refusal names
// WHY each seat in the target tier was turned away. It prefers the structured
// BlockedAccounts and falls back to the flat tag list when only tags are available.
func blockedAccountsSummary(a AccountCheck) string {
	if len(a.BlockedAccounts) > 0 {
		parts := make([]string, 0, len(a.BlockedAccounts))
		for _, b := range a.BlockedAccounts {
			tag := strings.TrimSpace(b.Tag)
			if tag == "" {
				tag = strings.TrimSpace(b.Account)
			}
			reason := strings.TrimSpace(b.Reason)
			switch {
			case tag != "" && reason != "":
				parts = append(parts, tag+"="+reason)
			case tag != "":
				parts = append(parts, tag)
			case reason != "":
				parts = append(parts, reason)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "; ")
		}
	}
	return strings.Join(nonEmpty(a.Blocked), ", ")
}

func nonEmpty(in []string) []string {
	out := make([]string, 0, len(in))
	for _, v := range in {
		v = strings.TrimSpace(v)
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

// hardCeiling clamps a RAISING capacity term to the hard ceilings only — host capacity
// (#1337) and, when seats are being folded, seat inventory — and never below zero. Both
// raising terms bound themselves this way: the #3368 predictive worker floor and the
// #3185 evidence-gated worktree raise. Sharing one clamp is what keeps the guarantee
// uniform — neither raise can overbook the box or the seat pool, and (deliberately)
// neither is bounded by the REACTIVE lease target each one exists to override.
//
// foldSeats is passed rather than inferred from seatTotal because the fold is gated on a
// POSITIVE inventory, not merely a present one: a seat total of 0 means "seat data not
// meaningful here", and must not clamp the ceiling to zero.
func hardCeiling(ceiling int, hostCap *int, foldSeats bool, seatTotal *int) int {
	if hostCap != nil {
		ceiling = minInt(ceiling, *hostCap)
	}
	if foldSeats {
		ceiling = minInt(ceiling, *seatTotal)
	}
	if ceiling < 0 {
		return 0
	}
	return ceiling
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
