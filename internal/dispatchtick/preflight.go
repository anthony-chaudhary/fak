package dispatchtick

import (
	"fmt"
	"strings"
)

const (
	PreflightSchema = "fleet-dispatch-preflight/1"

	PreflightOKVerdict       = "SPAWN_OK"
	PreflightRefuseHost      = "REFUSE_HOST"
	PreflightRefuseNoAccount = "REFUSE_NO_ACCOUNT"
	PreflightRefuseNoSeat    = "REFUSE_NO_SEAT"
	PreflightRefuseAtCap     = "REFUSE_AT_CAP"
	PreflightRefuseInspect   = "REFUSE_INSPECT"
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
	Available   bool     `json:"available"`
	Tag         string   `json:"tag,omitempty"`
	Dir         string   `json:"dir,omitempty"`
	Tier        any      `json:"tier,omitempty"`
	Model       string   `json:"model,omitempty"`
	Reason      string   `json:"reason,omitempty"`
	Error       string   `json:"error,omitempty"`
	Blocked     []string `json:"blocked,omitempty"`
	LoginStatus string   `json:"login_status,omitempty"`
	CanServe    *bool    `json:"can_serve,omitempty"`
}

type KernelCheck struct {
	Alive   *int   `json:"alive"`
	Target  *int   `json:"target"`
	Error   string `json:"error,omitempty"`
	Verdict string `json:"verdict,omitempty"`
}

type SeatCheck struct {
	Total    *int   `json:"total"`
	Free     *int   `json:"free,omitempty"`
	Leased   *int   `json:"leased,omitempty"`
	Depleted bool   `json:"depleted,omitempty"`
	Skipped  string `json:"skipped,omitempty"`
	Error    string `json:"error,omitempty"`
}

type PreflightInput struct {
	Workspace  string
	MaxWorkers int
	Host       HostCheck
	Account    AccountCheck
	Kernel     KernelCheck
	Seat       SeatCheck
	Resources  HostResources
	// Budgets are the per-worker host charges the capacity gradient folds; the
	// zero value means the built-in Host* consts. The shell passes
	// DefaultHostBudgets() so the FAK_HOST_* env knobs reach the fold.
	Budgets       HostBudgets
	OSWorkerProcs int
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
	Account       AccountCheck     `json:"account"`
	Kernel        KernelCheck      `json:"kernel"`
	OSWorkerProcs int              `json:"os_worker_procs"`
}

type CapTerms struct {
	ConfiguredCap int    `json:"configured_cap"`
	LeaseCap      *int   `json:"lease_cap"`
	HostCap       *int   `json:"host_cap"`
	SeatCap       *int   `json:"seat_cap"`
	EffectiveCap  int    `json:"effective_cap"`
	Limiting      string `json:"limiting"`
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
	for name, value := range info.Components {
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

	aliveKernelForCap := 0
	if in.Kernel.Target != nil && *in.Kernel.Target > 0 && in.Kernel.Alive != nil {
		aliveKernelForCap = *in.Kernel.Alive
	}
	live := maxInt(aliveKernelForCap, in.OSWorkerProcs)
	headroom := capacity - live

	seatsDepleted := false
	if foldSeats && in.Seat.Depleted && *in.Seat.Total <= capPreSeat && in.Seat.Leased != nil && *in.Seat.Leased > 0 {
		seatsDepleted = true
	}

	verdict, reason := classifyPreflight(in, capacity, live, seatsDepleted, hostCapInfo.HostCap)
	ok := verdict == PreflightOKVerdict
	return PreflightResult{
		Schema:        PreflightSchema,
		OK:            ok,
		Verdict:       verdict,
		Reason:        reason,
		Workspace:     in.Workspace,
		Cap:           capacity,
		Live:          live,
		Headroom:      headroom,
		CapTerms:      capTerms(in, hostCapInfo.HostCap, capacity),
		MaxWorkers:    in.MaxWorkers,
		HostCap:       hostCapInfo.HostCap,
		HostCapacity:  hostCapInfo,
		Seat:          in.Seat,
		Host:          in.Host,
		Account:       publicAccount(in.Account),
		Kernel:        in.Kernel,
		OSWorkerProcs: in.OSWorkerProcs,
	}
}

func capTerms(in PreflightInput, hostCap *int, effective int) CapTerms {
	terms := CapTerms{
		ConfiguredCap: in.MaxWorkers,
		LeaseCap:      positivePtr(in.Kernel.Target),
		HostCap:       copyIntPtr(hostCap),
		SeatCap:       positivePtr(in.Seat.Total),
		EffectiveCap:  effective,
		Limiting:      "configured",
	}
	best := in.MaxWorkers
	for _, candidate := range []struct {
		name  string
		value *int
	}{
		{name: "lease", value: terms.LeaseCap},
		{name: "host", value: terms.HostCap},
		{name: "seat", value: terms.SeatCap},
	} {
		if candidate.value != nil && *candidate.value < best {
			best = *candidate.value
			terms.Limiting = candidate.name
		}
	}
	if best < 0 {
		terms.Limiting = "configured"
	}
	return terms
}

func classifyPreflight(in PreflightInput, capacity, live int, seatsDepleted bool, hostCap *int) (string, string) {
	switch {
	case strings.TrimSpace(in.Host.Error) != "" || strings.TrimSpace(in.Kernel.Error) != "":
		reason := firstNonEmpty(in.Host.Error, in.Kernel.Error, "a preflight safety check could not run")
		return PreflightRefuseInspect, reason
	case !in.Host.Safe:
		names := strings.Join(in.Host.FlaggedNames, ", ")
		if strings.TrimSpace(names) == "" {
			names = "see proc_resource_guard"
		}
		return PreflightRefuseHost, fmt.Sprintf("host resource guard flagged %d process(es): %s - reap/inspect before growing the fleet", in.Host.Flagged, names)
	case seatsDepleted:
		total, leased := intValue(in.Seat.Total), intValue(in.Seat.Leased)
		return PreflightRefuseNoSeat, fmt.Sprintf("seat pool depleted: 0 of %d routable seat(s) free (%d leased to live worker(s), live=%d); a seat frees when a worker exits - refusing rather than double-book a busy seat", total, leased, live)
	case live >= capacity:
		return PreflightRefuseAtCap, fmt.Sprintf("live workers %d >= cap %d (kernel alive=%s, os procs=%d, dos target=%s, host_cap=%s, max-workers=%d)",
			live, capacity, ptrString(in.Kernel.Alive), in.OSWorkerProcs, ptrString(in.Kernel.Target), ptrString(hostCap), in.MaxWorkers)
	case !in.Account.Available:
		blocked := strings.Join(nonEmpty(in.Account.Blocked), ", ")
		reason := "switcher has no available worker account at the requested tier"
		if blocked != "" {
			reason += " (blocked: " + blocked + ")"
		}
		if detail := firstNonEmpty(in.Account.Reason, in.Account.Error); detail != "" {
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
		"configured_cap": c.ConfiguredCap,
		"lease_cap":      ptrAny(c.LeaseCap),
		"host_cap":       ptrAny(c.HostCap),
		"seat_cap":       ptrAny(c.SeatCap),
		"effective_cap":  c.EffectiveCap,
		"limiting":       c.Limiting,
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
		"available":    a.Available,
		"tag":          a.Tag,
		"dir":          a.Dir,
		"tier":         a.Tier,
		"model":        a.Model,
		"reason":       a.Reason,
		"error":        a.Error,
		"blocked":      a.Blocked,
		"login_status": a.LoginStatus,
		"can_serve":    a.CanServe,
	}
}

func (k KernelCheck) Map() map[string]any {
	return map[string]any{"alive": ptrAny(k.Alive), "target": ptrAny(k.Target), "error": k.Error, "verdict": k.Verdict}
}

func (s SeatCheck) Map() map[string]any {
	return map[string]any{"total": ptrAny(s.Total), "free": ptrAny(s.Free), "leased": ptrAny(s.Leased), "depleted": s.Depleted, "skipped": s.Skipped, "error": s.Error}
}

func publicAccount(a AccountCheck) AccountCheck {
	return AccountCheck{
		Available: a.Available, Tag: a.Tag, Dir: a.Dir, Tier: a.Tier, Model: a.Model,
		Reason: a.Reason, Error: a.Error, Blocked: append([]string(nil), a.Blocked...),
		LoginStatus: a.LoginStatus, CanServe: a.CanServe,
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

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
