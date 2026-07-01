package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// computeauto — `fak c --auto` (#939): health/cost/quota-aware automatic compute-target
// selection with failover, the third console-chat axis on top of the #937 registry and
// the #938 named selector.
//
// THE POLICY (documented, deterministic).
//  1. HEALTH (witnessed): probe every registered target's /healthz. A target that
//     answers is a candidate; one that is DOWN is excluded. A target with no /healthz
//     endpoint (the real Anthropic API) is unprobed and assumed reachable — it stays a
//     candidate, so the always-available metered backend is the last-resort fallback.
//  2. COST / LOCALITY (witnessed, from the registry): rank candidates by costClass —
//     local in-kernel (free) < your-own-Mac (free, remote hop) < paid GCP GPU <
//     metered Anthropic. Cheapest/most-local wins.
//  3. FIT: a remote target's /healthz proves LIVENESS, not model fit (and fak never reads
//     the body), so a remote fit is reported n/a — the remote host owns its memory. The
//     local in-kernel target consults internal/compute HostMemoryInfo (a precise
//     FitsMemoryPlan verdict needs a declared model, noted in the row).
//  4. QUOTA: a [stub] signal — NOT yet wired to a live `fak accounts` rotation read
//     (order=by_reset,near_cap_util,avoid_reserved). It is labeled honestly and never
//     marks a seat down, so --auto cannot claim a quota read it does not perform.
//
// The selection ENGINE is the previously-dormant internal/gateway Router/Decision/
// Fallbacks (routing.go) wired into a live path: each target becomes one Tier
// (CostPerMTok = costClass), the /healthz verdict drives SetHealth, and a CostBased
// Route returns the cheapest healthy winner plus the ordered failover ladder. This is a
// single launch-time decision (a launcher, not a daemon): a DOWN target is never a
// candidate, so the winner is always reachable — failover means the dead top choice is
// simply not selected and the next-cheapest healthy target is.

const computeTargetAutoSchema = "fak.computetarget.auto.v1"

// costClass returns a deterministic cost/locality ordinal for --auto ranking: lower is
// cheaper/more-local and is preferred. It is derived only from registry-declared fields
// (locality, kind, cost note), so the ranking is witnessed, never guessed. For the four
// built-ins it yields local(0) < mac(110) < gcp(120) < anthropic(150).
func (t computeTarget) costClass() int {
	rank := 0
	if t.Locality == localityRemote {
		rank += 100 // a remote hop costs more (latency + egress) than local compute
	}
	switch t.Kind {
	case targetLocalSpawn:
		rank += 0 // your own box, in-kernel
	case targetGatewayURL:
		rank += 10
	case targetProviderProxy:
		rank += 30 // a metered upstream provider is the last resort
	}
	note := strings.ToLower(t.CostNote)
	switch {
	case strings.Contains(note, "no per-token cost"):
		rank += 0 // own hardware, no marginal token cost
	case strings.Contains(note, "metered") || strings.Contains(note, "per token"):
		rank += 20
	case strings.Contains(note, "paid"):
		rank += 10
	}
	return rank
}

// autoSignal is one provenance-labeled signal in the ranked decision. Provenance is the
// honesty contract: "witnessed" (fak measured it this run), "stub" (a placeholder not
// wired to a live read — never trusted as truth), or "n/a" (does not apply here).
type autoSignal struct {
	State      string `json:"state"`
	Provenance string `json:"provenance"` // witnessed | stub | n/a
	Detail     string `json:"detail,omitempty"`
}

// autoTargetRow is one target's place in the ranked decision.
type autoTargetRow struct {
	Name      string         `json:"name"`
	Kind      targetKind     `json:"kind"`
	Locality  targetLocality `json:"locality"`
	CostClass int            `json:"cost_class"`
	Health    autoSignal     `json:"health"`
	Cred      autoSignal     `json:"cred"`
	Fit       autoSignal     `json:"fit"`
	Quota     autoSignal     `json:"quota"`
	Candidate bool           `json:"candidate"`      // passed the health+cred gate (launchable)
	Selected  bool           `json:"selected"`       // the winner
	Rank      int            `json:"rank,omitempty"` // 1=winner, 2..=fallback order; 0=excluded
}

// autoDecisionReport is the stable --json shape for `fak c --auto --json`.
type autoDecisionReport struct {
	Schema   string          `json:"schema"`
	Strategy string          `json:"strategy"`
	Targets  []autoTargetRow `json:"targets"`
	Winner   string          `json:"winner,omitempty"`
	Reason   string          `json:"reason"`
}

// healthSignal converts a probe verdict into a provenance-labeled health signal. up/down
// are witnessed (a real probe); n/a (no /healthz) is labeled n/a with the assumed-reachable
// caveat so it is never read as a witnessed "up".
func healthSignal(h targetHealth) autoSignal {
	switch h.State {
	case "up":
		detail := h.Detail
		if detail == "" {
			detail = "live /healthz 200"
		}
		return autoSignal{State: "up", Provenance: "witnessed", Detail: detail}
	case "down":
		return autoSignal{State: "down", Provenance: "witnessed", Detail: h.Detail}
	default: // n/a
		detail := h.Detail
		if detail == "" {
			detail = "no /healthz endpoint; unprobed, assumed reachable"
		}
		return autoSignal{State: "n/a", Provenance: "n/a", Detail: detail}
	}
}

// fitForTarget reports the memory-fit signal. A remote target's /healthz proves LIVENESS,
// not model fit — and probe() never reads the body, so a 200 (even from a mock-mode
// gateway) is no evidence the model fits — so a remote fit is reported n/a, never a
// fabricated witnessed "ok". The local in-kernel target consults internal/compute
// HostMemoryInfo for a witnessed host-memory read (a precise FitsMemoryPlan verdict needs
// a declared model, which the registry's local target does not carry — noted honestly).
func fitForTarget(t computeTarget) autoSignal {
	if t.Kind == targetLocalSpawn {
		total, free, known := compute.HostMemoryInfo(compute.Default())
		if !known {
			return autoSignal{State: "unknown", Provenance: "n/a", Detail: "host memory not probeable on this backend (cpu-ref floor)"}
		}
		return autoSignal{State: "advisory", Provenance: "witnessed",
			Detail: fmt.Sprintf("%s free of %s host memory; declare a model (fak serve --gguf) for a precise FitsMemoryPlan verdict",
				humanBytes(free), humanBytes(total))}
	}
	return autoSignal{State: "n/a", Provenance: "n/a", Detail: "remote backend; /healthz proves liveness, not model fit — the remote host's concern"}
}

// quotaForTarget reports the quota signal. It is a [stub] for a metered provider seat
// (Anthropic) — NOT wired to a live `fak accounts` rotation read — and n/a otherwise. It
// never marks a target down, so --auto cannot claim a quota read it does not perform.
func quotaForTarget(t computeTarget) autoSignal {
	if t.Kind == targetProviderProxy {
		return autoSignal{State: "assumed-available", Provenance: "stub",
			Detail: "not wired to a live fak accounts quota read (order=by_reset,near_cap_util,avoid_reserved); a near-cap seat is not yet detected here"}
	}
	return autoSignal{State: "n/a", Provenance: "n/a", Detail: "not a metered provider seat"}
}

// credForTarget reports whether a target's launch credential is present — the signal that
// completes the failover gate. A target can answer /healthz (which is unauthenticated on
// the Mac gateway) yet be un-launchable because its declared bearer env var is empty, so
// selecting on health alone crowns a target that then dies at launch. Exemptions mirror
// buildTUIAgentGatewayReport's own bearer tolerance: the anthropic provider-proxy uses
// OAuth via guard (no gateway bearer), and a target that declares no CredEnv (the local
// in-kernel serve) needs none. A target that DECLARES a CredEnv is taken at its word: the
// var must be set. present/absent are witnessed (fak read the environment this run).
func credForTarget(t computeTarget, getenv func(string) string) autoSignal {
	if t.Kind == targetProviderProxy {
		return autoSignal{State: "n/a", Provenance: "n/a", Detail: "OAuth via guard; no gateway bearer required here"}
	}
	env := strings.TrimSpace(t.CredEnv)
	if env == "" {
		return autoSignal{State: "n/a", Provenance: "n/a", Detail: "target declares no credential env"}
	}
	if strings.TrimSpace(getenv(env)) != "" {
		return autoSignal{State: "present", Provenance: "witnessed", Detail: "$" + env + " is set"}
	}
	return autoSignal{State: "absent", Provenance: "witnessed", Detail: "$" + env + " is empty — reachable but not launchable; export it or `fak c --list-targets`"}
}

// autoSelectComputeTarget probes every registered target, wires the dormant gateway
// Router (CostBased) over the candidates, and returns the ranked decision plus the
// winning target. It returns an error (no launch) when no target is launchable.
func autoSelectComputeTarget(parent context.Context, reg *targetRegistry, hc *http.Client, perProbe time.Duration) (autoDecisionReport, *computeTarget, error) {
	type probed struct {
		t      computeTarget
		health targetHealth
	}
	targets := reg.all()
	ps := make([]probed, 0, len(targets))
	for _, t := range targets {
		ctx, cancel := context.WithTimeout(parent, perProbe)
		ps = append(ps, probed{t: t, health: t.probe(ctx, hc)})
		cancel()
	}
	// Cost-ascending so the router's fallback chain (ordered by capacity, all unbounded)
	// preserves cost order: cheapest healthy first, then the next cheapest.
	sort.SliceStable(ps, func(i, j int) bool { return ps[i].t.costClass() < ps[j].t.costClass() })

	tiers := make([]gateway.Tier, 0, len(ps))
	byName := make(map[string]computeTarget, len(ps))
	for _, p := range ps {
		tiers = append(tiers, gateway.Tier{
			Name:            p.t.Name,
			Model:           p.t.Model,
			MaxPromptTokens: 0, // a compute target is not prompt-size-bounded here
			CostPerMTok:     float64(p.t.costClass()),
			Interactive:     true, // a chat launch is always interactive
		})
		byName[p.t.Name] = p.t
	}
	router, err := gateway.NewRouter(gateway.RouterConfig{Strategy: gateway.StrategyCostBased, Tiers: tiers})
	if err != nil {
		return autoDecisionReport{}, nil, err
	}
	// Launchability gate: a candidate must be BOTH reachable AND credentialed. A DOWN
	// probe is unhealthy; up and n/a (unprobed, assumed reachable) pass the health half.
	// The cred half excludes a target whose declared bearer env var is empty — a target
	// that passes an unauthenticated /healthz but would die at launch. Excluding it here
	// is what makes --auto fail OVER a credential-gated target instead of crowning it.
	// Quota is a [stub] and never marks a seat down.
	candidate := make(map[string]bool, len(ps))
	healthByName := make(map[string]targetHealth, len(ps))
	credByName := make(map[string]autoSignal, len(ps))
	for _, p := range ps {
		cred := credForTarget(p.t, os.Getenv)
		credByName[p.t.Name] = cred
		launchable := p.health.State != "down" && cred.State != "absent"
		router.SetHealth(p.t.Name, launchable)
		candidate[p.t.Name] = launchable
		healthByName[p.t.Name] = p.health
	}

	report := autoDecisionReport{Schema: computeTargetAutoSchema, Strategy: string(gateway.StrategyCostBased)}
	dec, routeErr := router.Route(gateway.RequestClass{Latency: gateway.LatencyInteractive})
	rank := map[string]int{}
	var winnerName string
	if routeErr == nil {
		winnerName = dec.Tier.Name
		rank[winnerName] = 1
		for i, fb := range dec.Fallbacks {
			rank[fb.Name] = i + 2
		}
		report.Winner = winnerName
		// "healthy" only when a real /healthz probe answered; a no-/healthz target (the
		// real Anthropic API) is "assumed-reachable", never claimed as witnessed-healthy.
		winnerDesc := "cheapest healthy target"
		if healthByName[winnerName].State == "n/a" {
			winnerDesc = "cheapest reachable target (assumed-reachable, no /healthz)"
		}
		report.Reason = fmt.Sprintf("%s: %s (cost_class=%d); %s",
			winnerDesc, winnerName, byName[winnerName].costClass(), dec.Reason)
	} else {
		report.Reason = routeErr.Error()
	}
	for _, p := range ps {
		report.Targets = append(report.Targets, autoTargetRow{
			Name:      p.t.Name,
			Kind:      p.t.Kind,
			Locality:  p.t.Locality,
			CostClass: p.t.costClass(),
			Health:    healthSignal(p.health),
			Cred:      credByName[p.t.Name],
			Fit:       fitForTarget(p.t),
			Quota:     quotaForTarget(p.t),
			Candidate: candidate[p.t.Name],
			Selected:  p.t.Name == winnerName,
			Rank:      rank[p.t.Name],
		})
	}
	if routeErr != nil {
		return report, nil, fmt.Errorf("no launchable compute target (every registered target is down or missing its credential): %w", routeErr)
	}
	w := byName[winnerName]
	return report, &w, nil
}

// renderAutoDecision writes the human ranked-decision table: the cost-ordered targets,
// each with its witnessed health/fit and the honestly-labeled quota stub, the winner, and
// the policy line. A [stub]/[n/a] signal is tagged inline so no placeholder reads as truth.
func renderAutoDecision(w io.Writer, rep autoDecisionReport) {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "fak c --auto\tstrategy=%s\n", rep.Strategy)
	fmt.Fprintln(tw, "RANK\tTARGET\tLOCALITY\tCOST\tHEALTH\tCRED\tFIT\tQUOTA")
	for _, t := range rep.Targets {
		rankCell := "-"
		if t.Rank > 0 {
			rankCell = strconv.Itoa(t.Rank)
		}
		name := t.Name
		if t.Selected {
			name += " *"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n",
			rankCell, name, t.Locality, t.CostClass,
			autoSignalCell(t.Health), autoSignalCell(t.Cred), autoSignalCell(t.Fit), autoSignalCell(t.Quota))
	}
	tw.Flush()
	if rep.Winner != "" {
		fmt.Fprintf(w, "winner: %s — %s\n", rep.Winner, rep.Reason)
	} else {
		fmt.Fprintf(w, "no winner: %s\n", rep.Reason)
	}
	fmt.Fprintln(w, "policy: launchable first — reachable (healthy or assumed-reachable, live /healthz where present) AND its credential present — then cheapest/most-local (cost_class asc: local<mac<gcp<anthropic).")
	fmt.Fprintln(w, "quota is a [stub] signal — not yet a live `fak accounts` rotation read; it never excludes a target.")
}

// autoSignalCell renders one signal for the human table, tagging a non-witnessed signal
// with its provenance so a placeholder is never mistaken for a measurement.
func autoSignalCell(s autoSignal) string {
	switch s.Provenance {
	case "witnessed":
		return s.State
	case "stub":
		return s.State + " [stub]"
	default:
		return s.State + " [n/a]"
	}
}
