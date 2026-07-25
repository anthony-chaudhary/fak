package executionroute

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// HealthFromFleetStatus is the live-read adapter that turns ONE current fleet
// health source — the per-account rows of `fak fleet-accounts status --json`
// (fleetaccounts.StatusReport.Accounts) — into the routing HealthReport. It proves
// a live fleet source can populate the health input the selector consumes.
//
// A harness candidate is a whole product (claude / codex / opencode), so the
// many seats of a product fold into ONE reading keyed by that product; the
// selector matches it to a profile by name or alias (opencode -> openai-generic).
// The fold is fail-open toward availability: a product is HealthAvailable when ANY
// counted seat is ready with a free slot. With no ready seat it reports the
// strongest exclusion its seats show, in a fixed precedence — draining (every
// seat at cap) over cooldown (a usage cap) over unavailable (an auth/access
// block) — so the reason is deterministic. Freshness is the freshest seat's
// registry age (seconds); an unstamped age is carried as -1 so a freshness bound
// treats it as stale. Provenance is the set of contributing status sources.
//
// maxAgeSeconds becomes the report's freshness bound (0 disables it). Only worker
// rows that count toward capacity are folded; duplicate / shared-pool / non-worker
// rows are ignored, matching the status report's own capacity view.
func HealthFromFleetStatus(rows []fleetaccounts.StatusAccount, maxAgeSeconds int64) HealthReport {
	type agg struct {
		anyReady    bool
		freeSlots   int
		sessionCap  int
		sawDraining bool
		sawCooldown bool
		coolDetail  string
		blockDetail string
		sources     map[string]bool
		ageSeconds  int64 // freshest (smallest) known age in seconds; valid only when ageKnown
		ageKnown    bool
	}
	byProduct := map[string]*agg{}
	order := []string{}
	get := func(product string) *agg {
		key := normalize(product)
		a := byProduct[key]
		if a == nil {
			a = &agg{sources: map[string]bool{}}
			byProduct[key] = a
			order = append(order, key)
		}
		return a
	}

	for _, row := range rows {
		if row.Kind != string(fleetaccounts.KindWorker) || !row.CapacityCounted {
			continue
		}
		product := strings.TrimSpace(row.Product)
		if product == "" {
			continue
		}
		a := get(product)
		if src := strings.TrimSpace(row.StatusSource); src != "" {
			a.sources[src] = true
		}
		if row.RegistryAgeMin != nil {
			age := int64(*row.RegistryAgeMin * 60)
			if age < 0 {
				age = 0
			}
			if !a.ageKnown || age < a.ageSeconds {
				a.ageSeconds, a.ageKnown = age, true
			}
		}
		switch row.State {
		case "ready", "leased":
			a.sessionCap += row.SessionCap
			a.freeSlots += row.FreeSlots
			if row.FreeSlots > 0 {
				a.anyReady = true
			}
		case "full":
			a.sessionCap += row.SessionCap
			a.sawDraining = true
		case "usage":
			a.sawCooldown = true
			if a.coolDetail == "" {
				a.coolDetail = statusDetail(row.Reason, row.Reset)
			}
		default: // auth / access / credit / blocked / any other block kind
			if a.blockDetail == "" {
				a.blockDetail = statusDetail(row.Reason, row.BlockKind)
			}
		}
	}

	report := HealthReport{MaxAgeSeconds: maxAgeSeconds}
	if len(order) == 0 {
		return report
	}
	report.Candidates = make(map[string]HarnessHealth, len(order))
	for _, key := range order {
		a := byProduct[key]
		h := HarnessHealth{Source: fleetSource(a.sources), AgeSeconds: -1}
		if a.ageKnown {
			h.AgeSeconds = a.ageSeconds
		}
		switch {
		case a.anyReady && a.freeSlots > 0:
			h.State = HealthAvailable
			h.FreeSlots = a.freeSlots
			h.SessionCap = a.sessionCap
		case a.sawDraining:
			h.State = HealthDraining
			h.SessionCap = a.sessionCap
			h.Detail = "every seat at session cap"
		case a.sawCooldown:
			h.State = HealthCooldown
			h.Detail = a.coolDetail
		default:
			h.State = HealthUnavailable
			h.Detail = a.blockDetail
			if h.Detail == "" {
				h.Detail = "no serving seat"
			}
		}
		report.Candidates[key] = h
	}
	return report
}

// statusDetail composes a compact reason from a status row's free-text reason and
// its structured fallback (reset time or block kind), preferring the reason.
func statusDetail(reason, fallback string) string {
	reason = strings.TrimSpace(reason)
	if reason != "" {
		return reason
	}
	fallback = strings.TrimSpace(fallback)
	return fallback
}

// fleetSource renders the provenance label from the set of contributing fleet
// status sources, sorted for determinism.
func fleetSource(set map[string]bool) string {
	if len(set) == 0 {
		return "fleet-account-status"
	}
	srcs := make([]string, 0, len(set))
	for s := range set {
		srcs = append(srcs, s)
	}
	sort.Strings(srcs)
	return "fleet-account-status:" + strings.Join(srcs, "+")
}
