package gateway

import (
	"net/http"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

const routerStatusSchema = "fak-router-status/v1"

type routerStatusResponse struct {
	Schema         string              `json:"schema"`
	RoutingEnabled bool                `json:"routing_enabled"`
	DefaultModel   string              `json:"default_model"`
	Routes         []routerRouteStatus `json:"routes"`
	Totals         routerTotalsStatus  `json:"totals"`
}

type routerRouteStatus struct {
	Model          string               `json:"model"`
	ResidencyFloor string               `json:"residency_floor,omitempty"`
	Primary        routerTargetStatus   `json:"primary"`
	Fallbacks      []routerTargetStatus `json:"fallbacks"`
}

type routerTargetStatus struct {
	Account             string `json:"account"`
	Kind                string `json:"kind"`
	UpstreamModel       string `json:"upstream_model"`
	Locality            string `json:"locality"`
	State               string `json:"state"`
	CooldownRemainingMs int64  `json:"cooldown_remaining_ms"`
}

type routerTotalsStatus struct {
	Attempts  int64 `json:"attempts"`
	Successes int64 `json:"successes"`
	Failovers int64 `json:"failovers"`
}

func targetLocalityString(t modelroute.Target) string {
	if t.Zone().SelfHosted() {
		return "self_hosted"
	}
	return "vendor"
}

func buildRouterTargetStatus(t modelroute.Target, snap routeCircuitSnapshot) routerTargetStatus {
	state := "healthy"
	var remMs int64
	if acct, ok := snap.accounts[t.Account]; ok && acct.cooldownRemaining > 0 {
		state = "cooldown"
		remMs = acct.cooldownRemaining.Milliseconds()
	}
	return routerTargetStatus{
		Account:             t.Account,
		Kind:                string(t.Kind),
		UpstreamModel:       t.UpstreamModel,
		Locality:            targetLocalityString(t),
		State:               state,
		CooldownRemainingMs: remMs,
	}
}

func (s *Server) handleRouterStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	snap := s.circuitTracker().snapshot()
	resp := routerStatusResponse{
		Schema:         routerStatusSchema,
		RoutingEnabled: s.roster != nil,
		DefaultModel:   s.model,
		Routes:         make([]routerRouteStatus, 0),
		Totals: routerTotalsStatus{
			Attempts:  snap.totalAttempts,
			Successes: snap.totalSuccesses,
			Failovers: snap.totalFailovers,
		},
	}
	if s.roster != nil {
		models := make([]string, 0, len(s.roster.Bindings))
		for _, b := range s.roster.Bindings {
			if b.Model != "" {
				models = append(models, b.Model)
			}
		}
		sort.Strings(models)
		for _, model := range models {
			chain, err := s.roster.ResolveChain(model)
			if err != nil || len(chain) == 0 {
				continue
			}
			fallbacks := make([]routerTargetStatus, 0, len(chain)-1)
			for _, fb := range chain[1:] {
				fallbacks = append(fallbacks, buildRouterTargetStatus(fb, snap))
			}
			resp.Routes = append(resp.Routes, routerRouteStatus{
				Model:          model,
				ResidencyFloor: chain[0].ResidencyFloor,
				Primary:        buildRouterTargetStatus(chain[0], snap),
				Fallbacks:      fallbacks,
			})
		}
	}
	writeJSON(w, http.StatusOK, resp)
}
