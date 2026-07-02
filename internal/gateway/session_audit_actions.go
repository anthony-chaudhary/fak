package gateway

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// handleFakSessionAuditActions serves the recent-session action ledger used by
// the guard/session pressure gate. It is read-only: no route or transcript is
// changed by this endpoint, and the gate verdict is advisory for callers.
func (s *Server) handleFakSessionAuditActions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	plan, status, msg := sessionAuditActionsFromRequest(r, time.Now())
	if msg != "" {
		writeErr(w, status, msg)
		return
	}
	writeJSON(w, http.StatusOK, plan)
}

func sessionAuditActionsFromRequest(r *http.Request, now time.Time) (sessionaudit.CompactActionPlan, int, string) {
	opts, includeSubagents, max, gate, status, msg := sessionAuditOptionsFromRequest(r)
	if msg != "" {
		return sessionaudit.CompactActionPlan{}, status, msg
	}
	rep, err := sessionaudit.BuildCompactReportFromDiscovery(opts, includeSubagents, max, now)
	if err != nil {
		return sessionaudit.CompactActionPlan{}, http.StatusInternalServerError, fmt.Sprintf("session audit discover: %v", err)
	}
	plan := sessionaudit.BuildCompactActionPlan(rep)
	plan, _ = sessionaudit.ApplyCompactActionGate(plan, gate)
	return plan, 0, ""
}

func sessionAuditOptionsFromRequest(r *http.Request) (sessionaudit.DiscoverOptions, bool, int, string, int, string) {
	q := r.URL.Query()
	sinceDays := 7.0
	if raw := firstQuery(q, "since_days", "days"); raw != "" {
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return sessionaudit.DiscoverOptions{}, false, 0, "", http.StatusBadRequest, "invalid since_days"
		}
		sinceDays = v
	}
	var since *float64
	if sinceDays >= 0 {
		v := sinceDays
		since = &v
	}
	max := 40
	if raw := strings.TrimSpace(q.Get("max")); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < 0 {
			return sessionaudit.DiscoverOptions{}, false, 0, "", http.StatusBadRequest, "invalid max"
		}
		max = v
	}
	gate := firstQuery(q, "fail_on", "gate", "action_gate")
	if gate == "" {
		gate = "high"
	}
	if _, ok := sessionaudit.CompactActionSeverityRank(gate); !ok {
		return sessionaudit.DiscoverOptions{}, false, 0, "", http.StatusBadRequest, "invalid gate"
	}
	includeSubagents := queryBool(firstQuery(q, "include_subagents", "subagents"))
	nsPrefix := strings.TrimSpace(firstQuery(q, "ns_prefix", "namespace_prefix"))
	allNS := queryBool(firstQuery(q, "all", "session_all"))
	if !allNS && nsPrefix == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return sessionaudit.DiscoverOptions{}, false, 0, "", http.StatusInternalServerError, fmt.Sprintf("current workspace namespace: %v", err)
		}
		nsPrefix = sessionaudit.ProjectNamespace(cwd)
	}
	return sessionaudit.DiscoverOptions{
		Roots:            trimQueryValues(q["root"]),
		SinceDays:        since,
		NamespacePrefix:  nsPrefix,
		IncludeSubagents: includeSubagents,
	}, includeSubagents, max, gate, 0, ""
}

func firstQuery(q map[string][]string, names ...string) string {
	for _, name := range names {
		for _, v := range q[name] {
			if strings.TrimSpace(v) != "" {
				return strings.TrimSpace(v)
			}
		}
	}
	return ""
}

func trimQueryValues(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}
