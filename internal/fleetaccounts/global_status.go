package fleetaccounts

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const GlobalStatusReportSchema = "fleet-account-global-status/1"

// StatusSnapshot is one node's portable `fak fleet-accounts status --json` result.
type StatusSnapshot struct {
	Node   string
	Path   string
	Report StatusReport
}

// GlobalStatusOptions controls a multi-node status fold.
type GlobalStatusOptions struct {
	Filter       StatusFilter
	GroupBy      []string
	FreshWithin  time.Duration
	IncludeStale bool
	Now          time.Time
}

// GlobalStatusReport is the "all nodes" account/switcher rollup. Totals and Rollups are
// free-now by default: stale snapshots are visible in Nodes/StaleTotals but excluded from
// capacity unless IncludeStale is true.
type GlobalStatusReport struct {
	Schema             string              `json:"schema"`
	GeneratedAt        string              `json:"generated_at"`
	FreshWithinSeconds int64               `json:"fresh_within_seconds"`
	IncludeStale       bool                `json:"include_stale"`
	Filters            StatusFilter        `json:"filters"`
	GroupBy            []string            `json:"group_by"`
	Totals             StatusTotals        `json:"totals"`
	StaleTotals        StatusTotals        `json:"stale_totals"`
	Nodes              []StatusNodeSummary `json:"nodes"`
	Rollups            []StatusRollup      `json:"rollups"`
	Accounts           []StatusAccount     `json:"accounts"`
	Warnings           []string            `json:"warnings"`
}

type StatusNodeSummary struct {
	Node         string       `json:"node"`
	Path         string       `json:"path,omitempty"`
	GeneratedAt  string       `json:"generated_at,omitempty"`
	Fresh        bool         `json:"fresh"`
	AgeSeconds   int64        `json:"age_seconds"`
	StaleReason  string       `json:"stale_reason,omitempty"`
	Totals       StatusTotals `json:"totals"`
	StaleTotals  StatusTotals `json:"stale_totals,omitempty"`
	Included     bool         `json:"included"`
	IncludeCause string       `json:"include_cause,omitempty"`
}

const defaultGlobalFreshWithin = 30 * time.Minute

// BuildGlobalStatusReport folds several node snapshots into one operator view.
func BuildGlobalStatusReport(snaps []StatusSnapshot, opts GlobalStatusOptions) GlobalStatusReport {
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	freshWithin := opts.FreshWithin
	if freshWithin <= 0 {
		freshWithin = defaultGlobalFreshWithin
	}
	filter := normalizeStatusFilter(opts.Filter)
	groupBy := normalizeGroupBy(opts.GroupBy)
	rep := GlobalStatusReport{
		Schema:             GlobalStatusReportSchema,
		GeneratedAt:        now.UTC().Format(time.RFC3339),
		FreshWithinSeconds: int64(freshWithin.Seconds()),
		IncludeStale:       opts.IncludeStale,
		Filters:            filter,
		GroupBy:            groupBy,
	}
	rollupByKey := map[string]*StatusRollup{}
	poolNodes := map[string]map[string]bool{}

	for i, snap := range snaps {
		node := statusSnapshotNode(snap, i)
		fresh, age, staleReason := statusSnapshotFreshness(snap.Report, now, freshWithin)
		nodeSummary := StatusNodeSummary{
			Node:        node,
			Path:        strings.TrimSpace(snap.Path),
			GeneratedAt: strings.TrimSpace(snap.Report.GeneratedAt),
			Fresh:       fresh,
			AgeSeconds:  age,
		}
		if !fresh {
			nodeSummary.StaleReason = staleReason
		}
		included := fresh || opts.IncludeStale
		nodeSummary.Included = included
		if !fresh && opts.IncludeStale {
			nodeSummary.IncludeCause = "stale included by request"
		}

		for _, acct := range snap.Report.Accounts {
			if acct.Node == "" {
				acct.Node = node
			}
			if !statusMatches(acct, filter) {
				continue
			}
			addStatusTotals(&nodeSummary.Totals, acct)
			if !fresh {
				addStatusTotals(&nodeSummary.StaleTotals, acct)
				addStatusTotals(&rep.StaleTotals, acct)
			}
			if !included {
				continue
			}
			addStatusTotals(&rep.Totals, acct)
			ru := statusRollupFor(rollupByKey, acct, groupBy)
			addStatusRollup(ru, acct)
			rep.Accounts = append(rep.Accounts, acct)
			if acct.CapacityCounted && acct.Pool != "" {
				key := acct.Provider + "|" + acct.Pool
				if poolNodes[key] == nil {
					poolNodes[key] = map[string]bool{}
				}
				poolNodes[key][node] = true
			}
		}
		rep.Nodes = append(rep.Nodes, nodeSummary)
	}

	for _, ru := range rollupByKey {
		ru.Mixed = ru.FreeSlots > 0 && ru.BlockedSlots > 0
		rep.Rollups = append(rep.Rollups, *ru)
	}
	sort.SliceStable(rep.Nodes, func(i, j int) bool {
		if rep.Nodes[i].Fresh != rep.Nodes[j].Fresh {
			return rep.Nodes[i].Fresh && !rep.Nodes[j].Fresh
		}
		return rep.Nodes[i].Node < rep.Nodes[j].Node
	})
	sort.SliceStable(rep.Rollups, func(i, j int) bool { return rep.Rollups[i].Key < rep.Rollups[j].Key })
	sort.SliceStable(rep.Accounts, func(i, j int) bool {
		if rep.Accounts[i].Node != rep.Accounts[j].Node {
			return rep.Accounts[i].Node < rep.Accounts[j].Node
		}
		return statusAccountLess(rep.Accounts[i], rep.Accounts[j])
	})
	rep.Warnings = globalStatusWarnings(rep, poolNodes)
	if rep.Nodes == nil {
		rep.Nodes = []StatusNodeSummary{}
	}
	if rep.Rollups == nil {
		rep.Rollups = []StatusRollup{}
	}
	if rep.Accounts == nil {
		rep.Accounts = []StatusAccount{}
	}
	if rep.Warnings == nil {
		rep.Warnings = []string{}
	}
	return rep
}

func statusSnapshotNode(snap StatusSnapshot, idx int) string {
	for _, v := range []string{snap.Node, snap.Report.Node} {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return "node-" + itoa(idx+1)
}

func statusSnapshotFreshness(rep StatusReport, now time.Time, freshWithin time.Duration) (bool, int64, string) {
	raw := strings.TrimSpace(rep.GeneratedAt)
	if raw == "" {
		return false, 0, "missing generated_at"
	}
	t, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return false, 0, "unparseable generated_at"
	}
	age := int64(now.Sub(t).Seconds())
	if age < 0 {
		age = 0
	}
	if now.Sub(t) > freshWithin {
		return false, age, "snapshot older than fresh window"
	}
	return true, age, ""
}

func globalStatusWarnings(rep GlobalStatusReport, poolNodes map[string]map[string]bool) []string {
	var warnings []string
	for _, node := range rep.Nodes {
		if node.Fresh {
			continue
		}
		if rep.IncludeStale {
			warnings = append(warnings, fmt.Sprintf("stale node %s included by request: %s", node.Node, node.StaleReason))
		} else {
			warnings = append(warnings, fmt.Sprintf("stale node %s excluded from free-now totals: %s", node.Node, node.StaleReason))
		}
	}
	for _, ru := range rep.Rollups {
		if ru.Mixed {
			warnings = append(warnings, fmt.Sprintf("mixed limit posture: %s has %d free, %d leased, %d blocked slot(s)",
				ru.Label, ru.FreeSlots, ru.LeasedSlots, ru.BlockedSlots))
		}
	}
	for pool, nodes := range poolNodes {
		if len(nodes) < 2 {
			continue
		}
		names := make([]string, 0, len(nodes))
		for node := range nodes {
			names = append(names, node)
		}
		sort.Strings(names)
		warnings = append(warnings, fmt.Sprintf("shared account pool %s appears on multiple included nodes: %s",
			pool, strings.Join(names, ", ")))
	}
	sort.Strings(warnings)
	return warnings
}

func RenderGlobalStatusReport(rep GlobalStatusReport, includeAccounts bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fleet account status (global")
	if filterText := renderStatusFilters(rep.Filters); filterText != "" {
		fmt.Fprintf(&b, " %s", filterText)
	}
	fmt.Fprintf(&b, ")\n")
	fmt.Fprintf(&b, "  fresh window: %ds   include_stale=%v\n", rep.FreshWithinSeconds, rep.IncludeStale)
	fmt.Fprintf(&b, "  slots: %d free / %d total   leased=%d blocked=%d   pools=%d\n",
		rep.Totals.FreeSlots, rep.Totals.TotalSlots, rep.Totals.LeasedSlots,
		rep.Totals.BlockedSlots, rep.Totals.Pools)
	if rep.StaleTotals.Accounts > 0 {
		fmt.Fprintf(&b, "  stale excluded: %d free / %d total   accounts=%d\n",
			rep.StaleTotals.FreeSlots, rep.StaleTotals.TotalSlots, rep.StaleTotals.Accounts)
	}

	b.WriteString("\nnodes:\n")
	if len(rep.Nodes) == 0 {
		b.WriteString("  (none)\n")
	}
	for _, node := range rep.Nodes {
		state := "stale"
		if node.Fresh {
			state = "fresh"
		}
		extra := ""
		if node.StaleReason != "" {
			extra = " " + node.StaleReason
		}
		fmt.Fprintf(&b, "  %-18s %-5s age=%ds slots=%d/%d free accounts=%d%s\n",
			node.Node, state, node.AgeSeconds, node.Totals.FreeSlots,
			node.Totals.TotalSlots, node.Totals.Accounts, extra)
	}

	fmt.Fprintf(&b, "\nrollups by %s:\n", strings.Join(rep.GroupBy, "+"))
	if len(rep.Rollups) == 0 {
		b.WriteString("  (none)\n")
	}
	for _, ru := range rep.Rollups {
		mix := ""
		if ru.Mixed {
			mix = " mixed"
		}
		fmt.Fprintf(&b, "  %-52s slots=%d/%d free leased=%d blocked=%d accounts ready=%d blocked=%d%s\n",
			ru.Label, ru.FreeSlots, ru.TotalSlots, ru.LeasedSlots, ru.BlockedSlots,
			ru.ReadyAccounts, ru.BlockedAccounts, mix)
	}
	if len(rep.Warnings) > 0 {
		b.WriteString("\nlimits:\n")
		for _, w := range rep.Warnings {
			fmt.Fprintf(&b, "  %s\n", w)
		}
	}
	if includeAccounts {
		b.WriteString("\naccounts:\n")
		if len(rep.Accounts) == 0 {
			b.WriteString("  (none)\n")
		}
		for _, a := range rep.Accounts {
			tier := "t?"
			if a.ModelTier != nil {
				tier = "t" + itoa(*a.ModelTier)
			}
			fmt.Fprintf(&b, "  %-14s [%-10s] %-9s %-18s %-4s %-10s slots=%d/%d free %s\n",
				a.Node, a.Provider, a.Product, a.Tag, tier, a.State, a.FreeSlots, a.SessionCap, a.Reason)
		}
	}
	return b.String()
}
