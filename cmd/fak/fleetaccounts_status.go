package main

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

type fleetAccountsStatusRequest struct {
	paths                              fleetaccounts.Paths
	rows                               []fleetaccounts.Account
	repoRoot                           string
	product, provider, state, account  *string
	modelFilter, freshWithin, groupBy  *string
	nodeLabel                          *string
	tier                               *int
	t1, t2, t3                         *bool
	includeStale, asJSON, showAccounts *bool
	snapshots                          fleetStatusSnapshotFlags
}

func runFleetAccountsStatus(stdout, stderr io.Writer, req fleetAccountsStatusRequest) int {
	rows, repoRoot := req.rows, req.repoRoot
	product, provider, tier := req.product, req.provider, req.tier
	state, account, modelFilter := req.state, req.account, req.modelFilter
	t1, t2, t3 := req.t1, req.t2, req.t3
	statusSnapshots, freshWithin := req.snapshots, req.freshWithin
	groupBy, includeStale := req.groupBy, req.includeStale
	asJSON, showAccounts, nodeLabel := req.asJSON, req.showAccounts, req.nodeLabel
	filter := fleetaccounts.StatusFilter{
		Product: *product, Provider: *provider, Tier: statusTierFilter(*tier, *t1, *t2, *t3),
		State: *state, Account: *account, Model: *modelFilter,
	}
	if len(statusSnapshots) > 0 {
		window, err := time.ParseDuration(*freshWithin)
		if err != nil || window <= 0 {
			fmt.Fprintf(stderr, "fleet-accounts status: invalid --fresh-within %q\n", *freshWithin)
			return 2
		}
		snaps, err := loadFleetAccountStatusSnapshots(statusSnapshots)
		if err != nil {
			fmt.Fprintf(stderr, "fleet-accounts status: %v\n", err)
			return 1
		}
		report := fleetaccounts.BuildGlobalStatusReport(snaps, fleetaccounts.GlobalStatusOptions{
			Filter:       filter,
			GroupBy:      fleetAccountsSplitCSV(*groupBy),
			FreshWithin:  window,
			IncludeStale: *includeStale,
			Now:          time.Now().UTC(),
		})
		if *asJSON {
			out, err := json.MarshalIndent(report, "", " ")
			if err != nil {
				fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
				return 1
			}
			fmt.Fprintln(stdout, string(out))
			return 0
		}
		fmt.Fprint(stdout, fleetaccounts.RenderGlobalStatusReport(report, *showAccounts || statusFilterRequested(filter)))
		return 0
	}
	report := fleetaccounts.BuildStatusReport(rows, fleetSeatLeases(repoRoot), fleetaccounts.StatusOptions{
		Filter:  filter,
		GroupBy: fleetAccountsSplitCSV(*groupBy),
	})
	fleetaccounts.StampStatusReport(&report, *nodeLabel, time.Now().UTC().Format(time.RFC3339))
	if *asJSON {
		out, err := json.MarshalIndent(report, "", " ")
		if err != nil {
			fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		return 0
	}
	fmt.Fprint(stdout, fleetaccounts.RenderStatusReport(report, *showAccounts || statusFilterRequested(filter)))
	return 0

}
