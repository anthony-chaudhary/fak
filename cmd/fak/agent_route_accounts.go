package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func loadAgentRouteOptionsWithAccounts(manifestPath, rosterPath string) (*modelroute.Manifest, *modelroute.Roster, []agent.RunOption, error) {
	manifest, opts, err := loadAgentRouteOptions(manifestPath)
	if err != nil {
		return nil, nil, nil, err
	}
	roster, _, err := loadAgentRouteAccounts(rosterPath)
	if err != nil {
		return nil, nil, nil, err
	}
	if roster == nil {
		return manifest, nil, opts, nil
	}
	return manifest, roster, append(opts, agent.WithRouteAccounts(roster), agent.WithRoutePrincipal("")), nil
}

func loadAgentRouteAccounts(path string) (*modelroute.Roster, modelroute.AccountReadinessReport, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, modelroute.AccountReadinessReport{}, nil
	}
	roster, err := modelroute.LoadRoster(path)
	if err != nil {
		return nil, modelroute.AccountReadinessReport{}, fmt.Errorf("fak agent: --route-accounts: %w", err)
	}
	return &roster, roster.Readiness(os.LookupEnv), nil
}

func announceAgentRouteAccounts(w io.Writer, path string, roster *modelroute.Roster) {
	if roster == nil {
		return
	}
	rep := roster.Readiness(os.LookupEnv)
	fmt.Fprintf(w, "fak agent: loaded model-account roster from %s", path)
	for _, row := range rep.Rows {
		if row.CredEnv != "" {
			fmt.Fprintf(w, " cred_env=%s", row.CredEnv)
		}
	}
	fmt.Fprintln(w)
}

func writeAgentRouteAccountsBail(w io.Writer, path string, err error) {
	if err != nil {
		fmt.Fprintf(w, "fak agent: cannot load model-account roster %s: %v\n", path, err)
	}
}
