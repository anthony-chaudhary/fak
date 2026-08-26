package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentquery"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

type agentsExecutionConfig struct {
	source, addr, key, journal, asOfText *string
	all, asJSON                          *bool
	nowUnix                              *int64
	listPlan                             agentquery.ListPlan
	queryPlan                            agentquery.QueryPlan
	grouped                              bool
}

func executeAgentsQuery(stdout, stderr io.Writer, cfg agentsExecutionConfig) int {
	source, addr, key, journal, asOfText := cfg.source, cfg.addr, cfg.key, cfg.journal, cfg.asOfText
	all, asJSON, nowUnix := cfg.all, cfg.asJSON, cfg.nowUnix
	listPlan, queryPlan, grouped := cfg.listPlan, cfg.queryPlan, cfg.grouped
	now := time.Now().UTC()
	if *nowUnix != 0 {
		now = time.Unix(*nowUnix, 0).UTC()
	}
	var asOf *time.Time
	if strings.TrimSpace(*asOfText) != "" {
		t, err := time.Parse(time.RFC3339, strings.TrimSpace(*asOfText))
		if err != nil {
			fmt.Fprintln(stderr, "fak agents: --as-of must be RFC3339")
			return 2
		}
		t = t.UTC()
		if t.After(now) {
			fmt.Fprintln(stderr, "fak agents: --as-of cannot be in the future")
			return 2
		}
		asOf = &t
		if *source != "history" || grouped {
			fmt.Fprintln(stderr, "fak agents: --as-of requires --source history and cannot be grouped")
			return 2
		}
		*all = true
	}
	if !*all && listPlan.State == "" && listPlan.Liveness == "" {
		listPlan.Liveness = "LIVE"
	}
	var live []agentquery.Row
	if *source != "history" {
		c := &sessionClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}
		list, err := c.list()
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: live source unavailable: %v\n", err)
			return 1
		}
		live = agentRowsFromLive(list, now)
	}
	var history []agentquery.Row
	var historyHealth *agentquery.SourceHealth
	if *source != "live" {
		events, health := sessionjournal.LoadFileReport(*journal)
		if health.ReadError != "" {
			fmt.Fprintf(stderr, "fak agents: history source %s\n", health.ReadError)
			return 1
		}
		cutoff := now
		if asOf != nil {
			cutoff = *asOf
		}
		events = sessionjournal.EventsAsOf(events, cutoff)
		foldNow := now
		if asOf != nil {
			foldNow = *asOf
		}
		history = agentRowsFromHistory(events, foldNow)
		historyHealth = agentHistoryHealth(health)
	}
	result := agentquery.Union(live, history, *source, false, 10000, now)
	result.Metadata.History = historyHealth
	if asOf != nil {
		value := asOf.Format(time.RFC3339)
		result.Metadata.AsOf = &value
	}
	if !grouped {
		filtered, truncated, err := agentquery.ApplyListPlan(result.Rows, listPlan, now)
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: list query rejected: %v\n", err)
			return 2
		}
		result.Rows = filtered
		result.Metadata.Truncated = truncated
		result.Metadata.Limit = listPlan.Limit
		result.Metadata.ListPlan = &listPlan
	}
	if grouped {
		groups := agentquery.GroupLaneStatePlan(result.Rows, queryPlan, now, *source, historyHealth)
		if err := agentquery.ValidateGroupResult(groups); err != nil {
			fmt.Fprintf(stderr, "fak agents: invalid grouped result: %v\n", err)
			return 1
		}
		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(groups); err != nil {
				fmt.Fprintf(stderr, "fak agents: encode: %v\n", err)
				return 1
			}
			return 0
		}
		renderAgentGroups(stdout, groups)
		return 0
	}
	if err := agentquery.ValidateResult(result); err != nil {
		fmt.Fprintf(stderr, "fak agents: invalid result: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak agents: encode: %v\n", err)
			return 1
		}
		return 0
	}
	renderAgents(stdout, result)
	return 0
}
