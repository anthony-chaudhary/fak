package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentquery"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func cmdAgents(argv []string) { os.Exit(runAgents(os.Stdout, os.Stderr, argv)) }

func runAgents(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("agents", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL for live rows")
	key := fs.String("key", defaultGatewayBearerToken(), "gateway bearer credential")
	journal := fs.String("journal", sessionjournal.DefaultPath(), "authoritative session lifecycle journal")
	source := fs.String("source", "live", "row source: live, history, or union")
	all := fs.Bool("all", false, "include terminal and stale rows")
	asJSON := fs.Bool("json", false, "emit fak-agents/1 JSON")
	limit := fs.Int("limit", 200, "maximum returned rows")
	historyWindowText := fs.String("history", "", "historical lookback window (for example 7d)")
	groupBy := fs.String("group-by", "", "aggregate grouping (currently lane,state)")
	count := fs.Bool("count", false, "include group row counts")
	queryText := fs.String("query", "", "constrained read-only grouped query text")
	nowUnix := fs.Int64("now", 0, "observation Unix timestamp (tests/replay)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak agents [--source live|history|union] [--journal FILE] [--all] [--json]")
		fmt.Fprintln(stderr, "       fak agents --history 168h --group-by lane,state --count [--json]")
		fmt.Fprintln(stderr, "       fak agents --query \"SELECT lane,state,count(*) AS agents,max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC\"")
		fmt.Fprintln(stderr, "--query is a constrained read-only grammar, not arbitrary SQL")
	}
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak agents: positional arguments are not supported")
		return 2
	}
	if *source != "live" && *source != "history" && *source != "union" {
		fmt.Fprintf(stderr, "fak agents: invalid --source %q (want live, history, or union)\n", *source)
		return 2
	}
	if *limit < 1 || *limit > 10000 {
		fmt.Fprintln(stderr, "fak agents: --limit must be between 1 and 10000")
		return 2
	}
	historyWindow, historyErr := parseAgentHistoryWindow(*historyWindowText)
	groupedFlags := *groupBy != "" || *historyWindowText != "" || *count
	if groupedFlags && *queryText != "" {
		fmt.Fprintln(stderr, "fak agents: --query cannot be combined with grouped flags")
		return 2
	}
	var queryPlan agentquery.QueryPlan
	grouped := groupedFlags || *queryText != ""
	if *queryText != "" {
		var err error
		queryPlan, err = agentquery.ParseQuery(*queryText)
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: query rejected: %v\n", err)
			return 2
		}
	} else if groupedFlags {
		if historyErr != nil || *groupBy != "lane,state" || !*count || historyWindow <= 0 {
			fmt.Fprintln(stderr, "fak agents: grouped query requires --history DURATION --group-by lane,state --count")
			return 2
		}
		var err error
		queryPlan, err = agentquery.GroupedPlan(historyWindow)
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: grouped query rejected: %v\n", err)
			return 2
		}
	}
	if grouped {
		*source = queryPlan.Source
		*all = true
	}
	now := time.Now().UTC()
	if *nowUnix != 0 {
		now = time.Unix(*nowUnix, 0).UTC()
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
		history = agentRowsFromHistory(events, now)
		historyHealth = agentHistoryHealth(health)
	}
	queryLimit := *limit
	if grouped {
		queryLimit = 10000
	}
	result := agentquery.Union(live, history, *source, !*all, queryLimit, now)
	result.Metadata.History = historyHealth
	if grouped {
		groups := agentquery.GroupLaneStatePlan(result.Rows, queryPlan, now, *source, historyHealth)
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

func parseAgentHistoryWindow(raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	if strings.HasSuffix(raw, "d") {
		days, err := strconv.Atoi(strings.TrimSuffix(raw, "d"))
		if err != nil || days <= 0 || days > 3650 {
			return 0, fmt.Errorf("invalid day window")
		}
		return time.Duration(days) * 24 * time.Hour, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("invalid duration")
	}
	return d, nil
}

func agentHistoryHealth(h sessionjournal.ParseHealth) *agentquery.SourceHealth {
	status := "ok"
	if h.Degraded() {
		status = "degraded"
	}
	return &agentquery.SourceHealth{Status: status, TotalRows: h.TotalRows, BlankRows: h.BlankRows, AcceptedRows: h.AcceptedRows, MalformedRows: h.MalformedRows, WrongSchemaRows: h.WrongSchemaRows, MissingIDRows: h.MissingIDRows, ScanError: h.ScanError, ReadError: h.ReadError}
}

func agentRowsFromLive(list gateway.SessionListResponse, now time.Time) []agentquery.Row {
	rows := make([]agentquery.Row, 0, len(list.Sessions))
	observed := now.Format(time.RFC3339)
	for _, s := range list.Sessions {
		id := s.TraceID
		elapsed := s.Time.ElapsedSeconds * 1000
		state := fmt.Sprint(s.Run)
		liveness := "LIVE"
		if strings.EqualFold(state, "stopped") || strings.EqualFold(state, "terminal") {
			liveness = "CLOSED"
		}
		r := agentquery.Row{AgentID: id, LogicalSessionID: id, State: state, Liveness: liveness, ObservedAt: observed, ElapsedMS: &elapsed, Source: "live", SourceVersion: "gateway-session-state/1"}
		if s.ParentTrace != "" {
			r.ParentID = agentStringPtr(s.ParentTrace)
		}
		if s.Reason != "" {
			r.StopReason = agentStringPtr(s.Reason)
		}
		rows = append(rows, r)
	}
	return rows
}

func agentRowsFromHistory(events []sessionjournal.Event, now time.Time) []agentquery.Row {
	cs := sessionjournal.Classify(sessionjournal.FoldEvents(events), sessionjournal.ClassifyConfig{Now: now, StaleAfter: 2 * time.Minute})
	rows := make([]agentquery.Row, 0, len(cs))
	observed := now.Format(time.RFC3339)
	for _, s := range cs {
		started := agentTimePtr(s.StartedAt)
		last := agentTimePtr(s.LastSeen)
		var ended *string
		if s.Closed {
			ended = last
		}
		elapsed := int64(0)
		end := now
		if s.Closed && !s.LastSeen.IsZero() {
			end = s.LastSeen
		}
		if !s.StartedAt.IsZero() && end.After(s.StartedAt) {
			elapsed = end.Sub(s.StartedAt).Milliseconds()
		}
		r := agentquery.Row{AgentID: s.ID, LogicalSessionID: s.ID, State: string(s.Status), Liveness: string(s.Status), StartedAt: started, LastProgressAt: last, EndedAt: ended, ObservedAt: observed, ElapsedMS: &elapsed, Source: "history", SourceVersion: sessionjournal.Schema, Stale: s.Status == sessionjournal.StatusStale}
		if s.Boot != "" {
			r.ExecutionEpoch = agentStringPtr(s.Boot)
		}
		if s.Host != "" {
			r.Host = agentStringPtr(s.Host)
		}
		if s.PID > 0 {
			p := s.PID
			r.PID = &p
		}
		if s.Model != "" {
			r.Model = agentStringPtr(s.Model)
		}
		if s.Agent != "" {
			r.Provider = agentStringPtr(s.Agent)
		}
		if s.CloseReason != "" {
			r.StopReason = agentStringPtr(s.CloseReason)
		}
		if x := s.Registration; x != nil {
			if x.RootRegistrationID != "" {
				r.RootID = agentStringPtr(x.RootRegistrationID)
			}
			if x.ParentRegistrationID != "" {
				r.ParentID = agentStringPtr(x.ParentRegistrationID)
			}
			if x.Lane != "" {
				r.Lane = agentStringPtr(x.Lane)
			}
			if x.TaskID != "" {
				r.Group = agentStringPtr(x.TaskID)
			}
			if x.HostID != "" {
				r.Host = agentStringPtr(x.HostID)
			}
		}
		rows = append(rows, r)
	}
	return rows
}
func agentStringPtr(s string) *string { return &s }
func agentTimePtr(t time.Time) *string {
	if t.IsZero() {
		return nil
	}
	s := t.UTC().Format(time.RFC3339)
	return &s
}
func agentField(p *string) string {
	if p == nil || *p == "" {
		return "-"
	}
	return *p
}
func renderAgents(w io.Writer, result agentquery.Result) {
	fmt.Fprintf(w, "agents source=%s rows=%d live=%d history=%d deduplicated=%d observed=%s\n", result.Metadata.Source, len(result.Rows), result.Metadata.LiveRows, result.Metadata.HistoryRows, result.Metadata.Deduplicated, result.Metadata.ObservedAt)
	if h := result.Metadata.History; h != nil {
		fmt.Fprintf(w, "history status=%s accepted=%d rejected=%d malformed=%d wrong_schema=%d missing_id=%d\n", h.Status, h.AcceptedRows, h.MalformedRows+h.WrongSchemaRows+h.MissingIDRows, h.MalformedRows, h.WrongSchemaRows, h.MissingIDRows)
	}
	if len(result.Rows) == 0 {
		fmt.Fprintln(w, "no matching agents")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "AGENT\tSTATE\tLIVE\tELAPSED\tLANE\tHOST\tSOURCE")
	for _, r := range result.Rows {
		elapsed := "-"
		if r.ElapsedMS != nil {
			elapsed = compactDuration(*r.ElapsedMS / 1000)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", r.AgentID, r.State, r.Liveness, elapsed, agentField(r.Lane), agentField(r.Host), r.Source)
	}
	tw.Flush()
}

func renderAgentGroups(w io.Writer, result agentquery.GroupResult) {
	fmt.Fprintf(w, "agent groups source=%s since=%s matched=%d input=%d observed=%s\n", result.Metadata.Source, result.Metadata.Since, result.Metadata.MatchedRows, result.Metadata.InputRows, result.Metadata.ObservedAt)
	if h := result.Metadata.History; h != nil {
		fmt.Fprintf(w, "history status=%s accepted=%d rejected=%d\n", h.Status, h.AcceptedRows, h.MalformedRows+h.WrongSchemaRows+h.MissingIDRows)
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LANE\tSTATE\tCOUNT\tMAX ELAPSED")
	for _, r := range result.Rows {
		lane := "-"
		if r.Lane != nil {
			lane = *r.Lane
		}
		elapsed := "-"
		if r.MaxElapsedMS != nil {
			elapsed = compactDuration(*r.MaxElapsedMS / 1000)
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\n", lane, r.State, r.Count, elapsed)
	}
	tw.Flush()
}
