package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/agentquery"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func cmdAgents(argv []string) { os.Exit(runAgents(os.Stdout, os.Stderr, argv)) }

func runAgents(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || (len(argv) > 0 && (argv[0] == "list" || argv[0] == "descriptors" || argv[0] == "declarative")) {
		if len(argv) > 0 {
			return runAgentsList(stdout, stderr, argv[1:])
		}
		return runAgentsList(stdout, stderr, argv)
	}
	for i, arg := range argv {
		if arg == "--descriptors" || arg == "--declarative" || arg == "--list" {
			remaining := append(append([]string(nil), argv[:i]...), argv[i+1:]...)
			return runAgentsList(stdout, stderr, remaining)
		}
		if arg == "--dir" {
			return runAgentsList(stdout, stderr, argv)
		}
	}
	fs := flag.NewFlagSet("agents", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL for live rows")
	key := fs.String("key", defaultGatewayBearerToken(), "gateway bearer credential")
	journal := fs.String("journal", sessionjournal.DefaultPath(), "authoritative session lifecycle journal")
	source := fs.String("source", "live", "row source: live, history, or union")
	all := fs.Bool("all", false, "include terminal and stale rows")
	stateFilter := fs.String("state", "", "exact lifecycle state filter")
	livenessFilter := fs.String("liveness", "", "exact liveness filter")
	ownerFilter := fs.String("owner", "", "exact owner/account filter")
	hostFilter := fs.String("host", "", "exact host filter")
	laneFilter := fs.String("lane", "", "exact lane filter")
	groupFilter := fs.String("group", "", "exact group filter")
	modelFilter := fs.String("model", "", "exact model filter")
	providerFilter := fs.String("provider", "", "exact provider filter")
	rootFilter := fs.String("root", "", "exact root identity filter")
	parentFilter := fs.String("parent", "", "exact parent identity filter")
	startedAfter := fs.String("started-after", "", "inclusive RFC3339 start lower bound")
	startedBefore := fs.String("started-before", "", "inclusive RFC3339 start upper bound")
	orderBy := fs.String("order-by", "elapsed_desc", "elapsed|progress_age|started|ended|cost|identity with _asc or _desc")
	asJSON := fs.Bool("json", false, "emit versioned machine JSON")
	showSchema := fs.Bool("schema", false, "emit the shared relation schema descriptor")
	benchmark := fs.Bool("benchmark", false, "run the deterministic relation-path benchmark")
	benchmarkSizes := fs.String("benchmark-sizes", "1000,10000,100000", "comma-separated even lifecycle-event counts")
	benchmarkRepetitions := fs.Int("benchmark-repetitions", 5, "samples per benchmark path (1..20)")
	limit := fs.Int("limit", 200, "maximum returned rows")
	historyWindowText := fs.String("history", "", "historical lookback window (for example 7d)")
	groupBy := fs.String("group-by", "", "aggregate grouping (currently lane,state)")
	count := fs.Bool("count", false, "include group row counts")
	allAggregates := fs.Bool("all-aggregates", false, "include min/max/sum/avg elapsed_ms")
	queryText := fs.String("query", "", "constrained read-only grouped query text")
	asOfText := fs.String("as-of", "", "historical state at RFC3339 instant")
	nowUnix := fs.Int64("now", 0, "observation Unix timestamp (tests/replay)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak agents [--source live|history|union] [--journal FILE] [--all] [--json]")
		fmt.Fprintln(stderr, "       fak agents list [--dir WORKSPACE] [--json]")
		fmt.Fprintln(stderr, "       fak agents --history 168h --group-by lane,state --count [--all-aggregates] [--json]")
		fmt.Fprintln(stderr, "       fak agents --query \"SELECT lane,state,count(*) AS agents,max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC\"")
		fmt.Fprintln(stderr, "--query is a constrained read-only grammar, not arbitrary SQL")
		fmt.Fprintf(stderr, "--schema --json emits %s\n", agentquery.DescriptorSchema)
		fmt.Fprintf(stderr, "--benchmark --json emits %s\n", sessionjournal.BenchmarkSchema)
	}
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *benchmark {
		if !*asJSON {
			fmt.Fprintf(stderr, "fak agents: --benchmark requires --json (%s)\n", sessionjournal.BenchmarkSchema)
			return 2
		}
		counts, err := parseBenchmarkSizes(*benchmarkSizes)
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: benchmark sizes: %v\n", err)
			return 2
		}
		now := time.Now().UTC()
		if *nowUnix != 0 {
			now = time.Unix(*nowUnix, 0).UTC()
		}
		report, err := sessionjournal.RunBenchmark(counts, *benchmarkRepetitions, now)
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: benchmark: %v\n", err)
			return 1
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak agents: encode benchmark: %v\n", err)
			return 1
		}
		return 0
	}
	if *showSchema {
		if !*asJSON {
			fmt.Fprintf(stderr, "fak agents: --schema requires --json (%s)\n", agentquery.DescriptorSchema)
			return 2
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(agentquery.SchemaDescriptor()); err != nil {
			fmt.Fprintf(stderr, "fak agents: encode schema: %v\n", err)
			return 1
		}
		return 0
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak agents: positional arguments are not supported")
		return 2
	}
	listFlagsSet := false
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "state", "liveness", "owner", "host", "lane", "group", "model", "provider", "root", "parent", "started-after", "started-before", "order-by":
			listFlagsSet = true
		}
	})
	if *source != "live" && *source != "history" && *source != "union" {
		fmt.Fprintf(stderr, "fak agents: invalid --source %q (want live, history, or union)\n", *source)
		return 2
	}
	if *limit < 1 || *limit > 10000 {
		fmt.Fprintln(stderr, "fak agents: --limit must be between 1 and 10000")
		return 2
	}
	listPlan := agentquery.ListPlan{Schema: agentquery.ListPlanSchema, State: *stateFilter, Liveness: *livenessFilter, Owner: *ownerFilter, Host: *hostFilter, Lane: *laneFilter, Group: *groupFilter, Model: *modelFilter, Provider: *providerFilter, RootID: *rootFilter, ParentID: *parentFilter, OrderBy: *orderBy, Limit: *limit}
	if *startedAfter != "" {
		listPlan.StartedAfter = agentStringPtr(*startedAfter)
	}
	if *startedBefore != "" {
		listPlan.StartedBefore = agentStringPtr(*startedBefore)
	}
	if err := agentquery.ValidateListPlan(listPlan); err != nil {
		fmt.Fprintf(stderr, "fak agents: list query rejected: %v\n", err)
		return 2
	}
	historyWindow, historyErr := parseAgentHistoryWindow(*historyWindowText)
	groupedFlags := *groupBy != "" || *historyWindowText != "" || *count || *allAggregates
	if groupedFlags && *queryText != "" {
		fmt.Fprintln(stderr, "fak agents: --query cannot be combined with grouped flags")
		return 2
	}
	var queryPlan agentquery.QueryPlan
	grouped := groupedFlags || *queryText != ""
	if grouped && listFlagsSet {
		fmt.Fprintln(stderr, "fak agents: grouped and list filter/sort flags cannot be combined")
		return 2
	}
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
		if err == nil && *allAggregates {
			queryPlan.Aggregates = []string{"count", "min_elapsed_ms", "max_elapsed_ms", "sum_elapsed_ms", "avg_elapsed_ms"}
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak agents: grouped query rejected: %v\n", err)
			return 2
		}
	}
	if grouped {
		*source = queryPlan.Source
		*all = true
	}
	return executeAgentsQuery(stdout, stderr, agentsExecutionConfig{
		source: source, all: all, asJSON: asJSON, nowUnix: nowUnix, asOfText: asOfText,
		addr: addr, key: key, journal: journal, listPlan: listPlan, queryPlan: queryPlan, grouped: grouped,
	})

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
		if s.Account != "" {
			r.Owner = agentStringPtr(s.Account)
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
	fmt.Fprintf(w, "agents source=%s rows=%d live=%d history=%d deduplicated=%d observed=%s", result.Metadata.Source, len(result.Rows), result.Metadata.LiveRows, result.Metadata.HistoryRows, result.Metadata.Deduplicated, result.Metadata.ObservedAt)
	if result.Metadata.AsOf != nil {
		fmt.Fprintf(w, " as_of=%s", *result.Metadata.AsOf)
	}
	fmt.Fprintln(w)
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
	fmt.Fprintln(tw, "LANE\tSTATE\tCOUNT\tMIN MS\tMAX MS\tSUM MS\tAVG MS")
	for _, r := range result.Rows {
		lane := "-"
		if r.Lane != nil {
			lane = *r.Lane
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\t%s\t%s\t%s\n", lane, r.State, r.Count, nullableInt(r.MinElapsedMS), nullableInt(r.MaxElapsedMS), nullableInt(r.SumElapsedMS), nullableFloat(r.AvgElapsedMS))
	}
	tw.Flush()
}

func nullableInt(v *int64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}
func nullableFloat(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.3f", *v)
}

func parseBenchmarkSizes(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	if len(parts) == 0 || len(parts) > 10 {
		return nil, fmt.Errorf("want 1..10 counts")
	}
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n < 2 || n > 1_000_000 || n%2 != 0 {
			return nil, fmt.Errorf("%q must be even and within 2..1000000", part)
		}
		out = append(out, n)
	}
	return out, nil
}

func runAgentsList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("agents list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", ".", "workspace directory to search for agent descriptors")
	asJSON := fs.Bool("json", false, "emit JSON format")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak agents list [--dir WORKSPACE] [--json]")
		fmt.Fprintln(stderr, "       lists declarative agent descriptors (.fak/agents/*.md and .agents/*.md)")
	}
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 && *dir == "." {
		*dir = fs.Arg(0)
	}

	descs, err := agent.DiscoverAgentDescriptors(*dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak agents list: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(descs); err != nil {
			fmt.Fprintf(stderr, "fak agents list: encode JSON: %v\n", err)
			return 1
		}
		return 0
	}

	if len(descs) == 0 {
		fmt.Fprintln(stdout, "no declarative agents found (.fak/agents/*.md, .agents/*.md)")
		return 0
	}

	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tMODE\tMODEL\tVARIANT\tTURNS\tMUTATION\tTOOLS\tPATH")
	for _, d := range descs {
		tools := "-"
		if len(d.Capabilities.Tools) > 0 {
			tools = strings.Join(d.Capabilities.Tools, ",")
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%t\t%s\t%s\n",
			d.Name, d.Mode, d.Model, d.Variant, d.MaxTurns, d.Capabilities.AllowMutation, tools, d.Path)
	}
	tw.Flush()
	return 0
}
