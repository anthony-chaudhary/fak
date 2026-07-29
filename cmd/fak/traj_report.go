package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/toolrollup"
	"github.com/anthony-chaudhary/fak/internal/toolseq"
	"github.com/anthony-chaudhary/fak/internal/tooltrend"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/trajhook"
)

// trajReportSchema stamps the combined session-analytics payload so a consumer can
// tell which fold produced it.
const trajReportSchema = "fak.traj.report.v1"

// trajReport is the combined session-analytics view over a trajectory corpus — the
// `fak traj report` front door (#2827). It folds the corpus three ways behind one
// command: per-tool rollup (internal/toolrollup, C2), tool-transition graph +
// top n-grams (internal/toolseq, C3), and the per-session tool-mix + output-shape
// TREND (internal/tooltrend, C4). All three read the same bridged tool-call rows,
// so the report is one bridge from a trajectory to three consistent folds.
type trajReport struct {
	Schema       string                `json:"schema"`
	Corpus       string                `json:"corpus"`
	Turns        int                   `json:"turns"`
	ToolTurns    int                   `json:"tool_turns"`
	Sessions     int                   `json:"sessions"`
	Tools        []toolrollup.ToolStat `json:"tools"`
	Transitions  []toolseq.Edge        `json:"transitions"`
	TopSequences []toolseq.SeqCount    `json:"top_sequences"`
	Trend        tooltrend.Trend       `json:"trend"`
}

// cmdTrajReport folds a trajectory corpus into the combined session-analytics
// report. It groups the corpus into per-session buckets (by trace, in first-seen
// order), bridges each tool-bearing turn to a toolrollup.ToolCall, and runs the
// three session-analytics folds over the shared rows.
func cmdTrajReport(args []string) {
	fs := flag.NewFlagSet("traj report", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	top := fs.Int("top", 10, "cap each list (tools, transitions, sequences, movers) at N")
	ngram := fs.Int("ngram", 3, "sequence length for the top-n-gram list")
	_ = fs.Parse(args)

	r := loadCorpus("report", *corpus)
	turns := r.Turns()

	order, byTrace := groupByTrace(turns)

	// One bridge from turns to tool-call rows, reused by all three folds.
	allCalls := make([]toolrollup.ToolCall, 0, len(turns))
	buckets := make([]tooltrend.Bucket, 0, len(order))
	sessions := make([][]string, 0, len(order))
	toolTurns := 0
	for _, id := range order {
		tt := byTrace[id]
		calls := make([]toolrollup.ToolCall, 0, len(tt))
		seq := make([]string, 0, len(tt))
		for _, t := range tt {
			if t.Tool == "" {
				continue // not a tool call — no place in the tool folds
			}
			c := bridgeCall(t)
			calls = append(calls, c)
			seq = append(seq, t.Tool)
		}
		toolTurns += len(calls)
		allCalls = append(allCalls, calls...)
		buckets = append(buckets, tooltrend.Bucket{Label: id, Calls: calls})
		sessions = append(sessions, seq)
	}

	rep := trajReport{
		Schema:       trajReportSchema,
		Corpus:       *corpus,
		Turns:        len(turns),
		ToolTurns:    toolTurns,
		Sessions:     len(order),
		Tools:        capStats(toolrollup.Rollup(allCalls), *top),
		Transitions:  capEdges(toolseq.Transitions(sessions), *top),
		TopSequences: toolseq.TopSequences(sessions, *ngram, *top),
		Trend:        tooltrend.FoldTopK(buckets, *top),
	}

	if *asJSON {
		emitJSON(rep)
		return
	}
	renderTrajReport(os.Stdout, rep, *ngram)
}

// groupByTrace groups turns by trace id, preserving first-seen trace order and
// sorting each trace's turns by Seq so a session reads in call order regardless of
// how the corpus lines were ordered on disk.
func groupByTrace(turns []trajectory.Turn) (order []string, byTrace map[string][]trajectory.Turn) {
	// The grouping half is trajhook.GroupByTrace (the #3096 refcount fold needs the same
	// first-seen trace order); the Seq sort below stays here because the two callers differ
	// on whether it may mutate the grouped slice — this one sorts in place, that one copies.
	order, byTrace = trajhook.GroupByTrace(turns)
	for id := range byTrace {
		g := byTrace[id]
		sort.SliceStable(g, func(i, j int) bool { return g[i].Seq < g[j].Seq })
	}
	return order, byTrace
}

// bridgeCall maps one trajectory Turn to a toolrollup.ToolCall. TokenEstimate is
// the turn's cost in tokens; it feeds the output-size shape class, so it maps to
// TokensOut. A turn is a success unless its verdict blocked it (see turnOK).
func bridgeCall(t trajectory.Turn) toolrollup.ToolCall {
	return toolrollup.ToolCall{
		Tool:      t.Tool,
		TokensOut: t.TokenEstimate,
		OK:        turnOK(t.Verdict),
	}
}

// turnOK reads a turn's adjudication verdict as success/failure. An empty verdict
// (a plain recorded call) is a success; only an explicitly blocking verdict is an
// error. The set is closed so an unknown verdict never silently reads as failure.
func turnOK(verdict string) bool {
	switch strings.ToUpper(strings.TrimSpace(verdict)) {
	case "DENY", "QUARANTINE", "BLOCK", "ERROR", "FAULT":
		return false
	default:
		return true
	}
}

func capStats(s []toolrollup.ToolStat, n int) []toolrollup.ToolStat {
	if n > 0 && len(s) > n {
		return s[:n]
	}
	return s
}

func capEdges(e []toolseq.Edge, n int) []toolseq.Edge {
	if n > 0 && len(e) > n {
		return e[:n]
	}
	return e
}

// renderTrajReport writes the human-readable report: a header line, the per-tool
// rollup, the top transitions and n-grams, and the tool-mix / output-shape movers.
func renderTrajReport(w *os.File, rep trajReport, ngram int) {
	fmt.Fprintf(w, "%d turns (%d tool calls) across %d session(s)\n", rep.Turns, rep.ToolTurns, rep.Sessions)

	fmt.Fprintf(w, "\ntop tools (by calls):\n")
	if len(rep.Tools) == 0 {
		fmt.Fprintln(w, "  (no tool calls in corpus)")
	}
	for _, s := range rep.Tools {
		fmt.Fprintf(w, "  %-20s %5d calls  %5.1f%% share  %5.1f%% errors  %8.0f mean tok\n",
			s.Tool, s.Calls, s.Share*100, s.ErrorRate*100, s.MeanTokensOut)
	}

	fmt.Fprintf(w, "\ntop transitions (from -> to):\n")
	if len(rep.Transitions) == 0 {
		fmt.Fprintln(w, "  (no within-session transitions)")
	}
	for _, e := range rep.Transitions {
		fmt.Fprintf(w, "  %-16s -> %-16s %4d  (%4.1f%%)\n", e.From, e.To, e.Count, e.Prob*100)
	}

	fmt.Fprintf(w, "\ntop %d-grams:\n", ngram)
	if len(rep.TopSequences) == 0 {
		fmt.Fprintln(w, "  (no sequences of that length)")
	}
	for _, sq := range rep.TopSequences {
		fmt.Fprintf(w, "  %4d  %s\n", sq.Count, strings.Join(sq.Seq, " -> "))
	}

	fmt.Fprintf(w, "\ntool-mix movers (first -> last session):\n")
	renderMovers(w, rep.Trend.ToolMovers)
	fmt.Fprintf(w, "\noutput-shape movers (first -> last session):\n")
	renderMovers(w, rep.Trend.ShapeMovers)
}

func renderMovers(w *os.File, movers []tooltrend.Move) {
	if len(movers) == 0 {
		fmt.Fprintln(w, "  (no change, or fewer than two sessions)")
		return
	}
	for _, m := range movers {
		sign := "+"
		if m.Delta < 0 {
			sign = ""
		}
		fmt.Fprintf(w, "  %-16s %5.1f%% -> %5.1f%%  (%s%.1f%% %s)\n",
			m.Key, m.From*100, m.To*100, sign, m.Delta*100, m.Direction)
	}
}
