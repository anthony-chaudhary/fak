package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/toolseq"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/trajhook"
)

type trajConceptReport struct {
	Schema   string            `json:"schema"`
	Corpus   string            `json:"corpus"`
	Sessions int               `json:"sessions"`
	Concepts []toolseq.Concept `json:"concepts"`
}

// cmdTrajConcepts surfaces recurring, explainable workflow families between
// aggregate dashboards and individual tool calls. It never mutates the corpus.
func cmdTrajConcepts(args []string) {
	fs := flag.NewFlagSet("traj concepts", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	asJSON := fs.Bool("json", false, "emit JSON")
	threshold := fs.Float64("threshold", .55, "workflow similarity threshold (0,1]")
	selected := fs.String("concept", "", "show source sessions for one workflow concept ID")
	_ = fs.Parse(args)

	turns := loadCorpus("concepts", *corpus).Turns()
	episodes := conceptEpisodes(turns)
	concepts := toolseq.Discover(episodes, *threshold)
	if *selected == "" {
		for i := range concepts {
			concepts[i].Members = nil
		}
	} else {
		found := false
		for i := range concepts {
			if concepts[i].ID == *selected {
				found = true
			} else {
				concepts[i].Members = nil
			}
		}
		if !found {
			fmt.Fprintf(os.Stderr, "fak traj concepts: unknown concept %q\n", *selected)
			return
		}
	}
	report := trajConceptReport{Schema: "fak.traj.concepts.v1", Corpus: *corpus, Sessions: len(episodes), Concepts: concepts}
	if *asJSON {
		emitJSON(report)
		return
	}
	printTrajConcepts(report)
}

func conceptEpisodes(turns []trajectory.Turn) []toolseq.Session {
	order, grouped := trajhook.GroupByTrace(turns)
	out := make([]toolseq.Session, 0, len(order))
	for _, id := range order {
		group := grouped[id]
		sort.SliceStable(group, func(i, j int) bool { return group[i].Seq < group[j].Seq })
		calls := make([]toolseq.Call, 0, len(group))
		for _, turn := range group {
			if turn.Tool != "" {
				calls = append(calls, toolseq.Call{Tool: turn.Tool, Error: !turnOK(turn.Verdict)})
			}
		}
		if len(calls) > 0 {
			out = append(out, toolseq.Session{ID: id, Calls: calls})
		}
	}
	return out
}

func printTrajConcepts(report trajConceptReport) {
	fmt.Fprintf(os.Stdout, "%d workflow concept(s) across %d session(s)\n", len(report.Concepts), report.Sessions)
	fmt.Fprintln(os.Stdout, "ID           SESSIONS  SHARE   ERRORS  LABEL  [EXEMPLARS]")
	for _, c := range report.Concepts {
		fmt.Fprintf(os.Stdout, "%-12s %8d  %5.1f%%  %5.1f%%  %s  [%s]\n", c.ID, c.Sessions, c.Share*100, c.ErrorRate*100, c.Label, strings.Join(c.Exemplars, ", "))
		fmt.Fprintf(os.Stdout, "  signature: %s\n", strings.Join(c.Signature, " -> "))
		if len(c.Members) > 0 {
			fmt.Fprintf(os.Stdout, "  members: %s\n", strings.Join(c.Members, ", "))
		}
	}
	fmt.Fprintln(os.Stdout, "drill down: fak traj concepts --corpus <path> --concept <ID>")
}
