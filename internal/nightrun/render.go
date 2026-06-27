package nightrun

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// PlanSchema / NextSchema tag the stable --json envelopes so a consumer can
// validate the shape.
const (
	PlanSchema = "fak.nightrun.plan.v1"
	NextSchema = "fak.nightrun.next.v1"
)

// PlanReport is the stable --json shape for `fak nightrun plan`.
type PlanReport struct {
	Schema       string       `json:"schema"`
	GeneratedAt  string       `json:"generated_at"`
	Capabilities Capabilities `json:"capabilities"`
	Feasible     int          `json:"feasible"`
	Total        int          `json:"total"`
	Ranked       []Scored     `json:"ranked"`
}

// NextReport is the stable --json shape for `fak nightrun next` — the single
// most important datum to collect, or an empty selection with the reason none is
// feasible.
type NextReport struct {
	Schema       string       `json:"schema"`
	GeneratedAt  string       `json:"generated_at"`
	Capabilities Capabilities `json:"capabilities"`
	HasNext      bool         `json:"has_next"`
	Next         *Scored      `json:"next,omitempty"`
	Note         string       `json:"note,omitempty"`
}

// RenderCapabilities writes the one-line box fact-sheet that heads every human
// view, so an operator immediately sees WHY a task is or isn't feasible here.
func RenderCapabilities(w io.Writer, c Capabilities) {
	creds := strings.Join(c.CredNames(), ",")
	if creds == "" {
		creds = "none"
	}
	fmt.Fprintf(w, "box %s — gpu=%s weights=%s datasets=%s net=%s creds=%s\n",
		c.Box, c.gpuOrNone(), yesno(c.Weights), yesno(c.Datasets), yesno(c.Net), creds)
}

// RenderNext writes the focused human answer to "what should I collect next."
func RenderNext(w io.Writer, c Capabilities, s Scored, ok bool) {
	RenderCapabilities(w, c)
	fmt.Fprintln(w)
	if !ok {
		fmt.Fprintln(w, "next: (nothing feasible on this box right now)")
		fmt.Fprintln(w, "  every candidate needs a capability this box lacks — see `fak nightrun plan` for the full list and why.")
		return
	}
	t := s.Task
	fmt.Fprintf(w, "next: %s  [%s · %s · score %.3f]\n", t.ID, t.Value, t.Source, s.Score)
	fmt.Fprintf(w, "  %s\n\n", t.Title)
	fmt.Fprintf(w, "  why : %s\n", s.Reason)
	fmt.Fprintf(w, "  run : %s\n", t.Run)
	fmt.Fprintf(w, "  done: %s\n", t.Acceptance)
	if t.Doc != "" {
		fmt.Fprintf(w, "  doc : %s\n", t.Doc)
	}
	fmt.Fprintf(w, "\ncollect it, then `fak nightrun run --apply` records it and picks the next — or `--loop` to run the night.\n")
}

// RenderPlan writes the ranked human table for the whole night's queue on this
// box: feasible tasks first (the runnable queue), then the blocked ones with the
// capability they wait on.
func RenderPlan(w io.Writer, c Capabilities, ranked []Scored) {
	RenderCapabilities(w, c)
	feasible := 0
	for _, s := range ranked {
		if s.Feasible {
			feasible++
		}
	}
	fmt.Fprintf(w, "%d task(s); %d feasible here, %d blocked.\n\n", len(ranked), feasible, len(ranked)-feasible)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "#\tOK\tSCORE\tVALUE\tID\tLAST\tWHY")
	for i, s := range ranked {
		fmt.Fprintf(tw, "%d\t%s\t%.3f\t%s\t%s\t%s\t%s\n",
			i+1, okMark(s.Feasible), s.Score, s.Task.Value, s.Task.ID, lastOrNever(s), truncate72(s.Reason))
	}
	_ = tw.Flush()
	fmt.Fprintln(w, "\nrun the feasible queue with `fak nightrun run --apply [--loop] [--max N]`.")
}

// RenderLedger writes the durable collection history newest-first — what the
// fleet has gathered, on which box, with what outcome.
func RenderLedger(w io.Writer, rows []CollectRow) {
	if len(rows) == 0 {
		fmt.Fprintln(w, "no collection rows yet — run `fak nightrun run --apply` to start the ledger.")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "DATE\tBOX\tOUTCOME\tVALUE\tTASK\tNUMBER")
	for i := len(rows) - 1; i >= 0; i-- {
		r := rows[i]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n",
			r.Date, blankDash(r.Box), r.Outcome, blankDash(r.Value), r.TaskID, blankDash(r.Number))
	}
	_ = tw.Flush()
}

func lastOrNever(s Scored) string {
	if s.LastCollected == "" {
		return "never"
	}
	return s.LastCollected
}

func okMark(ok bool) string {
	if ok {
		return "+"
	}
	return "-"
}

func yesno(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func blankDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func truncate72(s string) string {
	const n = 72
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
