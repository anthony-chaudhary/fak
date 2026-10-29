package modelperfobs

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
)

// RunReport is shared by both existing report entry points. Without --goal the
// historical JSON shape and Markdown renderer remain unchanged.
func RunReport(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("model-observe report", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	input := fs.String("input", "model-perf.jsonl", "observation JSONL")
	format := fs.String("format", "md", "md or json")
	goal := fs.String("goal", "", "versioned service-goal JSON with reusable run evidence")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			fs.SetOutput(out)
			fs.Usage()
			return nil
		}
		return err
	}
	if fs.NArg() != 0 || (*format != "md" && *format != "json") {
		return fmt.Errorf("report accepts --input FILE [--goal FILE] [--format md|json]")
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer f.Close()
	rows, err := ReadObservations(f)
	if err != nil {
		return err
	}
	if *goal == "" {
		s := Summarize(rows)
		if *format == "json" {
			return json.NewEncoder(out).Encode(s)
		}
		return WriteMarkdown(out, s)
	}
	gf, err := os.Open(*goal)
	if err != nil {
		return err
	}
	defer gf.Close()
	g, err := ReadServiceGoal(gf)
	if err != nil {
		return err
	}
	r, err := EvaluateServiceGoal(g, rows)
	if err != nil {
		return err
	}
	if *format == "json" {
		return json.NewEncoder(out).Encode(r)
	}
	return WriteServiceGoalMarkdown(out, r)
}

func WriteServiceGoalMarkdown(w io.Writer, r ServiceGoalReport) error {
	quality := "unknown"
	if r.QualityQualifiedGoodput != nil {
		quality = fmt.Sprintf("%.4f req/s / %.4f completion tok/s", r.QualityQualifiedGoodput.RequestsPerSecond, r.QualityQualifiedGoodput.CompletionTokensPerSecond)
	}
	cost := "unknown"
	if r.CostPerQualifiedRequest != nil {
		cost = fmt.Sprintf("%.6g %s", *r.CostPerQualifiedRequest, r.Goal.Run.CostUnit)
	}
	if _, err := fmt.Fprintf(w, "# Service goal: %s\n\n- Workload: %s\n- Offered / observed / missing: **%d / %d / %d** (%d warmups excluded; %d errors)\n- Joint SLO pass / fail / unknown: **%d / %d / %d**\n- Offered-request SLO attainment: **%.4f**\n- Measured SLO goodput: **%.4f req/s / %.4f completion tok/s**\n- External-quality-qualified goodput: **%s**\n- TTFT p95: **%s** (%d/%d offered requests measured)\n- TPOT p95: **%s** (%d/%d offered requests measured; %d one-token requests have undefined TPOT)\n- Cost per qualified SLO request: **%s**\n", r.Goal.ID, r.Goal.WorkloadRef,
		r.Goal.Run.OfferedRequests, r.ObservedRequests, r.MissingRequests, r.WarmupRequests, r.Errors,
		r.SLOPassedRequests, r.SLOFailedRequests, r.SLOUnknownRequests, r.SLOAttainment,
		r.MeasuredSLOGoodput.RequestsPerSecond, r.MeasuredSLOGoodput.CompletionTokensPerSecond, quality,
		serviceTimingText(r.TTFTP95), r.TTFTP95.MeasuredRequests, r.TTFTP95.OfferedRequests,
		serviceTimingText(r.TPOTP95), r.TPOTP95.MeasuredRequests, r.TPOTP95.OfferedRequests, r.TPOTNotApplicableRequests, cost); err != nil {
		return err
	}
	for _, unknown := range r.Unknowns {
		if _, err := fmt.Fprintf(w, "- Unknown: %s\n", unknown); err != nil {
			return err
		}
	}
	return nil
}

func serviceTimingText(c ServiceTimingCoverage) string {
	if c.ValueMS == nil {
		return "unknown"
	}
	return fmt.Sprintf("%.4f ms", *c.ValueMS)
}
