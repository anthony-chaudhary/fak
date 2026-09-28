package naivecontrol

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"
)

// Aggregate is one metric folded across an arm's runs. Total is MEASURED only when
// every run measured it: a sum that silently skipped unmeasured runs would be a
// smaller number that looks like a real one. The partial sum is disclosed instead.
type Aggregate[T Number] struct {
	Runs       int       `json:"runs"`
	Measured   int       `json:"measured"`
	Total      Metric[T] `json:"total"`
	PartialSum *T        `json:"partial_sum,omitempty"`
}

func aggregate[T Number](rows []Row, get func(Row) Metric[T]) Aggregate[T] {
	a := Aggregate[T]{Runs: len(rows)}
	if len(rows) == 0 {
		a.Total = Missing[T]("no runs recorded for this arm")
		return a
	}
	var sum T
	firstWhy := ""
	for _, r := range rows {
		m := get(r)
		if v, ok := m.Get(); ok {
			sum += v
			a.Measured++
			continue
		}
		if firstWhy == "" {
			firstWhy = fmt.Sprintf("run %s: %s", r.RunID, m.WhyMissing())
		}
	}
	if a.Measured == a.Runs {
		a.Total = Measured(sum)
		return a
	}
	a.Total = Missing[T](fmt.Sprintf("%d of %d run(s) unmeasured; first: %s", a.Runs-a.Measured, a.Runs, firstWhy))
	if a.Measured > 0 {
		a.PartialSum = &sum
	}
	return a
}

// ArmSummary is one arm's column of the comparison.
type ArmSummary struct {
	Arm             Arm                `json:"arm"`
	Runs            int                `json:"runs"`
	Harvested       int                `json:"harvested"`
	FirstTSUnixNano int64              `json:"first_ts_unix_nano,omitempty"`
	LastTSUnixNano  int64              `json:"last_ts_unix_nano,omitempty"`
	PicksOffered    Aggregate[int64]   `json:"picks_offered"`
	PicksShipped    Aggregate[int64]   `json:"picks_shipped"`
	CommitsClaimed  Aggregate[int64]   `json:"commits_claimed"`
	CommitsLanded   Aggregate[int64]   `json:"commits_landed"`
	WallMS          Aggregate[int64]   `json:"wall_ms"`
	ControllerMS    Aggregate[int64]   `json:"controller_ms"`
	CostUSD         Aggregate[float64] `json:"cost_usd"`
	// ShipRate is picks_shipped / picks_offered over the arm's runs, the job repo's
	// "92.6% pick-level" number.
	ShipRate Metric[float64] `json:"ship_rate"`
	// CostPerShippedUSD is cost_usd / picks_shipped, the "$ per shipped unit" number.
	CostPerShippedUSD Metric[float64] `json:"cost_per_shipped_usd"`
}

// Summarize folds one arm's rows (already reduced by Latest). Pure.
func Summarize(arm Arm, rows []Row) ArmSummary {
	var mine []Row
	for _, r := range rows {
		if r.Arm == arm {
			mine = append(mine, r)
		}
	}
	s := ArmSummary{Arm: arm, Runs: len(mine)}
	for _, r := range mine {
		if r.Stage == StageHarvested {
			s.Harvested++
		}
		if s.FirstTSUnixNano == 0 || r.TSUnixNano < s.FirstTSUnixNano {
			s.FirstTSUnixNano = r.TSUnixNano
		}
		if r.TSUnixNano > s.LastTSUnixNano {
			s.LastTSUnixNano = r.TSUnixNano
		}
	}
	s.PicksOffered = aggregate(mine, func(r Row) Metric[int64] { return r.PicksOffered })
	s.PicksShipped = aggregate(mine, func(r Row) Metric[int64] { return r.PicksShipped })
	s.CommitsClaimed = aggregate(mine, func(r Row) Metric[int64] { return r.CommitsClaimed })
	s.CommitsLanded = aggregate(mine, func(r Row) Metric[int64] { return r.CommitsLanded })
	s.WallMS = aggregate(mine, func(r Row) Metric[int64] { return r.WallMS })
	s.ControllerMS = aggregate(mine, func(r Row) Metric[int64] { return r.ControllerMS })
	s.CostUSD = aggregate(mine, func(r Row) Metric[float64] { return r.CostUSD })
	s.ShipRate = ratio(s.PicksShipped.Total, s.PicksOffered.Total, "picks shipped", "picks offered")
	s.CostPerShippedUSD = ratio(s.CostUSD.Total, s.PicksShipped.Total, "cost", "picks shipped")
	return s
}

func ratio[N Number, D Number](num Metric[N], den Metric[D], numName, denName string) Metric[float64] {
	n, okN := num.Get()
	d, okD := den.Get()
	switch {
	case !okN:
		return Missing[float64](numName + " unknown: " + num.WhyMissing())
	case !okD:
		return Missing[float64](denName + " unknown: " + den.WhyMissing())
	case d == 0:
		return Missing[float64]("0 " + denName + ": ratio undefined")
	}
	return Measured(float64(n) / float64(d))
}

// Comparison is the single side-by-side surface: both arms, always in Arms order,
// whether or not an arm has rows.
type Comparison struct {
	Schema string       `json:"schema"`
	Ledger LedgerHealth `json:"ledger"`
	// WindowStartUnixNano is set when the caller compared only rows recorded since
	// then (see Since); 0 means the whole ledger.
	WindowStartUnixNano int64        `json:"window_start_unix_nano,omitempty"`
	Arms                []ArmSummary `json:"arms"`
	Conclusive          bool         `json:"conclusive"`
	Verdict             string       `json:"verdict"`
}

// Compare folds a ledger into the comparison. Pure.
func Compare(rows []Row, health LedgerHealth) Comparison {
	latest := Latest(rows)
	c := Comparison{Schema: CompareSchema, Ledger: health}
	for _, arm := range Arms {
		c.Arms = append(c.Arms, Summarize(arm, latest))
	}
	c.Conclusive, c.Verdict = verdict(c.Arms)
	return c
}

// verdict states the one-screen conclusion, or names exactly which arm's ship rate
// blocks it. The reasons themselves are listed under the table, once.
func verdict(arms []ArmSummary) (bool, string) {
	var blocked []string
	for _, a := range arms {
		if !a.ShipRate.IsMeasured() {
			blocked = append(blocked, fmt.Sprintf("%s ship rate UNKNOWN", a.Arm))
		}
	}
	if len(blocked) > 0 {
		return false, "INCONCLUSIVE: " + strings.Join(blocked, ", ") + " (reasons below)"
	}
	var parts []string
	for _, a := range arms {
		shipped, _ := a.PicksShipped.Total.Get()
		offered, _ := a.PicksOffered.Total.Get()
		rate, _ := a.ShipRate.Get()
		part := fmt.Sprintf("%s ships %.1f%% of picks (%d/%d)", a.Arm, rate*100, shipped, offered)
		if cps, ok := a.CostPerShippedUSD.Get(); ok {
			part += fmt.Sprintf(" at $%.2f per shipped pick", cps)
		} else {
			part += ", cost per shipped pick UNKNOWN"
		}
		parts = append(parts, part)
	}
	return true, strings.Join(parts, "; ") + " (git-verified)"
}

// RenderComparison prints both arms side by side, then every reason a cell reads
// UNKNOWN. Pure over its input.
func RenderComparison(w io.Writer, c Comparison) error {
	var b strings.Builder
	fmt.Fprintf(&b, "naive-control compare (%s)\n", c.Schema)
	fmt.Fprintf(&b, "naive arm: %s\n", NaiveCommand)
	b.WriteString(renderLedger(c.Ledger) + "\n")
	if c.WindowStartUnixNano > 0 {
		fmt.Fprintf(&b, "window: runs recorded since %s\n", time.Unix(0, c.WindowStartUnixNano).UTC().Format(time.RFC3339))
	}
	b.WriteString("\n")

	tw := tabwriter.NewWriter(&b, 0, 0, 3, ' ', 0)
	header := []string{"METRIC"}
	for _, a := range c.Arms {
		header = append(header, strings.ToUpper(string(a.Arm)))
	}
	fmt.Fprintln(tw, strings.Join(header, "\t"))
	rows := []struct {
		name string
		cell func(ArmSummary) string
	}{
		{"runs", func(a ArmSummary) string {
			if a.Runs == 0 {
				return "0 (no data)"
			}
			return fmt.Sprintf("%d (%d harvested)", a.Runs, a.Harvested)
		}},
		{"picks offered", func(a ArmSummary) string { return countCell(a.PicksOffered) }},
		{"picks shipped (git)", func(a ArmSummary) string { return countCell(a.PicksShipped) }},
		{"ship rate", func(a ArmSummary) string { return pctCell(a.ShipRate) }},
		{"commits claimed", func(a ArmSummary) string { return countCell(a.CommitsClaimed) }},
		{"commits landed (git)", func(a ArmSummary) string { return countCell(a.CommitsLanded) }},
		{"wall time", func(a ArmSummary) string { return msCell(a.WallMS) }},
		{"controller time", func(a ArmSummary) string { return msCell(a.ControllerMS) }},
		{"cost (USD)", func(a ArmSummary) string { return usdCell(a.CostUSD) }},
		{"cost / shipped pick", func(a ArmSummary) string { return usdMetric(a.CostPerShippedUSD) }},
	}
	for _, r := range rows {
		cells := []string{r.name}
		for _, a := range c.Arms {
			cells = append(cells, r.cell(a))
		}
		fmt.Fprintln(tw, strings.Join(cells, "\t"))
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	fmt.Fprintf(&b, "\nverdict: %s\n", c.Verdict)
	b.WriteString("\nUNKNOWN = the field could not be read. It is never a 0.\n")
	for _, a := range c.Arms {
		if a.Runs == 0 {
			fmt.Fprintf(&b, "  %s: no runs recorded, so every field is UNKNOWN\n", a.Arm)
			continue
		}
		for _, f := range []struct {
			name string
			why  string
		}{
			{"picks_offered", a.PicksOffered.Total.WhyMissing()},
			{"picks_shipped", a.PicksShipped.Total.WhyMissing()},
			{"commits_claimed", a.CommitsClaimed.Total.WhyMissing()},
			{"commits_landed", a.CommitsLanded.Total.WhyMissing()},
			{"wall_ms", a.WallMS.Total.WhyMissing()},
			{"controller_ms", a.ControllerMS.Total.WhyMissing()},
			{"cost_usd", a.CostUSD.Total.WhyMissing()},
		} {
			if f.why != "" {
				fmt.Fprintf(&b, "  %s.%s: %s\n", a.Arm, f.name, f.why)
			}
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func renderLedger(h LedgerHealth) string {
	path := h.Path
	if path == "" {
		path = "(ledger)"
	}
	if h.Missing {
		return fmt.Sprintf("ledger: %s does not exist yet (0 rows)", path)
	}
	s := fmt.Sprintf("ledger: %s, %d row(s), %d rejected", path, h.Rows, h.Rejected)
	for _, e := range h.Errors {
		s += "\n  rejected " + e
	}
	return s
}

func partial[T Number](a Aggregate[T], render func(T) string) string {
	if a.PartialSum == nil {
		return Unknown
	}
	return fmt.Sprintf("%s (partial %s over %d/%d runs)", Unknown, render(*a.PartialSum), a.Measured, a.Runs)
}

func countCell(a Aggregate[int64]) string {
	if v, ok := a.Total.Get(); ok {
		return fmt.Sprint(v)
	}
	return partial(a, func(v int64) string { return fmt.Sprint(v) })
}

func msCell(a Aggregate[int64]) string {
	render := func(v int64) string { return (time.Duration(v) * time.Millisecond).String() }
	if v, ok := a.Total.Get(); ok {
		return render(v)
	}
	return partial(a, render)
}

func usdCell(a Aggregate[float64]) string {
	render := func(v float64) string { return fmt.Sprintf("$%.2f", v) }
	if v, ok := a.Total.Get(); ok {
		return render(v)
	}
	return partial(a, render)
}

func usdMetric(m Metric[float64]) string {
	if v, ok := m.Get(); ok {
		return fmt.Sprintf("$%.2f", v)
	}
	return Unknown
}

func pctCell(m Metric[float64]) string {
	if v, ok := m.Get(); ok {
		return fmt.Sprintf("%.1f%%", v*100)
	}
	return Unknown
}
