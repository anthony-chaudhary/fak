package operatorbrief

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// RenderCompact is the default operator snapshot: one terse line per item with
// no per-item detail or action prose, and none of the interpretation blocks
// (human-use, coherence, strengths, learning, generation lanes). It answers
// "what is my pace and what needs a look" in a handful of lines. `Render`
// (behind --full) carries the full explanation for a human who wants to drill
// in, and --json carries the whole model for agents. The point is that an
// operator scanning many briefs a day should not read a wall of text per item.
func RenderCompact(r Report) string {
	lines := []string{
		fmt.Sprintf("operator brief - %s %s  @%s  %s", r.Verdict, r.Pace, strmatch.DashIfBlank(r.Commit), strmatch.DashIfBlank(r.Date)),
		fmt.Sprintf("  load       human %d, agent %d, watch %d, background %d  |  attention %s %dm",
			r.Counts.Human, r.Counts.Agent, r.Counts.Watch, r.Counts.Background,
			strmatch.DashIfBlank(r.Attention.Level), r.Attention.BudgetMinutes),
	}
	if r.Delta != nil && r.Delta.Status == "changed" {
		lines = append(lines, "  since prev "+r.Delta.Summary)
	}
	lines = appendCompactSection(lines, "human", r.Human, 6)
	lines = appendCompactSection(lines, "agent", r.Agent, 6)
	lines = appendCompactSection(lines, "watch", r.Watch, 6)
	lines = appendDebtSection(lines, "origin_debt", r.OriginDebt, 6)
	lines = appendDebtSection(lines, "late_found_debt", r.LateFoundDebt, 6)
	lines = append(lines, "", "  -> "+r.NextAction, "  (--full for the full explanation; --json for agents)")
	return strings.Join(lines, "\n")
}

// appendCompactSection prints a bucket as a title plus one terse line per item
// (source: title only). Long buckets are capped with a "+N more" pointer so the
// compact view stays small; --full shows every item with its detail and action.
func appendCompactSection(lines []string, name string, items []Item, limit int) []string {
	if len(items) == 0 {
		return lines
	}
	lines = append(lines, "  "+name+":")
	shown, extra := items, 0
	if len(items) > limit {
		shown, extra = items[:limit], len(items)-limit
	}
	for _, it := range shown {
		lines = append(lines, "    - "+it.Source+": "+it.Title)
	}
	if extra > 0 {
		lines = append(lines, fmt.Sprintf("    ... +%d more (--full)", extra))
	}
	return lines
}

// appendDebtSection prints a debt bucket as a title plus one line per witness,
// capped like the item sections so the compact view stays scannable. The two
// buckets stay visibly separate: origin debt is cleanup, late-found debt is a
// missing control.
func appendDebtSection(lines []string, name string, records []DebtWitnessRecord, limit int) []string {
	if len(records) == 0 {
		return lines
	}
	lines = append(lines, "  "+name+":")
	shown, extra := records, 0
	if len(records) > limit {
		shown, extra = records[:limit], len(records)-limit
	}
	for _, record := range shown {
		lines = append(lines, "    - "+debtLine(record))
	}
	if extra > 0 {
		lines = append(lines, fmt.Sprintf("    ... +%d more (--full)", extra))
	}
	return lines
}

// Render produces the full human snapshot (every section, with per-item detail
// and action). It is the --full expansion of RenderCompact.
func Render(r Report) string {
	lines := []string{
		fmt.Sprintf("operator brief - %s (%s)  @%s  %s", r.Verdict, r.Finding, strmatch.DashIfBlank(r.Commit), strmatch.DashIfBlank(r.Date)),
		"",
		fmt.Sprintf("  pace       %s; human %d, agent %d, watch %d, background %d",
			r.Pace, r.Counts.Human, r.Counts.Agent, r.Counts.Watch, r.Counts.Background),
		"  state      " + strmatch.DashIfBlank(r.State.OperatorUse),
		"  attention  " + renderAttention(r.Attention),
		"  human use  " + renderHumanUse(r.HumanUse),
		"  coherence  " + renderCoherence(r.Coherence),
		"  sources    " + renderSources(r.Sources),
	}
	lines = appendGeneration(lines, r.Generation)
	lines = appendDelta(lines, r.Delta)
	lines = appendStrengths(lines, r.Strengths)
	lines = appendChoices(lines, r.Choices)
	lines = appendChallenges(lines, r.Challenges)
	lines = appendLearningAgenda(lines, r.Agenda)
	lines = appendLearning(lines, r.Learning)
	lines = appendSection(lines, "human", r.Human)
	lines = appendSection(lines, "agent", r.Agent)
	lines = appendSection(lines, "watch", r.Watch)
	lines = appendSection(lines, "background", capItems(r.Background, 4))
	lines = appendFullDebtSection(lines, "origin_debt", r.OriginDebt)
	lines = appendFullDebtSection(lines, "late_found_debt", r.LateFoundDebt)
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

func appendGeneration(lines []string, g *Generation) []string {
	if g == nil {
		return lines
	}
	lines = append(lines, "  generation "+g.Summary)
	if g.Attention != "" {
		lines = append(lines, "              "+g.Attention)
	}
	if g.Heat != "" {
		lines = append(lines, "              heat "+g.Heat)
	}
	for _, lane := range g.Lanes {
		if lane.Tracked == 0 {
			lines = append(lines, fmt.Sprintf("              %s: 0 tracked", lane.Generation))
			continue
		}
		parts := []string{fmt.Sprintf("%s: %d tracked", lane.Generation, lane.Tracked)}
		if lane.OpenDiscrete > 0 || lane.Discrete > 0 {
			parts = append(parts, fmt.Sprintf("%d open discrete", lane.OpenDiscrete))
			parts = append(parts, fmt.Sprintf("%.1f%% discrete", lane.OverallPct))
		}
		if lane.Programs > 0 {
			parts = append(parts, fmt.Sprintf("%d ongoing", lane.Programs))
		}
		if lane.ShipVelocity != "" {
			parts = append(parts, "velocity "+lane.ShipVelocity)
		}
		if lane.HeatScore > 0 {
			if lane.HeatReason != "" {
				parts = append(parts, fmt.Sprintf("heat %d (%s)", lane.HeatScore, lane.HeatReason))
			} else {
				parts = append(parts, fmt.Sprintf("heat %d", lane.HeatScore))
			}
		}
		if lane.StaleAge != "" {
			parts = append(parts, "stale age "+lane.StaleAge)
		}
		if lane.Errored > 0 {
			parts = append(parts, fmt.Sprintf("%d unreadable", lane.Errored))
		}
		if lane.DebtScore > 0 {
			if lane.DebtReason != "" {
				parts = append(parts, fmt.Sprintf("debt %d (%s)", lane.DebtScore, lane.DebtReason))
			} else {
				parts = append(parts, fmt.Sprintf("debt %d", lane.DebtScore))
			}
		}
		lines = append(lines, "              "+strings.Join(parts, "; "))
	}
	return lines
}

func appendDelta(lines []string, d *Delta) []string {
	if d == nil {
		return lines
	}
	lines = append(lines, "", "  since previous:")
	line := fmt.Sprintf("    - %s: %s", d.Status, d.Summary)
	if d.PaceChanged {
		line += fmt.Sprintf(" | pace: %s -> %s", d.PaceFrom, d.PaceTo)
	}
	lines = append(lines, line)
	lines = appendDeltaItems(lines, "new", d.New)
	lines = appendDeltaItems(lines, "resolved", d.Resolved)
	lines = appendDeltaItems(lines, "still present", d.Persistent)
	return lines
}

func appendDeltaItems(lines []string, label string, items []DeltaItem) []string {
	if len(items) == 0 {
		return lines
	}
	for _, it := range items {
		line := fmt.Sprintf("    - %s: [%s] %s: %s", label, it.Bucket, it.Source, it.Title)
		line = appendField(line, " - ", it.Detail)
		line = appendField(line, " | ", it.Action)
		lines = append(lines, line)
	}
	return lines
}

// appendField appends sep+value to line when value is non-empty, returning the
// (possibly unchanged) line. It captures the repeated
// `if value != "" { line += sep + value }` idiom used across the renderers.
func appendField(line, sep, value string) string {
	if value != "" {
		line += sep + value
	}
	return line
}

func appendStrengths(lines []string, strengths []Strength) []string {
	if len(strengths) == 0 {
		return lines
	}
	lines = append(lines, "", "  strengths:")
	for _, st := range strengths {
		line := fmt.Sprintf("    - %s: %s (%s)", st.Source, st.Title, st.Kind)
		line = appendField(line, " - ", st.Detail)
		line = appendField(line, " | ", st.Use)
		lines = append(lines, line)
	}
	return lines
}

func appendChoices(lines []string, choices []Choice) []string {
	if len(choices) == 0 {
		return lines
	}
	lines = append(lines, "", "  choices:")
	for _, ch := range choices {
		disp := string(ch.Disposition)
		if disp == "" {
			disp = string(choicetriage.FreshContext)
		}
		line := fmt.Sprintf("    - %s: %s [%s]", ch.Source, ch.Question, disp)
		line = appendField(line, " -> ", ch.Resolve)
		line = appendField(line, " | ", ch.Action)
		lines = append(lines, line)
	}
	return lines
}

func appendChallenges(lines []string, challenges []Challenge) []string {
	if len(challenges) == 0 {
		return lines
	}
	lines = append(lines, "", "  challenges:")
	for _, ch := range challenges {
		line := fmt.Sprintf("    - %s: %s (%s)", ch.Source, ch.Title, ch.Kind)
		line = appendField(line, " - ", ch.Detail)
		line = appendField(line, " | ", ch.Action)
		lines = append(lines, line)
	}
	return lines
}

func appendLearningAgenda(lines []string, agenda LearningAgenda) []string {
	if agenda.Focus == "" {
		return lines
	}
	line := fmt.Sprintf("    - focus: %s", agenda.Focus)
	line = appendField(line, " - ", agenda.Reason)
	if agenda.Practice != "" {
		line += " | practice: " + agenda.Practice
	}
	if agenda.Skip != "" {
		line += " | skip: " + agenda.Skip
	}
	if len(agenda.DrillDown) > 0 {
		line += " | drill: " + strings.Join(agenda.DrillDown, " -> ")
	}
	return append(lines, "", "  learning agenda:", line)
}

func appendLearning(lines []string, lessons []Learning) []string {
	if len(lessons) == 0 {
		return lines
	}
	lines = append(lines, "", "  learning:")
	for _, l := range lessons {
		line := fmt.Sprintf("    - %s: %s", l.Topic, l.Lesson)
		line = appendField(line, " | ", l.Apply)
		lines = append(lines, line)
	}
	return lines
}

func appendSection(lines []string, name string, items []Item) []string {
	if len(items) == 0 {
		return lines
	}
	lines = append(lines, "", "  "+name+":")
	for _, it := range items {
		line := fmt.Sprintf("    - %s: %s", it.Source, it.Title)
		line = appendField(line, " - ", it.Detail)
		line = appendField(line, " | ", it.Action)
		lines = append(lines, line)
	}
	return lines
}

// appendFullDebtSection is the --full expansion of appendDebtSection: every
// witness, uncapped, under a blank-line-separated heading like the other full
// sections.
func appendFullDebtSection(lines []string, name string, records []DebtWitnessRecord) []string {
	if len(records) == 0 {
		return lines
	}
	lines = append(lines, "", "  "+name+":")
	for _, record := range records {
		lines = append(lines, "    - "+debtLine(record))
	}
	return lines
}

func renderAttention(a Attention) string {
	parts := []string{
		strmatch.DashIfBlank(a.Level),
		fmt.Sprintf("%d min", a.BudgetMinutes),
		strmatch.DashIfBlank(a.Cadence),
	}
	if len(a.ReadOrder) > 0 {
		parts = append(parts, strings.Join(a.ReadOrder, " -> "))
	}
	if a.Summary != "" {
		parts = append(parts, a.Summary)
	}
	return strings.Join(parts, "; ")
}

func renderHumanUse(h HumanUse) string {
	parts := []string{
		"human: " + strmatch.DashIfBlank(h.UseHumanFor),
		"agents: " + strmatch.DashIfBlank(h.LetAgentsDo),
		"avoid: " + strmatch.DashIfBlank(h.Avoid),
		"escalate: " + strmatch.DashIfBlank(h.EscalateWhen),
	}
	return strings.Join(parts, "; ")
}

func renderCoherence(c Coherence) string {
	parts := []string{strmatch.DashIfBlank(c.Status), strmatch.DashIfBlank(c.Summary)}
	if c.Action != "" {
		parts = append(parts, c.Action)
	}
	return strings.Join(parts, "; ")
}

func capItems(items []Item, n int) []Item {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

func renderSources(srcs []SourceState) string {
	parts := make([]string, 0, len(srcs))
	for _, s := range srcs {
		parts = append(parts, s.Name+"="+s.Status)
	}
	return strings.Join(parts, ", ")
}
