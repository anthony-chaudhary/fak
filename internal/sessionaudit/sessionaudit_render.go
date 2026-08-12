package sessionaudit

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

func ReportMarkdown(sessions []Session, agg Aggregate, nsPrefix string, sinceDays *float64, includeSubagents bool, maxSessions int, discoveredCount int, excludedSubagents *Summary, generated time.Time) string {
	var b strings.Builder
	ok := validSessions(sessions)
	fmt.Fprintln(&b, "# Session-Transcript Audit - active scope")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "**Generated:** %s  \n", generated.Format("2006-01-02T15:04:05"))
	fmt.Fprintf(&b, "**Top-level sessions audited:** %d  .  **Tool:** `fak session-audit` (re-runnable)  \n", agg.NSessions)
	fmt.Fprintf(&b, "**Scope:** %s\n", scopeLine(ok, nsPrefix, sinceDays, includeSubagents, maxSessions))
	if note := maxClipNote(maxSessions, discoveredCount, nsPrefix != ""); note != "" {
		fmt.Fprintln(&b, note)
	}
	if note := subagentNote(excludedSubagents); note != "" {
		fmt.Fprintln(&b, note)
	}
	t := agg.Totals
	totalIn := t.Input + t.CacheRead + t.CacheCreate
	fmt.Fprintln(&b, "## Scope totals (EXACT token counts)")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- **Output tokens (the actual work generated):** %s\n", fmtInt(t.Output))
	fmt.Fprintf(&b, "- **Fresh input tokens (billed, non-cached):** %s\n", fmtInt(t.Input))
	fmt.Fprintf(&b, "- **Cache-read tokens (prompt-cache / KV reuse):** %s\n", fmtInt(t.CacheRead))
	fmt.Fprintf(&b, "- **Cache-creation tokens:** %s\n", fmtInt(t.CacheCreate))
	fmt.Fprintf(&b, "- **Total context ingested:** %s  ->  **scope I:O ratio = %.1f : 1**\n", fmtInt(totalIn), float64(totalIn)/float64(max64(t.Output, 1)))
	fmt.Fprintf(&b, "- **Cache-read share of all ingested context = %.1f%%**\n", float64(t.CacheRead)/float64(max64(totalIn, 1))*100)
	fmt.Fprintf(&b, "- **Web requests - server-tool (`server_tool_use`, billed):** search %s / fetch %s  .  **client tool:** WebSearch %s / WebFetch %s\n",
		fmtInt(t.WebSearch), fmtInt(t.WebFetch), fmtInt(agg.ToolMix["WebSearch"]), fmtInt(agg.ToolMix["WebFetch"]))
	fmt.Fprintf(&b, "- **Multi-iteration count:** %s\n", fmtInt(t.Iterations))
	fmt.Fprintf(&b, "- **Estimated Anthropic-billed cost:** $%s  _(cost uses an ASSUMED price table; token counts above are exact)_\n", fmtFloat(agg.TotalCostUSD, 2))
	if other := otherBuckets(agg.PerBucket); len(other) > 0 {
		parts := make([]string, 0, len(other))
		for _, bucket := range other {
			c := agg.PerBucket[bucket]
			parts = append(parts, fmt.Sprintf("%s (%s output tok, unpriced - add its card)", bucket, fmtInt(c.Output)))
		}
		fmt.Fprintf(&b, "- **Other billing buckets present (NOT in the total above - different invoices):** %s\n", strings.Join(parts, "; "))
	}
	if nb := agg.PerBucket["non-billed (harness)"]; nb.Turns > 0 {
		fmt.Fprintf(&b, "- **Non-billed `<synthetic>` turns (harness-injected, $0):** %s (%s output tok)\n", fmtInt(nb.Turns), fmtInt(nb.Output))
	}
	fmt.Fprintln(&b)
	renderModelMix(&b, agg)
	renderSelfHostedShare(&b, agg)
	writeDelegationShare(&b, agg)
	renderShellChoice(&b, agg)
	renderBuckets(&b, agg)
	renderModels(&b, agg)
	renderNamespaces(&b, agg)
	renderOpusHeavySessions(&b, ok)
	renderLongContextSessions(&b, ok)
	renderDistributions(&b, agg.Distributions)
	renderToolMix(&b, agg.ToolMix)
	renderTopSessions(&b, ok)
	return b.String()
}

func DeepMarkdown(s Session) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Trajectory: %s\n", s.Session)
	ioText := "null"
	if s.IORatio != nil {
		ioText = fmt.Sprintf("%.1f", *s.IORatio)
	}
	cacheText := "null"
	if s.CacheHitFrac != nil {
		cacheText = fmt.Sprintf("%.3f", *s.CacheHitFrac)
	}
	fmt.Fprintf(&b, "records=%d turns=%d tool_calls=%d output_tok=%s io=%s cache_hit=%s cost=$%.2f\n",
		s.NRecords, s.AssistantTurns, s.NToolUse, fmtInt(s.Tokens.Output), ioText, cacheText, s.CostUSD)
	fmt.Fprintf(&b, "tools=%v\n", s.Tools)
	fmt.Fprintln(&b, "\n## User asks (the trajectory), in order:")
	for i, p := range s.Prompts {
		txt := strings.Join(strings.Fields(p.Text), " ")
		if len(txt) > 200 {
			txt = txt[:200]
		}
		fmt.Fprintf(&b, "  [%2d] %s  %s\n", i, p.Timestamp, txt)
	}
	renderConfusion(&b, s.Confusion)
	return b.String()
}

// renderConfusion surfaces the prose Confusion lens for a single deep-audited session —
// the "read the trace for confusion" view: the confused-turn rate and every detected
// self-correction / dead-end / confusion marker with one verbatim example. A session
// with no markers prints a one-line clean verdict rather than an empty header.
func renderConfusion(b *strings.Builder, c Confusion) {
	fmt.Fprintln(b, "\n## Confusion / self-correction (assistant prose)")
	if c.TextTurns == 0 || c.TotalMarkers == 0 {
		fmt.Fprintf(b, "  none detected across %d prose turn(s)\n", c.TextTurns)
		return
	}
	fmt.Fprintf(b, "  score=%.3f  turns_with_confusion=%d/%d  self_correction_turns=%d  dead_end_turns=%d  confusion_turns=%d  total_markers=%d\n",
		c.Score, c.TurnsWithConfusion, c.TextTurns, c.SelfCorrectionTurns, c.DeadEndTurns, c.ConfusionTurns, c.TotalMarkers)
	for _, m := range c.Markers {
		fmt.Fprintf(b, "  - [%s] %s ×%d: %s\n", m.Category, m.Label, m.Count, m.Example)
	}
}

func scopeLine(sessions []Session, nsPrefix string, sinceDays *float64, includeSubagents bool, maxSessions int) string {
	names := sessionNamespaces(sessions)
	nsDesc := "none"
	if len(names) > 8 {
		nsDesc = strings.Join(names[:8], ", ") + fmt.Sprintf(", ... (+%d more)", len(names)-8)
	} else if len(names) > 0 {
		nsDesc = strings.Join(names, ", ")
	}
	nsFilter := nsPrefix
	if nsFilter == "" {
		nsFilter = "all non-excluded namespaces"
	}
	window := "all-time"
	if sinceDays != nil {
		window = "last " + trimFloat(*sinceDays) + " days"
	}
	kinds := "top-level session transcripts"
	if includeSubagents {
		// Says FOLDED IN, not "reported separately below", because that is what this flag
		// now does — and the distinction decides whether the delegation share can be
		// computed at all. This harness parks delegated turns in their own transcripts, so
		// a scope that leaves them out has no delegated volume to divide by.
		kinds += " (sub-agent / workflow transcripts folded in)"
	}
	cap := ""
	if maxSessions > 0 {
		cap = fmt.Sprintf("; max transcripts before analysis: %d", maxSessions)
	}
	return fmt.Sprintf("%d namespaces folded (%s); namespace filter: %s; time window: %s; %s%s", len(names), nsDesc, nsFilter, window, kinds, cap)
}

func maxClipNote(maxSessions, discoveredCount int, scoped bool) string {
	if maxSessions <= 0 || discoveredCount <= maxSessions {
		return ""
	}
	if scoped {
		return fmt.Sprintf("NOTE: `--max %d` clipped this scoped audit to the newest %d of %d discovered transcripts; raise `--max` before treating older sessions, model usage, or long-context rows inside this scope as absent.",
			maxSessions, maxSessions, discoveredCount)
	}
	return fmt.Sprintf("NOTE: `--max %d` clipped this audit to the newest %d of %d discovered transcripts; use `--ns-prefix <namespace>`, `--here`, or raise `--max` before treating missing namespaces or model usage as absent.",
		maxSessions, maxSessions, discoveredCount)
}

func subagentNote(summary *Summary) string {
	if summary == nil || summary.Count == 0 {
		return ""
	}
	return fmt.Sprintf("NOTE: +%d subagent transcripts uncounted; re-run with `--include-subagents` (about +$%s / +%s output tok).",
		summary.Count, fmtFloat(summary.CostUSD, 2), fmtInt(summary.Tokens.Output))
}

func otherBuckets(buckets map[string]ModelCounts) []string {
	var out []string
	for bucket, c := range buckets {
		if bucket != "Anthropic (Claude)" && bucket != "non-billed (harness)" && c.Output > 0 {
			out = append(out, bucket)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if buckets[out[i]].Output == buckets[out[j]].Output {
			return out[i] < out[j]
		}
		return buckets[out[i]].Output > buckets[out[j]].Output
	})
	return out
}

func renderModelMix(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Model-mix KPI (tier shares)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Tier | Output tok | Output share | Est. cost | Cost share |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|")
	totalOutput, totalCost := tierOutputCostTotals(agg)
	for _, tier := range sortedModelCounts(agg.PerTier) {
		c := agg.PerTier[tier]
		tierCost := modelCostByKey(agg, tier, ModelTier)
		fmt.Fprintf(b, "| %s | %s | %s | $%s | %s |\n", tier, fmtInt(c.Output), fmtPct(ratio(c.Output, totalOutput)), fmtFloat(tierCost, 2), fmtPct(floatRatio(tierCost, totalCost)))
	}
	fmt.Fprintln(b)
}

// renderSelfHostedShare prints epic #5416's headline number: the fraction of generated
// tokens served on hardware we operate, ALWAYS next to the coverage that qualifies it.
//
// The two never appear apart. "62% self-hosted" over 4% coverage is a misleading
// headline, and on today's corpus the honest reading is usually "we cannot tell yet" —
// which is a finding, not a gap to paper over with a zero.
func renderSelfHostedShare(b *strings.Builder, agg Aggregate) {
	s := FoldSelfHostedShare(agg.PerBucket)
	fmt.Fprintln(b, "## Self-hosted share (where the tokens were actually served)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Placement | Turns | Output tok | Cache-read tok |")
	fmt.Fprintln(b, "|---|---:|---:|---:|")
	for _, row := range []struct {
		label string
		c     ModelCounts
	}{
		{"self-hosted (device + fleet)", s.SelfHosted},
		{"vendor API", s.Vendor},
		{"unattributable (no placement signal)", s.Unattributable},
		{"non-billed (harness, not inference)", s.NonBilled},
	} {
		fmt.Fprintf(b, "| %s | %s | %s | %s |\n", row.label, fmtInt(row.c.Turns), fmtInt(row.c.Output), fmtInt(row.c.CacheRead))
	}
	fmt.Fprintln(b)
	if s.OutputShare == nil {
		fmt.Fprintf(b, "**Self-hosted share = UNKNOWN** - no turn in this corpus carries a placement signal, so %s output tok of real inference cannot be attributed either way. That is the wiring gap, not a 0%%.\n",
			fmtInt(s.Unattributable.Output))
	} else {
		fmt.Fprintf(b, "**Self-hosted share of attributed output = %s**  _(over %s of inference output; the rest carries no placement signal)_\n",
			fmtPct(s.OutputShare), fmtPct(s.Coverage))
	}
	if len(s.UnattributableBuckets) > 0 {
		fmt.Fprintf(b, "- **Missing the placement signal:** %s\n", strings.Join(s.UnattributableBuckets, "; "))
	}
	fmt.Fprintln(b)
}

// writeDelegationShare prints how much of the corpus's generated volume was produced for
// delegated work — epic #5416 track E's lever, sized rather than asserted.
//
// It sits directly under the self-hosted share because the two are read together: the
// self-hosted share says how much volume already runs on our own hardware, and this says
// how much of the remainder is the kind that could be moved without an engineer noticing.
//
// An uninstrumented corpus reports UNKNOWN, never 0%. A zero here would be the strongest
// possible argument against track E, and it must not be available for free to a transcript
// that simply never wrote the marker.
//
// Named write* rather than render* deliberately, so please leave it: a new render-family
// identifier is refused on admission, and renaming out of the family is the sanctioned
// escape.
func writeDelegationShare(b *strings.Builder, agg Aggregate) {
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)
	fmt.Fprintln(b, "## Delegation share (how much volume is sub-agent and background work)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Track | Turns | Output tok | Cache-read tok |")
	fmt.Fprintln(b, "|---|---:|---:|---:|")
	for _, row := range []struct {
		label string
		c     ModelCounts
	}{
		{"delegated (sub-agent + background)", d.Delegation},
		{"main thread (the operator's own session)", d.Main},
		{"untracked (no delegation signal)", d.Untracked},
	} {
		fmt.Fprintf(b, "| %s | %s | %s | %s |\n", row.label, fmtInt(row.c.Turns), fmtInt(row.c.Output), fmtInt(row.c.CacheRead))
	}
	fmt.Fprintln(b)
	switch {
	case d.UnderInstrumented:
		fmt.Fprintf(b, "**Delegation share = UNKNOWN** - this corpus made %s spawn-tool call(s) but not one turn carries the `isSidechain` marker, so delegated volume was generated and not recorded. That is an instrumentation gap, NOT a 0%%.\n",
			fmtInt(d.SpawnCalls))
	case d.OutputShare == nil:
		fmt.Fprintln(b, "**Delegation share = UNKNOWN** - no generated output in this corpus carries a delegation track.")
	default:
		fmt.Fprintf(b, "**Delegation share of tracked output = %s**  _(over %s of generated output; %s spawn-tool call(s) observed)_\n",
			fmtPct(d.OutputShare), fmtPct(d.Coverage), fmtInt(d.SpawnCalls))
	}
	fmt.Fprintln(b)
}

// renderShellChoice prints the shell-choice KPI (#3227): which shell the corpus
// actually reached for, and how often each one came back broken.
//
// It sits in the scope rollup, not in the tool-mix table at the bottom, because the
// tool mix answers "how many Bash calls" while this answers the two questions a fix
// can move: which shell agents PICK, and which one WORKS. Reading them apart is what
// left the friction an anecdote someone had to re-derive by eye.
//
// A shell nobody called still gets its row (that IS the choice signal), and a window
// with no shell calls reports UNKNOWN rather than a flawless 0%.
func renderShellChoice(b *strings.Builder, agg Aggregate) {
	sc := agg.ShellChoice
	fmt.Fprintln(b, "## Shell choice (KPI)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Shell | Calls | Call share | Errors | Error rate |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|")
	for _, s := range sc.Shells {
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s |\n", s.Tool, fmtInt(s.Calls), fmtPct(s.CallShare), fmtInt(s.Errors), fmtPct(s.ErrorRate))
	}
	fmt.Fprintln(b)
	if sc.Calls == 0 {
		fmt.Fprintln(b, "**Preferred shell = UNKNOWN** - this corpus ran no shell command at all, so there is no choice and no error rate to report. That is an empty window, NOT a 0% error rate.")
		fmt.Fprintln(b)
		return
	}
	fmt.Fprintf(b, "**Preferred shell:** %s  _(%s of %s shell call(s); all-shell error rate %s over %s errored result(s))_\n",
		sc.Preferred, fmtPct(ratio(shellCalls(sc, sc.Preferred), sc.Calls)), fmtInt(sc.Calls), fmtPct(sc.ErrorRate), fmtInt(sc.Errors))
	fmt.Fprintln(b)
}

// shellCalls returns one shell's call count from a folded KPI, or 0 for a name the
// fold does not carry.
func shellCalls(sc ShellChoice, tool string) int64 {
	for _, s := range sc.Shells {
		if s.Tool == tool {
			return s.Calls
		}
	}
	return 0
}

func renderBuckets(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Cost by billing bucket (provider) - never sum across these")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Billing bucket | Turns | Output tok | Cache-read tok | Est. cost | Priced? |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|:--:|")
	for _, bucket := range sortedModelCounts(agg.PerBucket) {
		c := agg.PerBucket[bucket]
		bcost := modelCostByKey(agg, bucket, ProviderBucket)
		costCell := "- (no card)"
		priced := ""
		if bucket == BucketAnthropic {
			costCell = "$" + fmtFloat(bcost, 2)
			priced = "yes"
		} else if bucket == BucketNonBilled {
			costCell = "$0.00"
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s | %s |\n", bucket, fmtInt(c.Turns), fmtInt(c.Output), fmtInt(c.CacheRead), costCell, priced)
	}
	fmt.Fprintln(b)
}

func renderModels(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Per-model breakdown (token-exact; cost Anthropic-assumed)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Model | Bucket | Turns | Output tok | Cache-read tok | Est. cost |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|")
	for _, model := range sortedModelCounts(agg.PerModel) {
		c := agg.PerModel[model]
		costCell := "- (no card)"
		if _, ok := PriceFor(model); ok {
			costCell = "$" + fmtFloat(ModelCost(model, c), 2)
		} else if nonBilled(model) {
			costCell = "$0.00"
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s | %s |\n", model, ProviderBucket(model), fmtInt(c.Turns), fmtInt(c.Output), fmtInt(c.CacheRead), costCell)
	}
	fmt.Fprintln(b)
}

func renderNamespaces(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Per-namespace rollup")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Namespace | Sessions | Output tok | Opus output share | Cache-read tok | Tool calls | Top model (by output) | Est. cost |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|---:|---|---:|")
	keys := make([]string, 0, len(agg.PerNamespace))
	for ns := range agg.PerNamespace {
		keys = append(keys, ns)
	}
	sort.Slice(keys, func(i, j int) bool {
		if agg.PerNamespace[keys[i]].Output == agg.PerNamespace[keys[j]].Output {
			return keys[i] < keys[j]
		}
		return agg.PerNamespace[keys[i]].Output > agg.PerNamespace[keys[j]].Output
	})
	for _, ns := range keys {
		v := agg.PerNamespace[ns]
		fmt.Fprintf(b, "| %s | %d | %s | %s | %s | %s | %s | $%s |\n",
			ns, v.Sessions, fmtInt(v.Output), fmtPctPtr(agg.PerNamespaceOpusShare[ns]), fmtInt(v.CacheRead), fmtInt(v.ToolUse), agg.PerNamespaceTopModel[ns], fmtFloat(agg.PerNamespaceCost[ns], 2))
	}
	fmt.Fprintln(b)
}

type opusSessionRow struct {
	Session   Session
	OpusOut   int64
	OpusCost  float64
	OpusShare *float64
	TopModel  string
	TotalCost float64
	TotalOut  int64
}

func renderOpusHeavySessions(b *strings.Builder, sessions []Session) {
	rows := opusHeavySessionRows(sessions)
	if len(rows) == 0 {
		return
	}
	fmt.Fprintln(b, "## Opus-heavy sessions")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Session | NS | Opus output tok | Opus share | Opus est.$ | Total output tok | Total est.$ | Top model |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|---:|---|")
	if len(rows) > 10 {
		rows = rows[:10]
	}
	for _, row := range rows {
		sid := shortSessionID(row.Session.Session)
		fmt.Fprintf(b, "| %s | %s | %s | %s | $%s | %s | $%s | %s |\n",
			sid,
			namespaceName(row.Session.Path),
			fmtInt(row.OpusOut),
			fmtPctPtr(row.OpusShare),
			fmtFloat(row.OpusCost, 2),
			fmtInt(row.TotalOut),
			fmtFloat(row.TotalCost, 2),
			row.TopModel)
	}
	fmt.Fprintln(b)
}

func opusHeavySessionRows(sessions []Session) []opusSessionRow {
	rows := make([]opusSessionRow, 0, len(sessions))
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		opusOut, opusCost := sessionTierOutputCost(s, "opus")
		if opusOut == 0 {
			continue
		}
		var share *float64
		if s.Tokens.Output > 0 {
			v := float64(opusOut) / float64(s.Tokens.Output)
			share = &v
		}
		rows = append(rows, opusSessionRow{
			Session:   s,
			OpusOut:   opusOut,
			OpusCost:  opusCost,
			OpusShare: share,
			TopModel:  topSessionModel(s),
			TotalCost: s.CostUSD,
			TotalOut:  s.Tokens.Output,
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].OpusOut == rows[j].OpusOut {
			if rows[i].OpusCost == rows[j].OpusCost {
				return rows[i].Session.Path < rows[j].Session.Path
			}
			return rows[i].OpusCost > rows[j].OpusCost
		}
		return rows[i].OpusOut > rows[j].OpusOut
	})
	return rows
}

func sessionTierOutputCost(s Session, tier string) (int64, float64) {
	var output int64
	var cost float64
	for model, counts := range s.PerModel {
		if ModelTier(model) != tier {
			continue
		}
		output += counts.Output
		cost += ModelCost(model, counts)
	}
	return output, cost
}

func topSessionModel(s Session) string {
	top := ""
	var topOut int64
	for model, counts := range s.PerModel {
		if counts.Output > topOut || (counts.Output == topOut && (top == "" || model < top)) {
			top = model
			topOut = counts.Output
		}
	}
	if top == "" {
		return "?"
	}
	return top
}

// shortSessionID truncates a session ID to its first 8 characters for
// compact table display.
func shortSessionID(sid string) string {
	if len(sid) > 8 {
		return sid[:8]
	}
	return sid
}

type longContextSessionRow struct {
	Session       Session
	TotalContext  int64
	CacheReadFrac *float64
}

func renderLongContextSessions(b *strings.Builder, sessions []Session) {
	rows := longContextSessionRows(sessions)
	if len(rows) == 0 {
		return
	}
	fmt.Fprintln(b, "## Long-context sessions")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Session | NS | Total context tok | Fresh input tok | Cache-read tok | Cache-read share | Output tok | I:O | Top model |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|---:|---:|---|")
	if len(rows) > 10 {
		rows = rows[:10]
	}
	for _, row := range rows {
		s := row.Session
		sid := shortSessionID(s.Session)
		ioCell := "-"
		if s.IORatio != nil {
			ioCell = fmt.Sprintf("%.1f", *s.IORatio)
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
			sid,
			namespaceName(s.Path),
			fmtInt(row.TotalContext),
			fmtInt(s.Tokens.Input),
			fmtInt(s.Tokens.CacheRead),
			fmtPctPtr(row.CacheReadFrac),
			fmtInt(s.Tokens.Output),
			ioCell,
			topSessionModel(s))
	}
	fmt.Fprintln(b)
}

func longContextSessionRows(sessions []Session) []longContextSessionRow {
	rows := make([]longContextSessionRow, 0, len(sessions))
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		total := totalContextTokens(s)
		if total == 0 {
			continue
		}
		rows = append(rows, longContextSessionRow{
			Session:       s,
			TotalContext:  total,
			CacheReadFrac: ratio(s.Tokens.CacheRead, total),
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].TotalContext == rows[j].TotalContext {
			return rows[i].Session.Path < rows[j].Session.Path
		}
		return rows[i].TotalContext > rows[j].TotalContext
	})
	return rows
}

func totalContextTokens(s Session) int64 {
	return s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
}

func renderDistributions(b *strings.Builder, d Distributions) {
	fmt.Fprintln(b, "## Distributions (per session)")
	fmt.Fprintln(b)
	fmt.Fprintf(b, "- **Tool calls/session:** median %s, mean %s, p90 %s, max %s\n",
		fmtStat(d.CallsPerSession.Median), fmtStat(d.CallsPerSession.Mean), fmtStat(d.CallsPerSession.P90), fmtStat(d.CallsPerSession.Max))
	fmt.Fprintf(b, "- **Output tokens/session:** median %s, p90 %s, max %s\n",
		fmtStatInt(d.OutputTokensPerSession.Median), fmtStatInt(d.OutputTokensPerSession.P90), fmtStatInt(d.OutputTokensPerSession.Max))
	fmt.Fprintf(b, "- **I:O ratio/session:** median %s, p90 %s\n", fmtStat(d.IORatio.Median), fmtStat(d.IORatio.P90))
	fmt.Fprintf(b, "- **Cache-hit fraction/session:** median %s, p10 %s, p90 %s\n",
		fmtStat(d.CacheHitFrac.Median), fmtStat(d.CacheHitFrac.P10), fmtStat(d.CacheHitFrac.P90))
	fmt.Fprintf(b, "- **Read-only tool fraction/session:** median %s\n", fmtStat(d.ReadOnlyFrac.Median))
	// Shell error rate is per SESSION here (#3227): the corpus-wide rate in the KPI
	// rollup averages an outlier away, and the max is the session that ate it.
	fmt.Fprintf(b, "- **Shell error rate/session:** median %s, p90 %s, max %s  _(sessions that ran a shell command)_\n",
		fmtPct(d.ShellErrorRate.Median), fmtPct(d.ShellErrorRate.P90), fmtPct(d.ShellErrorRate.Max))
	fmt.Fprintln(b)
}

func renderToolMix(b *strings.Builder, tools map[string]int64) {
	fmt.Fprintln(b, "## Global tool mix")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Tool | Calls | Read-only? |")
	fmt.Fprintln(b, "|---|---:|:--:|")
	keys := sortedCounts(tools)
	if len(keys) > 25 {
		keys = keys[:25]
	}
	for _, name := range keys {
		mark := ""
		if ReadOnlyTools[name] {
			mark = "yes"
		}
		fmt.Fprintf(b, "| %s | %s | %s |\n", name, fmtInt(tools[name]), mark)
	}
	fmt.Fprintln(b)
}

func renderTopSessions(b *strings.Builder, sessions []Session) {
	fmt.Fprintln(b, "## Top 15 sessions by output tokens")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Session | NS | Turns | Tool calls | Output tok | I:O | Cache-hit | Est.$ |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|---:|---:|")
	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].Tokens.Output == sessions[j].Tokens.Output {
			return sessions[i].Path < sessions[j].Path
		}
		return sessions[i].Tokens.Output > sessions[j].Tokens.Output
	})
	if len(sessions) > 15 {
		sessions = sessions[:15]
	}
	for _, s := range sessions {
		sid := shortSessionID(s.Session)
		ioCell := "-"
		if s.IORatio != nil {
			ioCell = fmt.Sprintf("%.0f", *s.IORatio)
		}
		chCell := "-"
		if s.CacheHitFrac != nil {
			chCell = fmt.Sprintf("%.0f%%", *s.CacheHitFrac*100)
		}
		fmt.Fprintf(b, "| %s | %s | %d | %d | %s | %s | %s | $%.2f |\n",
			sid, namespaceName(s.Path), s.AssistantTurns, s.NToolUse, fmtInt(s.Tokens.Output), ioCell, chCell, s.CostUSD)
	}
	fmt.Fprintln(b)
}
