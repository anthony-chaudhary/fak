package maturity

import (
	"encoding/json"
	"strings"
)

// Render is the terminal view: the headline, the rung distribution as a bar
// chart, and the top of the next-work backlog.
func Render(p ScorecardPayload) string {
	c := p.Corpus
	dist, _ := c["distribution"].(map[string]int)
	lines := []string{
		"maturity — " + p.Verdict + " (" + p.Finding + ")",
		"  maturity_debt (ladder-skips): " + anyStr(c["maturity_debt"]) +
			"   index " + anyStr(c["score"]) + "/100 [" + anyStr(c["grade"]) + "]   over " +
			anyStr(c["capabilities"]) + " capabilities",
		"  " + p.Reason,
		"",
		"  lifecycle ladder (count of capabilities per rung):",
	}
	max := 0
	for _, r := range RungName {
		if dist[r] > max {
			max = dist[r]
		}
	}
	for i, r := range RungName {
		lines = append(lines, "    "+itoa(i)+" "+pad(r, 12)+" "+bar(dist[r], max)+" "+itoa(dist[r]))
	}
	lines = append(lines, "", "  next work (advance one rung — skips first, then least-mature):")
	shown := 0
	for _, w := range p.Backlog {
		if shown >= 12 {
			lines = append(lines, "    … and "+itoa(len(p.Backlog)-shown)+" more (see `fak maturity next`)")
			break
		}
		mark := " "
		if w.Skip {
			mark = "!"
		}
		lines = append(lines, "    "+mark+" "+w.Title)
		shown++
	}
	if len(p.Backlog) == 0 {
		lines = append(lines, "    (none — every capability is at the top of the ladder)")
	}
	lines = append(lines, "", "  NEXT: "+p.NextAction)
	return strings.Join(lines, "\n")
}

// RenderNext is the focused backlog view — the agentic-culture queue an agent or
// the dispatch loop pulls from. One ticket-shaped line per capability.
func RenderNext(p ScorecardPayload) string {
	lines := []string{
		"maturity backlog — " + itoa(len(p.Backlog)) + " next work item(s) (ladder-skips first):",
		"",
	}
	if len(p.Backlog) == 0 {
		lines = append(lines, "  (none — every declared capability is at the top of the ladder)")
		return strings.Join(lines, "\n")
	}
	for _, w := range p.Backlog {
		tag := w.FromRung.String() + " → " + w.Gap.String()
		if w.Skip {
			tag = "SKIP " + tag
		}
		lines = append(lines, "  ["+tag+"] "+w.Title)
		lines = append(lines, "      witness: "+w.Witness)
	}
	return strings.Join(lines, "\n")
}

// Markdown renders the durable scorecard doc.
func Markdown(p ScorecardPayload) string {
	c := p.Corpus
	dist, _ := c["distribution"].(map[string]int)
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString(`title: "fak maturity scorecard — where each capability sits on its lifecycle ladder"` + "\n")
	b.WriteString(`description: "Every declared fak capability (one per internal/<leaf> lane) placed on a closed lifecycle ladder — proposed → prototyped → tested → dogfooded → default, with a benchmarked badge — and the next work item that would mature it. Immaturity is not a defect; a ladder-skip (fak relies on a capability yet leaves it untested) is. Re-derived from dos.toml + the tree's import graph + the CLI reference."` + "\n")
	b.WriteString("---\n\n")
	b.WriteString("# fak maturity scorecard — lifecycle, not just completeness\n\n")
	b.WriteString("**maturity_debt (ladder-skips): " + anyStr(c["maturity_debt"]) + "**; maturity index **" +
		anyStr(c["score"]) + "/100 (" + anyStr(c["grade"]) + ")** over **" + anyStr(c["capabilities"]) +
		"** declared capabilities" + benchmarkedSuffix(c) + ".\n\n")
	b.WriteString("> " + p.Reason + "\n\n")
	b.WriteString("A v1 prototype can be legitimately *complete* and still not be tested, dogfooded, " +
		"benchmarked, or the default. This scorecard makes that lifecycle visible: it places every declared " +
		"capability (one per `internal/<leaf>` lane in [`dos.toml`](../dos.toml) `[lanes.trees]`) on a closed " +
		"ladder, and for each one names the next step that would mature it. Every rung is gated by evidence " +
		"the author did not write — code on disk, a `*_test.go`, an edge in the running binary's transitive " +
		"import graph (fak itself runs it), a documented verb — so the only way up the ladder is to change " +
		"the real tree.\n\n")
	b.WriteString("**Immaturity is not a defect.** A capability honestly at `prototyped` is a complete v1 that " +
		"simply has not been matured yet — expected, and never counted against anyone. The one defect this " +
		"refuses is a **ladder-skip**: a capability that looks more mature than its evidence — concretely, one " +
		"fak relies on (dogfooded, a default surface, or benchmarked) yet leaves untested. That is the maturity " +
		"sibling of the product scorecard's verdict-overclaim and the readiness ladder's `READINESS_OVERCLAIM` " +
		"([#582](https://github.com/anthony-chaudhary/fak/issues/582) / grammar G1).\n\n")

	b.WriteString("## The lifecycle ladder\n\n")
	b.WriteString("| # | Rung | Reached when (evidence the author did not write) |\n")
	b.WriteString("|---|---|---|\n")
	b.WriteString("| 0 | `proposed` | a declared capability with no code on disk yet |\n")
	b.WriteString("| 1 | `prototyped` | a non-test `.go` file exists in the leaf — a complete v1 |\n")
	b.WriteString("| 2 | `tested` | the leaf carries a `*_test.go` (the QA rung) |\n")
	b.WriteString("| 3 | `dogfooded` | the leaf is on the running binary's transitive import graph — **fak itself runs it** |\n")
	b.WriteString("| 4 | `default` | the capability is a documented `fak` verb (`docs/cli-reference.md`) — the default surface |\n")
	b.WriteString("| · | `benchmarked` (badge) | a `func Benchmark*` in the leaf or a `BENCHMARK-AUTHORITY.md` row — the natural step after `default` |\n\n")

	b.WriteString("## Distribution\n\n")
	b.WriteString("| Rung | Capabilities |\n|---|---|\n")
	for _, r := range RungName {
		b.WriteString("| `" + r + "` | " + itoa(dist[r]) + " |\n")
	}
	b.WriteString("| `benchmarked` (badge) | " + anyStr(c["benchmarked"]) + " |\n")
	b.WriteString("\n")

	b.WriteString("## Next work — the agentic-culture backlog\n\n")
	b.WriteString("Each gap is a concrete, checkable next work item. `fak maturity next` is the queue an agent " +
		"(or the issue-dispatch loop) pulls from to advance the fleet one rung at a time. Ladder-skips first " +
		"(they are the real debt), then the least-mature capabilities (the most leverage).\n\n")
	if len(p.Backlog) == 0 {
		b.WriteString("_No next work — every declared capability is at the top of the ladder._\n\n")
	} else {
		b.WriteString("| | From → gap | Next work item | Witness |\n|---|---|---|---|\n")
		shown := 0
		for _, w := range p.Backlog {
			if shown >= 30 {
				b.WriteString("| | | _… and " + itoa(len(p.Backlog)-shown) + " more (run `fak maturity next`)_ | |\n")
				break
			}
			mark := ""
			if w.Skip {
				mark = "⚠"
			}
			b.WriteString("| " + mark + " | `" + w.FromRung.String() + " → " + w.Gap.String() + "` | " +
				mdEsc(w.Title) + " | " + mdEsc(w.Witness) + " |\n")
			shown++
		}
		b.WriteString("\n")
	}

	b.WriteString("## Run it\n\n```bash\n")
	b.WriteString("go run ./cmd/fak maturity              # the lifecycle scorecard\n")
	b.WriteString("go run ./cmd/fak maturity next         # the next-work backlog (ladder-skips first)\n")
	b.WriteString("go run ./cmd/fak maturity --markdown    # regenerate this doc\n")
	b.WriteString("go run ./cmd/fak maturity --json        # machine payload (control-pane / dispatch loop)\n")
	b.WriteString("go test ./internal/maturity/...        # prove the ladder + skip detection + next-work fold\n")
	b.WriteString("```\n\n")
	b.WriteString("**Next:** " + p.NextAction + "\n")
	return b.String()
}

// Compare proves movement against a pinned --json baseline: capabilities promoted,
// ladder-skips retired, and the index lift.
func Compare(current ScorecardPayload, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := anyInt(bc["maturity_debt"])
	cDebt := anyInt(current.Corpus["maturity_debt"])
	bScore := anyInt(bc["score"])
	cScore := anyInt(current.Corpus["score"])
	bDefault := anyInt(bc["at_default"])
	cDefault := anyInt(current.Corpus["at_default"])
	lines := []string{
		"maturity compare:",
		"  maturity_debt (ladder-skips): " + itoa(bDebt) + " -> " + itoa(cDebt) +
			"  (retired " + itoa(bDebt-cDebt) + ")",
		"  index: " + itoa(bScore) + " -> " + itoa(cScore) + "  grade " + anyStr(bc["grade"]) +
			" -> " + anyStr(current.Corpus["grade"]),
		"  at default rung: " + itoa(bDefault) + " -> " + itoa(cDefault),
	}
	switch {
	case bDebt > 0 && cDebt == 0:
		lines = append(lines, "  VERDICT: all ladder-skips retired")
	case cDefault > bDefault:
		lines = append(lines, "  VERDICT: matured ("+itoa(cDefault-bDefault)+" more capability(ies) at the default rung)")
	case cScore > bScore || cDebt < bDebt:
		lines = append(lines, "  VERDICT: improved (index "+itoa(bScore)+" -> "+itoa(cScore)+
			", skips "+itoa(bDebt)+" -> "+itoa(cDebt)+")")
	case cScore == bScore && cDebt == bDebt:
		lines = append(lines, "  VERDICT: flat")
	default:
		lines = append(lines, "  VERDICT: regressed")
	}
	return strings.Join(lines, "\n")
}

// ---- small render helpers ---------------------------------------------------

func benchmarkedSuffix(c map[string]any) string {
	b := anyInt(c["benchmarked"])
	if b == 0 {
		return ""
	}
	return " (" + itoa(b) + " carry the benchmarked badge)"
}

func bar(n, max int) string {
	const width = 24
	if max <= 0 {
		return strings.Repeat("·", width)
	}
	cells := int(float64(width)*float64(n)/float64(max) + 0.5)
	if cells < 0 {
		cells = 0
	}
	if cells > width {
		cells = width
	}
	if n > 0 && cells == 0 {
		cells = 1
	}
	return strings.Repeat("█", cells) + strings.Repeat("·", width-cells)
}

func pad(s string, w int) string {
	if len(s) >= w {
		return s
	}
	return s + strings.Repeat(" ", w-len(s))
}

func mdEsc(s string) string {
	return strings.ReplaceAll(s, "|", "\\|")
}

func anyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return itoa(x)
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func anyInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	default:
		return 0
	}
}
