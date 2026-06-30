package dispatchtick

import (
	"fmt"
	"strings"
)

const PromptSchema = "fleet-issue-worker-prompt/1"

type IssuePromptInput struct {
	Number     int
	Title      string
	Body       string
	Labels     []string
	Lane       string
	Workspace  string
	FetchError string
}

type IssuePromptRecord struct {
	Schema      string `json:"schema"`
	Issue       int    `json:"issue"`
	Lane        string `json:"lane"`
	Title       string `json:"title,omitempty"`
	FetchError  string `json:"fetch_error,omitempty"`
	Prompt      string `json:"prompt"`
	PromptChars int    `json:"prompt_chars"`
}

func BuildIssuePrompt(in IssuePromptInput) IssuePromptRecord {
	prompt := RenderIssuePrompt(in)
	return IssuePromptRecord{
		Schema:      PromptSchema,
		Issue:       in.Number,
		Lane:        in.Lane,
		Title:       strings.TrimSpace(in.Title),
		FetchError:  strings.TrimSpace(in.FetchError),
		Prompt:      prompt,
		PromptChars: len(prompt),
	}
}

func RenderIssuePrompt(in IssuePromptInput) string {
	title := strings.TrimSpace(in.Title)
	if title == "" {
		title = fmt.Sprintf("issue #%d", in.Number)
	}
	agentBrief := renderAgentIssueBrief(in.Body)
	body := strings.TrimSpace(in.Body)
	if len(body) > 1800 {
		body = body[:1800] + "\n...(truncated - read the full issue with `gh issue view`)"
	}
	if body == "" {
		body = "(no body - read the title and `gh issue view` for the full thread)"
	}
	labels := labelsLine(in.Labels)
	return fmt.Sprintf(`your goal: resolve GitHub issue #%[1]d (%[2]s) with the smallest correct change that genuinely closes it, then ship it on `+"`main`"+` citing `+"`#%[1]d`"+` in the commit subject - OR end with a final report naming the exact gate you could not reach and why. Do NOT fabricate a pass.

read first: run `+"`gh issue view %[1]d`"+` for the live issue, then orient with `+"`AGENTS.md`"+` (build/test/run + the hard rules) and `+"`llms.txt`"+` (the doc map). Then run `+"`python tools/memory_read.py`"+` for the committed fleet memory (lane quirks, known blockers, host gotchas) - a Claude worker gets this auto-injected, an opencode worker does NOT, so this read is how both backends start warm (it is a harmless no-op if the mirror is absent). This issue routed to the `+"`%[3]s`"+` lane (its file-tree). Labels: %[4]s.

%[7]s

issue body (verbatim, may be truncated - re-read live):
---
%[5]s
---

how to work it:
- Take the lane lease first if siblings may collide: `+"`dos arbitrate --workspace . --lane %[3]s --kind cluster`"+` (honor a REFUSE - pick nothing and stop; do not --force onto a held lane).
- Make the SMALLEST change that resolves the issue's actual ask. Prefer one leaf / one file. If the issue is a docs/observability/test ask, that is often a single file. If it is a large epic you cannot land whole, land the smallest honest increment and say in your report what remains.
- Run the gate yourself before claiming done: the lane's own test (`+"`go test ./... -count=1`"+` for the touched package, or the doc/lint check the issue names). A claim with no gate run is not done.

git laws (enforced below the agent - breaking them refuses your commit):
- Work on `+"`main`"+` ONLY. Never branch / new-worktree (the OFF_TRUNK guard refuses it).
- `+"`git commit -s -- <explicit paths>`"+` - sign-off (DCO), commit BY PATH only. NEVER `+"`git add -A`"+` (shared multi-session tree - a blanket add steals a sibling's in-flight files). Stage only the files you wrote.
- Reference `+"`#%[1]d`"+` in the subject AND end it with a `+"`(fak %[3]s)`"+` trailer, lead with a verb (e.g. `+"`fix(%[3]s): ... (#%[1]d) (fak %[3]s)`"+`; use add/fix/implement/test, NEVER a noun-led description). The `+"`#%[1]d`"+` binds your commit to the issue; the verb-led subject + `+"`(fak %[3]s)`"+` trailer is what makes `+"`dos commit-audit`"+` grade it `+"`diff-witnessed`"+` instead of ABSTAIN - and the closure auditor closes the issue ONLY on a witnessed commit. Miss either the `+"`#%[1]d`"+` or the trailer and your resolved issue never closes.
- No push / tag / force-push / history-rewrite / reset / clean / checkout-of-tracked-files. Just commit on main.

acceptance (your stop condition): a committed change on `+"`main`"+` whose subject cites `+"`#%[1]d`"+` and whose gate you actually ran is green - OR a final report that names the specific missing artifact/host capability and the smallest next step. Honesty over a green-looking lie: the repo keeps a witness ledger and a self-authored "done" is re-checked against git. If you discovered a durable fact worth keeping (a lane quirk, a host gotcha, a blocker), surface it explicitly in your final message so an operator or Claude peer can record it to the memory mirror - an opencode worker has no auto-memory write path of its own.

workspace: %[6]s. lane: %[3]s. issue: #%[1]d.
`, in.Number, title, strings.TrimSpace(in.Lane), labels, body, strings.TrimSpace(in.Workspace), agentBrief)
}

func labelsLine(labels []string) string {
	out := make([]string, 0, len(labels))
	for _, label := range labels {
		label = strings.TrimSpace(label)
		if label != "" {
			out = append(out, label)
		}
	}
	if len(out) == 0 {
		return "(none)"
	}
	return strings.Join(out, ", ")
}

func renderAgentIssueBrief(body string) string {
	sections := promptMarkdownSections(body)
	brief := []struct {
		Label string
		Names []string
	}{
		{"Work unit", []string{"work unit", "work-unit shape", "issue shape"}},
		{"Expected steps", []string{"expected steps", "step budget"}},
		{"Working spine", []string{"working spine"}},
		{"Assumptions", []string{"assumptions"}},
		{"Confusion risks", []string{"confusion risks", "known confusion", "unknowns"}},
		{"Coordination", []string{"coordination", "coordination notes", "handoff notes"}},
		{"Trigger", []string{"trigger", "creation trigger"}},
		{"Batch policy", []string{"batch policy", "noise control", "spam control"}},
		{"In scope", []string{"in scope"}},
		{"Out of scope", []string{"out of scope"}},
		{"Done condition", []string{"done condition"}},
		{"Witness", []string{"witness"}},
		{"Acceptance gate", []string{"acceptance gate"}},
	}
	lines := []string{}
	for _, row := range brief {
		value := firstPromptSection(sections, row.Names...)
		value = promptBriefValue(value)
		if value != "" {
			lines = append(lines, fmt.Sprintf("- %s: %s", row.Label, value))
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return "agent issue brief (parsed from standard sections):\n" + strings.Join(lines, "\n") + "\n"
}

func promptMarkdownSections(body string) map[string]string {
	out := map[string]string{}
	current := ""
	var buf []string
	flush := func() {
		if current != "" {
			out[current] = strings.TrimSpace(strings.Join(buf, "\n"))
		}
	}
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(raw)
		if heading, ok := promptHeading(line); ok {
			flush()
			current = heading
			buf = nil
			continue
		}
		if current != "" {
			buf = append(buf, raw)
		}
	}
	flush()
	return out
}

func promptHeading(line string) (string, bool) {
	if !strings.HasPrefix(line, "#") {
		return "", false
	}
	i := 0
	for i < len(line) && line[i] == '#' {
		i++
	}
	if i == 0 || i > 6 || i >= len(line) || line[i] != ' ' {
		return "", false
	}
	return normalizePromptHeading(line[i+1:]), true
}

func normalizePromptHeading(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "`*_:# ")
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

func firstPromptSection(sections map[string]string, names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(sections[normalizePromptHeading(name)]); value != "" {
			return value
		}
	}
	return ""
}

func promptBriefValue(section string) string {
	parts := []string{}
	for _, raw := range strings.Split(section, "\n") {
		line := strings.TrimSpace(raw)
		line = strings.TrimPrefix(line, "- ")
		line = strings.TrimPrefix(line, "* ")
		line = strings.TrimSpace(line)
		if line == "" || promptPlaceholder(line) {
			continue
		}
		parts = append(parts, line)
	}
	if len(parts) == 0 {
		return ""
	}
	value := strings.Join(parts, " / ")
	if len(value) > 260 {
		value = value[:260] + "..."
	}
	return value
}

func promptPlaceholder(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "not specified", "not specified.", "none", "none.", "none named", "none named.", "no special coordination", "no special coordination beyond the lane lease.":
		return true
	default:
		return false
	}
}
