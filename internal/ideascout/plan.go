package ideascout

// Issue planning: rendering a surviving candidate into a triage-ready issue, and
// the score - dedup - threshold - cap pipeline that decides which ones get filed.

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

func RenderIssue(c Candidate, score int, reasons []string, topic Topic, today string) IssuePlan {
	rawTitle := strings.TrimSuffix(strings.TrimSpace(c.Title), ".")
	rawTitle = trimRunes(rawTitle, 100)
	title := "idea-scout: " + rawTitle
	summary := trimRunes(strings.TrimSpace(c.Summary), 700)
	var facts []string
	switch c.Source {
	case "arxiv":
		if authors := stringSliceFromExtra(c.Extra, "authors"); len(authors) > 0 {
			facts = append(facts, "**Authors:** "+strings.Join(authors, ", "))
		}
		if c.Published != "" {
			facts = append(facts, "**Submitted:** "+firstN(c.Published, 10))
		}
	case "hackernews", "reddit":
		if sub := stringFromExtra(c.Extra, "subreddit"); sub != "" {
			facts = append(facts, "**Subreddit:** r/"+sub)
		}
		if points := intFromExtra(c.Extra, "points"); points > 0 {
			facts = append(facts, fmt.Sprintf("**Points:** %d", points))
		}
		if comments := intFromExtra(c.Extra, "num_comments"); comments > 0 {
			facts = append(facts, fmt.Sprintf("**Comments:** %d", comments))
		}
		if disc := stringFromExtra(c.Extra, "discussion"); disc != "" {
			facts = append(facts, "**Discussion:** "+disc)
		}
		if c.Published != "" {
			facts = append(facts, "**Posted:** "+firstN(c.Published, 10))
		}
	default:
		if stars := intFromExtra(c.Extra, "stars"); stars > 0 {
			facts = append(facts, fmt.Sprintf("**Stars:** %d", stars))
		}
		if lang := stringFromExtra(c.Extra, "language"); lang != "" {
			facts = append(facts, "**Language:** "+lang)
		}
		if pushed := stringFromExtra(c.Extra, "pushed_at"); pushed != "" {
			facts = append(facts, "**Last push:** "+firstN(pushed, 10))
		}
	}
	why := strings.Join(reasons, "; ")
	if why == "" {
		why = "matched topic query"
	}
	body := "> Auto-filed by the daily **idea-scout** (`fak idea-scout`, " + today + "). A candidate RELATED idea found on " + sourceLabel(c.Source) + "; **needs human triage** - close as `wontfix`/`duplicate` if it is not worth pursuing.\n\n" +
		"**Source:** " + c.URL + "\n\n"
	if len(facts) > 0 {
		body += strings.Join(facts, "\n") + "\n\n"
	}
	body += fmt.Sprintf("**Why surfaced** (topic `%s`, score %d): %s\n\n", topic.Key, score, why) +
		"### Dispatchability\n" +
		"- dispatchability: `triage_only`\n" +
		"- reason: idea-scout candidates need human scope, lane, witness, and acceptance criteria before they become worker-ready leaves.\n\n"
	if summary != "" {
		body += "**Summary**\n\n" + summary + "\n\n"
	}
	body += "---\n" +
		"_Triage hint: is this a capability fak should adopt, a threat it should defend against, or prior art to cite? If none, close it._\n" +
		"<!-- idea-scout-source: " + c.SourceID + " -->"
	labels := []string{ScoutLabel, TriageLabel, TriageOnlyLabel, "research"}
	if topic.Area != "" {
		labels = append(labels, topic.Area)
	}
	return IssuePlan{Title: title, Body: body, Labels: labels, SourceID: c.SourceID, URL: c.URL, Score: score, Topic: topic.Key}
}

// PlanIssues runs score → dedup → threshold → CAP. It returns the issues to file,
// the per-rung counts, and `dropped`, which names the rung that stopped each
// individual source_id so an auditor can re-run a dry-run and check BY NAME that a
// known-already-triaged source is being caught, and by which rung.
func PlanIssues(candidates []Candidate, topicsByKey map[string]Topic, seen map[string]SeenRecord, stamped map[string]struct{}, titleSets []map[string]struct{}, bodiesJoined string, cfg Config, today string, now time.Time) ([]IssuePlan, map[string]int, []DroppedSource) {
	stats := map[string]int{"seen-cache": 0, "filed-stamp": 0, "issue-body": 0, "title-near": 0, "below-min": 0, "within-run-dup": 0}
	var scored []IssuePlan
	var dropped []DroppedSource
	runSeen := map[string]struct{}{}
	for _, cand := range candidates {
		if _, ok := runSeen[cand.SourceID]; ok {
			stats["within-run-dup"]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: "within-run-dup"})
			continue
		}
		runSeen[cand.SourceID] = struct{}{}
		topic, ok := topicsByKey[cand.Topic]
		if !ok {
			topic = Topic{Key: cand.Topic}
		}
		if rung := IsDuplicate(cand, seen, stamped, titleSets, bodiesJoined, cfg.DupJaccard); rung != "" {
			stats[rung]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: rung})
			continue
		}
		score, reasons := ScoreCandidate(cand, topic, cfg, now)
		if score < cfg.MinScore {
			stats["below-min"]++
			dropped = append(dropped, DroppedSource{SourceID: cand.SourceID, Rung: "below-min"})
			continue
		}
		scored = append(scored, RenderIssue(cand, score, reasons, topic, today))
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].SourceID < scored[j].SourceID
	})
	sort.Slice(dropped, func(i, j int) bool {
		if dropped[i].Rung != dropped[j].Rung {
			return dropped[i].Rung < dropped[j].Rung
		}
		return dropped[i].SourceID < dropped[j].SourceID
	})
	if cfg.MaxIssues >= 0 && len(scored) > cfg.MaxIssues {
		scored = scored[:cfg.MaxIssues]
	}
	return scored, stats, dropped
}
