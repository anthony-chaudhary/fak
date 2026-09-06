package steerpr

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

var (
	benchUnitsSink       []Unit
	benchCommitsSink     []Commit
	benchLandingSink     Landing
	benchIssuesSink      []string
	benchWaveIndexSink   map[string]string
	benchTitleSink       string
	benchBandSink        Band
	benchBoolSink        bool
	benchActivePauseSink map[string]Pause
	benchExpectationSink Expectation
)

func makeBenchLog(n int) string {
	leaves := []string{"gateway", "adjudicator", "ctxmmu", "steerpr", "dispatchtick", "vdso", "model", "engine"}
	types := []string{"feat", "fix", "docs", "test", "refactor"}
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sha := fmt.Sprintf("%040x", i+1)
		leaf := leaves[i%len(leaves)]
		typ := types[i%len(types)]
		var subject string
		if i%9 == 0 {
			subject = fmt.Sprintf("%s: unassigned work resolving #%d", typ, 100+i)
		} else {
			subject = fmt.Sprintf("%s(%s): commit %d resolves #%d (fak %s)", typ, leaf, i, 100+i, leaf)
		}
		body := fmt.Sprintf("Body detail for commit %d mentioning #%d and #%d.\nCo-authored-by: bot <bot@example.com>", i, 200+i, 300+i)
		files := []string{
			fmt.Sprintf("internal/%s/file_%d.go", leaf, i%5),
			fmt.Sprintf("internal/%s/file_%d_test.go", leaf, i%5),
		}
		sb.WriteString("\x1e" + sha + "\x1f" + subject + "\x1f" + body + "\x1f" + strings.Join(files, "\n"))
	}
	return sb.String()
}

func BenchmarkParseLog(b *testing.B) {
	raw := makeBenchLog(100)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		commits := ParseLog(raw)
		if len(commits) == 0 {
			b.Fatal("unexpected empty commits")
		}
		benchCommitsSink = commits
	}
}

func BenchmarkFoldUnits(b *testing.B) {
	raw := makeBenchLog(100)
	commits := ParseLog(raw)
	for i := range commits {
		switch i % 3 {
		case 0:
			commits[i].Verdict = VerdictWitnessed
		case 1:
			commits[i].Verdict = VerdictUnwitnessed
		default:
			commits[i].Verdict = VerdictAbstain
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		units, unstamped := FoldUnits(commits)
		if len(units) == 0 {
			b.Fatal("unexpected empty units")
		}
		benchUnitsSink = units
		benchCommitsSink = unstamped
	}
}

func BenchmarkFoldUnitsByWave(b *testing.B) {
	raw := makeBenchLog(100)
	commits := ParseLog(raw)
	bindings := make([]WaveBinding, 5)
	for w := 0; w < 5; w++ {
		bindings[w].Index = w
		for j := 0; j < 20; j++ {
			bindings[w].Issues = append(bindings[w].Issues, fmt.Sprintf("#%d", 100+w*20+j))
		}
	}
	waveOf := WaveIndex(bindings)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		units, unstamped := FoldUnitsByWave(commits, waveOf)
		if len(units) == 0 {
			b.Fatal("unexpected empty units")
		}
		benchUnitsSink = units
		benchCommitsSink = unstamped
	}
}

func BenchmarkSortWorstFirst(b *testing.B) {
	raw := makeBenchLog(100)
	commits := ParseLog(raw)
	for i := range commits {
		if i%2 == 0 {
			commits[i].Verdict = VerdictWitnessed
		} else {
			commits[i].Verdict = VerdictUnwitnessed
		}
	}
	templateUnits, _ := FoldUnits(commits)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		units := make([]Unit, len(templateUnits))
		copy(units, templateUnits)
		SortWorstFirst(units)
		benchUnitsSink = units
	}
}

func BenchmarkAssignLanded(b *testing.B) {
	sha := "297938d8dcea096d1b443ab92b8077015b76fb51"
	subject := "feat(gateway): implement adaptive rate limiter for #402 (fak gateway)"
	body := "Closes #402 and mentions #403.\nSigned-off-by: Dev <dev@example.com>"
	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		landing, err := AssignLanded(sha, subject, body, now)
		if err != nil {
			b.Fatal(err)
		}
		benchLandingSink = landing
	}
}

func BenchmarkIssues(b *testing.B) {
	text := "Resolves #101, #102, #103, mentions #104, related to #105 and #106"
	exclude := []string{"#101", "#105"}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		issues := Issues(text, exclude)
		if len(issues) != 4 {
			b.Fatalf("got %d issues, want 4", len(issues))
		}
		benchIssuesSink = issues
	}
}

func BenchmarkWaveIndex(b *testing.B) {
	bindings := make([]WaveBinding, 10)
	for w := 0; w < 10; w++ {
		bindings[w].Index = w
		for j := 0; j < 15; j++ {
			bindings[w].Issues = append(bindings[w].Issues, fmt.Sprintf("#%d", w*15+j))
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		idx := WaveIndex(bindings)
		if len(idx) == 0 {
			b.Fatal("unexpected empty wave index")
		}
		benchWaveIndexSink = idx
	}
}

func BenchmarkUnitTitle(b *testing.B) {
	unit := Unit{
		Leaf:    "gateway",
		Commits: make([]Commit, 12),
		Types: map[string]int{
			"feat": 6,
			"fix":  4,
			"docs": 2,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		title := UnitTitle(unit)
		if title == "" {
			b.Fatal("unexpected empty unit title")
		}
		benchTitleSink = title
	}
}

func BenchmarkFoldBand(b *testing.B) {
	commits := make([]Commit, 60)
	for i := 0; i < 60; i++ {
		commits[i].SHA = fmt.Sprintf("%040x", i+1)
		switch {
		case i < 40:
			commits[i].Verdict = VerdictWitnessed
		case i < 55:
			commits[i].Verdict = VerdictAbstain
		default:
			commits[i].Verdict = VerdictUnwitnessed
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		band := FoldBand(commits)
		if band != BandResidual {
			b.Fatalf("got %v, want %v", band, BandResidual)
		}
		benchBandSink = band
	}
}

func BenchmarkAckCovers(b *testing.B) {
	shas := make([]string, 20)
	for i := 0; i < 20; i++ {
		shas[i] = fmt.Sprintf("%040x", i+1)
	}
	ack, err := NewAck("gateway", "operator-jane", shas, "verified on dev", time.Now())
	if err != nil {
		b.Fatalf("NewAck failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		covers := ack.Covers(shas)
		if !covers {
			b.Fatal("expected ack to cover shas")
		}
		benchBoolSink = covers
	}
}

func BenchmarkActivePauses(b *testing.B) {
	pauses := make([]Pause, 50)
	now := time.Now()
	leaves := []string{"gateway", "adjudicator", "ctxmmu", "steerpr", "dispatchtick"}
	for i := 0; i < 50; i++ {
		leaf := leaves[i%len(leaves)]
		action := PauseActionPause
		if i%4 == 0 {
			action = PauseActionResume
		}
		pauses[i] = Pause{
			Schema: PauseSchema,
			Action: action,
			Leaf:   leaf,
			By:     "operator",
			At:     now.Add(time.Duration(i) * time.Minute).Format(time.RFC3339),
			Issue:  fmt.Sprintf("#%d", 200+i),
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		active := ActivePauses(pauses)
		benchActivePauseSink = active
	}
}

func BenchmarkDeriveExpected(b *testing.B) {
	issues := make([]IntentIssue, 100)
	marker := FanoutMarker("gateway")
	for i := 0; i < 100; i++ {
		num := 500 + i
		body := fmt.Sprintf("Issue description for task %d", num)
		if i == 0 {
			body = "Spine issue for gateway rollout"
		} else if i%3 == 0 {
			body = fmt.Sprintf("Follow-on subtask with marker: %s", marker)
		}
		issues[i] = IntentIssue{
			Number: num,
			Body:   body,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		exp, ok := DeriveExpected("gateway", "#500", issues)
		if !ok || exp.Total == 0 {
			b.Fatal("expected successful expectation derivation")
		}
		benchExpectationSink = exp
	}
}
