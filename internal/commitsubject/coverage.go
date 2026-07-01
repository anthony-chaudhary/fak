// Package commitsubject reports witness-gradeable commit subject coverage.
package commitsubject

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	Schema      = "fleet-commit-subject-coverage/1"
	DefaultLast = 50
)

var versionRE = regexp.MustCompile(`^v\d+\.\d+\.\d+`)

var exemptPrefixes = []string{"Merge ", "Revert ", "fixup! ", "squash! ", "amend! "}

type AbstainSubject struct {
	Subject string `json:"subject"`
	Reason  string `json:"reason"`
}

type Coverage struct {
	Total           int              `json:"total"`
	Gradeable       int              `json:"gradeable"`
	Abstain         int              `json:"abstain"`
	Coverage        *float64         `json:"coverage"`
	AbstainSubjects []AbstainSubject `json:"abstain_subjects"`
}

type Payload struct {
	Schema          string           `json:"schema"`
	OK              bool             `json:"ok"`
	Verdict         string           `json:"verdict"`
	Reason          string           `json:"reason"`
	Workspace       string           `json:"workspace"`
	CoveragePct     *float64         `json:"coverage_pct"`
	MinCoverage     *float64         `json:"min_coverage"`
	Total           int              `json:"total"`
	Gradeable       int              `json:"gradeable"`
	Abstain         int              `json:"abstain"`
	Coverage        *float64         `json:"coverage"`
	AbstainSubjects []AbstainSubject `json:"abstain_subjects"`
}

// FirstLine returns the first non-empty, non-comment line from a commit message.
func FirstLine(text string) string {
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if s != "" && !strings.HasPrefix(s, "#") {
			return s
		}
	}
	return ""
}

// IsExempt reports merge/revert/fixup/squash/version-bump subjects that make no gradeable claim.
func IsExempt(subject string) bool {
	for _, p := range exemptPrefixes {
		if strings.HasPrefix(subject, p) {
			return true
		}
	}
	return versionRE.MatchString(subject)
}

func Fold(subjects []string) Coverage {
	total, gradeable := 0, 0
	var abstains []AbstainSubject
	for _, raw := range subjects {
		s := FirstLine(raw)
		if s == "" || IsExempt(s) {
			continue
		}
		total++
		ok, why := hooks.CommitMsgVerdict(s)
		if ok {
			gradeable++
		} else {
			abstains = append(abstains, AbstainSubject{Subject: s, Reason: why})
		}
	}
	return Coverage{
		Total:           total,
		Gradeable:       gradeable,
		Abstain:         len(abstains),
		Coverage:        fracPtr(gradeable, total),
		AbstainSubjects: abstains,
	}
}

type SubjectFetcher func(root string, last int) []string

func RecentSubjects(root string, last int) []string {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "log", fmt.Sprintf("-%d", last), "--no-merges", "--pretty=format:%s")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var subjects []string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) != "" {
			subjects = append(subjects, line)
		}
	}
	return subjects
}

func BuildPayload(root string, cov Coverage, minCoverage *float64) Payload {
	var pct *float64
	if cov.Coverage != nil {
		v := round1(*cov.Coverage * 100)
		pct = &v
	}
	ok := true
	verdict := "OK"
	reason := ""
	if pct == nil {
		verdict = "NO_GRADEABLE_COMMITS"
		reason = "no witness-gradeable ship commits in the window (all merges/reverts/bumps?)"
	} else if minCoverage != nil && *pct < *minCoverage {
		ok = false
		verdict = "BELOW_FLOOR"
		reason = fmt.Sprintf("%.1f%% of recent ship subjects are witness-gradeable, below the %.1f%% floor - %d ABSTAIN-prone subject(s) (noun-led / non-conventional) the witness cannot grade",
			*pct, *minCoverage, cov.Abstain)
	} else {
		reason = fmt.Sprintf("%.1f%% of recent ship subjects are witness-gradeable (%d/%d)", *pct, cov.Gradeable, cov.Total)
	}
	return Payload{
		Schema:          Schema,
		OK:              ok,
		Verdict:         verdict,
		Reason:          reason,
		Workspace:       root,
		CoveragePct:     pct,
		MinCoverage:     minCoverage,
		Total:           cov.Total,
		Gradeable:       cov.Gradeable,
		Abstain:         cov.Abstain,
		Coverage:        cov.Coverage,
		AbstainSubjects: cov.AbstainSubjects,
	}
}

func Collect(root string, last int, minCoverage *float64, fetcher SubjectFetcher) Payload {
	if last < 1 {
		last = 1
	}
	if fetcher == nil {
		fetcher = RecentSubjects
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	return BuildPayload(root, Fold(fetcher(root, last)), minCoverage)
}

func Render(p Payload) string {
	lines := []string{
		fmt.Sprintf("commit-subject coverage: %s (%s)", p.Verdict, boolWord(p.OK)),
		"  " + p.Reason,
		fmt.Sprintf("  gradeable=%d/%d  coverage=%s%%  abstain=%d", p.Gradeable, p.Total, pctString(p.CoveragePct), p.Abstain),
	}
	if len(p.AbstainSubjects) > 0 {
		lines = append(lines, "  ABSTAIN-prone (fix: lead the description with a recognized verb):")
		for i, a := range p.AbstainSubjects {
			if i >= 12 {
				break
			}
			subj := a.Subject
			if len(subj) > 78 {
				subj = subj[:78]
			}
			lines = append(lines, "    - "+subj)
		}
	}
	return strings.Join(lines, "\n")
}

func fracPtr(n, d int) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}

func round1(v float64) float64 {
	if v >= 0 {
		return float64(int(v*10+0.5)) / 10
	}
	return float64(int(v*10-0.5)) / 10
}

func pctString(v *float64) string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%.1f", *v)
}

func boolWord(ok bool) string {
	if ok {
		return "ok"
	}
	return "ACTION"
}
