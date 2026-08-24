package pagespublish

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const FreshnessTargetsSchema = "fak-pages-freshness-targets/1"

type FreshnessTarget struct {
	Path            string `json:"path"`
	Class           string `json:"class"`
	ReviewAfterDays int    `json:"review_after_days,omitempty"`
	Check           string `json:"check"`
}

type FreshnessTargets struct {
	Schema  string            `json:"schema"`
	Roots   []string          `json:"roots"`
	Targets []FreshnessTarget `json:"targets"`
}

// FreshnessEntry records the committed age and review contract for one overdue asset.
type FreshnessEntry struct {
	Path            string    `json:"path"`
	LastChange      time.Time `json:"last_change"`
	AgeDays         int       `json:"age_days"`
	ReviewAfterDays int       `json:"review_after_days"`
	Check           string    `json:"check"`
}

// FreshnessReport is the review queue for explicitly time-sensitive Pages content.
type FreshnessReport struct {
	Schema  string           `json:"schema"`
	Checked int              `json:"checked"`
	Durable int              `json:"durable"`
	Due     []FreshnessEntry `json:"due"`
}

func LoadFreshnessTargets(path string) (FreshnessTargets, error) {
	var targets FreshnessTargets
	b, err := os.ReadFile(path)
	if err != nil {
		return targets, err
	}
	if err := json.Unmarshal(b, &targets); err != nil {
		return targets, fmt.Errorf("decode freshness targets: %w", err)
	}
	return targets, nil
}

// AuditFreshness checks only declared review targets. Durable assets are inventoried but
// never expire merely because their content remains stable.
func AuditFreshness(root string, targets FreshnessTargets, now time.Time) (FreshnessReport, error) {
	shallow, err := gitOutput(root, "rev-parse", "--is-shallow-repository")
	if err != nil {
		return FreshnessReport{}, err
	}
	if strings.TrimSpace(shallow) == "true" {
		return FreshnessReport{}, fmt.Errorf("freshness audit requires full git history (checkout must use fetch-depth: 0)")
	}
	report := FreshnessReport{Schema: "fak-pages-freshness/2", Due: []FreshnessEntry{}}
	if targets.Schema != FreshnessTargetsSchema {
		return report, fmt.Errorf("freshness targets schema %q, want %q", targets.Schema, FreshnessTargetsSchema)
	}
	trackedOut, err := gitOutput(root, append([]string{"ls-files", "--"}, targets.Roots...)...)
	if err != nil {
		return report, err
	}
	tracked := map[string]bool{}
	for _, path := range strings.Fields(strings.ReplaceAll(trackedOut, "\\", "/")) {
		tracked[path] = true
	}
	declared := map[string]bool{}
	for _, target := range targets.Targets {
		target.Path = filepath.ToSlash(strings.TrimSpace(target.Path))
		if target.Path == "" || declared[target.Path] {
			return report, fmt.Errorf("freshness target path is empty or duplicated: %q", target.Path)
		}
		declared[target.Path] = true
		if !tracked[target.Path] {
			return report, fmt.Errorf("freshness target is not tracked under a configured root: %s", target.Path)
		}
		if strings.TrimSpace(target.Check) == "" {
			return report, fmt.Errorf("%s: freshness target needs a concrete check", target.Path)
		}
		switch target.Class {
		case "durable":
			if target.ReviewAfterDays != 0 {
				return report, fmt.Errorf("%s: durable target must not set review_after_days", target.Path)
			}
			report.Durable++
			continue
		case "review":
			if target.ReviewAfterDays <= 0 {
				return report, fmt.Errorf("%s: review target needs positive review_after_days", target.Path)
			}
		default:
			return report, fmt.Errorf("%s: class must be durable or review", target.Path)
		}
		stamp, err := gitOutput(root, "log", "-1", "--format=%ct", "--", target.Path)
		if err != nil {
			return report, err
		}
		stamp = strings.TrimSpace(stamp)
		if stamp == "" {
			return report, fmt.Errorf("%s: no git history (checkout must use fetch-depth: 0)", target.Path)
		}
		seconds, err := strconv.ParseInt(stamp, 10, 64)
		if err != nil {
			return report, fmt.Errorf("%s: invalid git timestamp %q", target.Path, stamp)
		}
		changed := time.Unix(seconds, 0).UTC()
		report.Checked++
		if changed.Before(now.UTC().Add(-time.Duration(target.ReviewAfterDays) * 24 * time.Hour)) {
			report.Due = append(report.Due, FreshnessEntry{Path: target.Path, LastChange: changed, AgeDays: int(now.UTC().Sub(changed) / (24 * time.Hour)), ReviewAfterDays: target.ReviewAfterDays, Check: target.Check})
		}
	}
	var missing []string
	for path := range tracked {
		if !declared[path] {
			missing = append(missing, path)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		return report, fmt.Errorf("tracked freshness roots contain unclassified assets: %s", strings.Join(missing, ", "))
	}
	sort.Slice(report.Due, func(i, j int) bool { return report.Due[i].Path < report.Due[j].Path })
	if len(report.Due) > 0 {
		names := make([]string, len(report.Due))
		for i, entry := range report.Due {
			names[i] = entry.Path
		}
		return report, fmt.Errorf("%d published assets need freshness review; verify the declared check, then update, archive, or retain with a new witnessed review commit: %s", len(report.Due), strings.Join(names, ", "))
	}
	return report, nil
}

func gitOutput(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	configureDispatchHelperCommand(cmd)
	cmd.Dir = root
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

func WriteFreshnessJSON(report FreshnessReport) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
