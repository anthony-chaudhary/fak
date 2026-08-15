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

// FreshnessEntry records the last committed change to a published marketing asset.
type FreshnessEntry struct {
	Path       string    `json:"path"`
	LastChange time.Time `json:"last_change"`
	AgeDays    int       `json:"age_days"`
}

// FreshnessReport is the deletion queue for time-sensitive Pages content.
type FreshnessReport struct {
	Schema         string           `json:"schema"`
	MaximumAgeDays int              `json:"maximum_age_days"`
	Checked        int              `json:"checked"`
	Stale          []FreshnessEntry `json:"stale"`
}

// AuditFreshness finds tracked files whose latest commit predates the maximum age.
// Git history, rather than filesystem mtimes, keeps the result reproducible in CI.
func AuditFreshness(root string, paths []string, maximumAge time.Duration, now time.Time) (FreshnessReport, error) {
	report := FreshnessReport{Schema: "fak-pages-freshness/1", MaximumAgeDays: int(maximumAge / (24 * time.Hour)), Stale: []FreshnessEntry{}}
	if maximumAge <= 0 {
		return report, fmt.Errorf("maximum age must be positive")
	}
	if len(paths) == 0 {
		return report, fmt.Errorf("at least one path is required")
	}
	filesOut, err := gitOutput(root, append([]string{"ls-files", "--"}, paths...)...)
	if err != nil {
		return report, err
	}
	files := strings.Fields(strings.ReplaceAll(filesOut, "\\", "/"))
	sort.Strings(files)
	cutoff := now.UTC().Add(-maximumAge)
	for _, path := range files {
		if _, statErr := os.Stat(filepath.Join(root, filepath.FromSlash(path))); statErr != nil {
			if os.IsNotExist(statErr) {
				continue
			}
			return report, fmt.Errorf("%s: %w", path, statErr)
		}
		stamp, err := gitOutput(root, "log", "-1", "--format=%ct", "--", path)
		if err != nil {
			return report, err
		}
		stamp = strings.TrimSpace(stamp)
		if stamp == "" {
			return report, fmt.Errorf("%s: no git history (checkout must use fetch-depth: 0)", path)
		}
		seconds, err := strconv.ParseInt(stamp, 10, 64)
		if err != nil {
			return report, fmt.Errorf("%s: invalid git timestamp %q", path, stamp)
		}
		changed := time.Unix(seconds, 0).UTC()
		report.Checked++
		if changed.Before(cutoff) {
			report.Stale = append(report.Stale, FreshnessEntry{Path: filepath.ToSlash(path), LastChange: changed, AgeDays: int(now.UTC().Sub(changed) / (24 * time.Hour))})
		}
	}
	if len(report.Stale) > 0 {
		names := make([]string, len(report.Stale))
		for i, entry := range report.Stale {
			names[i] = entry.Path
		}
		return report, fmt.Errorf("%d published marketing assets exceed %d days; delete or substantively refresh: %s", len(report.Stale), report.MaximumAgeDays, strings.Join(names, ", "))
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

// WriteFreshnessJSON emits the report even when stale entries make the audit fail.
func WriteFreshnessJSON(report FreshnessReport) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
