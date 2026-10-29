package treedoctor

import (
	"fmt"
	"os"
	"path/filepath"
)

// GoCachePlacement is a point-in-time diagnostic, not resource admission. The
// caller must pass the exact GOCACHE and filesystem budget used by the build.
// It does not observe later mount changes, other writers, or quota boundaries.
type GoCachePlacement struct {
	Schema                 string `json:"schema"`
	CachePath              string `json:"cache_path"`
	FilesystemPath         string `json:"filesystem_path"`
	FilesystemName         string `json:"filesystem_name,omitempty"`
	PlanSHA256             string `json:"plan_sha256,omitempty"`
	SameFilesystem         bool   `json:"same_filesystem"`
	CacheGrowthBudgetBytes uint64 `json:"filesystem_sampled_budget_bytes"`
	Verdict                string `json:"verdict"`
	Note                   string `json:"note"`
}

// CheckGoCachePlacement detects the concrete contradiction between a cache that
// may grow and a zero-growth plan for the filesystem containing it. A positive
// result is not admission: the caller must still check every other budget,
// retained allocation, ownership, live pressure, and exact child environment.
func CheckGoCachePlacement(cachePath, filesystemPath string, growthBudget uint64) (GoCachePlacement, error) {
	report := GoCachePlacement{Schema: "fak-go-cache-placement/1", CacheGrowthBudgetBytes: growthBudget,
		Note: "Point-in-time diagnostic only; not admission, ownership, or proof of the child environment. Recheck before use."}
	cache, err := placementDirectory(cachePath)
	if err != nil {
		return report, fmt.Errorf("cache: %w", err)
	}
	filesystem, err := placementDirectory(filesystemPath)
	if err != nil {
		return report, fmt.Errorf("filesystem: %w", err)
	}
	report.CachePath, report.FilesystemPath = cache, filesystem
	same, err := samePlacementFilesystem(cache, filesystem)
	if err != nil {
		return report, err
	}
	report.SameFilesystem = same
	switch {
	case !same:
		report.Verdict = "different_filesystem_requires_own_budget"
	case growthBudget == 0:
		report.Verdict = "zero_growth_conflict"
	default:
		report.Verdict = "positive_growth_declared_not_admission"
	}
	return report, nil
}

func placementDirectory(path string) (string, error) {
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("require an explicit absolute directory")
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("not a directory: %s", resolved)
	}
	return resolved, nil
}
