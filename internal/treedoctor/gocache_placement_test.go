package treedoctor

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1s
func TestGoCachePlacementZeroGrowthAndSymlink(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("filesystem identity is Linux-only")
	}
	dir := t.TempDir()
	cache := filepath.Join(dir, "cache")
	if err := os.Mkdir(cache, 0700); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(dir, "alias")
	if err := os.Symlink(cache, alias); err != nil {
		t.Fatal(err)
	}
	got, err := CheckGoCachePlacement(alias, dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !got.SameFilesystem || got.CachePath != cache || got.Verdict != "zero_growth_conflict" {
		t.Fatalf("zero-budget cache placement = %+v", got)
	}
	got, err = CheckGoCachePlacement(alias, dir, 1)
	if err != nil || got.Verdict != "positive_growth_declared_not_admission" {
		t.Fatalf("positive declaration = %+v, %v", got, err)
	}
	if _, err := CheckGoCachePlacement("relative", dir, 0); err == nil {
		t.Fatal("relative cache path must fail")
	}
	if _, err := CheckGoCachePlacement(filepath.Join(dir, "missing"), dir, 0); err == nil {
		t.Fatal("missing cache must fail")
	}
}

// fak-test:runtime fast est=1s
func TestGoCachePlanBindsExactBudgetAndRejectsAmbiguity(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("filesystem identity is Linux-only")
	}
	dir := t.TempDir()
	plan := filepath.Join(dir, "plan.json")
	write := func(body string) {
		t.Helper()
		if err := os.WriteFile(plan, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	body := fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"workspace","path":%q,"sampled_budget_bytes":0}]}`, dir, dir)
	write(body)
	got, err := CheckGoCachePlan(plan)
	if err != nil || got.Verdict != "zero_growth_conflict" || got.FilesystemName != "workspace" || len(got.PlanSHA256) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST PlanSHA256 is a sha256 hex digest; 64 is its fixed width
		t.Fatalf("zero-growth plan = %+v, %v", got, err)
	}
	write(fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"workspace","path":%q,"sampled_budget_bytes":3}]}`, dir, dir))
	got, err = CheckGoCachePlan(plan)
	if err != nil || got.Verdict != "positive_growth_declared_not_admission" || got.PlanSHA256 == "" {
		t.Fatalf("positive plan = %+v, %v", got, err)
	}
	write(fmt.Sprintf(`{"environment":{"GOCACHE":%q,"GOCACHE":%q},"filesystems":[{"name":"workspace","path":%q,"sampled_budget_bytes":0}]}`, dir, dir, dir))
	if _, err := CheckGoCachePlan(plan); err == nil {
		t.Fatal("duplicate key must fail")
	}
	write(fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"workspace","path":%q}]}`, dir, dir))
	if _, err := CheckGoCachePlan(plan); err == nil {
		t.Fatal("missing budget must fail")
	}
	write(fmt.Sprintf(`{"environment":{"GOCACHE":"off"},"filesystems":[{"name":"workspace","path":%q,"sampled_budget_bytes":1}]}`, dir))
	if _, err := CheckGoCachePlan(plan); err == nil {
		t.Fatal("GOCACHE=off must not be treated as a directory")
	}
	write(fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"one","path":%q,"sampled_budget_bytes":1},{"name":"two","path":%q,"sampled_budget_bytes":1}]}`, dir, dir, dir))
	if _, err := CheckGoCachePlan(plan); err == nil {
		t.Fatal("two matching filesystem records must fail")
	}
	if separate, err := CheckGoCachePlacement(dir, "/proc", 1); err == nil && !separate.SameFilesystem {
		write(fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"other","path":"/proc","sampled_budget_bytes":1}]}`, dir))
		if _, err := CheckGoCachePlan(plan); err == nil {
			t.Fatal("unmatched cache filesystem must fail")
		}
	}
	badPlans := map[string]string{
		"duplicate names":        fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"x","path":%q,"sampled_budget_bytes":1},{"name":"x","path":%q,"sampled_budget_bytes":1}]}`, dir, dir, dir),
		"external cache program": fmt.Sprintf(`{"environment":{"GOCACHE":%q,"GOCACHEPROG":"helper"},"filesystems":[{"name":"x","path":%q,"sampled_budget_bytes":1}]}`, dir, dir),
		"null budget":            fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"x","path":%q,"sampled_budget_bytes":null}]}`, dir, dir),
		"negative budget":        fmt.Sprintf(`{"environment":{"GOCACHE":%q},"filesystems":[{"name":"x","path":%q,"sampled_budget_bytes":-1}]}`, dir, dir),
		"trailing JSON":          body + `{}`,
		"escaped duplicate":      fmt.Sprintf(`{"environment":{"GOCACHE":%q,"\u0047OCACHE":%q},"filesystems":[{"name":"x","path":%q,"sampled_budget_bytes":1}]}`, dir, dir, dir),
	}
	for name, input := range badPlans {
		t.Run(name, func(t *testing.T) {
			write(input)
			if _, err := CheckGoCachePlan(plan); err == nil {
				t.Fatal("ambiguous or invalid plan must fail")
			}
		})
	}
	if _, err := CheckGoCachePlan(dir); err == nil {
		t.Fatal("non-regular plan must fail")
	}
}

// fak-test:runtime fast est=1s
func TestGoCachePlacementUnsupportedPlatform(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("Linux supports filesystem identity")
	}
	dir := t.TempDir()
	if _, err := CheckGoCachePlacement(dir, dir, 1); err == nil {
		t.Fatal("unsupported platform must not report success")
	}
}
