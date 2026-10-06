package grafana

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestFleetJobScrapeTimeoutContract pins an explicit scrape_timeout on every
// exporter job whose fold is known to exceed the Prometheus 10s default.
//
// The defect this guards (observed 2026-09-30): `fak fleet metrics --serve`
// re-folds the durable session registry + the local CLI-invocation journal per
// scrape (measured 8-21s warm). With no `scrape_timeout` on the fak_fleet job the
// target inherited Prometheus's 10s default, so it read DOWN with
// `context deadline exceeded` while the exporter was actually up and a direct
// curl returned 200. The dashboard then looks healthy but has no data — the
// worst failure mode for an observability board.
//
// A job that needs more than the default MUST say so inline, and the value must
// cover the observed fold. The check is deliberately a floor, not equality, so
// raising the timeout further stays legal.
func TestFleetJobScrapeTimeoutContract(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(file), "prometheus.yml"))
	if err != nil {
		t.Fatalf("read prometheus.yml: %v", err)
	}
	cfg := string(raw)

	// Slow-fold jobs and the minimum timeout their fold requires. Every entry
	// names a job whose per-scrape fold can exceed the 10s default.
	minTimeout := map[string]int{
		"fak_fleet": 30, // registry + usage-log fold; measured 8-21s
		"fak_ops":   30, // ops plane fold; scrape_timeout 45s observed
	}

	for job, floor := range minTimeout {
		block := jobBlock(cfg, job)
		if block == "" {
			t.Fatalf("prometheus.yml has no %q job block", job)
		}
		got := scrapeTimeoutSeconds(block)
		if got < 0 {
			t.Errorf("job %q has no explicit scrape_timeout; it inherits the 10s Prometheus default and times out on a fold that exceeds it", job)
			continue
		}
		if got < floor {
			t.Errorf("job %q scrape_timeout=%ds is below the %ds floor its fold needs", job, got, floor)
		}
	}
}

// jobBlock returns the YAML lines of the scrape_configs entry named job, from
// the `- job_name: <job>` line to the next job or end of file.
func jobBlock(cfg, job string) string {
	lines := strings.Split(cfg, "\n")
	start := -1
	for i, l := range lines {
		if strings.TrimSpace(l) == "- job_name: "+job {
			start = i
			break
		}
	}
	if start < 0 {
		return ""
	}
	for j := start + 1; j < len(lines); j++ {
		if strings.HasPrefix(strings.TrimSpace(lines[j]), "- job_name: ") {
			return strings.Join(lines[start:j], "\n")
		}
	}
	return strings.Join(lines[start:], "\n")
}

var scrapeTimeoutRe = regexp.MustCompile(`(?m)^\s*scrape_timeout:\s*(\d+)s`)

// scrapeTimeoutSeconds returns the job block's scrape_timeout in seconds, or -1
// when none is set.
func scrapeTimeoutSeconds(block string) int {
	m := scrapeTimeoutRe.FindStringSubmatch(block)
	if m == nil {
		return -1
	}
	n := 0
	for _, r := range m[1] {
		n = n*10 + int(r-'0')
	}
	return n
}

// TestScrapeTimeoutWithinInterval enforces Prometheus's own config invariant:
// scrape_timeout MUST be <= scrape_interval, or the server rejects the whole
// file at reload/startup and silently keeps the previous config. That rejection
// is exactly how the fak_fleet timeout fix first failed to apply (observed
// 2026-09-30: `scrape timeout greater than scrape interval for scrape config
// with job name "fak_fleet"`), so it is pinned here rather than rediscovered at
// reload time.
func TestScrapeTimeoutWithinInterval(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(file), "prometheus.yml"))
	if err != nil {
		t.Fatalf("read prometheus.yml: %v", err)
	}
	cfg := string(raw)

	global := intervalFrom(cfg, 0)
	for _, job := range []string{"fak_fleet", "fak_ops", "fak_ops_workers", "fak_cachevalue", "fak_gateway"} {
		block := jobBlock(cfg, job)
		if block == "" {
			continue
		}
		timeout := scrapeTimeoutSeconds(block)
		if timeout < 0 {
			continue // inherits the global/interval default; the timeout contract test covers jobs that need more
		}
		interval := intervalFrom(block, global)
		if timeout > interval {
			t.Errorf("job %q scrape_timeout=%ds exceeds scrape_interval=%ds; Prometheus rejects the file and keeps the old config", job, timeout, interval)
		}
	}
}

var scrapeIntervalRe = regexp.MustCompile(`(?m)^\s*scrape_interval:\s*(\d+)s`)

// intervalFrom returns the scrape_interval in seconds from block, or fallback
// when the block does not set one. Pass the whole config as block to read the
// global value.
func intervalFrom(block string, fallback int) int {
	m := scrapeIntervalRe.FindStringSubmatch(block)
	if m == nil {
		return fallback
	}
	n := 0
	for _, r := range m[1] {
		n = n*10 + int(r-'0')
	}
	return n
}
