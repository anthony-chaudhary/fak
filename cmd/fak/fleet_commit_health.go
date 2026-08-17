package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
)

func writeFleetCommitThroughputMetrics(w *promWriter, metric fleetmetrics.CommitThroughput, activeWorkers int) {
	health := metric.Health(activeWorkers)
	w.gauge("fak_fleet_commits_per_10m", "Real file-changing commits landed on HEAD's first-parent history in the current 10-minute window.", float64(metric.Current))
	w.gauge("fak_fleet_commits_previous_10m", "Real file-changing commits landed in the preceding 10-minute window; used to distinguish a fresh stall from a sustained block.", float64(metric.Previous))
	w.gauge("fak_fleet_commit_throughput_healthy", "1 when commit throughput is measurable and positive for an active fleet; idle fleets are neutral.", boolGauge(health.Healthy))
	w.gauge("fak_fleet_commit_throughput_measured", "1 when repository first-parent commit history was read successfully.", boolGauge(metric.Measured))
	latest := float64(0)
	if !metric.LatestCommitAt.IsZero() {
		latest = float64(metric.LatestCommitAt.Unix())
	}
	w.gauge("fak_fleet_latest_commit_unixtime", "Unix time of the latest real commit observed in the current or previous 10-minute window; 0 when neither window contains one.", latest)
	for _, state := range []string{"healthy", "stalled", "blocked", "unknown", "idle"} {
		w.gauge("fak_fleet_commit_throughput_state", "Top-level commit throughput health state for the active fleet.", boolGauge(health.State == state), "state", state)
	}
}

// formatCommitThroughput is the compact contract shared by text surfaces.
func formatCommitThroughput(metric fleetmetrics.CommitThroughput, activeWorkers int, now time.Time) string {
	health := metric.Health(activeWorkers)
	age := "none"
	if !metric.LatestCommitAt.IsZero() {
		age = now.Sub(metric.LatestCommitAt).Round(time.Second).String()
	}
	return "commits/10m=" + strconv.Itoa(metric.Current) + " previous=" + strconv.Itoa(metric.Previous) + " state=" + health.State + " latest_age=" + age
}

type fleetCommitHealthReport struct {
	Schema        string                        `json:"schema"`
	ActiveWorkers int                           `json:"active_workers"`
	Throughput    fleetmetrics.CommitThroughput `json:"throughput"`
	Health        fleetmetrics.CommitHealth     `json:"health"`
}

var fleetCommitHealthNow = time.Now
var fleetCommitHealthMeasure = fleetmetrics.MeasureCommitThroughput

func runFleetCommitHealth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak fleet health", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the machine-readable health report")
	registry := fs.String("registry", defaultSessionRegistryPath(), "durable session registry used to count active workers")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	now := fleetCommitHealthNow()
	source := fleetMetricsSources{registryPath: *registry, staleWindow: defaultSessionStaleWindow, maxSessions: defaultFleetMetricsMaxSessions, stderr: stderr}
	inv, _, readable := source.liveInventory(now)
	metric := fleetCommitHealthMeasure(repoRoot(), now)
	if !readable {
		metric.Measured = false
		metric.Error = "durable session registry unreadable"
	}
	report := fleetCommitHealthReport{Schema: "fak-fleet-commit-health/1", ActiveWorkers: inv.Count, Throughput: metric, Health: metric.Health(inv.Count)}
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak fleet health: encode: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprintln(stdout, formatCommitThroughput(metric, inv.Count, now))
		fmt.Fprintf(stdout, "active_workers=%d healthy=%t reason=%s\n", inv.Count, report.Health.Healthy, report.Health.Reason)
		if report.Health.NextAction != "" {
			fmt.Fprintln(stdout, "next="+report.Health.NextAction)
		}
	}
	if !report.Health.Healthy {
		return 3
	}
	return 0
}
