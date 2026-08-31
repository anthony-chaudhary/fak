package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/looptrigger"
)

func runSuperloopTrigger(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop trigger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	loop := fs.String("loop", "super-loop", "loop identifier")
	observed := fs.String("observed-at", "", "RFC3339 observation time (default now)")
	eligible := fs.Int("eligible", 0, "eligible demand units")
	oldest := fs.Duration("oldest-age", 0, "age of oldest eligible demand")
	sourceAge := fs.Duration("source-age", 0, "age of trigger evidence")
	maxSourceAge := fs.Duration("max-source-age", 5*time.Minute, "maximum useful evidence age")
	overlap := fs.Int("overlap", 0, "other owners of the same effect")
	offered := fs.Int("capacity", 0, "available capacity")
	required := fs.Int("required-capacity", 1, "capacity required")
	since := fs.Duration("since-last-run", 0, "elapsed time since last completed run")
	cooldown := fs.Duration("cooldown", 0, "minimum interval between runs")
	window := fs.Duration("service-window", 0, "maximum useful demand age")
	value := fs.Float64("expected-value", 0, "expected useful movement")
	floor := fs.Float64("value-floor", 0, "minimum useful movement")
	wall := fs.Duration("estimated-wall", 0, "estimated wall-clock cost")
	attention := fs.Duration("estimated-attention", 0, "estimated operator attention cost")
	evidence := fs.String("evidence", "", "comma-separated stable evidence references")
	asJSON := fs.Bool("json", false, "emit the trigger receipt as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	now := time.Now().UTC()
	if *observed != "" {
		parsed, err := time.Parse(time.RFC3339Nano, *observed)
		if err != nil {
			fmt.Fprintf(stderr, "fak superloop trigger: invalid --observed-at: %v\n", err)
			return 2
		}
		now = parsed
	}
	refs := []string{}
	for _, v := range strings.Split(*evidence, ",") {
		if v = strings.TrimSpace(v); v != "" {
			refs = append(refs, v)
		}
	}
	r := looptrigger.Evaluate(looptrigger.Input{Loop: *loop, ObservedAt: now, EligibleUnits: *eligible, OldestAge: *oldest, SourceAge: *sourceAge, MaxSourceAge: *maxSourceAge, OverlapCount: *overlap, OfferedCapacity: *offered, RequiredCapacity: *required, SinceLastRun: *since, Cooldown: *cooldown, ServiceWindow: *window, ExpectedValue: *value, ValueFloor: *floor, EstimatedWall: *wall, EstimatedAttention: *attention, EvidenceRefs: refs})
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, r, "fak superloop trigger")
	}
	fmt.Fprintf(stdout, "%s %s: %s (%s) demand=%d freshness=%s ownership=%s capacity=%d/%d timing=%s lateness=%ds value=%.2f/%.2f\n", r.Loop, r.Decision, r.Reason, r.Schema, r.Demand.EligibleUnits, r.Freshness.State, r.Ownership.State, r.Capacity.Offered, r.Capacity.Required, r.Timing.State, r.Timing.LatenessSeconds, r.Cost.ExpectedValue, r.Cost.ValueFloor)
	return 0
}
