package benchsweep

import (
	"strconv"
	"testing"
	"time"
)

// testCfg is the shared data-parameter config: crashed/cancelled are hard
// excluded, ok/degraded are whitelisted, and the duration floor is 5s.
var testCfg = Config{
	HardExcluded: []string{"crashed", "cancelled"},
	Whitelist:    []string{"ok", "degraded"},
	MinDuration:  5 * time.Second,
}

// oc builds one outcome with the given combo id, status, artifact count, and
// duration. No clock is read anywhere: durations are fixed values.
func oc(id, status string, artifacts int, dur time.Duration) Outcome {
	arts := make([]string, 0, artifacts)
	for i := 0; i < artifacts; i++ {
		arts = append(arts, "art-"+id+"-"+strconv.Itoa(i))
	}
	return Outcome{Combo: Combo{ID: id}, Status: status, Artifacts: arts, Duration: dur}
}

// capturingLogger returns a Logger that records entries into *entries.
func capturingLogger(entries *[]RescueLogEntry) Logger {
	return func(e RescueLogEntry) { *entries = append(*entries, e) }
}

func TestEligibilityChain(t *testing.T) {
	t.Run("accept_when_every_check_passes", func(t *testing.T) {
		var logged []RescueLogEntry
		ch := NewChain(testCfg, capturingLogger(&logged))
		v := ch.Evaluate(oc("a1", "ok", 2, 10*time.Second))
		if !v.IsAccepted() || v.Reason != "" || v.Check != "" || v.Rescued {
			t.Fatalf("want plain accept, got %+v", v)
		}
		if len(logged) != 0 {
			t.Fatalf("accept must not log rescues, got %d entries", len(logged))
		}
	})

	t.Run("reject_no_artifacts_is_the_first_decisive_check", func(t *testing.T) {
		ch := NewChain(testCfg, nil)
		// Long duration and whitelisted status: only the missing artifacts
		// can decide, and it must be the first check to do so.
		v := ch.Evaluate(oc("a1", "ok", 0, 60*time.Second))
		if !v.IsRejected() || v.Reason != ReasonNoArtifacts || v.Check != "artifacts-exist" {
			t.Fatalf("want NO_ARTIFACTS from artifacts-exist, got %+v", v)
		}
	})

	t.Run("hard_exclude_beats_whitelist_listing", func(t *testing.T) {
		// "crashed" appears in BOTH sets: the hard-exclude check runs first,
		// so listing the status in the whitelist cannot admit it.
		cfg := Config{
			HardExcluded: []string{"crashed"},
			Whitelist:    []string{"ok", "crashed"},
			MinDuration:  5 * time.Second,
		}
		ch := NewChain(cfg, nil)
		v := ch.Evaluate(oc("b1", "crashed", 1, 60*time.Second))
		if !v.IsRejected() || v.Reason != ReasonHardExcludedStatus || v.Check != "hard-exclude-statuses" {
			t.Fatalf("want HARD_EXCLUDED_STATUS from hard-exclude-statuses, got %+v", v)
		}
	})

	t.Run("status_not_on_whitelist_is_rejected", func(t *testing.T) {
		ch := NewChain(testCfg, nil)
		v := ch.Evaluate(oc("c1", "weird", 1, 60*time.Second))
		if !v.IsRejected() || v.Reason != ReasonStatusNotWhitelisted || v.Check != "status-whitelist" {
			t.Fatalf("want STATUS_NOT_WHITELISTED from status-whitelist, got %+v", v)
		}
	})

	t.Run("short_duration_is_rescued_flagged_and_logged", func(t *testing.T) {
		var logged []RescueLogEntry
		ch := NewChain(testCfg, capturingLogger(&logged))
		v := ch.Evaluate(oc("d1", "ok", 1, time.Second))
		if !v.IsRescued() || !v.Rescued || v.Check != "duration-rescue" || v.Reason != "" {
			t.Fatalf("want flagged rescue from duration-rescue, got %+v", v)
		}
		if len(logged) != 1 {
			t.Fatalf("rescue must log exactly one entry, got %d", len(logged))
		}
		e := logged[0]
		if e.ComboID != "d1" || e.Status != "ok" || e.Duration != time.Second || e.Check != "duration-rescue" {
			t.Fatalf("log entry wrong: %+v", e)
		}
	})

	t.Run("duration_at_exact_floor_is_accepted", func(t *testing.T) {
		ch := NewChain(testCfg, nil)
		v := ch.Evaluate(oc("e1", "ok", 1, 5*time.Second))
		if !v.IsAccepted() {
			t.Fatalf("duration == MinDuration must accept (strictly shorter rescues), got %+v", v)
		}
	})

	t.Run("whitelist_and_min_duration_are_data_parameters", func(t *testing.T) {
		// Same outcomes, different configs: the verdicts move with the data,
		// proving no literal is baked into the checks.
		outcomes := []Outcome{
			oc("x1", "slow_ok", 1, 30*time.Second),
			oc("x2", "ok", 1, 30*time.Second),
		}
		strict := NewChain(Config{Whitelist: []string{"slow_ok"}, MinDuration: 1 * time.Millisecond}, nil)
		lax := NewChain(Config{Whitelist: []string{"ok", "slow_ok"}, MinDuration: 1 * time.Hour}, nil)
		if v := strict.Evaluate(outcomes[0]); !v.IsAccepted() {
			t.Fatalf("strict chain should accept slow_ok, got %+v", v)
		}
		if v := strict.Evaluate(outcomes[1]); !v.IsRejected() || v.Reason != ReasonStatusNotWhitelisted {
			t.Fatalf("strict chain should reject ok, got %+v", v)
		}
		if v := lax.Evaluate(outcomes[0]); !v.IsRescued() {
			t.Fatalf("lax chain should rescue below its hour floor, got %+v", v)
		}
	})

	t.Run("first_decisive_check_wins_later_checks_never_run", func(t *testing.T) {
		probeCalls := 0
		ch := NewChain(testCfg, nil).Append(Check{
			Name: "probe",
			Inspect: func(o Outcome) CheckAnswer {
				probeCalls++
				return Pass
			},
		})
		v := ch.Evaluate(oc("f1", "ok", 0, 60*time.Second)) // rejected at check 1
		if !v.IsRejected() || v.Reason != ReasonNoArtifacts {
			t.Fatalf("want NO_ARTIFACTS, got %+v", v)
		}
		if probeCalls != 0 {
			t.Fatalf("decisive check must stop the walk; probe ran %d times", probeCalls)
		}
	})

	t.Run("appended_campaign_check_extends_the_chain", func(t *testing.T) {
		ch := NewChain(testCfg, nil).Append(Check{
			Name: "no-hot-restart-axis",
			Inspect: func(o Outcome) CheckAnswer {
				for _, wp := range o.Combo.WorkloadParams {
					if wp.Name == "hot_restart" {
						return RejectAnswer("CAMPAIGN_EXCLUDE")
					}
				}
				return Pass
			},
		})
		plain := oc("g1", "ok", 1, 60*time.Second)
		if v := ch.Evaluate(plain); !v.IsAccepted() {
			t.Fatalf("campaign check must not fire without its marker, got %+v", v)
		}
		marked := oc("g2", "ok", 1, 60*time.Second)
		marked.Combo.WorkloadParams = []Param{{Name: "hot_restart", Value: "true"}}
		v := ch.Evaluate(marked)
		if !v.IsRejected() || v.Reason != "CAMPAIGN_EXCLUDE" || v.Check != "no-hot-restart-axis" {
			t.Fatalf("want CAMPAIGN_EXCLUDE from no-hot-restart-axis, got %+v", v)
		}
	})

	t.Run("empty_whitelist_is_fail_closed", func(t *testing.T) {
		ch := NewChain(Config{MinDuration: time.Second}, nil)
		v := ch.Evaluate(oc("h1", "ok", 1, 60*time.Second))
		if !v.IsRejected() || v.Reason != ReasonStatusNotWhitelisted {
			t.Fatalf("empty whitelist must reject every status, got %+v", v)
		}
	})

	t.Run("rescues_log_in_evaluation_order", func(t *testing.T) {
		var logged []RescueLogEntry
		ch := NewChain(testCfg, capturingLogger(&logged))
		_ = ch.Evaluate(oc("i2", "ok", 1, 2*time.Second))
		_ = ch.Evaluate(oc("i1", "degraded", 1, time.Second))
		if len(logged) != 2 || logged[0].ComboID != "i2" || logged[1].ComboID != "i1" {
			t.Fatalf("rescue log order wrong: %+v", logged)
		}
	})

	t.Run("verdict_helpers_cover_the_closed_vocabulary", func(t *testing.T) {
		cases := []struct {
			v        Verdict
			accepted bool
			rejected bool
			rescued  bool
		}{
			{Verdict{Kind: VerdictAccept}, true, false, false},
			{Verdict{Kind: VerdictReject, Reason: ReasonNoArtifacts}, false, true, false},
			{Verdict{Kind: VerdictRescue, Rescued: true}, false, false, true},
		}
		for _, c := range cases {
			if c.v.IsAccepted() != c.accepted || c.v.IsRejected() != c.rejected || c.v.IsRescued() != c.rescued {
				t.Fatalf("helpers disagree on %+v", c.v)
			}
		}
	})
}
