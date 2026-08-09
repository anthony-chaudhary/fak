package slackoutbox

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"
)

// HealthWindowSample is one cumulative-free observation in a rolling cache-health
// window. UpgradeAttempts is the sum of authored upgrades and refusals.
type HealthWindowSample struct {
	At              time.Time `json:"at"`
	UpgradeAttempts uint64    `json:"upgrade_attempts"`
	UpgradeRefusals uint64    `json:"upgrade_refusals"`
	PrefixTurns     uint64    `json:"prefix_turns"`
	ColdPrefixTurns uint64    `json:"cold_prefix_turns"`
}

// HealthWindowConfig defines the sustained-breach contract. Every sample in
// the trailing Window must breach a threshold; one transient spike cannot page.
type HealthWindowConfig struct {
	Window                    int     `json:"window"`
	MinUpgradeAttempts        uint64  `json:"min_upgrade_attempts"`
	MinPrefixTurns            uint64  `json:"min_prefix_turns"`
	MaxUpgradeRefusalFraction float64 `json:"max_upgrade_refusal_fraction"`
	MaxColdPrefixFraction     float64 `json:"max_cold_prefix_fraction"`
	Destination               string  `json:"destination"`
}

// HealthWindowAlert is the deterministic witness posted to the outbox.
type HealthWindowAlert struct {
	Kind                   string    `json:"kind"`
	WindowStart            time.Time `json:"window_start"`
	WindowEnd              time.Time `json:"window_end"`
	Samples                int       `json:"samples"`
	UpgradeRefusalFraction float64   `json:"upgrade_refusal_fraction"`
	ColdPrefixFraction     float64   `json:"cold_prefix_fraction"`
	UpgradeRefusalBreached bool      `json:"upgrade_refusal_breached"`
	ColdPrefixBreached     bool      `json:"cold_prefix_breached"`
}

func (c HealthWindowConfig) validate() error {
	if c.Window < 1 {
		return errors.New("cache-health alert window must be positive")
	}
	if c.Destination == "" {
		return errors.New("cache-health alert destination is required")
	}
	for name, v := range map[string]float64{"upgrade-refusal threshold": c.MaxUpgradeRefusalFraction, "cold-prefix threshold": c.MaxColdPrefixFraction} {
		if math.IsNaN(v) || v < 0 || v > 1 {
			return fmt.Errorf("%s must be within [0,1]", name)
		}
	}
	return nil
}

// EvaluateHealthWindow returns an alert only for a full, sustained trailing
// window. A metric with an undersized denominator is ignored rather than paged.
func EvaluateHealthWindow(samples []HealthWindowSample, cfg HealthWindowConfig) (HealthWindowAlert, bool, error) {
	if err := cfg.validate(); err != nil {
		return HealthWindowAlert{}, false, err
	}
	if len(samples) < cfg.Window {
		return HealthWindowAlert{}, false, nil
	}
	w := samples[len(samples)-cfg.Window:]
	alert := HealthWindowAlert{Kind: "cache-health-threshold", WindowStart: w[0].At, WindowEnd: w[len(w)-1].At, Samples: len(w)}
	var attempts, refusals, prefix, cold uint64
	refusalSustained, coldSustained := true, true
	for i, s := range w {
		if i > 0 && s.At.Before(w[i-1].At) {
			return HealthWindowAlert{}, false, errors.New("cache-health samples are not time ordered")
		}
		attempts += s.UpgradeAttempts
		refusals += s.UpgradeRefusals
		prefix += s.PrefixTurns
		cold += s.ColdPrefixTurns
		refusalSustained = refusalSustained && s.UpgradeAttempts >= cfg.MinUpgradeAttempts && fraction(s.UpgradeRefusals, s.UpgradeAttempts) > cfg.MaxUpgradeRefusalFraction
		coldSustained = coldSustained && s.PrefixTurns >= cfg.MinPrefixTurns && fraction(s.ColdPrefixTurns, s.PrefixTurns) > cfg.MaxColdPrefixFraction
	}
	alert.UpgradeRefusalFraction = fraction(refusals, attempts)
	alert.ColdPrefixFraction = fraction(cold, prefix)
	alert.UpgradeRefusalBreached, alert.ColdPrefixBreached = refusalSustained, coldSustained
	return alert, refusalSustained || coldSustained, nil
}

func fraction(n, d uint64) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// EnqueueHealthWindowAlert evaluates the rolling window and durably posts one
// alert when breached. Healthy and incomplete windows leave the outbox alone.
func EnqueueHealthWindowAlert(out *Outbox, samples []HealthWindowSample, cfg HealthWindowConfig) (bool, error) {
	alert, breached, err := EvaluateHealthWindow(samples, cfg)
	if err != nil || !breached {
		return false, err
	}
	payload, err := json.Marshal(alert)
	if err != nil {
		return false, err
	}
	_, err = out.Enqueue(Row{Nonce: fmt.Sprintf("cache-health-%d", alert.WindowEnd.UnixNano()), Channel: cfg.Destination, Text: string(payload), Source: "cache-health-alert", EnqueuedAt: alert.WindowEnd.Format(time.RFC3339Nano)})
	return err == nil, err
}
