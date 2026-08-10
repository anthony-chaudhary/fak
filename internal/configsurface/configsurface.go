// Package configsurface scores fak.toml discoverability and default coverage.
// It keeps user flexibility from turning into an unbounded mystery-knob file.
package configsurface

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/configguide"
	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

const (
	Schema      = "fak-config-surface/1"
	MaxKeys     = 32
	MaxPostures = 8
)

type Finding struct {
	Key    string `json:"key,omitempty"`
	Reason string `json:"reason"`
}

type Report struct {
	Schema              string    `json:"schema"`
	Keys                int       `json:"keys"`
	Postures            int       `json:"postures"`
	MaxKeys             int       `json:"max_keys"`
	MaxPostures         int       `json:"max_postures"`
	DefaultCoverage     float64   `json:"default_coverage"`
	DescriptionCoverage float64   `json:"description_coverage"`
	GuideCoverage       float64   `json:"guide_coverage"`
	Discoverable        bool      `json:"discoverable"`
	Findings            []Finding `json:"findings"`
}

func Audit() Report {
	descriptors := deploymanifest.Descriptors()
	postures := configguide.Names()
	report := Report{Schema: Schema, Keys: len(descriptors), Postures: len(postures), MaxKeys: MaxKeys, MaxPostures: MaxPostures}
	defaults, descriptions := 0, 0
	guided := make(map[string]bool)
	for _, posture := range postures {
		result, err := configguide.Guide(configguide.Options{Posture: posture})
		if err != nil {
			report.Findings = append(report.Findings, Finding{Reason: fmt.Sprintf("posture %s does not generate: %v", posture, err)})
			continue
		}
		for _, change := range result.Changes {
			guided[change.Field] = true
		}
	}
	for _, descriptor := range descriptors {
		dotted := descriptor.Key.Dotted()
		// The typed descriptor always carries a concrete default, including false,
		// zero, and empty string; nil is the only undefaulted state.
		if descriptor.Default != nil {
			defaults++
		} else {
			report.Findings = append(report.Findings, Finding{Key: dotted, Reason: "missing built-in default"})
		}
		if descriptor.Description != "" {
			descriptions++
		} else {
			report.Findings = append(report.Findings, Finding{Key: dotted, Reason: "missing discoverable description"})
		}
	}
	if report.Keys > MaxKeys {
		report.Findings = append(report.Findings, Finding{Reason: fmt.Sprintf("config key budget exceeded: %d > %d", report.Keys, MaxKeys)})
	}
	if report.Postures > MaxPostures {
		report.Findings = append(report.Findings, Finding{Reason: fmt.Sprintf("guided posture budget exceeded: %d > %d", report.Postures, MaxPostures)})
	}
	if report.Keys != 0 {
		report.DefaultCoverage = float64(defaults) / float64(report.Keys)
		report.DescriptionCoverage = float64(descriptions) / float64(report.Keys)
		report.GuideCoverage = float64(len(guided)) / float64(report.Keys)
	}
	report.Discoverable = len(report.Findings) == 0
	return report
}

func (r Report) Check() error {
	if r.Discoverable {
		return nil
	}
	return fmt.Errorf("config surface is not discoverable: %d finding(s)", len(r.Findings))
}
