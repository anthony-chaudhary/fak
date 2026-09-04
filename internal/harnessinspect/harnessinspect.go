// Package harnessinspect produces operator-facing inspection reports and rendered summaries
// from resolved harness product locks.
package harnessinspect

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// Schema defines the canonical schema identifier for harness inspection reports.
const Schema = "fak-harness-inspection/1"

// Asset represents an effective capability granted or restricted within the inspected harness.
type Asset struct {
	Capability string   `json:"capability"`
	Source     string   `json:"source"`
	Control    string   `json:"control"`
	Detail     string   `json:"detail,omitempty"`
	Grants     []string `json:"grants,omitempty"`
	Denies     []string `json:"denies,omitempty"`
}

// Component represents a resolved harness component and its provenance.
type Component struct {
	ID       string   `json:"id"`
	Version  string   `json:"version"`
	Source   string   `json:"source"`
	Reason   string   `json:"reason"`
	Provides []string `json:"provides,omitempty"`
}

// Report holds the structured inspection of a resolved harness lock.
type Report struct {
	Schema      string                     `json:"schema"`
	Verified    bool                       `json:"verified"`
	LockID      string                     `json:"lock_id"`
	Environment harnessresolve.Environment `json:"environment"`
	Budget      harnessresolve.Budget      `json:"budget"`
	Components  []Component                `json:"components"`
	Assets      []Asset                    `json:"assets"`
	Decisions   int                        `json:"decisions"`
	Controls    []string                   `json:"controls"`
}

// Inspect projects and sorts components, capabilities, and controls from a resolved lock into an inspection report.
func Inspect(lock harnessresolve.Lock, lockPath string) Report {
	report := Report{Schema: Schema, Verified: true, LockID: lock.ID, Environment: lock.Environment, Budget: lock.Budget, Decisions: len(lock.Decisions)}
	for _, component := range lock.Components {
		report.Components = append(report.Components, Component{ID: component.ID, Version: component.Version, Source: component.Source, Reason: component.Reason, Provides: component.Provides})
	}
	for _, asset := range lock.Assets {
		report.Assets = append(report.Assets, inspectAsset(asset))
	}
	sort.Slice(report.Components, func(i, j int) bool { return report.Components[i].ID < report.Components[j].ID })
	sort.Slice(report.Assets, func(i, j int) bool { return report.Assets[i].Capability < report.Assets[j].Capability })
	report.Controls = []string{
		fmt.Sprintf("compare a candidate: fak harness preview --current %s --candidate <candidate.lock.json>", quote(lockPath)),
		"change the harness: edit the product manifest or layer selection, then run fak harness resolve",
		"verify again: fak harness inspect --lock <candidate.lock.json>",
	}
	return report
}

func inspectAsset(asset harnesscompose.EffectiveAsset) Asset {
	control := "changeable by re-resolve"
	if asset.Mandatory {
		control = "mandatory"
	} else if asset.Locked {
		control = "locked by source"
	}
	detail := asset.Value
	if detail == "" {
		detail = asset.Ref
	}
	if detail == "" {
		detail = asset.Boundary
	}
	return Asset{Capability: asset.Kind + ":" + asset.ID, Source: asset.Source, Control: control, Detail: detail, Grants: asset.Grants, Denies: asset.Denies}
}

// Render formats an inspection report into operator-readable summary text.
func Render(report Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "HARNESS INSPECT | VERIFIED\nlock: %s\nenvironment: %s/%s | contract %s\nbudget: context=%d tokens | memory=%d MiB | workers=%d\n", report.LockID, report.Environment.OS, report.Environment.Arch, report.Environment.Contract, report.Budget.ContextTokens, report.Budget.MemoryMiB, report.Budget.Workers)
	fmt.Fprintf(&b, "components (%d):\n", len(report.Components))
	for _, component := range report.Components {
		fmt.Fprintf(&b, "- %s@%s | from %s | %s", component.ID, component.Version, component.Source, component.Reason)
		if len(component.Provides) > 0 {
			fmt.Fprintf(&b, " | provides %s", strings.Join(component.Provides, ", "))
		}
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "effective capabilities (%d):\n", len(report.Assets))
	for _, asset := range report.Assets {
		fmt.Fprintf(&b, "- %s | from %s | %s", asset.Capability, asset.Source, asset.Control)
		if asset.Detail != "" {
			fmt.Fprintf(&b, " | %s", asset.Detail)
		}
		if len(asset.Grants) > 0 {
			fmt.Fprintf(&b, " | grants %s", strings.Join(asset.Grants, ", "))
		}
		if len(asset.Denies) > 0 {
			fmt.Fprintf(&b, " | denies %s", strings.Join(asset.Denies, ", "))
		}
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "resolver decisions: %d\ncontrols:\n", report.Decisions)
	for _, control := range report.Controls {
		fmt.Fprintf(&b, "- %s\n", control)
	}
	return b.String()
}

func quote(path string) string {
	if strings.ContainsAny(path, " \t") {
		return fmt.Sprintf("%q", path)
	}
	return path
}
