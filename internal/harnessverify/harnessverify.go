package harnessverify

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// ObservationSchema is the schema identifier for harness runtime observations.
const ObservationSchema = "fak-harness-runtime-observation/1"

// ReportSchema is the schema identifier for harness runtime verification reports.
const ReportSchema = "fak-harness-runtime-verification/1"

// Observation captures the runtime capabilities and execution events recorded for a harness run.
type Observation struct {
	Schema       string       `json:"schema"`
	LockID       string       `json:"lock_id"`
	RunID        string       `json:"run_id"`
	Capabilities []Capability `json:"capabilities"`
	Events       []Event      `json:"events,omitempty"`
}

// Capability represents an individual runtime capability declaration and its active bindings.
type Capability struct {
	Capability string   `json:"capability"`
	Source     string   `json:"source"`
	Value      string   `json:"value,omitempty"`
	Ref        string   `json:"ref,omitempty"`
	Boundary   string   `json:"boundary,omitempty"`
	Grants     []string `json:"grants,omitempty"`
	Denies     []string `json:"denies,omitempty"`
}

// Event records a runtime invocation or policy gate outcome for a capability.
type Event struct {
	Kind       string `json:"kind"`
	Capability string `json:"capability"`
	Source     string `json:"source"`
	Outcome    string `json:"outcome"`
}

// Finding describes a capability-level match, difference, omission, or addition.
type Finding struct {
	Status         string `json:"status"`
	Capability     string `json:"capability"`
	ExpectedSource string `json:"expected_source,omitempty"`
	RuntimeSource  string `json:"runtime_source,omitempty"`
	Difference     string `json:"difference,omitempty"`
}

// Report aggregates verification findings, counts, and final verdict for a run against a lock.
type Report struct {
	Schema   string    `json:"schema"`
	Verdict  string    `json:"verdict"`
	LockID   string    `json:"lock_id"`
	RunID    string    `json:"run_id"`
	Findings []Finding `json:"findings"`
	Events   []Event   `json:"events,omitempty"`
	Matched  int       `json:"matched"`
	Changed  int       `json:"changed"`
	Added    int       `json:"added"`
	Omitted  int       `json:"omitted"`
}

// Verify checks observed runtime capabilities against a resolved lock, detecting additions, omissions, or changes.
func Verify(lock harnessresolve.Lock, observation Observation) (Report, error) {
	if observation.Schema != ObservationSchema {
		return Report{}, fmt.Errorf("observation schema must be %q", ObservationSchema)
	}
	if strings.TrimSpace(observation.RunID) == "" {
		return Report{}, fmt.Errorf("observation run_id is required")
	}
	if observation.LockID != lock.ID {
		return Report{}, fmt.Errorf("observation lock_id %q does not match verified lock %q", observation.LockID, lock.ID)
	}
	expected := make(map[string]harnesscompose.EffectiveAsset, len(lock.Assets))
	for _, asset := range lock.Assets {
		expected[asset.Kind+":"+asset.ID] = asset
	}
	observed := make(map[string]Capability, len(observation.Capabilities))
	for _, capability := range observation.Capabilities {
		capability.Capability = strings.TrimSpace(capability.Capability)
		if capability.Capability == "" || !strings.Contains(capability.Capability, ":") {
			return Report{}, fmt.Errorf("runtime capability %q must be kind:id", capability.Capability)
		}
		if _, exists := observed[capability.Capability]; exists {
			return Report{}, fmt.Errorf("runtime capability %q is duplicated", capability.Capability)
		}
		observed[capability.Capability] = capability
	}
	report := Report{Schema: ReportSchema, Verdict: "verified", LockID: lock.ID, RunID: observation.RunID, Events: append([]Event(nil), observation.Events...)}
	keys := make([]string, 0, len(expected)+len(observed))
	seen := map[string]bool{}
	for key := range expected {
		keys = append(keys, key)
		seen[key] = true
	}
	for key := range observed {
		if !seen[key] {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		want, expectedOK := expected[key]
		got, observedOK := observed[key]
		finding := Finding{Capability: key}
		switch {
		case expectedOK && !observedOK:
			finding.Status, finding.ExpectedSource = "omitted", want.Source
			report.Omitted++
		case !expectedOK && observedOK:
			finding.Status, finding.RuntimeSource = "added", got.Source
			report.Added++
		default:
			finding.ExpectedSource, finding.RuntimeSource = want.Source, got.Source
			finding.Difference = difference(want, got)
			if finding.Difference == "" {
				finding.Status = "matched"
				report.Matched++
			} else {
				finding.Status = "changed"
				report.Changed++
			}
		}
		report.Findings = append(report.Findings, finding)
	}
	if report.Added+report.Changed+report.Omitted > 0 {
		report.Verdict = "deviation"
	}
	return report, nil
}

func difference(want harnesscompose.EffectiveAsset, got Capability) string {
	var fields []string
	if want.Source != got.Source {
		fields = append(fields, "source")
	}
	if want.Value != got.Value {
		fields = append(fields, "value")
	}
	if want.Ref != got.Ref {
		fields = append(fields, "ref")
	}
	if want.Boundary != got.Boundary {
		fields = append(fields, "boundary")
	}
	if strings.Join(want.Grants, "\x00") != strings.Join(got.Grants, "\x00") {
		fields = append(fields, "grants")
	}
	if strings.Join(want.Denies, "\x00") != strings.Join(got.Denies, "\x00") {
		fields = append(fields, "denies")
	}
	return strings.Join(fields, ",")
}

// Render formats a verification report into a plain-text audit summary.
func Render(report Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "HARNESS VERIFY RUN | %s\nlock: %s\nrun: %s\ncapabilities: matched=%d changed=%d added=%d omitted=%d\n", strings.ToUpper(report.Verdict), report.LockID, report.RunID, report.Matched, report.Changed, report.Added, report.Omitted)
	for _, finding := range report.Findings {
		fmt.Fprintf(&b, "- %s | %s", finding.Capability, finding.Status)
		if finding.ExpectedSource != "" {
			fmt.Fprintf(&b, " | expected %s", finding.ExpectedSource)
		}
		if finding.RuntimeSource != "" {
			fmt.Fprintf(&b, " | runtime %s", finding.RuntimeSource)
		}
		if finding.Difference != "" {
			fmt.Fprintf(&b, " | changed %s", finding.Difference)
		}
		b.WriteByte('\n')
	}
	if len(report.Events) > 0 {
		b.WriteString("runtime decisions:\n")
		for _, event := range report.Events {
			fmt.Fprintf(&b, "- %s | %s | from %s | %s\n", event.Kind, event.Capability, event.Source, event.Outcome)
		}
	}
	return b.String()
}
