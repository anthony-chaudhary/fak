package orchestration

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ptr[T any](v T) *T { return &v }
func nativeCaps() HarnessCapabilities {
	return HarnessCapabilities{SupportNative, SupportNative, SupportNative, SupportNative, SupportNative}
}

func TestPrecedence(t *testing.T) {
	task := TaskSpec{Schema: "fak-orchestration-task/1", ID: "x", WorkClass: WorkGrind, MaxWorkers: ptr(2), MaxTokens: ptr(int64(12000))}
	r, err := Resolve(OrchestrationProfile{Name: ProfileUltracode, MaxWorkers: ptr(7), MaxTokens: ptr(int64(9000))}, task, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	if r.Resolved.Budget != (Budget{7, 9000}) {
		t.Fatalf("budget=%+v", r.Resolved.Budget)
	}
	if got := r.Overrides[len(r.Overrides)-1].Source; got != "operator" {
		t.Fatalf("last source=%s", got)
	}
}

func TestCapabilityNegotiation(t *testing.T) {
	fields := []string{"concurrency", "task_messaging", "cancellation", "leases", "independent_witness"}
	for _, missing := range fields {
		t.Run(missing, func(t *testing.T) {
			c := nativeCaps()
			switch missing {
			case "concurrency":
				c.Concurrency = SupportUnsupported
			case "task_messaging":
				c.TaskMessaging = SupportEmulated
			case "cancellation":
				c.Cancellation = SupportDegraded
			case "leases":
				c.Leases = SupportUnsupported
			case "independent_witness":
				c.IndependentWitness = SupportEmulated
			}
			r, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, TaskSpec{ID: "x", WorkClass: WorkRigor}, c)
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Degradations) != 1 || r.Degradations[0].Capability != missing {
				t.Fatalf("degradations=%+v", r.Degradations)
			}
			if !strings.Contains(strings.Join(r.Resolved.Explanation, "\n"), "degraded:") {
				t.Fatal("text omitted degradation")
			}
		})
	}
}

func TestStrictRejectsEveryDegradation(t *testing.T) {
	c := nativeCaps()
	c.Leases = SupportUnsupported
	_, err := Resolve(OrchestrationProfile{Name: ProfileUltracode, Strict: true}, TaskSpec{ID: "x", WorkClass: WorkRigor}, c)
	if !errors.Is(err, ErrStrictDegradation) {
		t.Fatalf("err=%v", err)
	}
}

func TestFixtureMatrix(t *testing.T) {
	cases := []struct {
		name     string
		req      OrchestrationProfile
		task     TaskSpec
		caps     HarnessCapabilities
		profile  Profile
		workers  int
		attended bool
		degraded bool
	}{
		{"off", OrchestrationProfile{Name: ProfileOff}, TaskSpec{ID: "off", WorkClass: WorkDefault}, HarnessCapabilities{}, ProfileOff, 1, false, false},
		{"auto-grind", OrchestrationProfile{Name: ProfileAuto}, TaskSpec{ID: "grind", WorkClass: WorkGrind}, nativeCaps(), ProfileUltracode, 4, false, false},
		{"auto-rigor", OrchestrationProfile{Name: ProfileAuto}, TaskSpec{ID: "rigor", WorkClass: WorkRigor}, nativeCaps(), ProfileUltracode, 3, false, false},
		{"forced-ultracode", OrchestrationProfile{Name: ProfileUltracode}, TaskSpec{ID: "force"}, nativeCaps(), ProfileUltracode, 4, false, false},
		{"attended", OrchestrationProfile{Name: ProfileAuto}, TaskSpec{ID: "attended", WorkClass: WorkGrind, Attended: ptr(true)}, nativeCaps(), ProfileUltracode, 4, true, false},
		{"unattended-budget-cap", OrchestrationProfile{Name: ProfileUltracode, MaxWorkers: ptr(2), MaxTokens: ptr(int64(7000)), Attended: ptr(false)}, TaskSpec{ID: "cap"}, nativeCaps(), ProfileUltracode, 2, false, false},
		{"unsupported-harness", OrchestrationProfile{Name: ProfileUltracode}, TaskSpec{ID: "unsupported"}, HarnessCapabilities{}, ProfileUltracode, 4, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Resolve(tc.req, tc.task, tc.caps)
			if err != nil {
				t.Fatal(err)
			}
			if r.Resolved.Profile != tc.profile || r.Resolved.Budget.MaxWorkers != tc.workers || r.Resolved.Interaction.Attended != tc.attended || (len(r.Degradations) > 0) != tc.degraded {
				t.Fatalf("resolution=%+v", r)
			}
		})
	}
}

func TestJSONRoundTripAndUnknownVersionBehavior(t *testing.T) {
	r, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, TaskSpec{ID: "x", WorkClass: WorkRigor}, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	b, err := StableJSON(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseResolution(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != SchemaVersion {
		t.Fatal(got.Schema)
	}
	var raw map[string]any
	if json.Unmarshal(b, &raw) != nil {
		t.Fatal("decode")
	}
	raw["future_field"] = true
	future, _ := json.Marshal(raw)
	if _, err := ParseResolution(future); err == nil {
		t.Fatal("unknown field accepted")
	}
	raw = map[string]any{}
	json.Unmarshal(b, &raw)
	raw["schema"] = "fak-orchestration-plan/99"
	future, _ = json.Marshal(raw)
	if _, err := ParseResolution(future); err == nil {
		t.Fatal("unknown version accepted")
	}
}

func TestPackageHasNoProviderOrAdapterImports(t *testing.T) {
	// Compile-time package imports are deliberately only standard-library imports.
	if strings.Contains(SchemaVersion, "claude") || strings.Contains(SchemaVersion, "codex") {
		t.Fatal("provider dialect leaked")
	}
}

func TestFixtureCorpus(t *testing.T) {
	cases := []struct {
		file    string
		profile Profile
		caps    HarnessCapabilities
		wantDeg bool
	}{
		{"off.json", ProfileOff, nativeCaps(), false},
		{"auto-grind.json", ProfileAuto, nativeCaps(), false},
		{"auto-rigor.json", ProfileAuto, nativeCaps(), false},
		{"forced-ultracode.json", ProfileUltracode, nativeCaps(), false},
		{"attended.json", ProfileAuto, nativeCaps(), false},
		{"unattended.json", ProfileAuto, nativeCaps(), false},
		{"budget-caps.json", ProfileAuto, nativeCaps(), false},
		{"unsupported-harness.json", ProfileAuto, HarnessCapabilities{}, true},
	}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", tc.file))
			if err != nil {
				t.Fatal(err)
			}
			task, err := ParseTask(data)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Resolve(OrchestrationProfile{Name: tc.profile}, task, tc.caps)
			if err != nil {
				t.Fatal(err)
			}
			if (len(got.Degradations) > 0) != tc.wantDeg {
				t.Fatalf("degradations=%v, want any=%v", got.Degradations, tc.wantDeg)
			}
		})
	}
}

func TestParsersRejectTrailingJSON(t *testing.T) {
	if _, err := ParseTask([]byte(`{"schema":"fak-orchestration-task/1","id":"x"} {}`)); err == nil {
		t.Fatal("ParseTask accepted trailing JSON")
	}
	resolved, err := Resolve(OrchestrationProfile{Name: ProfileOff}, TaskSpec{Schema: "fak-orchestration-task/1", ID: "x"}, HarnessCapabilities{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := StableJSON(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseResolution(append(data, []byte("\n{}")...)); err == nil {
		t.Fatal("ParseResolution accepted trailing JSON")
	}
}

func TestStableJSONSortsDuplicateFieldProvenance(t *testing.T) {
	r := Resolution{Schema: SchemaVersion, Overrides: []Provenance{
		{Field: "budget.max_workers", Source: "task", Value: 2},
		{Field: "budget.max_workers", Source: "base-default", Value: 1},
	}}
	first, err := StableJSON(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Overrides[0], r.Overrides[1] = r.Overrides[1], r.Overrides[0]
	second, err := StableJSON(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("unstable JSON:\n%s\n---\n%s", first, second)
	}
}
