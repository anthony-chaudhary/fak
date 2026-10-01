package buildwitness

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestBuildGHAMatrixCoalescesOnlyEquivalentRows(t *testing.T) {
	// Decode the public wire shape so this regression compiles against the old
	// generator, which lacks the additive covered-variant list.
	type entry struct {
		Variant  string   `json:"variant"`
		Variants []string `json:"variants"`
		Target   string   `json:"target"`
		GOOS     string   `json:"goos"`
		GOARCH   string   `json:"goarch"`
		Tags     string   `json:"tags"`
		Advisory bool     `json:"advisory"`
	}
	check := func(t *testing.T, manifest *VariantManifest, pureOnly bool, want map[string][]string) {
		t.Helper()
		before, err := json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(BuildGHAMatrix(manifest, pureOnly))
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Include []entry `json:"include"`
		}
		if err := json.Unmarshal(encoded, &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Include) != len(want) {
			t.Errorf("canonical compile cells = %d, want %d", len(got.Include), len(want))
		}
		seen := map[string]bool{}
		covered := map[string]bool{}
		for _, row := range got.Include {
			key := row.Variant + "@" + row.Target
			names, exists := want[key]
			if !exists || seen[key] {
				t.Errorf("unexpected or repeated canonical cell %s", key)
			}
			seen[key] = true
			if !reflect.DeepEqual(row.Variants, names) {
				t.Errorf("%s covered variants = %v, want %v", key, row.Variants, names)
			}
			canonical := manifest.FindVariant(row.Variant)
			if canonical == nil {
				t.Errorf("unknown canonical variant %q", row.Variant)
				continue
			}
			if row.GOOS+"/"+row.GOARCH != row.Target || row.Tags != canonical.Tags || row.Advisory != canonical.IsAdvisory() {
				t.Errorf("canonical execution/gating fields changed: %+v", row)
			}
			target := ReleaseTarget{GOOS: row.GOOS, GOARCH: row.GOARCH}
			plan := PlanCompile(TargetPackage, *canonical, target, NullDevice())
			for _, name := range row.Variants {
				identity := name + "@" + row.Target
				if covered[identity] {
					t.Errorf("logical variant-target covered twice: %s", identity)
				}
				covered[identity] = true
				variant := manifest.FindVariant(name)
				if variant == nil {
					t.Errorf("unknown covered variant %q", name)
					continue
				}
				if variant.TargetOSArch != "" && variant.TargetOSArch != "all" && variant.TargetOSArch != row.Target {
					t.Errorf("%s covered on wrong target %s", name, row.Target)
				}
				other := PlanCompile(TargetPackage, *variant, target, NullDevice())
				if !reflect.DeepEqual(plan.Command, other.Command) || !reflect.DeepEqual(plan.Env, other.Env) || variant.IsAdvisory() != row.Advisory {
					t.Errorf("%s was coalesced with a different full compile command, env or gate", identity)
				}
				if pureOnly && !variant.IsPureGo() {
					t.Errorf("pure-only matrix admitted cgo variant %s", name)
				}
			}
		}
		wantCoverage := 0
		for key, names := range want {
			if !seen[key] {
				t.Errorf("missing canonical cell %s", key)
			}
			target := strings.SplitN(key, "@", 2)[1]
			for _, name := range names {
				wantCoverage++
				if !covered[name+"@"+target] {
					t.Errorf("missing logical coverage %s@%s", name, target)
				}
			}
		}
		if len(covered) != wantCoverage {
			t.Errorf("logical coverage count = %d, want %d", len(covered), wantCoverage)
		}
		after, err := json.Marshal(manifest)
		if err != nil || string(before) != string(after) {
			t.Error("generating the matrix modified the declared taxonomy")
		}
	}

	t.Run("current_taxonomy", func(t *testing.T) {
		manifest, _, err := LoadVariantManifest(repoRoot(t))
		if err != nil {
			t.Fatal(err)
		}
		check(t, manifest, true, map[string][]string{
			"default@linux/amd64":            {"default", "linux-amd64"},
			"default@linux/arm64":            {"default"},
			"default@darwin/amd64":           {"default"},
			"default@darwin/arm64":           {"default", "darwin-arm64"},
			"default@windows/amd64":          {"default", "windows-amd64"},
			"wip_sessionfleet@linux/amd64":   {"wip_sessionfleet"},
			"wip_sessionfleet@linux/arm64":   {"wip_sessionfleet"},
			"wip_sessionfleet@darwin/amd64":  {"wip_sessionfleet"},
			"wip_sessionfleet@darwin/arm64":  {"wip_sessionfleet"},
			"wip_sessionfleet@windows/amd64": {"wip_sessionfleet"},
		})
	})

	// Identical-looking labels cannot collapse a different compiler configuration
	// or turn an advisory failure into trunk-gating evidence (or vice versa).
	manifest := &VariantManifest{
		ReleaseTargets: []ReleaseTarget{{GOOS: "linux", GOARCH: "amd64"}, {GOOS: "windows", GOARCH: "amd64"}},
		Variants: []Variant{
			{Name: "base", CGOEnabled: "0", TargetOSArch: "all", Gate: "matrix"},
			{Name: "alias", CGOEnabled: "0", TargetOSArch: "linux/amd64", Gate: "matrix"},
			{Name: "tagged", CGOEnabled: "0", TargetOSArch: "linux/amd64", Tags: "wip_probe", Gate: "matrix"},
			{Name: "cgo", CGOEnabled: "1", TargetOSArch: "linux/amd64", Gate: "matrix"},
			{Name: "advisory", CGOEnabled: "0", TargetOSArch: "linux/amd64", Gate: "advisory"},
		},
	}
	for _, pureOnly := range []bool{false, true} {
		name := "distinct_configs"
		if pureOnly {
			name = "pure_filter"
		}
		t.Run(name, func(t *testing.T) {
			want := map[string][]string{
				"base@linux/amd64":     {"base", "alias"},
				"base@windows/amd64":   {"base"},
				"tagged@linux/amd64":   {"tagged"},
				"advisory@linux/amd64": {"advisory"},
			}
			if !pureOnly {
				want["cgo@linux/amd64"] = []string{"cgo"}
			}
			check(t, manifest, pureOnly, want)
		})
	}
}
