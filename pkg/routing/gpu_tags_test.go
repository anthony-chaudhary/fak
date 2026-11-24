package routing

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"
)

// gpu_tags_test.go is the adversarial witness for the GPU-tag contract:
// EngineDiscovery, the VRAMClass/CCFamily closed vocabularies, the tag
// builders, TagsFromDiscovery, (*Engine).PopulateTags, and RequireGPUClass
// selection riding the EXISTING RequireTags -> HasTag -> ReasonMissingTag
// fabric. Assertions are closed-token/sentinel only: errors.Is against
// ErrInvalidDiscoveryFact, exact strings, exact slices. Never error prose.

// pinnedVRAMClasses is the contract vocabulary MEMBERS, written out
// independently of the implementation so a registry drift fails here. The ""
// unknown sentinel is deliberately not a listed member (contract: "IsKnown
// true for each listed, false for \"\""; house style matches Surfaces()): the
// accessors list representable members only.
var pinnedVRAMClasses = []VRAMClass{"lt8", "8g", "16g", "32g", "64g", "96g"}

// pinnedCCFamilies is the contract vocabulary members, written out
// independently. "" (unknown) is not a listed member; see above.
var pinnedCCFamilies = []CCFamily{"pascal", "volta", "turing", "ampere", "ada", "hopper", "blackwell"}

// gpuTagsEngine builds a healthy local worker-capable engine for GPU-tag tests.
func gpuTagsEngine(id string) *Engine {
	return &Engine{
		ID:               id,
		Surface:          SurfaceLocal,
		Model:            "test-model",
		Roles:            []string{"worker"},
		Healthy:          true,
		CapacityHeadroom: 0.5,
	}
}

// gpuTagsCandidate builds a minimal routing candidate for a table case.
func gpuTagsCandidate(id string) *TaskCandidate {
	return &TaskCandidate{ID: id, Lane: "docs"}
}

// mustTag fails the test when a tag builder refuses a fact the contract says
// is valid.
func mustTag(t *testing.T, tag string, err error) string {
	t.Helper()
	if err != nil {
		t.Fatalf("tag builder refused a contract-valid fact: %v", err)
	}
	return tag
}

// fak-test:runtime fast est=1s
func TestVRAMClassForBoundaries(t *testing.T) {
	cases := []struct {
		name    string
		gib     int
		want    VRAMClass
		wantErr bool
	}{
		{"zero is unknown", 0, "", false},
		{"one is lt8", 1, "lt8", false},
		{"seven is lt8", 7, "lt8", false},
		{"eight is 8g", 8, "8g", false},
		{"fifteen is 8g", 15, "8g", false},
		{"sixteen is 16g", 16, "16g", false},
		{"thirty-one is 16g", 31, "16g", false},
		{"thirty-two is 32g", 32, "32g", false},
		{"sixty-three is 32g", 63, "32g", false},
		{"sixty-four is 64g", 64, "64g", false},
		{"ninety-five is 64g", 95, "64g", false},
		{"ninety-six is 96g", 96, "96g", false},
		{"one-thousand is 96g", 1000, "96g", false},
		{"max int32 is 96g", math.MaxInt32, "96g", false},
		{"max int64 is 96g", math.MaxInt64, "96g", false},
		{"negative one refused", -1, "", true},
		{"negative seven refused", -7, "", true},
		{"min int32 refused", math.MinInt32, "", true},
		{"min int64 refused", math.MinInt64, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := VRAMClassFor(tc.gib)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("VRAMClassFor(%d) = %q, nil; want refusal", tc.gib, got)
				}
				if !errors.Is(err, ErrInvalidDiscoveryFact) {
					t.Fatalf("VRAMClassFor(%d) error does not wrap ErrInvalidDiscoveryFact: %v", tc.gib, err)
				}
				if got.IsKnown() {
					t.Fatalf("VRAMClassFor(%d) refused but returned known class %q", tc.gib, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("VRAMClassFor(%d) error = %v, want nil", tc.gib, err)
			}
			if got != tc.want {
				t.Fatalf("VRAMClassFor(%d) = %q, want %q", tc.gib, got, tc.want)
			}
			if tc.want == "" && got.IsKnown() {
				t.Fatalf("VRAMClassFor(%d) returned %q which must not be known", tc.gib, got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCCFamilyForMapped(t *testing.T) {
	cases := []struct {
		cc   string
		want CCFamily
	}{
		{"6.0", "pascal"},
		{"6.1", "pascal"},
		{"6.2", "pascal"},
		{"7.0", "volta"},
		{"7.2", "volta"},
		{"7.5", "turing"},
		{"8.0", "ampere"},
		{"8.6", "ampere"},
		{"8.7", "ampere"},
		{"8.9", "ada"},
		{"9.0", "hopper"},
		{"10.0", "blackwell"},
		{"12.0", "blackwell"},
	}
	for _, tc := range cases {
		t.Run(tc.cc, func(t *testing.T) {
			got, err := CCFamilyFor(tc.cc)
			if err != nil {
				t.Fatalf("CCFamilyFor(%q) error = %v, want nil", tc.cc, err)
			}
			if got != tc.want {
				t.Fatalf("CCFamilyFor(%q) = %q, want %q", tc.cc, got, tc.want)
			}
		})
	}
	t.Run("empty cc is unknown not an error", func(t *testing.T) {
		got, err := CCFamilyFor("")
		if err != nil {
			t.Fatalf("CCFamilyFor(\"\") error = %v, want nil", err)
		}
		if got != "" || got.IsKnown() {
			t.Fatalf("CCFamilyFor(\"\") = %q (known=%t), want unknown", got, got.IsKnown())
		}
	})
}

// fak-test:runtime fast est=1s
func TestCCFamilyForRefusals(t *testing.T) {
	cases := []struct {
		name string
		cc   string
	}{
		{"unmapped well-formed 5.0", "5.0"},
		{"unmapped well-formed 8.3", "8.3"},
		{"unmapped well-formed 11.0", "11.0"},
		{"unmapped well-formed 7.3", "7.3"},
		{"unmapped well-formed 9.1", "9.1"},
		{"malformed no dot", "8"},
		{"malformed trailing dot", "8.6."},
		{"malformed alpha", "abc"},
		{"malformed suffix", "8.6a"},
		{"malformed leading space", " 8.6"},
		{"malformed trailing space", "8.6 "},
		{"malformed double dot", "8..6"},
		{"malformed leading dot", ".6"},
		{"malformed bare dot", "."},
		{"malformed negative", "-1.0"},
		{"malformed newline suffix", "8.6\n"},
		{"malformed integer blackwell", "10"},
		{"malformed hex", "0x8"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := CCFamilyFor(tc.cc)
			if err == nil {
				t.Fatalf("CCFamilyFor(%q) = %q, nil; want refusal", tc.cc, got)
			}
			if !errors.Is(err, ErrInvalidDiscoveryFact) {
				t.Fatalf("CCFamilyFor(%q) error does not wrap ErrInvalidDiscoveryFact: %v", tc.cc, err)
			}
			if got.IsKnown() {
				t.Fatalf("CCFamilyFor(%q) refused but returned known family %q", tc.cc, got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestTagPrefixConstants(t *testing.T) {
	if TagPrefixVRAM != "vram:" {
		t.Errorf("TagPrefixVRAM = %q, want %q", TagPrefixVRAM, "vram:")
	}
	if TagPrefixCC != "cc:" {
		t.Errorf("TagPrefixCC = %q, want %q", TagPrefixCC, "cc:")
	}
	if TagPrefixTP != "tp:" {
		t.Errorf("TagPrefixTP = %q, want %q", TagPrefixTP, "tp:")
	}
	if TagPrefixMIG != "mig:" {
		t.Errorf("TagPrefixMIG = %q, want %q", TagPrefixMIG, "mig:")
	}
}

// fak-test:runtime fast est=1s
func TestVRAMClassTag(t *testing.T) {
	for _, c := range []VRAMClass{"lt8", "8g", "16g", "32g", "64g", "96g"} {
		t.Run(string(c), func(t *testing.T) {
			got, err := VRAMClassTag(c)
			if err != nil {
				t.Fatalf("VRAMClassTag(%q) error = %v, want nil", c, err)
			}
			if want := TagPrefixVRAM + string(c); got != want {
				t.Fatalf("VRAMClassTag(%q) = %q, want %q", c, got, want)
			}
		})
	}
	for _, bad := range []VRAMClass{"", "8g+", "12g", "LT8", "lt9", "volta2"} {
		t.Run("refuses_"+string(bad), func(t *testing.T) {
			got, err := VRAMClassTag(bad)
			if err == nil {
				t.Fatalf("VRAMClassTag(%q) = %q, nil; want refusal", bad, got)
			}
			if !errors.Is(err, ErrInvalidDiscoveryFact) {
				t.Fatalf("VRAMClassTag(%q) error does not wrap ErrInvalidDiscoveryFact: %v", bad, err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCCFamilyTag(t *testing.T) {
	for _, f := range []CCFamily{"pascal", "volta", "turing", "ampere", "ada", "hopper", "blackwell"} {
		t.Run(string(f), func(t *testing.T) {
			got, err := CCFamilyTag(f)
			if err != nil {
				t.Fatalf("CCFamilyTag(%q) error = %v, want nil", f, err)
			}
			if want := TagPrefixCC + string(f); got != want {
				t.Fatalf("CCFamilyTag(%q) = %q, want %q", f, got, want)
			}
		})
	}
	for _, bad := range []CCFamily{"", "volta2", "PASCAL", "gpu", "hopper2"} {
		t.Run("refuses_"+string(bad), func(t *testing.T) {
			got, err := CCFamilyTag(bad)
			if err == nil {
				t.Fatalf("CCFamilyTag(%q) = %q, nil; want refusal", bad, got)
			}
			if !errors.Is(err, ErrInvalidDiscoveryFact) {
				t.Fatalf("CCFamilyTag(%q) error does not wrap ErrInvalidDiscoveryFact: %v", bad, err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestTPTag(t *testing.T) {
	cases := []struct {
		name    string
		width   int
		want    string
		wantErr bool
	}{
		{"one", 1, "tp:1", false},
		{"two", 2, "tp:2", false},
		{"eight", 8, "tp:8", false},
		{"one-thousand-twenty-four", 1024, "tp:1024", false},
		{"max int32", math.MaxInt32, "tp:2147483647", false},
		{"zero refused", 0, "", true},
		{"negative one refused", -1, "", true},
		{"min int32 refused", math.MinInt32, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := TPTag(tc.width)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("TPTag(%d) = %q, nil; want refusal", tc.width, got)
				}
				if !errors.Is(err, ErrInvalidDiscoveryFact) {
					t.Fatalf("TPTag(%d) error does not wrap ErrInvalidDiscoveryFact: %v", tc.width, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("TPTag(%d) error = %v, want nil", tc.width, err)
			}
			if got != tc.want {
				t.Fatalf("TPTag(%d) = %q, want %q", tc.width, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestMIGProfileTag(t *testing.T) {
	cases := []struct {
		name    string
		profile string
		want    string
		wantErr bool
	}{
		{"trimmed and lowercased", " 1G.5GB ", "mig:1g.5gb", false},
		{"upper case lowered", "1G.5GB", "mig:1g.5gb", false},
		{"already canonical", "1g.5gb", "mig:1g.5gb", false},
		{"tab and newline trimmed", "\t1g.5gb\n", "mig:1g.5gb", false},
		{"non-whitespace odd chars pass through", "1g_5gb", "mig:1g_5gb", false},
		{"empty refused", "", "", true},
		{"spaces only refused", "   ", "", true},
		{"tabs and newlines only refused", "\t\n", "", true},
		{"interior space refused", "1g 5gb", "", true},
		{"interior double space refused", "1g  5gb", "", true},
		{"interior tab refused", "1g\t5gb", "", true},
		{"interior space with padding refused", " 1g 5gb ", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MIGProfileTag(tc.profile)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("MIGProfileTag(%q) = %q, nil; want refusal", tc.profile, got)
				}
				if !errors.Is(err, ErrInvalidDiscoveryFact) {
					t.Fatalf("MIGProfileTag(%q) error does not wrap ErrInvalidDiscoveryFact: %v", tc.profile, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("MIGProfileTag(%q) error = %v, want nil", tc.profile, err)
			}
			if got != tc.want {
				t.Fatalf("MIGProfileTag(%q) = %q, want %q", tc.profile, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestTagsFromDiscovery(t *testing.T) {
	cases := []struct {
		name string
		d    EngineDiscovery
		want []string
	}{
		{"zero value emits nothing", EngineDiscovery{}, nil},
		{
			"full facts in canonical order vram cc tp mig",
			EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: 8, MIGProfile: "1g.5gb"},
			[]string{"vram:64g", "cc:hopper", "tp:8", "mig:1g.5gb"},
		},
		{"vram only", EngineDiscovery{VRAMGiB: 24}, []string{"vram:16g"}},
		{"cc only", EngineDiscovery{ComputeCapability: "8.6"}, []string{"cc:ampere"}},
		{"tp only", EngineDiscovery{TPWidth: 2}, []string{"tp:2"}},
		{"mig only", EngineDiscovery{MIGProfile: " 1G.5GB "}, []string{"mig:1g.5gb"}},
		{
			"absent tp and mig emit nothing",
			EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0"},
			[]string{"vram:64g", "cc:hopper"},
		},
		{
			"absent vram and cc emit nothing",
			EngineDiscovery{TPWidth: 4, MIGProfile: "1g.5gb"},
			[]string{"tp:4", "mig:1g.5gb"},
		},
		{
			"vram zero is absent not lt8",
			EngineDiscovery{VRAMGiB: 0, TPWidth: 8},
			[]string{"tp:8"},
		},
		{
			"tp zero is absent and does not error",
			EngineDiscovery{VRAMGiB: 24, TPWidth: 0},
			[]string{"vram:16g"},
		},
		{
			"mig empty is absent and does not error",
			EngineDiscovery{VRAMGiB: 24, MIGProfile: ""},
			[]string{"vram:16g"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := TagsFromDiscovery(tc.d)
			if err != nil {
				t.Fatalf("TagsFromDiscovery(%+v) error = %v, want nil", tc.d, err)
			}
			if tc.want == nil {
				if got != nil {
					t.Fatalf("TagsFromDiscovery(%+v) = %v, want nil", tc.d, got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("TagsFromDiscovery(%+v) = %v, want %v", tc.d, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestTagsFromDiscoveryInvalidFacts(t *testing.T) {
	cases := []struct {
		name string
		d    EngineDiscovery
	}{
		{"negative vram", EngineDiscovery{VRAMGiB: -1}},
		{"malformed cc", EngineDiscovery{ComputeCapability: "8"}},
		{"whitespace cc", EngineDiscovery{ComputeCapability: " 8.6"}},
		{"unmapped cc", EngineDiscovery{ComputeCapability: "8.3"}},
		{"negative tp", EngineDiscovery{TPWidth: -1}},
		{"whitespace-only mig", EngineDiscovery{MIGProfile: "   "}},
		{"interior-whitespace mig", EngineDiscovery{MIGProfile: "1g 5gb"}},
		{
			"valid prefix facts plus later invalid fact yields no partial result",
			EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: -1},
		},
		{
			"multi invalid yields nil tags",
			EngineDiscovery{VRAMGiB: -1, ComputeCapability: "abc", TPWidth: -4, MIGProfile: "  "},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := TagsFromDiscovery(tc.d)
			if err == nil {
				t.Fatalf("TagsFromDiscovery(%+v) = %v, nil; want refusal", tc.d, got)
			}
			if !errors.Is(err, ErrInvalidDiscoveryFact) {
				t.Fatalf("TagsFromDiscovery(%+v) error does not wrap ErrInvalidDiscoveryFact: %v", tc.d, err)
			}
			if got != nil {
				t.Fatalf("TagsFromDiscovery(%+v) returned partial tags %v on refusal", tc.d, got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestPopulateTags(t *testing.T) {
	full := EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: 8, MIGProfile: "1g.5gb"}
	fullTags := []string{"vram:64g", "cc:hopper", "tp:8", "mig:1g.5gb"}

	t.Run("fresh engine gets canonical order", func(t *testing.T) {
		e := gpuTagsEngine("e1")
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() error = %v, want nil", err)
		}
		if !reflect.DeepEqual(e.Tags, fullTags) {
			t.Fatalf("Tags = %v, want %v", e.Tags, fullTags)
		}
	})

	t.Run("pre-existing tags preserved before appended in original order", func(t *testing.T) {
		e := gpuTagsEngine("e2")
		e.Tags = []string{"primary", "small"}
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() error = %v, want nil", err)
		}
		want := []string{"primary", "small", "vram:64g", "cc:hopper", "tp:8", "mig:1g.5gb"}
		if !reflect.DeepEqual(e.Tags, want) {
			t.Fatalf("Tags = %v, want %v", e.Tags, want)
		}
	})

	t.Run("already present derived tag is not duplicated", func(t *testing.T) {
		e := gpuTagsEngine("e3")
		e.Tags = []string{"cc:hopper"}
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() error = %v, want nil", err)
		}
		want := []string{"cc:hopper", "vram:64g", "tp:8", "mig:1g.5gb"}
		if !reflect.DeepEqual(e.Tags, want) {
			t.Fatalf("Tags = %v, want %v", e.Tags, want)
		}
	})

	t.Run("out of place existing tag stays in place", func(t *testing.T) {
		e := gpuTagsEngine("e4")
		e.Tags = []string{"tp:8", "primary"}
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() error = %v, want nil", err)
		}
		want := []string{"tp:8", "primary", "vram:64g", "cc:hopper", "mig:1g.5gb"}
		if !reflect.DeepEqual(e.Tags, want) {
			t.Fatalf("Tags = %v, want %v", e.Tags, want)
		}
	})

	t.Run("same family different value appends tag-level", func(t *testing.T) {
		e := gpuTagsEngine("e5")
		e.Tags = []string{"vram:16g"}
		if err := e.PopulateTags(EngineDiscovery{VRAMGiB: 80}); err != nil {
			t.Fatalf("PopulateTags() error = %v, want nil", err)
		}
		want := []string{"vram:16g", "vram:64g"}
		if !reflect.DeepEqual(e.Tags, want) {
			t.Fatalf("Tags = %v, want %v", e.Tags, want)
		}
	})

	t.Run("idempotent second call appends nothing", func(t *testing.T) {
		e := gpuTagsEngine("e6")
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() #1 error = %v, want nil", err)
		}
		before := append([]string(nil), e.Tags...)
		if err := e.PopulateTags(full); err != nil {
			t.Fatalf("PopulateTags() #2 error = %v, want nil", err)
		}
		if !reflect.DeepEqual(e.Tags, before) {
			t.Fatalf("Tags after second call = %v, want %v", e.Tags, before)
		}
	})

	t.Run("nil receiver is refused", func(t *testing.T) {
		var e *Engine
		if err := e.PopulateTags(full); err == nil {
			t.Fatalf("(*Engine)(nil).PopulateTags() = nil, want refusal")
		}
	})

	t.Run("invalid discovery leaves tags unchanged", func(t *testing.T) {
		bad := []struct {
			name string
			d    EngineDiscovery
		}{
			{"negative vram", EngineDiscovery{VRAMGiB: -1}},
			{"malformed cc", EngineDiscovery{ComputeCapability: "8"}},
			{"negative tp", EngineDiscovery{TPWidth: -1}},
			{"whitespace-only mig", EngineDiscovery{MIGProfile: "  "}},
			{
				"valid facts before an invalid one must not partially append",
				EngineDiscovery{VRAMGiB: 80, ComputeCapability: "abc", TPWidth: 8, MIGProfile: "1g.5gb"},
			},
			{
				"multi invalid",
				EngineDiscovery{VRAMGiB: -1, ComputeCapability: "abc", TPWidth: -4, MIGProfile: "  "},
			},
		}
		for _, tc := range bad {
			t.Run(tc.name, func(t *testing.T) {
				e := gpuTagsEngine("e7")
				e.Tags = []string{"primary"}
				before := append([]string(nil), e.Tags...)
				err := e.PopulateTags(tc.d)
				if err == nil {
					t.Fatalf("PopulateTags(%+v) = nil, want refusal", tc.d)
				}
				if !errors.Is(err, ErrInvalidDiscoveryFact) {
					t.Fatalf("PopulateTags(%+v) error does not wrap ErrInvalidDiscoveryFact: %v", tc.d, err)
				}
				if !reflect.DeepEqual(e.Tags, before) {
					t.Fatalf("PopulateTags(%+v) mutated Tags to %v, want unchanged %v", tc.d, e.Tags, before)
				}
			})
		}
	})
}

// fak-test:runtime fast est=1s
func TestRequireGPUClass(t *testing.T) {
	for _, f := range []CCFamily{"pascal", "volta", "turing", "ampere", "ada", "hopper", "blackwell"} {
		t.Run(string(f), func(t *testing.T) {
			got, err := RequireGPUClass(f)
			if err != nil {
				t.Fatalf("RequireGPUClass(%q) error = %v, want nil", f, err)
			}
			want := []string{TagPrefixCC + string(f)}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("RequireGPUClass(%q) = %v, want %v", f, got, want)
			}
		})
	}
	for _, bad := range []CCFamily{"", "volta2", "gpu", "8.6"} {
		t.Run("refuses_"+string(bad), func(t *testing.T) {
			got, err := RequireGPUClass(bad)
			if err == nil {
				t.Fatalf("RequireGPUClass(%q) = %v, nil; want refusal", bad, got)
			}
			if !errors.Is(err, ErrInvalidDiscoveryFact) {
				t.Fatalf("RequireGPUClass(%q) error does not wrap ErrInvalidDiscoveryFact: %v", bad, err)
			}
			if got != nil {
				t.Fatalf("RequireGPUClass(%q) returned %v on refusal", bad, got)
			}
		})
	}
	t.Run("result is a fresh slice", func(t *testing.T) {
		got, err := RequireGPUClass(CCFamily("hopper"))
		if err != nil {
			t.Fatalf("RequireGPUClass(hopper) error = %v, want nil", err)
		}
		got[0] = "mutated"
		again, err := RequireGPUClass(CCFamily("hopper"))
		if err != nil {
			t.Fatalf("RequireGPUClass(hopper) #2 error = %v, want nil", err)
		}
		if !reflect.DeepEqual(again, []string{"cc:hopper"}) {
			t.Fatalf("RequireGPUClass(hopper) #2 = %v, want [cc:hopper]", again)
		}
	})
}

// fak-test:runtime fast est=1s
func TestGPUClassSelectionThroughFabric(t *testing.T) {
	hopper := gpuTagsEngine("h80")
	if err := hopper.PopulateTags(EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: 8}); err != nil {
		t.Fatalf("PopulateTags(hopper) error = %v", err)
	}
	ampere := gpuTagsEngine("a24")
	if err := ampere.PopulateTags(EngineDiscovery{VRAMGiB: 24, ComputeCapability: "8.6", TPWidth: 2}); err != nil {
		t.Fatalf("PopulateTags(ampere) error = %v", err)
	}
	hopperReq, err := RequireGPUClass(CCFamily("hopper"))
	if err != nil {
		t.Fatalf("RequireGPUClass(hopper) error = %v", err)
	}
	ampereReq, err := RequireGPUClass(CCFamily("ampere"))
	if err != nil {
		t.Fatalf("RequireGPUClass(ampere) error = %v", err)
	}
	vram64Tag, err := VRAMClassTag(VRAMClass("64g"))
	vram64 := mustTag(t, vram64Tag, err)
	vram16Tag, err := VRAMClassTag(VRAMClass("16g"))
	vram16 := mustTag(t, vram16Tag, err)

	if !IsKnownReason(ReasonMissingTag) {
		t.Fatalf("ReasonMissingTag must stay in the closed rejection vocabulary")
	}

	t.Run("hopper requirement routes to hopper, ampere rejected MISSING_TAG", func(t *testing.T) {
		assignments, unrouted, err := Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T1"), Role: "worker", RequireTags: hopperReq},
		}, []Engine{*ampere, *hopper})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if len(unrouted) != 0 {
			t.Fatalf("unrouted = %v, want none", unrouted)
		}
		if len(assignments) != 1 {
			t.Fatalf("assignments = %d, want 1", len(assignments))
		}
		a := assignments[0]
		if a.EngineID != "h80" {
			t.Fatalf("assigned engine = %q, want %q", a.EngineID, "h80")
		}
		wantRejected := []Rejection{{EngineID: "a24", Reason: ReasonMissingTag}}
		if !reflect.DeepEqual(a.Rejected, wantRejected) {
			t.Fatalf("Rejected = %v, want %v", a.Rejected, wantRejected)
		}
	})

	t.Run("ampere requirement reverses the pick", func(t *testing.T) {
		assignments, unrouted, err := Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T2"), Role: "worker", RequireTags: ampereReq},
		}, []Engine{*hopper, *ampere})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if len(unrouted) != 0 || len(assignments) != 1 {
			t.Fatalf("assignments=%d unrouted=%v, want 1 and none", len(assignments), unrouted)
		}
		if assignments[0].EngineID != "a24" {
			t.Fatalf("assigned engine = %q, want %q", assignments[0].EngineID, "a24")
		}
		wantRejected := []Rejection{{EngineID: "h80", Reason: ReasonMissingTag}}
		if !reflect.DeepEqual(assignments[0].Rejected, wantRejected) {
			t.Fatalf("Rejected = %v, want %v", assignments[0].Rejected, wantRejected)
		}
	})

	t.Run("vram tag requirement matches only the right engine", func(t *testing.T) {
		assignments, unrouted, err := Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T3"), Role: "worker", RequireTags: []string{vram64}},
		}, []Engine{*ampere, *hopper})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if len(unrouted) != 0 || len(assignments) != 1 {
			t.Fatalf("assignments=%d unrouted=%v, want 1 and none", len(assignments), unrouted)
		}
		if assignments[0].EngineID != "h80" {
			t.Fatalf("assigned engine = %q, want %q", assignments[0].EngineID, "h80")
		}
		wantRejected := []Rejection{{EngineID: "a24", Reason: ReasonMissingTag}}
		if !reflect.DeepEqual(assignments[0].Rejected, wantRejected) {
			t.Fatalf("Rejected = %v, want %v", assignments[0].Rejected, wantRejected)
		}

		assignments, unrouted, err = Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T3b"), Role: "worker", RequireTags: []string{vram16}},
		}, []Engine{*ampere, *hopper})
		if err != nil {
			t.Fatalf("Route() vram16 error = %v", err)
		}
		if len(unrouted) != 0 || len(assignments) != 1 || assignments[0].EngineID != "a24" {
			t.Fatalf("vram16: assignments=%v unrouted=%v, want a24 placed", assignments, unrouted)
		}
	})

	t.Run("superset of tags is still eligible", func(t *testing.T) {
		sup := gpuTagsEngine("sup")
		if err := sup.PopulateTags(EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: 8}); err != nil {
			t.Fatalf("PopulateTags(sup) error = %v", err)
		}
		sup.Tags = append([]string{"primary", "small"}, sup.Tags...)
		sup.Tags = append(sup.Tags, "extra")
		if !sup.HasTag(hopperReq) {
			t.Fatalf("HasTag(%v) = false on superset engine with tags %v", hopperReq, sup.Tags)
		}
		assignments, unrouted, err := Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T4"), Role: "worker", RequireTags: hopperReq},
		}, []Engine{*sup})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if len(unrouted) != 0 || len(assignments) != 1 || assignments[0].EngineID != "sup" {
			t.Fatalf("assignments=%v unrouted=%v, want sup placed", assignments, unrouted)
		}
	})

	t.Run("no matching engine is unrouted with MISSING_TAG rejections", func(t *testing.T) {
		assignments, unrouted, err := Route([]TaskRequest{
			{Candidate: gpuTagsCandidate("T5"), Role: "worker", RequireTags: hopperReq},
		}, []Engine{*ampere})
		if err != nil {
			t.Fatalf("Route() error = %v", err)
		}
		if len(assignments) != 0 {
			t.Fatalf("assignments = %v, want none", assignments)
		}
		if len(unrouted) != 1 || unrouted[0].Reason != ReasonNoEligibleEngine {
			t.Fatalf("unrouted = %v, want one %s", unrouted, ReasonNoEligibleEngine)
		}
		wantRejected := []Rejection{{EngineID: "a24", Reason: ReasonMissingTag}}
		if !reflect.DeepEqual(unrouted[0].Rejected, wantRejected) {
			t.Fatalf("Rejected = %v, want %v", unrouted[0].Rejected, wantRejected)
		}
	})
}

// fak-test:runtime fast est=1s
func TestVRAMClassVocabularyClosed(t *testing.T) {
	got := VRAMClasses()
	if !reflect.DeepEqual(got, pinnedVRAMClasses) {
		t.Fatalf("VRAMClasses() = %v, want pinned vocabulary %v", got, pinnedVRAMClasses)
	}
	for _, c := range got {
		if !c.IsKnown() {
			t.Errorf("IsKnown(%q) = false, want true for every listed member", c)
		}
	}
	if VRAMClass("").IsKnown() {
		t.Errorf("IsKnown(\"\") = true, want false (unknown is NOT known)")
	}
	for _, c := range got {
		if c == "" {
			t.Errorf("VRAMClasses() lists the unknown sentinel; it is not a member")
		}
	}
	for _, invented := range []VRAMClass{"volta2", "8g+", "lt9", "12g", "LT8", "96G"} {
		if invented.IsKnown() {
			t.Errorf("IsKnown(%q) = true, want false (invented value)", invented)
		}
	}
	got[1] = "mutated"
	if again := VRAMClasses(); !reflect.DeepEqual(again, pinnedVRAMClasses) {
		t.Fatalf("VRAMClasses() after mutating a returned slice = %v, want %v (must be a copy)", again, pinnedVRAMClasses)
	}
}

// fak-test:runtime fast est=1s
func TestCCFamiliesVocabularyClosed(t *testing.T) {
	got := CCFamilies()
	if !reflect.DeepEqual(got, pinnedCCFamilies) {
		t.Fatalf("CCFamilies() = %v, want pinned vocabulary %v", got, pinnedCCFamilies)
	}
	for _, f := range got {
		if !f.IsKnown() {
			t.Errorf("IsKnown(%q) = false, want true for every listed member", f)
		}
	}
	if CCFamily("").IsKnown() {
		t.Errorf("IsKnown(\"\") = true, want false (unknown is NOT known)")
	}
	for _, f := range got {
		if f == "" {
			t.Errorf("CCFamilies() lists the unknown sentinel; it is not a member")
		}
	}
	for _, invented := range []CCFamily{"volta2", "8g+", "PASCAL", "hopper2", "blackwell-2"} {
		if invented.IsKnown() {
			t.Errorf("IsKnown(%q) = true, want false (invented value)", invented)
		}
	}
	got[1] = "mutated"
	if again := CCFamilies(); !reflect.DeepEqual(again, pinnedCCFamilies) {
		t.Fatalf("CCFamilies() after mutating a returned slice = %v, want %v (must be a copy)", again, pinnedCCFamilies)
	}
}

// fak-test:runtime fast est=1s
func TestEngineDiscoveryJSON(t *testing.T) {
	full := EngineDiscovery{VRAMGiB: 80, ComputeCapability: "9.0", TPWidth: 8, MIGProfile: "1g.5gb"}

	b, err := json.Marshal(full)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	const fullJSON = `{"vram_gib":80,"compute_capability":"9.0","tp_width":8,"mig_profile":"1g.5gb"}`
	if string(b) != fullJSON {
		t.Fatalf("Marshal() = %s, want %s", b, fullJSON)
	}

	var back EngineDiscovery
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if !reflect.DeepEqual(back, full) {
		t.Fatalf("round trip = %+v, want %+v", back, full)
	}

	zero, err := json.Marshal(EngineDiscovery{})
	if err != nil {
		t.Fatalf("Marshal(zero) error = %v", err)
	}
	if string(zero) != "{}" {
		t.Fatalf("Marshal(zero) = %s, want {} (all fields omitempty)", zero)
	}

	partial, err := json.Marshal(EngineDiscovery{TPWidth: 2})
	if err != nil {
		t.Fatalf("Marshal(partial) error = %v", err)
	}
	if string(partial) != `{"tp_width":2}` {
		t.Fatalf("Marshal(partial) = %s, want {\"tp_width\":2}", partial)
	}

	var fromLiteral EngineDiscovery
	if err := json.Unmarshal([]byte(fullJSON), &fromLiteral); err != nil {
		t.Fatalf("Unmarshal(literal) error = %v", err)
	}
	if !reflect.DeepEqual(fromLiteral, full) {
		t.Fatalf("Unmarshal(literal) = %+v, want %+v", fromLiteral, full)
	}
}

// fak-test:runtime fast est=1s
// TestGPUTagsContractAmbiguityProbes pins readings the contract states
// loosely. A failure here is a CONTRACT AMBIGUITY finding for the operator,
// not necessarily an implementation bug:
//
//   - "6.x" / "10.x, 12.x": wildcard minor (probed here) vs only real minors.
//   - zero-padded well-formed CCs ("08.6", "8.06"): parsed numerically
//     (probed here) vs exact string match against the table.
//   - nil-receiver PopulateTags: "every refusal above wraps it" (probed here)
//     vs a plain error, since the sentinel is named "invalid discovery fact".
func TestGPUTagsContractAmbiguityProbes(t *testing.T) {
	t.Run("cc wildcard minor 6.5 maps to pascal", func(t *testing.T) {
		got, err := CCFamilyFor("6.5")
		if err != nil || got != CCFamily("pascal") {
			t.Fatalf("CCFamilyFor(\"6.5\") = %q, %v; want pascal, nil (wildcard-minor reading)", got, err)
		}
	})
	t.Run("cc wildcard minor 10.5 maps to blackwell", func(t *testing.T) {
		got, err := CCFamilyFor("10.5")
		if err != nil || got != CCFamily("blackwell") {
			t.Fatalf("CCFamilyFor(\"10.5\") = %q, %v; want blackwell, nil (wildcard-minor reading)", got, err)
		}
	})
	t.Run("cc wildcard minor 12.9 maps to blackwell", func(t *testing.T) {
		got, err := CCFamilyFor("12.9")
		if err != nil || got != CCFamily("blackwell") {
			t.Fatalf("CCFamilyFor(\"12.9\") = %q, %v; want blackwell, nil (wildcard-minor reading)", got, err)
		}
	})
	t.Run("cc zero-padded major 08.6 maps to ampere", func(t *testing.T) {
		got, err := CCFamilyFor("08.6")
		if err != nil || got != CCFamily("ampere") {
			t.Fatalf("CCFamilyFor(\"08.6\") = %q, %v; want ampere, nil (numeric-parse reading)", got, err)
		}
	})
	t.Run("cc zero-padded minor 8.06 maps to ampere", func(t *testing.T) {
		got, err := CCFamilyFor("8.06")
		if err != nil || got != CCFamily("ampere") {
			t.Fatalf("CCFamilyFor(\"8.06\") = %q, %v; want ampere, nil (numeric-parse reading)", got, err)
		}
	})
	t.Run("nil receiver PopulateTags wraps ErrInvalidDiscoveryFact", func(t *testing.T) {
		var e *Engine
		err := e.PopulateTags(EngineDiscovery{})
		if !errors.Is(err, ErrInvalidDiscoveryFact) {
			t.Fatalf("(*Engine)(nil).PopulateTags() error does not wrap ErrInvalidDiscoveryFact: %v", err)
		}
	})
}
