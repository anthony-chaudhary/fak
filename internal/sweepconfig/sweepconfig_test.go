package sweepconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func sampleProfile() SweepProfile {
	return SweepProfile{
		Name:        "nightly",
		Description: "the nightly sweep",
		Models: []ModelConfig{
			{
				Name:      "zai/glm-4.7-flash",
				Provider:  "zai",
				BaseURL:   "https://api.example/v1",
				APIKeyEnv: "ZAI_KEY",
				PriceHint: &PriceHint{Input: 0.5, Output: 1.5, Source: "docs"},
				Enabled:   true,
			},
			{Name: "local/qwen", Provider: "local", LocalShim: "shims/qwen.sh", Enabled: false},
		},
		Workload:  WorkloadConfig{MaxTurns: 20, Trials: 3, TimeoutS: 900, TranscriptPath: "t/x.jsonl"},
		OutputDir: "out/here",
		SkipAPI:   true,
		Tags:      []string{"a", "b"},
		Public:    false,
	}
}

func TestSaveThenLoadRecoversEveryField(t *testing.T) {
	p := sampleProfile()
	path := filepath.Join(t.TempDir(), "nightly.yaml")
	if err := SaveProfile(p, path); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, p) {
		t.Fatalf("profile mismatch\ngot:  %+v\nwant: %+v", got, p)
	}
}

func TestMinimalJSONProfileUsesDocumentedDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bare.json")
	b, _ := json.Marshal(map[string]any{"name": "bare"})
	if err := osWrite(path, b); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "bare" || got.Description != "" || len(got.Models) != 0 || got.OutputDir != "fak/experiments/agent-live/sweep" || got.SkipAPI || !got.Public {
		t.Fatalf("defaults wrong: %+v", got)
	}
	if got.Workload.MaxTurns != 12 || got.Workload.Trials != 1 || got.Workload.TimeoutS != 600 || got.Workload.TranscriptPath != "" {
		t.Fatalf("workload defaults wrong: %+v", got.Workload)
	}
}

func TestModelProviderDefaultsToUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	b, _ := json.Marshal(map[string]any{"name": "p", "models": []any{map[string]any{"name": "x"}}})
	if err := osWrite(path, b); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Models) != 1 || got.Models[0].Provider != "unknown" || !got.Models[0].Enabled || got.Models[0].PriceHint != nil {
		t.Fatalf("model defaults wrong: %+v", got.Models)
	}
}

func TestProfilePaths(t *testing.T) {
	dir := t.TempDir()
	if err := osWrite(filepath.Join(dir, "p.yml"), []byte("name: p\n")); err != nil {
		t.Fatal(err)
	}
	if got := GetProfilePath("p", dir); got != filepath.Join(dir, "p.yml") {
		t.Fatalf("path = %q", got)
	}
	if err := osWrite(filepath.Join(dir, "p.yaml"), []byte("name: p\n")); err != nil {
		t.Fatal(err)
	}
	if got := GetProfilePath("p", dir); got != filepath.Join(dir, "p.yaml") {
		t.Fatalf("path = %q", got)
	}
	if got := GetProfilePath("nope", dir); got != filepath.Join(dir, "nope.yaml") {
		t.Fatalf("absent path = %q", got)
	}
}

func TestListProfilesLoadsAllInDir(t *testing.T) {
	dir := t.TempDir()
	if err := SaveProfile(DefaultProfile("one"), filepath.Join(dir, "one.yaml")); err != nil {
		t.Fatal(err)
	}
	if err := SaveProfile(DefaultProfile("two"), filepath.Join(dir, "two.yaml")); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, p := range ListProfiles(dir) {
		names = append(names, p.Name)
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, []string{"one", "two"}) {
		t.Fatalf("names = %+v", names)
	}
}

func TestExistingSimpleYAMLShapeLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quick.yaml")
	body := []byte("name: quick\nworkload:\n  max_turns: 5\nmodels:\n  - name: m\n    provider: p\n    price_hint:\n      input: 0.07\n      output: 0.40\n      source: docs\ntags:\n  - smoke\npublic: true\n")
	if err := osWrite(path, body); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "quick" || got.Workload.MaxTurns != 5 || len(got.Models) != 1 || got.Models[0].PriceHint == nil || got.Tags[0] != "smoke" {
		t.Fatalf("parsed yaml wrong: %+v", got)
	}
}

func TestMalformedOrUnsupportedYAMLReturnsLineAwareError(t *testing.T) {
	tests := []struct {
		name string
		body string
		line int
	}{
		{name: "missing colon", body: "name: valid\nthis is not yaml\n", line: 2},
		{name: "block scalar", body: "name: valid\ndescription: |\n  unsupported\n", line: 2},
		{name: "flow sequence", body: "name: valid\ntags: [one, two]\n", line: 2},
		{name: "wrong indentation", body: "name: valid\nworkload:\n   max_turns: 5\n", line: 3},
		{name: "unknown key", body: "name: valid\nfuture_option: value\n", line: 2},
		{name: "unterminated quote", body: "name: valid\ndescription: \"unfinished\n", line: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.yaml")
			if err := osWrite(path, []byte(tt.body)); err != nil {
				t.Fatal(err)
			}
			_, err := LoadProfile(path)
			if err == nil {
				t.Fatal("LoadProfile succeeded for malformed or unsupported YAML")
			}
			want := "line " + strconv.Itoa(tt.line)
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error %q does not contain %q", err, want)
			}
		})
	}
}

func TestQuotedScalarsAndInlineCommentsLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quoted.yaml")
	body := []byte(`name: "quick: smoke #1" # profile comment
description: 'It''s # literal: yes' # trailing comment
models:
  - name: "model:#1" # trailing comment
    provider: 'provider #1'
    base_url: "https://example.test/v1?q=a#frag" # trailing comment
    api_key_env: "KEY:VALUE"
    local_shim: 'C:\tools\shim #1'
    price_hint:
      input: 0.5 # trailing comment
      output: 1.25 # trailing comment
      source: 'docs #1'
    enabled: true # trailing comment
workload:
  max_turns: 5 # trailing comment
  trials: 2
  timeout_s: 30
  transcript_path: " transcripts/run #1.jsonl " # spaces are data
output_dir: ' out: #1 '
skip_api: false # trailing comment
skip_offline: true
skip_local_shim: false
fail_fast: true
tags:
  - "smoke #1" # trailing comment
  - 'it''s: quoted'
  - owner's-tag
public: false # trailing comment
`)
	if err := osWrite(path, body); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := SweepProfile{
		Name:        "quick: smoke #1",
		Description: "It's # literal: yes",
		Models: []ModelConfig{{
			Name:      "model:#1",
			Provider:  "provider #1",
			BaseURL:   "https://example.test/v1?q=a#frag",
			APIKeyEnv: "KEY:VALUE",
			LocalShim: `C:\tools\shim #1`,
			PriceHint: &PriceHint{Input: 0.5, Output: 1.25, Source: "docs #1"},
			Enabled:   true,
		}},
		Workload:    WorkloadConfig{MaxTurns: 5, Trials: 2, TimeoutS: 30, TranscriptPath: " transcripts/run #1.jsonl "},
		OutputDir:   " out: #1 ",
		SkipOffline: true,
		FailFast:    true,
		Tags:        []string{"smoke #1", "it's: quoted", "owner's-tag"},
		Public:      false,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("profile mismatch\ngot:  %+v\nwant: %+v", got, want)
	}
}

func TestSaveThenLoadPreservesYAMLIndicatorStrings(t *testing.T) {
	special := ` leading: # "quote" \ trailing `
	p := SweepProfile{
		Name:        special,
		Description: special,
		Models: []ModelConfig{{
			Name:      special,
			Provider:  special,
			BaseURL:   special,
			APIKeyEnv: special,
			LocalShim: special,
			PriceHint: &PriceHint{Input: 0.5, Output: 1.5, Source: special},
			Enabled:   true,
		}},
		Workload:      WorkloadConfig{MaxTurns: 7, Trials: 2, TimeoutS: 45, TranscriptPath: special},
		OutputDir:     special,
		SkipAPI:       true,
		SkipOffline:   true,
		SkipLocalShim: true,
		FailFast:      true,
		Tags:          []string{special},
		Public:        true,
	}
	path := filepath.Join(t.TempDir(), "special.yaml")
	if err := SaveProfile(p, path); err != nil {
		t.Fatal(err)
	}
	got, err := LoadProfile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, p) {
		t.Fatalf("profile mismatch\ngot:  %+v\nwant: %+v", got, p)
	}
}

func osWrite(path string, b []byte) error {
	return os.WriteFile(path, b, 0o644)
}
