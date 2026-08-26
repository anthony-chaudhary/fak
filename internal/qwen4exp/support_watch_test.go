package qwen4exp

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
)

type supportWatchFixture struct {
	Schema       string              `json:"schema"`
	Baseline     map[string]string   `json:"baseline"`
	Observed     map[string]string   `json:"observed"`
	Dependencies map[string][]string `json:"dependencies"`
	WantChanged  []string            `json:"want_changed"`
	WantStale    []string            `json:"want_stale"`
}

func TestQwen4ExpSupportWatchDocument(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve support watch test path")
	}
	packageDir := filepath.Dir(filename)
	docPath := filepath.Join(packageDir, "..", "..", "docs", "notes", "QWEN4EXP-SUPPORT-ROLLBACK-WATCH-2026-08-26.md")
	docBefore, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatal(err)
	}

	required := []string{
		"[#9214](https://github.com/anthony-chaudhary/fak/issues/9214)",
		"[#9204](https://github.com/anthony-chaudhary/fak/issues/9204)",
		"[#9122](https://github.com/anthony-chaudhary/fak/issues/9122)",
		"study_207f3c56d6e23d2ccfb0d0881fde3a3a8ca1f81d7952897d1a87f61a61a4d383",
		"513aa6e18a335296fc13e538232a8735b230877d",
		"f5d08274bafd880402bd16f5e3e6c514136ec06c",
		"## Milestones and phase exits",
		"## Support matrix",
		"## Exact launch and comparator recipes",
		"## Observability and immutable receipt contract",
		"## Known limits and rejection reasons",
		"## Rollback",
		"## Upstream watch and stale transition",
		"Transformers",
		"vLLM",
		"SGLang",
		"llama.cpp",
		"MLX",
		"engine `fak-native/qwen4exp`",
		"`QWEN4EXP_UPSTREAM_STALE`",
	}
	for _, text := range required {
		if !bytes.Contains(docBefore, []byte(text)) {
			t.Errorf("support watch document missing %q", text)
		}
	}

	var fixture supportWatchFixture
	fixtureBytes, err := os.ReadFile(filepath.Join(packageDir, "testdata", "support_watch_changed.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fixtureBytes, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "fak-qwen4exp-support-watch/1" {
		t.Fatalf("fixture schema = %q", fixture.Schema)
	}

	baselineBefore := cloneStringMap(fixture.Baseline)
	changed, stale := staleSupportRows(fixture.Baseline, fixture.Observed, fixture.Dependencies)
	if !reflect.DeepEqual(changed, fixture.WantChanged) {
		t.Errorf("changed pins = %v, want %v", changed, fixture.WantChanged)
	}
	if !reflect.DeepEqual(stale, fixture.WantStale) {
		t.Errorf("stale rows = %v, want %v", stale, fixture.WantStale)
	}
	if !reflect.DeepEqual(fixture.Baseline, baselineBefore) {
		t.Fatal("stale evaluation rewrote the immutable baseline")
	}

	docAfter, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(docBefore, docAfter) {
		t.Fatal("stale evaluation rewrote the historical support document")
	}

	for _, runtimeName := range []string{"transformers", "vllm", "sglang", "llamacpp", "mlx"} {
		if _, ok := fixture.Baseline[runtimeName]; !ok {
			t.Errorf("upstream watch omits %s", runtimeName)
		}
	}
	for _, line := range strings.Split(string(docBefore), "\n") {
		if strings.Contains(line, "https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/") && !strings.Contains(line, "513aa6e18a335296fc13e538232a8735b230877d") {
			t.Errorf("upstream repository link is not immutable: %s", line)
		}
		if strings.Contains(line, "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/") && !strings.Contains(line, "f5d08274bafd880402bd16f5e3e6c514136ec06c") {
			t.Errorf("checkpoint link is not immutable: %s", line)
		}
	}
}

func staleSupportRows(baseline, observed map[string]string, dependencies map[string][]string) ([]string, []string) {
	changedSet := make(map[string]struct{})
	staleSet := make(map[string]struct{})
	for source, oldPin := range baseline {
		if observed[source] == oldPin {
			continue
		}
		changedSet[source] = struct{}{}
		for _, row := range dependencies[source] {
			staleSet[row] = struct{}{}
		}
	}
	return sortedKeys(changedSet), sortedKeys(staleSet)
}

func sortedKeys(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func cloneStringMap(source map[string]string) map[string]string {
	clone := make(map[string]string, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}
