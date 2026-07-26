package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func meta(keys ...string) map[string]ggufload.Value {
	m := make(map[string]ggufload.Value, len(keys))
	for _, k := range keys {
		m[k] = ggufload.Value{}
	}
	return m
}

// TestSelectKeysNoFilterSelectsEverything pins the distinction between "no
// filter" and "a filter that matched nothing". Collapsing the two would make a
// typo'd -grep silently print the whole header instead of an empty result.
func TestSelectKeysNoFilterSelectsEverything(t *testing.T) {
	m := meta("general.architecture", "tokenizer.ggml.model", "llama.block_count")

	all := selectKeys(m, nil)
	if len(all) != 3 {
		t.Errorf("no filter selected %d keys, want all 3: %v", len(all), all)
	}

	none := selectKeys(m, []string{"no-such-key"})
	if len(none) != 0 {
		t.Errorf("a non-matching filter selected %v, want nothing", none)
	}
}

func TestSelectKeysIsSorted(t *testing.T) {
	// Map iteration order is randomized, so an unsorted implementation would
	// produce a different diff on every run against the same checkpoint.
	m := meta("zeta", "alpha", "mid", "beta")
	got := selectKeys(m, nil)
	want := []string{"alpha", "beta", "mid", "zeta"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("selectKeys = %v, want %v", got, want)
	}
}

func TestSelectKeysMatchesAnySubstring(t *testing.T) {
	m := meta(
		"general.architecture",
		"tokenizer.ggml.model",
		"tokenizer.chat_template",
		"llama.block_count",
		"llama.attention.head_count",
	)
	got := selectKeys(m, []string{"tokenizer", "block_count"})
	want := []string{"llama.block_count", "tokenizer.chat_template", "tokenizer.ggml.model"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("selectKeys = %v, want %v", got, want)
	}
}

// TestSplitFiltersIgnoresEmptyFields: a trailing comma or a stray space must not
// produce an empty-string filter, which strings.Contains matches against EVERY
// key — silently turning a narrow -grep into "print everything".
func TestSplitFiltersIgnoresEmptyFields(t *testing.T) {
	cases := map[string][]string{
		"":                   {},
		",":                  {},
		"   ":                {},
		"tokenizer":          {"tokenizer"},
		"tokenizer,":         {"tokenizer"},
		" tokenizer , chat ": {"tokenizer", "chat"},
		",,arch,,":           {"arch"},
	}
	for in, want := range cases {
		got := splitFilters(in)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("splitFilters(%q) = %v, want %v", in, got, want)
		}
	}

	// The consequence, stated directly: an empty filter list means "everything",
	// so splitFilters must never hand selectKeys a list containing "".
	m := meta("a", "b")
	if got := selectKeys(m, splitFilters("tokenizer,")); len(got) != 0 {
		t.Errorf("a trailing comma widened the filter to %v", got)
	}
}

func TestMatchesAny(t *testing.T) {
	if matchesAny("general.architecture", nil) {
		t.Error("an empty filter list must not match via matchesAny (selectKeys handles the everything case)")
	}
	if !matchesAny("tokenizer.ggml.model", []string{"nope", "ggml"}) {
		t.Error("a later filter must still match")
	}
	if matchesAny("general.name", []string{"tokenizer"}) {
		t.Error("unexpected match")
	}
}
