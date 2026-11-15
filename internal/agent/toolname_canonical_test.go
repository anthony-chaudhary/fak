package agent

import (
	"errors"
	"testing"
)

func TestToolnameCanonical(t *testing.T) {
	t.Run("streaming_and_nonstreaming_yield_identical_canonical_name", func(t *testing.T) {
		// Same aliased call, delivered two ways: one complete completion
		// (non-streaming) and the same name split into streaming deltas.
		// Both parse paths must produce the identical canonical name.
		nonStream, err := ParseToolCallNonStreaming("functions.search_code", `{"query":"x"}`)
		if err != nil {
			t.Fatalf("non-streaming parse: %v", err)
		}
		stream, err := ParseToolCallStreaming([]string{"fun", "ctions.search", "_code"}, `{"query":"x"}`)
		if err != nil {
			t.Fatalf("streaming parse: %v", err)
		}
		if nonStream.Name != stream.Name {
			t.Fatalf("parse-path divergence: non-streaming %q != streaming %q", nonStream.Name, stream.Name)
		}
		if nonStream.Name != "search" {
			t.Fatalf("canonical name = %q, want %q", nonStream.Name, "search")
		}
	})

	t.Run("alias_resolution", func(t *testing.T) {
		cases := map[string]string{
			"web_search":         "search_web",
			"code_interpreter":   "interpreter",
			"str_replace_editor": "edit_file",
			"write_text_file":    "write_file",
			"browser_use":        "browser",
			"finish":             "finish",
		}
		for raw, want := range cases {
			got, err := CanonicalToolName(raw)
			if err != nil {
				t.Fatalf("CanonicalToolName(%q): %v", raw, err)
			}
			if got != want {
				t.Errorf("CanonicalToolName(%q) = %q, want %q", raw, got, want)
			}
		}
	})

	t.Run("prefix_normalization", func(t *testing.T) {
		cases := map[string]string{
			"functions.search_code":  "search",
			"tools.web_search":       "search_web",
			"namespace::browser_use": "browser",
			// Stacked prefixes collapse on every parse path.
			"functions.tools.search_code": "search",
		}
		for raw, want := range cases {
			got, err := CanonicalToolName(raw)
			if err != nil {
				t.Fatalf("CanonicalToolName(%q): %v", raw, err)
			}
			if got != want {
				t.Errorf("CanonicalToolName(%q) = %q, want %q", raw, got, want)
			}
		}
	})

	t.Run("plain_names_pass_through", func(t *testing.T) {
		for _, raw := range []string{"read_file", "run_command", "  search  "} {
			got, err := CanonicalToolName(raw)
			if err != nil {
				t.Fatalf("CanonicalToolName(%q): %v", raw, err)
			}
			want := raw
			if raw == "  search  " {
				want = "search"
			}
			if got != want {
				t.Errorf("CanonicalToolName(%q) = %q, want %q", raw, got, want)
			}
		}
	})

	t.Run("fail_closed_on_unknown_alias", func(t *testing.T) {
		// An unrecognized alias namespace must be refused, not passed
		// through half-normalized.
		for _, raw := range []string{"vendor.x.search", "ns::weird_tool", "functions.", "::", ""} {
			_, err := CanonicalToolName(raw)
			if !errors.Is(err, ErrUnknownToolAlias) {
				t.Errorf("CanonicalToolName(%q) error = %v, want ErrUnknownToolAlias", raw, err)
			}
		}
		// Both parse paths must refuse identically.
		if _, err := ParseToolCallNonStreaming("vendor.x.search", "{}"); !errors.Is(err, ErrUnknownToolAlias) {
			t.Errorf("non-streaming: error = %v, want ErrUnknownToolAlias", err)
		}
		if _, err := ParseToolCallStreaming([]string{"vendor.", "x.search"}, "{}"); !errors.Is(err, ErrUnknownToolAlias) {
			t.Errorf("streaming: error = %v, want ErrUnknownToolAlias", err)
		}
	})
}
