package harnessselect

import (
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestParseRequiresOneManifest(t *testing.T) {
	t.Parallel()
	const raw = `{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"company","scope":"company","lock":["audit"]}]}`
	want := Manifest{Schema: Schema, Layers: []Layer{{ID: "company", Scope: "company", Lock: []string{"audit"}}}}
	for _, suffix := range []string{"", " \t\r\n"} {
		got, err := Parse([]byte(raw + suffix))
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("Parse with JSON whitespace %q = %#v, %v; want %#v", suffix, got, err, want)
		}
	}
	for _, suffix := range []string{
		` {"layers":[{"id":"task","scope":"task","remove":["audit"]}]}`,
		` {}`, ` null`, ` 1`, ` ]`, ` garbage`, "\u00a0", "\xff",
	} {
		got, err := Parse([]byte(raw + suffix))
		if err == nil || !reflect.DeepEqual(got, Manifest{}) || !strings.HasPrefix(err.Error(), "parse manifest:") {
			t.Errorf("Parse with trailing %q = %#v, %v; want zero manifest and parse error", suffix, got, err)
		}
	}
}
