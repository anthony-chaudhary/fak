package harnessselect

import (
	"reflect"
	"strings"
	"testing"
)

func TestResolveOverlappingHarnessLayers(t *testing.T) {
	raw := `{"schema":"fak.harness-selection/v1alpha1","layers":[
		{"id":"acme-floor","scope":"company","capabilities":["audit","generic-writing"],"lock":["audit"]},
		{"id":"litigation","scope":"domain","when":{"tags":["legal"]},"capabilities":["legal-citations"],"remove":["generic-writing"]},
		{"id":"alice","scope":"person","capabilities":["alice-style"]},
		{"id":"matter-7","scope":"project","when":{"path_prefixes":["C:/matters/7"]},"capabilities":["matter-memory"]},
		{"id":"coding","scope":"domain","when":{"tags":["coding"]},"capabilities":["shell"]}
	]}`
	m, err := Parse([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(m, Context{Path: `C:\matters\7\briefs`, Tags: []string{"legal"}})
	if err != nil {
		t.Fatal(err)
	}
	wantLayers := []string{"acme-floor", "alice", "matter-7", "litigation"}
	if !reflect.DeepEqual(got.Layers, wantLayers) {
		t.Fatalf("layers = %#v, want %#v", got.Layers, wantLayers)
	}
	want := []Capability{
		{Name: "alice-style", Source: "alice"},
		{Name: "audit", Source: "acme-floor", Locked: true},
		{Name: "legal-citations", Source: "litigation"},
		{Name: "matter-memory", Source: "matter-7"},
	}
	if !reflect.DeepEqual(got.Capabilities, want) {
		t.Fatalf("capabilities = %#v, want %#v", got.Capabilities, want)
	}
}

func TestResolveNormalizesForeignPathSeparators(t *testing.T) {
	m := Manifest{Schema: Schema, Layers: []Layer{{
		ID:    "project",
		Scope: "project",
		When:  Match{PathPrefixes: []string{`C:/matters/7`}},
	}}}
	got, err := Resolve(m, Context{Path: `C:\matters\7\briefs`})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"project"}; !reflect.DeepEqual(got.Layers, want) {
		t.Fatalf("layers = %#v, want %#v", got.Layers, want)
	}
}

func TestResolveIsPermutationInvariant(t *testing.T) {
	base := Manifest{Schema: Schema, Layers: []Layer{
		{ID: "team", Scope: "team", Capabilities: []string{"review"}},
		{ID: "repo", Scope: "repo", Capabilities: []string{"tests"}},
		{ID: "person", Scope: "person", Capabilities: []string{"style"}},
	}}
	a, err := Resolve(base, Context{})
	if err != nil {
		t.Fatal(err)
	}
	base.Layers[0], base.Layers[2] = base.Layers[2], base.Layers[0]
	b, err := Resolve(base, Context{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("permuted input changed result:\n%#v\n%#v", a, b)
	}
}

func TestResolveCannotRemoveCompanyLock(t *testing.T) {
	m := Manifest{Schema: Schema, Layers: []Layer{
		{ID: "company", Scope: "company", Lock: []string{"audit"}},
		{ID: "task", Scope: "task", Remove: []string{"audit"}},
	}}
	_, err := Resolve(m, Context{})
	if err == nil || !strings.Contains(err.Error(), "cannot remove locked capability") {
		t.Fatalf("error = %v", err)
	}
}

func TestParseRejectsUnknownFields(t *testing.T) {
	_, err := Parse([]byte(`{"schema":"fak.harness-selection/v1alpha1","surprise":true}`))
	if err == nil {
		t.Fatal("expected unknown field rejection")
	}
}

func TestResolveCannotOverrideCompanyLock(t *testing.T) {
	m := Manifest{Schema: Schema, Layers: []Layer{
		{ID: "company", Scope: "company", Lock: []string{"audit"}},
		{ID: "task", Scope: "task", Capabilities: []string{"audit"}},
	}}
	got, err := Resolve(m, Context{})
	if err != nil {
		t.Fatal(err)
	}
	want := []Capability{{Name: "audit", Source: "company", Locked: true}}
	if !reflect.DeepEqual(got.Capabilities, want) {
		t.Fatalf("capabilities = %#v, want %#v", got.Capabilities, want)
	}
}
