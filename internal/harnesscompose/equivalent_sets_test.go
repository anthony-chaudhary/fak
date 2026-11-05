package harnesscompose

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestLockedPolicyRequiresExactNormalizedSets(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                string
		grants, laterGrants []string
		denies, laterDenies []string
		wantConflict        bool
	}{
		{name: "grant delimiter collision", grants: []string{"a\x00b"}, laterGrants: []string{"a", "b"}, wantConflict: true},
		{name: "deny delimiter collision", grants: []string{"read"}, laterGrants: []string{"read"}, denies: []string{"a\x00b"}, laterDenies: []string{"a", "b"}, wantConflict: true},
		{name: "normalized equivalent", grants: []string{"b", " a "}, laterGrants: []string{"a", "b", " a "}, denies: []string{"z", " y "}, laterDenies: []string{" y ", "z", "y"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest := Manifest{Schema: Schema, Layers: []Layer{
				{ID: "company", Scope: "company", Assets: []Asset{{Kind: "policy", ID: "tools", Grants: tc.grants, Denies: tc.denies, Lock: true}}},
				{ID: "task", Scope: "task", Assets: []Asset{{Kind: "policy", ID: "tools", Grants: tc.laterGrants, Denies: tc.laterDenies}}},
			}}
			// JSON escapes NUL legally; exercise the same parser as --assets.
			raw, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := Parse(raw)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Compose(parsed, []string{"company", "task"})
			if tc.wantConflict {
				if err == nil || !strings.Contains(err.Error(), `layer "task" policy/tools: cannot change locked asset from company`) || !reflect.DeepEqual(got, Result{}) {
					t.Fatalf("Compose = %#v, %v; want locked-asset conflict with no result", got, err)
				}
				return
			}
			want := []EffectiveAsset{{Kind: "policy", ID: "tools", Grants: []string{"a", "b"}, Denies: []string{"y", "z"}, Source: "company", Locked: true}}
			if err != nil || !reflect.DeepEqual(got.Assets, want) {
				t.Fatalf("Compose = %#v, %v; want preserved normalized asset %#v", got, err, want)
			}
		})
	}
}
