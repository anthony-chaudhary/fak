package harnesscompose

import (
	"reflect"
	"strings"
	"testing"
)

func TestMixedLayerCompositionDeterministic(t *testing.T) {
	manifest := Manifest{Schema: Schema, Layers: []Layer{
		{ID: "company", Scope: "company", Assets: []Asset{
			{Kind: "policy", ID: "tools", Grants: []string{"search", "shell"}, Denies: []string{"refund"}},
			{Kind: "secret", ID: "legal-db", Ref: "vault://legal/read", Lock: true},
			{Kind: "workflow", ID: "audit", Value: "record-every-call", Mandatory: true},
		}},
		{ID: "person", Scope: "person", Assets: []Asset{{Kind: "ui", ID: "density", Value: "compact"}}},
		{ID: "project", Scope: "project", Assets: []Asset{{Kind: "memory", ID: "matter", Boundary: "project", Value: "matter-7"}, {Kind: "policy", ID: "tools", Denies: []string{"shell"}}}},
		{ID: "legal", Scope: "domain", Assets: []Asset{{Kind: "instruction", ID: "citations", Value: "cite-primary-sources"}, {Kind: "tool", ID: "research", Value: "legal-search"}, {Kind: "route", ID: "model", Value: "legal-reviewed"}}},
		{ID: "task", Scope: "task", Assets: []Asset{{Kind: "ui", ID: "density", Operation: "replace", Value: "focused"}}},
	}}
	selected := []string{"company", "person", "project", "legal", "task"}
	got, err := Compose(manifest, selected)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Layers, selected) {
		t.Fatalf("layers=%v", got.Layers)
	}
	wantKinds := []string{"instruction", "memory", "policy", "route", "secret", "tool", "ui", "workflow"}
	var kinds []string
	for _, asset := range got.Assets {
		kinds = append(kinds, asset.Kind)
	}
	if !reflect.DeepEqual(kinds, wantKinds) {
		t.Fatalf("kinds=%v want=%v", kinds, wantKinds)
	}
	for _, asset := range got.Assets {
		if asset.Kind == "policy" && (!reflect.DeepEqual(asset.Grants, []string{"search"}) || !reflect.DeepEqual(asset.Denies, []string{"refund", "shell"})) {
			t.Fatalf("policy=%#v", asset)
		}
		if asset.Kind == "ui" && (asset.Value != "focused" || asset.Source != "task") {
			t.Fatalf("ui=%#v", asset)
		}
	}
}

func TestPolicyWideningFailsClosed(t *testing.T) {
	m := Manifest{Schema: Schema, Layers: []Layer{
		{ID: "company", Scope: "company", Assets: []Asset{{Kind: "policy", ID: "tools", Grants: []string{"search"}, Denies: []string{"shell"}}}},
		{ID: "task", Scope: "task", Assets: []Asset{{Kind: "policy", ID: "tools", Grants: []string{"shell"}}}},
	}}
	_, err := Compose(m, []string{"company", "task"})
	if err == nil || !strings.Contains(err.Error(), "privilege widening") || !strings.Contains(err.Error(), `layer "task" policy/tools`) {
		t.Fatalf("err=%v", err)
	}
}

func TestSecretReplacementAndMandatoryRemovalFailClosed(t *testing.T) {
	for name, tc := range map[string]struct {
		later Asset
		want  string
	}{
		"secret":  {later: Asset{Kind: "secret", ID: "db", Operation: "replace", Ref: "vault://other"}, want: "secret references cannot be replaced"},
		"witness": {later: Asset{Kind: "workflow", ID: "audit", Operation: "remove"}, want: "cannot remove locked or mandatory"},
	} {
		t.Run(name, func(t *testing.T) {
			base := Asset{Kind: "secret", ID: "db", Ref: "vault://company"}
			if name == "witness" {
				base = Asset{Kind: "workflow", ID: "audit", Value: "record", Mandatory: true}
			}
			m := Manifest{Schema: Schema, Layers: []Layer{{ID: "company", Scope: "company", Assets: []Asset{base}}, {ID: "task", Scope: "task", Assets: []Asset{tc.later}}}}
			_, err := Compose(m, []string{"company", "task"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestInstructionToolAmbiguityRequiresExplicitReplace(t *testing.T) {
	for _, kind := range []string{"instruction", "tool"} {
		t.Run(kind, func(t *testing.T) {
			m := Manifest{Schema: Schema, Layers: []Layer{{ID: "team", Scope: "team", Assets: []Asset{{Kind: kind, ID: "shared", Value: "one"}}}, {ID: "project", Scope: "project", Assets: []Asset{{Kind: kind, ID: "shared", Value: "two"}}}}}
			_, err := Compose(m, []string{"team", "project"})
			if err == nil || !strings.Contains(err.Error(), "ambiguous duplicate requires explicit replace") {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

func TestMemoryCannotCrossProjectOrDomainBoundary(t *testing.T) {
	raw := `{"schema":"fak.harness-assets/v1alpha1","layers":[{"id":"matter-7","scope":"project","assets":[{"kind":"memory","id":"matter","boundary":"matter-8","value":"private"}]}]}`
	_, err := Parse([]byte(raw))
	if err == nil || !strings.Contains(err.Error(), "must equal owning project layer") {
		t.Fatalf("err=%v", err)
	}
}

func TestAssetKindOperationMatrix(t *testing.T) {
	kinds := []string{"instruction", "tool", "memory", "policy", "route", "secret", "workflow", "ui"}
	operations := []string{"add", "replace", "remove"}
	for _, kind := range kinds {
		for _, operation := range operations {
			t.Run(kind+"/"+operation, func(t *testing.T) {
				base := validAsset(kind, "shared", "company")
				later := validAsset(kind, "shared", "task")
				if operation == "add" && kind != "policy" {
					later.Value += "-different"
				}
				later.Operation = operation
				if kind == "policy" {
					later.Grants = nil
					later.Denies = []string{"search"}
				}
				m := Manifest{Schema: Schema, Layers: []Layer{{ID: "company", Scope: "company", Assets: []Asset{base}}, {ID: "task", Scope: "task", Assets: []Asset{later}}}}
				_, err := Compose(m, []string{"company", "task"})
				switch operation {
				case "add":
					if kind != "policy" && err == nil {
						t.Fatal("duplicate add unexpectedly succeeded")
					}
				case "replace":
					if kind == "secret" && err == nil {
						t.Fatal("secret replacement unexpectedly succeeded")
					}
				case "remove":
					if err != nil {
						t.Fatalf("remove err=%v", err)
					}
				}
			})
		}
	}
}

func validAsset(kind, id, boundary string) Asset {
	a := Asset{Kind: kind, ID: id, Value: "value"}
	switch kind {
	case "secret":
		a.Value = ""
		a.Ref = "vault://" + boundary + "/" + id
	case "memory":
		a.Boundary = boundary
	case "policy":
		a.Value = ""
		a.Grants = []string{"search", "shell"}
	}
	return a
}
