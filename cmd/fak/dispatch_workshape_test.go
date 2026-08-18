package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDispatchPriceBroadWorkShapeReservesParentAndChildren(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	dir := t.TempDir()
	agents := filepath.Join(dir, "agents.json")
	shape := filepath.Join(dir, "shape.json")
	if err := os.WriteFile(agents, []byte(`{"agents":[{"name":"owner","tree":["internal/x/**"]}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	contract := `{"work_shape":{"kind":"BROAD","evidence":["three independent surfaces"],"root_spine":"freeze API","integration_owner":"owner","fits_deadline":false,"packets":[{"id":"docs","role":"LEAF_CHILD","tree":["docs/**"],"witness":"doc check"},{"id":"tests","role":"LEAF_CHILD","tree":["internal/x/*_test.go"],"witness":"go test"}]}}`
	if err := os.WriteFile(shape, []byte(contract), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runDispatchPrice(&out, &errb, []string{"--workspace", dir, "--in", agents, "--work-shape", shape, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	ws := mapAt(got, "work_shape")
	if got["capacity_reserved"] != float64(3) || ws["verdict"] != "ADMIT_BROAD" || ws["parent_capacity"] != float64(1) || ws["child_capacity"] != float64(2) {
		t.Fatalf("price = %#v", got)
	}
}

func TestDispatchPriceSerializesSemanticCollisionOnDisjointTrees(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	dir := t.TempDir()
	agents := filepath.Join(dir, "agents.json")
	shape := filepath.Join(dir, "shape.json")
	_ = os.WriteFile(agents, []byte(`{"agents":[{"name":"owner","tree":["internal/root/**"]}]}`), 0o644)
	contract := `{"work_shape":{"kind":"BROAD","root_spine":"own schema","integration_owner":"owner","packets":[{"id":"api","tree":["internal/api/**"],"witness":"api test","shared_contracts":["schema/v1"]},{"id":"docs","tree":["docs/**"],"witness":"doc test","shared_contracts":["schema/v1"]}]}}`
	_ = os.WriteFile(shape, []byte(contract), 0o644)
	var out, errb bytes.Buffer
	if code := runDispatchPrice(&out, &errb, []string{"--workspace", dir, "--in", agents, "--work-shape", shape, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	ws := mapAt(got, "work_shape")
	if ws["semantic_verdict"] != "SEMANTIC_SERIALIZED" || len(ws["serialized"].([]any)) != 2 {
		t.Fatalf("price = %#v", got)
	}
}
