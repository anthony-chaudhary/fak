package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseIDList(t *testing.T) {
	got, err := parseIDList(" 248045,8678,, 198 ")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{248045, 8678, 198}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parseIDList = %v, want %v", got, want)
	}
	if _, err := parseIDList("1,nope"); err == nil {
		t.Fatal("parseIDList accepted a non-integer token id")
	}
}

func TestTopKLogitsSortsByLogitThenID(t *testing.T) {
	got := topKLogits([]float32{1, 3, 3, 2}, 3)
	want := []topLogit{{ID: 1, Logit: 3}, {ID: 2, Logit: 3}, {ID: 3, Logit: 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("topKLogits = %#v, want %#v", got, want)
	}
}

func TestCompareIDs(t *testing.T) {
	if err := compareIDs([]int{248068, 198}, []int{248068, 198}); err != nil {
		t.Fatalf("compareIDs matching sequence: %v", err)
	}
	if err := compareIDs([]int{248068, 198}, []int{248068, 271}); err == nil || !strings.Contains(err.Error(), "step 1") {
		t.Fatalf("compareIDs mismatch error = %v, want step detail", err)
	}
	if err := compareIDs([]int{248068}, []int{248068, 198}); err == nil || !strings.Contains(err.Error(), "generated 1 ids") {
		t.Fatalf("compareIDs length error = %v, want length detail", err)
	}
}

func TestWriteResultJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "result.json")
	match := true
	res := checkResult{
		Model:        modelSummary{Source: "model.gguf", Type: "qwen35", QuantLoaded: true},
		PromptIDs:    []int{1, 2, 3},
		GeneratedIDs: []int{248068},
		ExpectedIDs:  []int{248068},
		ExpectMatch:  &match,
		Steps: []stepResult{{
			Index:   0,
			TokenID: 248068,
			Top:     []topLogit{{ID: 248068, Logit: 42}},
		}},
	}
	if err := writeResult(path, res); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got checkResult
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal: %v\n%s", err, raw)
	}
	if !reflect.DeepEqual(got, res) {
		t.Fatalf("decoded result = %#v, want %#v", got, res)
	}
}
