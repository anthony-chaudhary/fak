package bench

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestActivationPatchL3(t *testing.T) {
	data, err := os.ReadFile("testdata/activation_patch_l3_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []ActivationPatchCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	rows, err := RunActivationPatchL3(cases)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ActivationPatchJSONL(rows)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/activation_patch_l3_witness.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("witness drift\ngot:\n%s\nwant:\n%s", got, want)
	}
	t.Logf("%s", got)
}

func TestActivationPatchL3RejectsWrongDirection(t *testing.T) {
	_, err := RunActivationPatchL3([]ActivationPatchCase{{ID: "wrong", Layer: 3, ControlLayer: 1, Clean: []float32{-1, 0}, Corrupt: []float32{1, 0}, Target: []float32{1, 0}, MinEffect: .1, MaxControl: 0}})
	if err == nil {
		t.Fatal("meaning-worsening patch passed causal gate")
	}
}
