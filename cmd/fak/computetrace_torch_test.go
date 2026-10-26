package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// fak-test:runtime fast est=100ms lane=default
func TestComputeTraceTorchOracleFirstDivergence(t *testing.T) {
	dir := t.TempDir()
	oraclePath := filepath.Join(dir, "oracle.json")
	out := filepath.Join(dir, "ref.json")
	oracle, err := referenceoracle.ReadFile("../../pkg/referenceoracle/testdata/v41-reference-oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	oracle.Prompts = oracle.Prompts[:1]
	oracle.Prompts[0].InputTokenIDs = []int{1, 2}
	oracle.TraceRefs = nil
	b, err := json.Marshal(oracle)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(oraclePath, b, 0o600); err != nil {
		t.Fatal(err)
	}
	fixture := "../../internal/computetrace/testdata/sglang-dumper-tiny"
	var stdout, stderr bytes.Buffer
	if code := runComputeTrace(&stdout, &stderr, []string{"from-sglang", fixture, "--stage-map", "LlamaForCausalLM", "--out", out, "--oracle", oraclePath}); code != 0 {
		t.Fatalf("convert code=%d: %s", code, stderr.String())
	}
	attached, err := referenceoracle.ReadFile(oraclePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(attached.TraceRefs) != 1 || attached.TraceRefs[0].SHA256 == "" || attached.TraceRefs[0].StageMap != "sglang:LlamaForCausalLM" {
		t.Fatal(attached.TraceRefs)
	}
	ref, _, err := loadDivergenceReference(oraclePath, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := ref
	for i := range got.Events {
		got.Events[i].InputDigest = computetrace.Digest("actual-input")
		got.Events[i].WeightDigest = computetrace.Digest("actual-weight")
	}
	gotPath := filepath.Join(dir, "got.json")
	for _, divergent := range []bool{false, true} {
		if divergent {
			for i := range got.Events {
				e := &got.Events[i]
				if e.Layer == 1 && e.Phase == "residual_streams" {
					for j := range e.SampleValues {
						e.SampleValues[j] += 100
					}
				}
			}
		}
		b, err = json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(gotPath, b, 0o600); err != nil {
			t.Fatal(err)
		}
		stdout.Reset()
		stderr.Reset()
		code := runComputeTrace(&stdout, &stderr, []string{"first-divergence", "--ref", oraclePath, "--got", gotPath, "--adopt-digests", "--json"})
		var report struct {
			Verdict        computetrace.Verdict          `json:"verdict"`
			ParityLayers   int                           `json:"parity_layers"`
			FirstDivergent *computetrace.ActivationStage `json:"first_divergent"`
		}
		if err = json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("code=%d decode=%v stderr=%s stdout=%s", code, err, stderr.String(), stdout.String())
		}
		if !divergent {
			if report.Verdict != computetrace.TraceEqual || report.ParityLayers != 2 {
				t.Fatal(stdout.String())
			}
		} else if report.Verdict != computetrace.TraceDivergent || report.ParityLayers != 1 || report.FirstDivergent == nil || *report.FirstDivergent != (computetrace.ActivationStage{Layer: 1, Stage: "residual_streams"}) {
			t.Fatal(stdout.String())
		}
	}
	stdout.Reset()
	stderr.Reset()
	if code := runComputeTrace(&stdout, &stderr, []string{"from-vllm", fixture, "--stage-map", "LlamaForCausalLM", "--step", "0", "--out", filepath.Join(dir, "vllm.json")}); code != 0 {
		t.Fatalf("vllm code=%d: %s", code, stderr.String())
	}
}
