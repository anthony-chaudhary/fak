package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

func TestComputeTraceFirstDivergenceCLI(t *testing.T) {
	dir := t.TempDir()
	writeFile := func(path string, data []byte) {
		t.Helper()
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	dump := filepath.Join(dir, "dump")
	if err := os.Mkdir(dump, 0o755); err != nil {
		t.Fatal(err)
	}
	vals := make([]byte, 4*8*2)
	for k := 0; k < 16; k++ {
		binary.LittleEndian.PutUint32(vals[4*k:], math.Float32bits(float32(k+1)))
	}
	var index strings.Builder
	for seq, name := range []string{"model.input_embed", "ffn_out-0", "ffn_out-1", "l_last-1", "hc_mixed-1"} {
		file := name + ".bin"
		writeFile(filepath.Join(dump, file), vals)
		ne := `[8,2,1,1]`
		if name == "l_last-1" {
			ne = `[8,1,2,1]`
		}
		index.WriteString(`{"seq":` + strconv.Itoa(seq) + `,"name":"` + name + `","op":"ADD","type":"f32","ne":` + ne + `,"dumped":true,"file":"` + file + `"}` + "\n")
	}
	writeFile(filepath.Join(dump, "index.jsonl"), []byte(index.String()))
	writeFile(filepath.Join(dump, "tokens.json"), []byte(`{"prompt":"p","tokens":[4,2]}`))

	refPath := filepath.Join(dir, "ref.json")
	var out, errOut bytes.Buffer
	if code := runComputeTrace(&out, &errOut, []string{"from-llamacpp", dump, "--out", refPath, "--stage-map", "qwen4exp"}); code != 0 {
		t.Fatalf("from-llamacpp exit=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "events=8") {
		t.Fatalf("qwen4exp from-llamacpp stdout=%q", out.String())
	}
	if code := runComputeTrace(&out, &errOut, []string{"from-llamacpp", dump, "--out", refPath + ".x", "--stage-map", "llama"}); code != 1 {
		t.Fatalf("unknown stage map exit=%d", code)
	}

	ref, err := readComputeTraceFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	fak := computetrace.Artifact{Schema: computetrace.Schema, InputTokens: ref.InputTokens}
	for _, e := range ref.Events {
		e.InputDigest, e.WeightDigest = "sha256:in", "sha256:w"
		if e.Layer == 1 && e.Phase == "moe_sum" {
			e.SampleValues = append([]float32(nil), e.SampleValues...)
			e.SampleValues[0] += 3
		}
		fak.Events = append(fak.Events, e)
	}
	fakPath := filepath.Join(dir, "fak.json")
	data, _ := json.Marshal(fak)
	writeFile(fakPath, data)

	refData, _ := os.ReadFile(refPath)
	sum := sha256.Sum256(refData)
	oracle := referenceoracle.Oracle{
		Schema:    referenceoracle.Schema,
		Reference: referenceoracle.Reference{Repo: "r", SHA: "s"},
		Artifact:  referenceoracle.Artifact{Repo: "a"},
		Prompts:   []referenceoracle.Prompt{{Text: "p", InputTokenIDs: []int{4, 2}, GreedyIDs: []int{9}}},
		TraceRefs: []referenceoracle.TraceRef{{Prompt: 0, Path: "ref.json", SHA256: "sha256:" + hex.EncodeToString(sum[:]), StageMap: "qwen4exp"}},
	}
	oracleData, _ := json.Marshal(oracle)
	oraclePath := filepath.Join(dir, "oracle.json")
	writeFile(oraclePath, oracleData)

	for _, refArg := range []string{refPath, oraclePath} {
		out.Reset()
		code := runComputeTrace(&out, &errOut, []string{"first-divergence", "--ref", refArg, "--got", fakPath, "--adopt-digests", "--tol", "1e-3", "--json"})
		var r computetrace.DivergenceReport
		if err := json.Unmarshal(out.Bytes(), &r); err != nil {
			t.Fatalf("%s: %v stdout=%s stderr=%s", refArg, err, out.String(), errOut.String())
		}
		if code != 1 || r.Verdict != computetrace.TraceDivergent || r.FirstDivergent == nil || r.FirstDivergent.Layer != 1 || r.FirstDivergent.Stage != "moe_sum" || r.ParityLayers != 1 {
			t.Fatalf("%s: exit=%d report=%+v", refArg, code, r)
		}
	}

	out.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"first-divergence", "--ref", refPath, "--got", fakPath, "--adopt-digests", "--tol", "1"}); code != 0 {
		t.Fatalf("loose tolerance exit=%d stdout=%s", code, out.String())
	}
	if !strings.Contains(out.String(), "verdict=EQUAL first_divergent_layer=none") {
		t.Fatalf("text report=%q", out.String())
	}

	oracle.TraceRefs[0].SHA256 = strings.Repeat("0", 64)
	oracleData, _ = json.Marshal(oracle)
	writeFile(oraclePath, oracleData)
	if code := runComputeTrace(&out, &errOut, []string{"first-divergence", "--ref", oraclePath, "--got", fakPath}); code != 1 || !strings.Contains(errOut.String(), "oracle pins") {
		t.Fatalf("pinned digest mismatch exit=%d stderr=%q", code, errOut.String())
	}
	if code := runComputeTrace(&out, &errOut, []string{"first-divergence", "--ref", oraclePath}); code != 2 {
		t.Fatalf("missing --got exit=%d", code)
	}
}
