package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func TestComputeTraceActivationReferenceCLI(t *testing.T) {
	dir := t.TempDir()
	dump := filepath.Join(dir, "dump")
	if err := os.Mkdir(dump, 0o755); err != nil {
		t.Fatal(err)
	}
	vals := make([]byte, 4*16)
	for k := 0; k < 16; k++ {
		binary.LittleEndian.PutUint32(vals[4*k:], math.Float32bits(float32(k)))
	}
	writeFile := func(path string, data []byte) {
		t.Helper()
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeFile(filepath.Join(dump, "ffn.bin"), vals)
	writeFile(filepath.Join(dump, "tokens.json"), []byte(`{"prompt":"p","tokens":[4,2]}`))
	writeFile(filepath.Join(dump, "index.jsonl"), []byte(`{"seq":0,"name":"ffn_out-3","op":"ADD","type":"f32","ne":[8,2,1,1],"dumped":true,"file":"ffn.bin","sum":0,"sumsq":0,"maxabs":0,"nonfinite":0}`+"\n"))

	// The fak side: same tokens and sample rule, fak digests.
	ref0, err := computetrace.ConvertLlamaCppDump(dump)
	if err != nil {
		t.Fatal(err)
	}
	fak := computetrace.Artifact{Schema: computetrace.Schema, InputTokens: []int{4, 2}}
	for _, e := range ref0.Events {
		e.InputDigest, e.WeightDigest = "sha256:in", "sha256:w"
		fak.Events = append(fak.Events, e)
	}
	fakPath := filepath.Join(dir, "fak.json")
	data, _ := json.Marshal(fak)
	writeFile(fakPath, data)

	refPath := filepath.Join(dir, "ref.json")
	var out, errOut bytes.Buffer
	if code := runComputeTrace(&out, &errOut, []string{"from-llamacpp", dump, "--out", refPath, "--adopt-digests", fakPath}); code != 0 {
		t.Fatalf("from-llamacpp exit=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "events=2") || !strings.Contains(out.String(), "digests_adopted=true") {
		t.Fatalf("from-llamacpp stdout=%q", out.String())
	}

	out.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"profile", "--ref", refPath, "--got", fakPath, "--json"}); code != 0 {
		t.Fatalf("profile exit=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var p computetrace.ActivationProfile
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Verdict != computetrace.TraceEqual || len(p.Rows) != 1 || p.Rows[0].Layer != 3 || p.Rows[0].Stage != "moe_sum" || p.Rows[0].Tokens != 2 {
		t.Fatalf("profile=%+v", p)
	}

	out.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"profile", "--ref", refPath, "--got", refPath + ".missing"}); code != 1 {
		t.Fatalf("missing actual trace exit=%d, want 1", code)
	}
}
