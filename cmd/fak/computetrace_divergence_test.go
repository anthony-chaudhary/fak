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
	"slices"
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

func computeTraceStepArtifacts() (computetrace.Artifact, computetrace.Artifact) {
	actual := computetrace.Artifact{
		Schema: computetrace.Schema, InputTokens: []int{7, 3, 9},
		ActivationCapture: &computetrace.ActivationCapture{
			ID: "step-current", Kind: computetrace.ActivationCaptureIncrementalStep,
			Outcome: computetrace.ActivationCaptureCommitted, Token: 2, ExpectedLayers: 2,
			ExpectedStages: []string{"q_latent", "kv_latent", "attention_projected", "moe_sum", "residual_streams"},
		},
	}
	for layer := 0; layer < 2; layer++ {
		for _, stage := range actual.ActivationCapture.ExpectedStages {
			actual.Events = append(actual.Events, computetrace.Event{
				RunID: "step-current", Operation: "v41.activation", Phase: stage, Layer: layer, Token: 2,
				InputDigest: "input", WeightDigest: "weights", ProvenanceDigest: "provenance",
				Shapes: [][]int{{2}}, SampleIndices: []int{0, 1}, SampleValues: []float32{1, 2},
			})
			actual.RetainedSampleValues += 2
		}
	}
	ref := computetrace.Artifact{Schema: computetrace.Schema, InputTokens: slices.Clone(actual.InputTokens)}
	for token := range ref.InputTokens {
		for _, e := range actual.Events {
			e.Token, e.RunID = token, "full-forward"
			e.SampleValues = slices.Clone(e.SampleValues)
			ref.Events = append(ref.Events, e)
			ref.RetainedSampleValues += 2
		}
	}
	return ref, actual
}

func writeComputeTraceStepFixture(t *testing.T, path string, a computetrace.Artifact) {
	t.Helper()
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/computetrace*.go
// fak-test:runtime fast est=50ms lane=default
func TestComputeTraceFirstDivergenceStepCapture(t *testing.T) {
	refPath, gotPath := filepath.Join(t.TempDir(), "ref.json"), filepath.Join(t.TempDir(), "actual.json")
	run := func(ref, got computetrace.Artifact, flags ...string) (int, []byte, string) {
		t.Helper()
		writeComputeTraceStepFixture(t, refPath, ref)
		writeComputeTraceStepFixture(t, gotPath, got)
		var out, errOut bytes.Buffer
		args := []string{"first-divergence", "--ref", refPath, "--got", gotPath, "--json"}
		code := runComputeTrace(&out, &errOut, append(args, flags...))
		return code, out.Bytes(), errOut.String()
	}
	paired := []string{"--step-capture-id", "step-current", "--step-token", "2"}
	for _, corrupt := range []bool{false, true} {
		name := "software_sample_agreement"
		if corrupt {
			name = "attention_sample_divergence"
		}
		t.Run(name, func(t *testing.T) {
			ref, got := computeTraceStepArtifacts()
			if corrupt {
				for i := range got.Events {
					if got.Events[i].Layer == 1 && got.Events[i].Phase == "attention_projected" {
						got.Events[i].SampleValues[0] += 10
					}
				}
			}
			code, data, stderr := run(ref, got, paired...)
			var report struct {
				computetrace.DivergenceReport
				EvidenceScope string `json:"evidence_scope"`
				CaptureID     string `json:"capture_id"`
				CaptureToken  *int   `json:"capture_token"`
			}
			if err := json.Unmarshal(data, &report); err != nil {
				t.Fatalf("decode: %v stderr=%s", err, stderr)
			}
			if report.EvidenceScope != "committed_step_capture" || report.CaptureID != "step-current" || report.CaptureToken == nil || *report.CaptureToken != 2 {
				t.Fatal("step comparison did not disclose the admitted caller identity and evidence scope")
			}
			if corrupt {
				want := computetrace.ActivationIdentity{Layer: 1, Token: 2, Stage: "attention_projected"}
				if code != 1 || report.Verdict != computetrace.TraceDivergent || report.FirstDivergent == nil || report.FirstDivergent.Layer != 1 || report.FirstDivergent.Stage != want.Stage || report.ParityLayers != 1 || report.Strict.Divergent != want {
					t.Fatalf("wrong first divergence: code=%d verdict=%s strict=%v", code, report.Verdict, report.Strict.Divergent)
				}
			} else if code != 0 || report.Verdict != computetrace.TraceEqual || report.ParityLayers != 2 {
				t.Fatalf("valid capture refused: code=%d verdict=%s stderr=%s", code, report.Verdict, stderr)
			}
		})
	}
	cases := []struct {
		name    string
		flags   []string
		corrupt func(*computetrace.Artifact, *computetrace.Artifact)
	}{
		{name: "no_caller_identity"},
		{name: "id_only", flags: []string{"--step-capture-id", "step-current"}},
		{name: "token_only", flags: []string{"--step-token", "2"}},
		{name: "stale_artifact_after_unpublished_step", flags: []string{"--step-capture-id", "unpublished-new-step", "--step-token", "2"}},
		{name: "wrong_token", flags: []string{"--step-capture-id", "step-current", "--step-token", "1"}},
		{name: "missing_metadata", flags: paired, corrupt: func(r, a *computetrace.Artifact) { a.ActivationCapture = nil }},
		{name: "aborted_actual", flags: paired, corrupt: func(r, a *computetrace.Artifact) { a.ActivationCapture.Outcome = computetrace.ActivationCaptureAborted }},
		{name: "incomplete_actual", flags: paired, corrupt: func(r, a *computetrace.Artifact) { a.Events = a.Events[1:]; a.RetainedSampleValues -= 2 }},
		{name: "duplicate_actual", flags: paired, corrupt: func(r, a *computetrace.Artifact) { a.Events[1] = a.Events[0] }},
		{name: "dropped_actual", flags: paired, corrupt: func(r, a *computetrace.Artifact) { a.DroppedSampleValues++ }},
		{name: "earlier_prefix_same_last_token", flags: paired, corrupt: func(r, a *computetrace.Artifact) { r.InputTokens[0]++ }},
		{name: "incomplete_reference", flags: paired, corrupt: func(r, a *computetrace.Artifact) { r.Events = r.Events[:len(r.Events)-1]; r.RetainedSampleValues -= 2 }},
		{name: "duplicate_reference", flags: paired, corrupt: func(r, a *computetrace.Artifact) { r.Events[len(r.Events)-1] = r.Events[len(r.Events)-2] }},
		{name: "dropped_reference", flags: paired, corrupt: func(r, a *computetrace.Artifact) { r.Dropped++ }},
		{name: "aborted_reference_capture", flags: paired, corrupt: func(r, a *computetrace.Artifact) {
			_, *r = computeTraceStepArtifacts()
			r.ActivationCapture.Outcome = computetrace.ActivationCaptureAborted
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref, actual := computeTraceStepArtifacts()
			if tc.corrupt != nil {
				tc.corrupt(&ref, &actual)
			}
			code, data, _ := run(ref, actual, tc.flags...)
			if code == 0 || bytes.Contains(data, []byte("\"evidence_scope\":\"committed_step_capture\"")) {
				t.Fatal("invalid capture or caller receipt admitted as committed step evidence")
			}
		})
	}
	t.Run("legacy_keeps_sampled_scope", func(t *testing.T) {
		_, actual := computeTraceStepArtifacts()
		actual.ActivationCapture = nil
		code, data, stderr := run(actual, actual)
		var report struct {
			Verdict       computetrace.Verdict `json:"verdict"`
			EvidenceScope string               `json:"evidence_scope"`
		}
		if err := json.Unmarshal(data, &report); err != nil || code != 0 || report.Verdict != computetrace.TraceEqual || report.EvidenceScope != "sampled_numerical" {
			t.Fatalf("legacy scope changed: code=%d decode=%v stderr=%s", code, err, stderr)
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestComputeTraceFirstDivergenceStepTokenPresence(t *testing.T) {
	refPath, gotPath := filepath.Join(t.TempDir(), "ref.json"), filepath.Join(t.TempDir(), "actual.json")
	_, actual := computeTraceStepArtifacts()
	actual.InputTokens, actual.ActivationCapture.Token = []int{0}, 0
	for i := range actual.Events {
		actual.Events[i].Token = 0
		actual.Events[i].SampleValues[0] = 0
	}
	reference := actual
	reference.ActivationCapture = nil
	writeComputeTraceStepFixture(t, refPath, reference)
	raw, err := json.Marshal(actual)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"explicit_zero", "absent", "null", "null_input_token", "reference_null_input", "reference_null_samples", "reference_null_indices"} {
		t.Run(field, func(t *testing.T) {
			writeComputeTraceStepFixture(t, refPath, reference)
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(raw, &envelope); err != nil {
				t.Fatal(err)
			}
			var capture map[string]json.RawMessage
			if err := json.Unmarshal(envelope["activation_capture"], &capture); err != nil {
				t.Fatal(err)
			}
			switch field {
			case "absent":
				delete(capture, "token")
			case "null":
				capture["token"] = json.RawMessage("null")
			case "null_input_token":
				envelope["input_tokens"] = json.RawMessage("[null]")
			}
			envelope["activation_capture"], err = json.Marshal(capture)
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(envelope)
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(field, "reference_null_") {
				var refEnvelope map[string]json.RawMessage
				if err := json.Unmarshal(raw, &refEnvelope); err != nil {
					t.Fatal(err)
				}
				delete(refEnvelope, "activation_capture")
				if field == "reference_null_input" {
					refEnvelope["input_tokens"] = json.RawMessage("[null]")
				} else {
					var events []map[string]json.RawMessage
					if err := json.Unmarshal(refEnvelope["events"], &events); err != nil {
						t.Fatal(err)
					}
					key, value := "sample_values", "[null,2]"
					if field == "reference_null_indices" {
						key, value = "sample_indices", "[null,1]"
					}
					events[0][key] = json.RawMessage(value)
					refEnvelope["events"], err = json.Marshal(events)
					if err != nil {
						t.Fatal(err)
					}
				}
				refData, err := json.Marshal(refEnvelope)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(refPath, refData, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(gotPath, data, 0o644); err != nil {
				t.Fatal(err)
			}
			var out, stderr bytes.Buffer
			code := runComputeTrace(&out, &stderr, []string{"first-divergence", "--ref", refPath, "--got", gotPath, "--step-capture-id", "step-current", "--step-token", "0", "--json"})
			if field == "explicit_zero" {
				if code != 0 || !bytes.Contains(out.Bytes(), []byte("\"evidence_scope\":\"committed_step_capture\"")) {
					t.Fatalf("explicit token zero rejected: exit=%d stderr=%s", code, stderr.String())
				}
			} else if code == 0 || bytes.Contains(out.Bytes(), []byte("\"evidence_scope\":\"committed_step_capture\"")) {
				t.Fatal("missing or null token admitted as an explicit position zero")
			}
		})
	}
}
