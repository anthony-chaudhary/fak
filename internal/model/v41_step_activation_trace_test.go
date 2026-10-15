package model

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41StepActivationTrace(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	prefix, next := v41IncrementalPrefix(m.Cfg, 3)
	withV41ActivationTraceEnv(t, "")
	control := m.NewSession()
	t.Cleanup(control.Close)
	control.Prefill(prefix)
	want := append([]float32(nil), control.Step(next)...)
	second := (next + 1) % m.Cfg.VocabSize
	wantSecond := append([]float32(nil), control.Step(second)...)

	path := filepath.Join(t.TempDir(), "step.json")
	withV41ActivationTraceEnv(t, path)
	s := m.NewSession()
	t.Cleanup(s.Close)
	if len(s.Prefill(prefix)) == 0 || !s.v41IncrementalEligible() {
		t.Fatal("prefill did not seed an incremental session")
	}
	read, restore := v41SessionIncrementalCalls(t)
	defer restore()
	got := s.Step(next)
	if read() != 1 || !slices.Equal(got, want) {
		t.Fatal("tracing changed the incremental execution or logits")
	}
	coverage := s.NativeForwardCoverage()
	if coverage == nil || coverage.Validate() != nil || coverage.LogitsSource != NativeForwardLogitsSourceV41Incremental {
		t.Fatal("step did not produce completed incremental forward coverage")
	}
	full := append(append([]int(nil), prefix...), next)
	if !slices.Equal(s.v41Forward.history, full) {
		t.Fatal("step did not commit its token exactly once")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a, err := computetrace.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(a.InputTokens, full) {
		t.Fatalf("step capture input tokens = %v, want full committed prefix %v", a.InputTokens, full)
	}
	readMetadata := func(t *testing.T, path string) (id, kind, outcome string, token, layers int, stages []string) {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var envelope struct {
			Capture *struct {
				ID             string   `json:"id"`
				Kind           string   `json:"kind"`
				Outcome        string   `json:"outcome"`
				Token          int      `json:"token"`
				ExpectedLayers int      `json:"expected_layers"`
				ExpectedStages []string `json:"expected_stages"`
			} `json:"activation_capture"`
		}
		if err := json.Unmarshal(data, &envelope); err != nil || envelope.Capture == nil {
			t.Fatal("step artifact has no capture metadata")
		}
		c := envelope.Capture
		return c.ID, c.Kind, c.Outcome, c.Token, c.ExpectedLayers, c.ExpectedStages
	}
	id, kind, outcome, token, layers, expected := readMetadata(t, path)
	receipt := s.LastV41ActivationCapture()
	if receipt.ID != id || receipt.Token != len(prefix) || receipt.Outcome != "committed" || receipt.Path != path || !receipt.Published || !receipt.Current || receipt.PublicationError != "" {
		t.Fatal("caller receipt does not identify its published committed capture")
	}
	if id == "" || kind != "incremental_step" || outcome != "committed" || token != len(prefix) || layers != m.Cfg.NumLayers {
		t.Fatal("capture metadata does not identify the completed step")
	}
	stages := map[string]bool{"q_latent": true, "kv_latent": true, "attention_projected": true, "moe_sum": true, "residual_streams": true}
	if len(expected) != len(stages) {
		t.Fatal("capture does not declare all expected stages")
	}
	declared := make(map[string]bool, len(expected))
	for _, stage := range expected {
		if !stages[stage] || declared[stage] {
			t.Fatal("capture expected stages are missing or repeated")
		}
		declared[stage] = true
	}
	if a.Dropped != 0 || a.DroppedSampleValues != 0 || len(a.Events) != m.Cfg.NumLayers*len(stages) {
		t.Fatal("step capture is missing its complete per-layer stage set")
	}
	seen := make(map[computetrace.ActivationIdentity]bool, len(a.Events))
	for _, event := range a.Events {
		identity := computetrace.ActivationIdentity{Layer: event.Layer, Token: event.Token, Stage: event.Phase}
		if event.RunID != id || event.Operation != "v41.activation" || event.Layer < 0 || event.Layer >= m.Cfg.NumLayers || event.Token != len(prefix) || !stages[event.Phase] || seen[identity] {
			t.Fatalf("unexpected or duplicate step activation identity: %v", identity)
		}
		seen[identity] = true
		if event.InputDigest != m.v41ActivationInputIdentity(full) || event.WeightDigest == "" || len(event.Shapes) != 1 || len(event.Shapes[0]) != 1 || event.Shapes[0][0] <= 0 || len(event.SampleValues) == 0 || len(event.SampleIndices) != len(event.SampleValues) {
			t.Fatal("step activation lacks full-prefix provenance or shaped samples")
		}
		for i, value := range event.SampleValues {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || event.SampleIndices[i] < 0 || event.SampleIndices[i] >= event.Shapes[0][0] {
				t.Fatal("step activation contains invalid samples")
			}
		}
	}
	for layer := 0; layer < m.Cfg.NumLayers; layer++ {
		if s.v41Forward.layerState(layer).nextWindowPos != len(full) {
			t.Fatal("capture changed the retained continuation cursor")
		}
	}
	if computetrace.Enabled() {
		t.Fatal("step left a process recorder enabled")
	}
	if got := s.Step(second); !slices.Equal(got, wantSecond) || read() != 2 {
		t.Fatal("later capture changed or replayed continuation")
	}
	secondID, _, secondOutcome, secondToken, _, _ := readMetadata(t, path)
	secondReceipt := s.LastV41ActivationCapture()
	if secondReceipt.ID != secondID || !secondReceipt.Current || !secondReceipt.Published || secondReceipt.PublicationError != "" {
		t.Fatal("continuation receipt did not advance to the new published capture")
	}
	if secondID == "" || secondID == id || secondOutcome != "committed" || secondToken != len(full) {
		t.Fatal("later step retained the previous capture identity")
	}
	if !slices.Equal(s.v41Forward.history, append(slices.Clone(full), second)) {
		t.Fatal("later capture changed committed token history")
	}

	t.Run("full forward software self parity", func(t *testing.T) {
		referencePath := filepath.Join(t.TempDir(), "full-forward.json")
		withV41ActivationTraceEnv(t, referencePath)
		_ = m.Forward(full)
		data, err := os.ReadFile(referencePath)
		if err != nil {
			t.Fatal(err)
		}
		reference, err := computetrace.Read(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		projected, err := computetrace.ProjectActivationReference(reference, a)
		if err != nil {
			t.Fatal(err)
		}
		if projected.ActivationCapture != nil || !slices.Equal(projected.InputTokens, full) {
			t.Fatal("full forward projection fabricated a Step identity or discarded the prefix")
		}
		// Shared fixture arithmetic establishes software self parity only.
		tolerance := computetrace.ActivationTolerance{Atol: 1e-6, Rtol: 1e-6}
		if r := computetrace.CompareActivationTraces(projected, a, tolerance); r.Verdict != computetrace.TraceEqual || r.ComparedSamples == 0 {
			t.Fatalf("full forward and incremental samples differ: verdict=%s first=%v", r.Verdict, r.Divergent)
		}
		for i := range projected.Events {
			if projected.Events[i].Layer == 1 && projected.Events[i].Phase == "attention_projected" {
				projected.Events[i].SampleValues[0] += 10
			}
		}
		r := computetrace.FirstDivergence(projected, a, computetrace.DivergenceTolerance{RelL2: 1e-3}, tolerance)
		want := computetrace.ActivationIdentity{Layer: 1, Token: len(prefix), Stage: "attention_projected"}
		if r.Verdict != computetrace.TraceDivergent || r.FirstDivergent == nil || r.FirstDivergent.Layer != 1 || r.FirstDivergent.Stage != want.Stage || r.Strict.Divergent != want {
			t.Fatal("corrupt attention sample did not identify its exact first divergence")
		}
		reference.InputTokens[0] = (reference.InputTokens[0] + 1) % m.Cfg.VocabSize
		if _, err := computetrace.ProjectActivationReference(reference, a); !errors.Is(err, computetrace.ErrActivationCaptureInvalid) {
			t.Fatal("same final token concealed a mismatched earlier prefix")
		}
	})

	t.Run("publication failure never replays a committed token", func(t *testing.T) {
		withV41ActivationTraceEnv(t, "")
		failed := m.NewSession()
		t.Cleanup(failed.Close)
		failed.Prefill(prefix)
		failedPath := filepath.Join(t.TempDir(), "missing", "step.json")
		withV41ActivationTraceEnv(t, failedPath)
		before := read()
		var logits []float32
		if err := recoverError(func() { logits = failed.Step(next) }); err != nil {
			t.Fatalf("diagnostic publication escaped as execution failure: %v", err)
		}
		failedReceipt := failed.LastV41ActivationCapture()
		if failedReceipt.ID == "" || failedReceipt.ID == id || failedReceipt.Token != len(prefix) || failedReceipt.Outcome != "committed" || failedReceipt.Path != failedPath || failedReceipt.Published || failedReceipt.Current || failedReceipt.PublicationError == "" {
			t.Fatal("publication failure concealed the committed token or advertised a current artifact")
		}
		if read() != before+1 || !slices.Equal(logits, want) || !slices.Equal(failed.v41Forward.history, full) {
			t.Fatal("diagnostic publication changed or replayed the committed token")
		}
		if got := failed.Step(second); !slices.Equal(got, wantSecond) || read() != before+2 {
			t.Fatal("continuation after failed publication differs from the control")
		}
	})

	t.Run("head failure cannot complete an all-layer capture", func(t *testing.T) {
		withV41ActivationTraceEnv(t, "")
		failed := m.NewSession()
		t.Cleanup(failed.Close)
		failed.Prefill(prefix)
		positions := v41AllNextWindowPos(failed.v41Forward)
		fault := errors.New("selected head failure")
		failed.v41State().finalNorm = func([]float32) ([]float32, error) { return nil, fault }
		path := filepath.Join(t.TempDir(), "aborted.json")
		withV41ActivationTraceEnv(t, path)
		if err := recoverError(func() { failed.Step(next) }); !errors.Is(err, fault) {
			t.Fatal("capture changed the primary head failure")
		}
		_, _, outcome, _, _, _ := readMetadata(t, path)
		abortedReceipt := failed.LastV41ActivationCapture()
		if abortedReceipt.ID == "" || abortedReceipt.Outcome != "aborted" || abortedReceipt.Token != len(prefix) || !abortedReceipt.Published || !abortedReceipt.Current {
			t.Fatal("head failure did not publish its own aborted attempt receipt")
		}
		if outcome != "aborted" || !slices.Equal(failed.v41Forward.history, prefix) || !slices.Equal(v41AllNextWindowPos(failed.v41Forward), positions) {
			t.Fatal("failed head produced completed capture or retained token mutation")
		}
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		aborted, err := computetrace.Read(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(aborted.Events) != m.Cfg.NumLayers*len(stages) || computetrace.CompareActivationTraces(aborted, aborted, computetrace.ActivationTolerance{}).Verdict != computetrace.TraceIncomparable {
			t.Fatal("an aborted all-stage capture established numerical agreement")
		}
	})
}

// This reuses checked host-resident cohort fixtures; it is software evidence.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41StepActivationTraceCohort(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	m := v41CompressorTestFixtureIndex(t, 128)
	for _, cancelLane := range []bool{false, true} {
		name := "unequal_prefixes_commit_separate_captures"
		if cancelLane {
			name = "cancelled_lane_does_not_contaminate_peer"
		}
		t.Run(name, func(t *testing.T) {
			withV41ActivationTraceEnv(t, "")
			backend := &v41CohortTestBackend{v41PrefillStepBatchBackend: newV41PrefillStepBatchBackend()}
			prefixes := [][]int{{1, 2, 3}, {1, 2, 3, 4}}
			lanes := make([]v41StepCohortLane, 0, len(prefixes))
			var want [][]float32
			for lane, prefix := range prefixes {
				source := m.NewSession()
				t.Cleanup(source.Close)
				source.Prefill(prefix)
				snapshot := captureV41ForwardSnapshot(source.v41State())
				session, control := v41DenseTestSession(t, m, backend), v41DenseTestSession(t, m, backend)
				session.v41Forward, control.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
				session.v41State()
				control.v41State()
				id := 6 + lane
				logits, _, err := m.forwardV41Step(id, control.v41State(), nil)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, slices.Clone(logits))
				lanes = append(lanes, v41StepCohortLane{session: session, id: id})
			}
			before := captureV41ForwardSnapshot(lanes[1].session.v41Forward)
			if cancelLane {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				lanes[1].ctx = ctx
				st := lanes[1].session.v41State()
				project := st.denseProjection()
				st.setDenseProjection(func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == 0 && leaf == "attn.wq_a.weight" {
						cancel()
					}
					return project(layer, leaf, panel, out, in, rows)
				})
			}
			path := filepath.Join(t.TempDir(), "cohort.json")
			withV41ActivationTraceEnv(t, path)
			backend.reset()
			var owner sync.Mutex
			got, err := m.forwardV41StepCohort(&owner, lanes)
			if err != nil || len(got) != 2 || got[0].err != nil || !got[0].stats.Committed || !slices.Equal(got[0].logits, want[0]) {
				t.Fatalf("healthy cohort lane failed: %v", err)
			}
			receipts := []V41ActivationCaptureReceipt{lanes[0].session.LastV41ActivationCapture(), lanes[1].session.LastV41ActivationCapture()}
			if receipts[0].ID == "" || receipts[1].ID == "" || receipts[0].ID == receipts[1].ID || receipts[0].Token != len(prefixes[0]) || receipts[1].Token != len(prefixes[1]) || !receipts[0].Published || !receipts[1].Published {
				t.Fatal("interleaved lanes shared an identity, position, or lost publication")
			}
			currentLane := 1
			if cancelLane {
				currentLane = 0
				if !errors.Is(got[1].err, context.Canceled) || got[1].stats.Committed || got[1].logits != nil || receipts[1].Outcome != "aborted" || !reflect.DeepEqual(captureV41ForwardSnapshot(lanes[1].session.v41Forward), before) {
					t.Fatal("cancelled capture retained state or changed its primary cancellation")
				}
			} else if got[1].err != nil || !got[1].stats.Committed || !slices.Equal(got[1].logits, want[1]) || receipts[1].Outcome != "committed" {
				t.Fatal("second healthy capture differs from serial execution")
			}
			if !receipts[currentLane].Current || receipts[1-currentLane].Current || receipts[0].Outcome != "committed" {
				t.Fatal("caller receipt failed to distinguish an overwritten path from current capture")
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			a, err := computetrace.Read(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			if err := computetrace.ValidateActivationCapture(a); err != nil || a.ActivationCapture == nil || a.ActivationCapture.ID != receipts[currentLane].ID || !slices.Equal(a.InputTokens, append(slices.Clone(prefixes[currentLane]), lanes[currentLane].id)) {
				t.Fatal("published cohort artifact mixes lane identity, prefix, or stage coverage")
			}
			if computetrace.Enabled() || backend.active {
				t.Fatal("cohort capture leaked a process recorder or recording window")
			}
		})
	}
}
