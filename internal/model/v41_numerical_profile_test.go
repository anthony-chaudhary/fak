package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strconv"
	"testing"
)

// This is a request fixture, not an admitted profile or a parity claim.
func v41UnqualifiedNumericalRequest() *DeepSeekV41NumericalProfileRequest {
	return &DeepSeekV41NumericalProfileRequest{
		SchemaVersion:       1,
		TargetRepository:    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
		TargetRevision:      "1bae0e1088993b9b82de08f59de5be5a34e42170",
		StorageRepository:   "mrfakename/DeepSeek-V4.1-Flash-1017-GGUF",
		StorageRevision:     "1df54e7d1b1e442051ad393ce871870a929f243e",
		StorageFormat:       "GGUF/Q4_K_M",
		ReferenceRepository: "deepseek-ai/DeepInfer",
		ReferenceRevision:   "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01",
		Semantics:           "unqualified-test-semantics-v1",
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41NumericalProfileRequestBoundary(t *testing.T) {
	t.Parallel()
	_, official := readDeepSeekV41Config(t)
	gguf := Config{ModelType: "deepseek41", NumLayers: 3, HiddenSize: 32, NumHeads: 1, NumKVHeads: 1,
		CompressRatios: []int{0, 1, 1}, IndexerTypes: []string{"dense", "full", "shared"}}
	var err error
	gguf.DeepSeekV41, err = NewDeepSeekV41GGUFConfig(gguf, V41GGUFTopology{})
	if err != nil {
		t.Fatal(err)
	}
	for _, cfg := range []Config{official, gguf} {
		t.Run(cfg.ModelType, func(t *testing.T) {
			legacy, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			var absent map[string]json.RawMessage
			if err := json.Unmarshal(legacy, &absent); err != nil {
				t.Fatal(err)
			}
			if _, ok := absent["fak_v41_numerical_profile"]; ok {
				t.Fatal("legacy config acquired a numerical profile")
			}
			cfg.DeepSeekV41.NumericalProfile = v41UnqualifiedNumericalRequest()
			encoded, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			var doc map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &doc); err != nil {
				t.Fatal(err)
			}
			if len(doc["fak_v41_numerical_profile"]) == 0 {
				t.Fatal("root profile request was dropped")
			}
			var got Config
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			if got.DeepSeekV41 == nil || !reflect.DeepEqual(got.DeepSeekV41.NumericalProfile, cfg.DeepSeekV41.NumericalProfile) {
				t.Fatal("numerical request changed during Config JSON round trip")
			}
			if full, err := v41ForwardGeometry(got); full || !errors.Is(err, ErrV41NumericalProfile) || !errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("unqualified request reached geometry selection: full=%v err=%v", full, err)
			}
			// Match encoding/json's case-insensitive field selection at the root,
			// then emit the canonical spelling without losing the request.
			requestRaw := doc["fak_v41_numerical_profile"]
			delete(doc, "fak_v41_numerical_profile")
			doc["FaK_v41_NuMeRiCaL_PrOfIlE"] = requestRaw
			folded, err := json.Marshal(doc)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(folded, &got); err != nil {
				t.Fatal(err)
			}
			if got.DeepSeekV41 == nil || !reflect.DeepEqual(got.DeepSeekV41.NumericalProfile, cfg.DeepSeekV41.NumericalProfile) {
				t.Fatal("case-folded root request was dropped")
			}
			canonical, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(canonical, &got); err != nil {
				t.Fatal(err)
			}
			if full, err := v41ForwardGeometry(got); full || !errors.Is(err, ErrV41NumericalProfile) {
				t.Fatalf("case-folded round trip admitted request: full=%v err=%v", full, err)
			}
			delete(doc, "FaK_v41_NuMeRiCaL_PrOfIlE")
			doc["fak_v41_numerical_profile"] = requestRaw
			var unknown map[string]json.RawMessage
			if err := json.Unmarshal(doc["fak_v41_numerical_profile"], &unknown); err != nil {
				t.Fatal(err)
			}
			unknown["schema_version"] = json.RawMessage("2")
			futureSchema, err := json.Marshal(unknown)
			if err != nil {
				t.Fatal(err)
			}
			unknown["schema_version"] = json.RawMessage("1")
			unknown["unknown_axis"] = json.RawMessage("true")
			unknownRaw, err := json.Marshal(unknown)
			if err != nil {
				t.Fatal(err)
			}
			for _, bad := range []string{"null", "{}", `{"schema_version":1}`, string(futureSchema), string(unknownRaw)} {
				doc["fak_v41_numerical_profile"] = json.RawMessage(bad)
				raw, err := json.Marshal(doc)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, &got); !errors.Is(err, ErrV41NumericalProfile) {
					t.Fatalf("invalid request %s was ignored: %v", bad, err)
				}
			}
		})
	}
	t.Run("no silent placement or topology loss", func(t *testing.T) {
		request := v41UnqualifiedNumericalRequest()
		for _, doc := range []map[string]any{
			{"model_type": "deepseek41", "fak_v41_numerical_profile": request},
			{"model_type": "deepseek_v41", "text_config": map[string]any{"model_type": "deepseek_v41_text", "fak_v41_numerical_profile": request}},
			{"model_type": "llama", "fak_v41_numerical_profile": request},
		} {
			raw, err := json.Marshal(doc)
			if err != nil {
				t.Fatal(err)
			}
			var cfg Config
			if err := json.Unmarshal(raw, &cfg); !errors.Is(err, ErrV41NumericalProfile) {
				t.Fatalf("explicit request was dropped: %v", err)
			}
		}
	})
	t.Run("misplaced reserved key before numerical execution", func(t *testing.T) {
		raw, _ := readDeepSeekV41Config(t)
		var original map[string]json.RawMessage
		if err := json.Unmarshal(raw, &original); err != nil {
			t.Fatal(err)
		}
		// Unknown metadata remains legal, including numbers outside float64.
		original["future_metadata"] = json.RawMessage(`{"large":1e1000,"array":[{"opaque":true}]}`)
		control, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		publications, projections := 0, 0
		run := func(raw []byte) error {
			var cfg Config
			if err := json.Unmarshal(raw, &cfg); err != nil {
				return err
			}
			m := &Model{Cfg: cfg}
			scratch := &v41ProjScratch{gatherOps: &v41DeviceGatherOps{}}
			scratch.gatherOps.setBF16Publication(func(layer int, _ string, row []float32) ([]float32, error) {
				publications++
				return v41LatentNormBF16Copy(layer, "placement probe", "publication", row)
			})
			scratch.groupedOutput = v41GroupedOutputFunc(func(_ int, _ []float32, _, _, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
				projections++
				return []float32{math.Float32frombits(0x3f818001)}, v41ProjectionHandled, nil
			})
			input := make([]float32, cfg.HeadDim)
			input[0] = math.Float32frombits(0x3f818001)
			_, err := m.v41GroupedOutputProjector(0, 1, cfg.HeadDim, 1, 1, 1, scratch)(input)
			return err
		}
		if err := run(control); err != nil || publications != 2 || projections != 1 {
			t.Fatalf("legacy positive control did not execute numerical seams: publications=%d projections=%d err=%v", publications, projections, err)
		}
		request, err := json.Marshal(v41UnqualifiedNumericalRequest())
		if err != nil {
			t.Fatal(err)
		}
		for _, placement := range []struct{ parent, key string }{
			{"text_config", "fak_v41_numerical_profile"},
			{"vision_config", "fak_v41_numerical_profile"},
			{"quantization_config", "fak_v41_numerical_profile"},
			{"future_metadata", "fak_v41_numerical_profile"},
			{"vision_config", "FAK_V41_NUMERICAL_PROFILE"},
			{"quantization_config", "FaK_v41_NuMeRiCaL_PrOfIlE"},
		} {
			t.Run(placement.parent+"/"+placement.key, func(t *testing.T) {
				var doc map[string]json.RawMessage
				if err := json.Unmarshal(control, &doc); err != nil {
					t.Fatal(err)
				}
				var nested map[string]json.RawMessage
				if err := json.Unmarshal(doc[placement.parent], &nested); err != nil {
					t.Fatal(err)
				}
				nested[placement.key] = request
				if placement.parent == "future_metadata" {
					delete(nested, "fak_v41_numerical_profile")
					nested["array"] = append(append([]byte(`[{"fak_v41_numerical_profile":`), request...), []byte(`}]`)...)
				}
				doc[placement.parent], err = json.Marshal(nested)
				if err != nil {
					t.Fatal(err)
				}
				bad, err := json.Marshal(doc)
				if err != nil {
					t.Fatal(err)
				}
				publications, projections = 0, 0
				if err := run(bad); !errors.Is(err, ErrV41NumericalProfile) || publications != 0 || projections != 0 {
					t.Fatalf("misplaced request reached arithmetic: publications=%d projections=%d err=%v", publications, projections, err)
				}
			})
		}
		// A last-key-wins map walk would erase this misplaced first occurrence.
		duplicate := append(append([]byte(`{"future_metadata":{"fak_v41_numerical_profile":`), request...), []byte(`},`)...)
		duplicate = append(duplicate, bytes.TrimSpace(control)[1:]...)
		publications, projections = 0, 0
		if err := run(duplicate); !errors.Is(err, ErrV41NumericalProfile) || publications != 0 || projections != 0 {
			t.Fatalf("duplicate parent hid request: publications=%d projections=%d err=%v", publications, projections, err)
		}
	})
	for _, headDim := range []int{32, v41KVLoraRank} {
		t.Run(strconv.Itoa(headDim), func(t *testing.T) {
			cfg := Config{ModelType: "deepseek41", HeadDim: headDim}
			full, err := v41ForwardGeometry(cfg)
			if err != nil || full != (headDim == v41KVLoraRank) {
				t.Fatalf("legacy geometry changed: full=%v err=%v", full, err)
			}
			m := &Model{Cfg: cfg}
			input := make([]float32, headDim)
			input[0] = math.Float32frombits(0x3f818001)
			want := uint32(0x3f818001)
			if full {
				want = 0x3f820000
			}
			calls := 0
			scratch := &v41ProjScratch{groupedOutput: v41GroupedOutputFunc(func(_ int, row []float32, _, _, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
				calls++
				if math.Float32bits(row[0]) != want {
					t.Fatal("legacy pre-wo_a boundary changed")
				}
				return []float32{input[0]}, v41ProjectionHandled, nil
			})}
			project := m.v41GroupedOutputProjector(0, 1, headDim, 1, 1, 1, scratch)
			out, err := project(input)
			if err != nil || calls != 1 || len(out) != 1 || math.Float32bits(out[0]) != want {
				t.Fatalf("legacy grouped boundary changed: out=%v calls=%d err=%v", out, calls, err)
			}
			publications := 0
			scratch.gatherOps = &v41DeviceGatherOps{}
			scratch.gatherOps.setBF16Publication(func(int, string, []float32) ([]float32, error) {
				publications++
				return nil, errors.New("unexpected numerical publication")
			})
			for _, request := range []*DeepSeekV41NumericalProfileRequest{v41UnqualifiedNumericalRequest(), {}} {
				calls = 0
				m.Cfg.DeepSeekV41 = &DeepSeekV41Config{NumericalProfile: request}
				if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41NumericalProfile) {
					t.Fatalf("request reached weight admission: %v", err)
				}
				project := m.v41GroupedOutputProjector(0, 1, headDim, 1, 1, 1, scratch)
				out, err := project(input)
				if out != nil || calls != 0 || publications != 0 || !errors.Is(err, ErrV41NumericalProfile) {
					t.Fatalf("request reached projection: out=%v calls=%d publications=%d err=%v", out, calls, publications, err)
				}
				if math.Float32bits(input[0]) != 0x3f818001 {
					t.Fatal("refusal modified input")
				}
			}
		})
	}
}
