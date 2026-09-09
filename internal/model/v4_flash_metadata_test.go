package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
)

func TestDeepSeekV4FlashMetadataRetainsOfficialForwardAxes(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}

	want := map[string]any{
		"compress_ratios":   deepSeekV4FlashCompressionRatios(),
		"hc_mult":           4,
		"hc_eps":            1e-6,
		"hc_sinkhorn_iters": 20,
		"o_groups":          8,
		"o_lora_rank":       1024,
	}
	for key, value := range want {
		wantJSON, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got[key], wantJSON) {
			t.Errorf("round-tripped %s = %s, want %s from %s@%s", key, got[key], wantJSON, deepSeekV4FlashModelID, deepSeekV4FlashRevision)
		}
	}
}

func TestDeepSeekV4FlashMetadataAdmissionRunsBeforeWeightIO(t *testing.T) {
	swappedSchedule := deepSeekV4FlashCompressionRatios()
	swappedSchedule[2] = 128
	nonzeroDSparkTail := deepSeekV4FlashCompressionRatios()
	nonzeroDSparkTail[len(nonzeroDSparkTail)-1] = 4
	cases := []struct {
		name  string
		key   string
		value any
	}{
		{name: "compression schedule length", key: "compress_ratios", value: []int{0, 4, 128}},
		{name: "compression schedule value", key: "compress_ratios", value: swappedSchedule},
		{name: "compression DSpark tail", key: "compress_ratios", value: nonzeroDSparkTail},
		{name: "hyperconnection streams", key: "hc_mult", value: 3},
		{name: "hyperconnection epsilon", key: "hc_eps", value: 0},
		{name: "sinkhorn iterations", key: "hc_sinkhorn_iters", value: 0},
		{name: "output groups", key: "o_groups", value: 7},
		{name: "output low rank width", key: "o_lora_rank", value: 512},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := deepSeekV4FlashConfigMutation(t, tc.key, tc.value)
			opened := 0
			opener := func(string) (*safetensorsFile, error) {
				opened++
				return nil, errors.New("weight opener reached")
			}
			if _, err := loadSafetensorsQuantDir(t.TempDir(), cfg, opener); !errors.Is(err, ErrV4ConfigAdmission) {
				t.Fatalf("malformed %s error = %v, want ErrV4ConfigAdmission", tc.key, err)
			}
			if opened != 0 {
				t.Fatalf("malformed %s opened %d weight files before admission", tc.key, opened)
			}
		})
	}

	if err := AdmitDeepSeekV4Config(pinnedV4Config()); err != nil {
		t.Fatalf("pinned Pro config regressed when Flash metadata became required: %v", err)
	}
}

func deepSeekV4FlashCompressionRatios() []int {
	return []int{0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0}
}

func deepSeekV4FlashConfigMutation(t *testing.T, key string, value any) Config {
	t.Helper()
	raw, _ := readDeepSeekV4FlashConfig(t)
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	doc[key] = value
	mutated, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(mutated, &cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}
