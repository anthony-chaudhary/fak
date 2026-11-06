package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func TestServeSelectDirLoadArm(t *testing.T) {
	cases := []struct {
		name    string
		files   map[string]string
		want    serveDirLoadArm
		wantErr error
	}{
		{"fp8 qwen keeps safetensors arm", map[string]string{
			"config.json": `{"model_type":"qwen3_5","quantization_config":{"quant_method":"fp8","fmt":"e4m3"}}`,
		}, serveDirArmSafetensors, nil},
		{"modelopt nvfp4 in config.json", map[string]string{
			"config.json": `{"model_type":"qwen3_5","quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16}}`,
		}, serveDirArmModelOptNVFP4, nil},
		{"modelopt nvfp4 in hf_quant_config.json", map[string]string{
			"config.json":          `{"model_type":"qwen3_5"}`,
			"hf_quant_config.json": `{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"NVFP4","group_size":16}}`,
		}, serveDirArmModelOptNVFP4, nil},
		{"modelopt fp8 refuses typed", map[string]string{
			"config.json": `{"model_type":"qwen3_5","quantization_config":{"quant_method":"modelopt","quant_algo":"FP8"}}`,
		}, "", fakmodel.ErrModelOptNVFP4Config},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, body := range tc.files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			got, err := serveSelectDirLoadArm(dir)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("arm = %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}
