package model

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestAuditAllNativeLoadersRefuseV41BeforeWeightIO(t *testing.T) {
	raw, cfg := readDeepSeekV41Config(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}

	tests := map[string]func() error{
		"awq directory": func() error { _, err := LoadAWQ(dir); return err },
		"awq file": func() error {
			_, err := loadAWQSafetensors(filepath.Join(dir, "absent.safetensors"), cfg)
			return err
		},
		"awq group":       func() error { _, err := loadAWQGroupSafetensors(nil, cfg); return err },
		"awq pytorch bin": func() error { _, err := loadAWQPytorchBin("absent.bin", cfg); return err },
		"exl2 directory":  func() error { _, err := LoadEXL2(dir); return err },
		"exl2 file":       func() error { _, err := loadEXL2Safetensors(nil, cfg); return err },
	}
	for name, load := range tests {
		t.Run(name, func(t *testing.T) {
			if err := load(); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("error=%v, want ErrV41NativeUnsupported before weight I/O", err)
			}
		})
	}
}
