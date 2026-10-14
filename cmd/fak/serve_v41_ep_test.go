package main

import (
	"strings"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms lane=default
func TestServeV41ExpertParallelRemainsUnqualified(t *testing.T) {
	for _, identity := range []string{"deepseek41", "deepseek_v41", "deepseek_v41_text"} {
		err := admitServeV41ExpertParallel(fakmodel.Config{ModelType: identity})
		if err == nil || !strings.Contains(err.Error(), "digest-bound mixed Q2/Q3") {
			t.Fatalf("identity %s: %v", identity, err)
		}
	}
	if err := admitServeV41ExpertParallel(fakmodel.Config{ModelType: "glm_moe_dsa"}); err != nil {
		t.Fatalf("existing GLM path changed: %v", err)
	}
}
