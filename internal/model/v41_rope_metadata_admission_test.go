package model

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestV41RopeMetadataAdmission reaches the real forward admission gate without
// allocating weights. The nonempty placeholder manifest passes the weightless
// fence; malformed rotary metadata must refuse before any tensor is inspected.
// fak-test:runtime fast est=5ms lane=default
func TestV41RopeMetadataAdmission(t *testing.T) {
	config := func(ratio int) Config {
		return Config{
			// A valid ordinary base must not hide corrupt compressed metadata.
			ModelType: "deepseek41", NumLayers: 1, HeadDim: 32, RopeTheta: 10000,
			DeepSeekV41: &DeepSeekV41Config{
				CompressRatios: []int{ratio}, CompressRopeTheta: 160000,
			},
		}
	}
	for _, ratio := range []int{0, 1, 2} {
		for _, bad := range []struct {
			name  string
			theta float64
		}{
			{"zero", 0}, {"negative", -1}, {"nan", math.NaN()},
			{"positive_infinity", math.Inf(1)}, {"negative_infinity", math.Inf(-1)},
		} {
			t.Run(fmt.Sprintf("ratio%d/%s", ratio, bad.name), func(t *testing.T) {
				cfg := config(ratio)
				key := "compress_rope_theta"
				if ratio == 0 {
					cfg.RopeTheta = bad.theta
					key = "rope_theta"
				} else {
					cfg.DeepSeekV41.CompressRopeTheta = bad.theta
				}
				m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{"placeholder": {}}}
				err := m.v41ForwardAdmitted()
				var stage *V41ForwardError
				if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &stage) || stage.Stage != v41StageAttention || stage.Layer != 0 || !strings.Contains(err.Error(), key) {
					t.Fatalf("admission error = %v, want layer-0 attention refusal naming %s", err, key)
				}
			})
		}
	}

	// Valid active bases pass. An unused base or unreachable suffix does not
	// become an execution requirement for a deliberately narrowed fixture.
	for _, ratio := range []int{0, 1, 2} {
		cfg := config(ratio)
		if ratio == 0 {
			cfg.DeepSeekV41.CompressRopeTheta = math.NaN()
			cfg.DeepSeekV41.CompressRatios = []int{0, 2}
		} else {
			cfg.RopeTheta = math.NaN()
		}
		if err := v41RopeForwardAdmitted(cfg); err != nil {
			t.Fatalf("valid selected base for ratio %d: %v", ratio, err)
		}
	}
}
