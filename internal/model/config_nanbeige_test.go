package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"
)

func TestNanbeige42ConfigAdmission(t *testing.T) {
	validJSON := `{
		"model_type": "nanbeige",
		"num_hidden_layers": 22,
		"num_loops": 2,
		"head_dim": 128,
		"hidden_size": 3072,
		"num_attention_heads": 48,
		"num_key_value_heads": 8
	}`

	t.Run("valid Nanbeige4.2 config", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(validJSON), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if cfg.NumLayers != 22 {
			t.Fatalf("NumLayers = %d, want 22", cfg.NumLayers)
		}
		if cfg.NumLoops != 2 {
			t.Fatalf("NumLoops = %d, want 2", cfg.NumLoops)
		}
		if cfg.HeadDim != 128 {
			t.Fatalf("HeadDim = %d, want 128", cfg.HeadDim)
		}
	})

	t.Run("valid Nanbeige4.2 default loops", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if cfg.NumLayers != 22 {
			t.Fatalf("NumLayers = %d, want 22", cfg.NumLayers)
		}
		if cfg.NumLoops != 2 {
			t.Fatalf("NumLoops = %d, want default 2", cfg.NumLoops)
		}
		if cfg.HeadDim != 128 {
			t.Fatalf("HeadDim = %d, want 128", cfg.HeadDim)
		}
	})

	t.Run("valid Nanbeige4.2 loops hint", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if cfg.NumLoops != 2 {
			t.Fatalf("NumLoops = %d, want 2", cfg.NumLoops)
		}
	})

	t.Run("valid Nanbeige4.2 recurrent_loops hint", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"recurrent_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if cfg.NumLoops != 2 {
			t.Fatalf("NumLoops = %d, want 2", cfg.NumLoops)
		}
	})

	t.Run("valid Nanbeige4.2 shared_cache false", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"num_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8,
			"shared_cache": false
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if cfg.NumLayers != 22 || cfg.NumLoops != 2 || cfg.HeadDim != 128 {
			t.Fatalf("config mismatch: %+v", cfg)
		}
	})

	t.Run("rejection: num_hidden_layers != 22", func(t *testing.T) {
		for _, layers := range []int{16, 24, 32} {
			js := fmt.Sprintf(`{
				"model_type": "nanbeige",
				"num_hidden_layers": %d,
				"num_loops": 2,
				"head_dim": 128,
				"hidden_size": 3072,
				"num_attention_heads": 48,
				"num_key_value_heads": 8
			}`, layers)
			var cfg Config
			err := json.Unmarshal([]byte(js), &cfg)
			if err == nil {
				t.Fatalf("layers=%d accepted, want rejection", layers)
			}
			var unvErr *UnsupportedNanbeigeVariantError
			if !errors.As(err, &unvErr) {
				t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
			}
		}
	})

	t.Run("rejection: num_loops != 2", func(t *testing.T) {
		testCases := []string{
			`{"model_type":"nanbeige","num_hidden_layers":22,"num_loops":1,"head_dim":128,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			`{"model_type":"nanbeige","num_hidden_layers":22,"num_loops":3,"head_dim":128,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			`{"model_type":"nanbeige","num_hidden_layers":22,"loops":4,"head_dim":128,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			`{"model_type":"nanbeige","num_hidden_layers":22,"recurrent_loops":5,"head_dim":128,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
		}
		for _, js := range testCases {
			var cfg Config
			err := json.Unmarshal([]byte(js), &cfg)
			if err == nil {
				t.Fatalf("config %s accepted, want rejection", js)
			}
			var unvErr *UnsupportedNanbeigeVariantError
			if !errors.As(err, &unvErr) {
				t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
			}
		}
	})

	t.Run("rejection: head_dim != 128", func(t *testing.T) {
		testCases := []struct {
			name string
			js   string
		}{
			{
				name: "explicit head_dim 64",
				js:   `{"model_type":"nanbeige","num_hidden_layers":22,"num_loops":2,"head_dim":64,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			},
			{
				name: "omitted head_dim (must NOT derive 3072/48=64)",
				js:   `{"model_type":"nanbeige","num_hidden_layers":22,"num_loops":2,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			},
			{
				name: "explicit head_dim 256",
				js:   `{"model_type":"nanbeige","num_hidden_layers":22,"num_loops":2,"head_dim":256,"hidden_size":3072,"num_attention_heads":48,"num_key_value_heads":8}`,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				var cfg Config
				err := json.Unmarshal([]byte(tc.js), &cfg)
				if err == nil {
					t.Fatalf("config %s accepted, want rejection", tc.js)
				}
				var unvErr *UnsupportedNanbeigeVariantError
				if !errors.As(err, &unvErr) {
					t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
				}
			})
		}
	})

	t.Run("rejection: shared_cache=true", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"num_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8,
			"shared_cache": true
		}`
		var cfg Config
		err := json.Unmarshal([]byte(js), &cfg)
		if err == nil {
			t.Fatal("shared_cache=true accepted, want rejection")
		}
		var unvErr *UnsupportedNanbeigeVariantError
		if !errors.As(err, &unvErr) {
			t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
		}
	})

	t.Run("rejection: loop_share_kv=true", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"num_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8,
			"loop_share_kv": true
		}`
		var cfg Config
		err := json.Unmarshal([]byte(js), &cfg)
		if err == nil {
			t.Fatal("loop_share_kv=true accepted, want rejection")
		}
		var unvErr *UnsupportedNanbeigeVariantError
		if !errors.As(err, &unvErr) {
			t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
		}
	})

	t.Run("rejection: skip_loop_final_norm=true", func(t *testing.T) {
		js := `{
			"model_type": "nanbeige",
			"num_hidden_layers": 22,
			"num_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8,
			"skip_loop_final_norm": true
		}`
		var cfg Config
		err := json.Unmarshal([]byte(js), &cfg)
		if err == nil {
			t.Fatal("skip_loop_final_norm=true accepted, want rejection")
		}
		var unvErr *UnsupportedNanbeigeVariantError
		if !errors.As(err, &unvErr) {
			t.Fatalf("err = %v (%T), want *UnsupportedNanbeigeVariantError", err, err)
		}
	})

	t.Run("valid Nanbeige architecture array seam", func(t *testing.T) {
		js := `{
			"architectures": ["NanbeigeForCausalLM"],
			"num_hidden_layers": 22,
			"num_loops": 2,
			"head_dim": 128,
			"hidden_size": 3072,
			"num_attention_heads": 48,
			"num_key_value_heads": 8
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unexpected unmarshal error: %v", err)
		}
		if !cfg.IsNanbeige() {
			t.Fatal("IsNanbeige() = false, want true for architectures: [NanbeigeForCausalLM]")
		}
		if cfg.NumLayers != 22 || cfg.NumLoops != 2 || cfg.HeadDim != 128 {
			t.Fatalf("config mismatch: %+v", cfg)
		}
	})
}
