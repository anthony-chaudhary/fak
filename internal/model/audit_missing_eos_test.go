package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"
)

const auditMissingEOSConfigJSON = `{
	"model_type":"llama",
	"vocab_size":4,
	"hidden_size":32,
	"intermediate_size":64,
	"num_hidden_layers":1,
	"num_attention_heads":2,
	"num_key_value_heads":2,
	"head_dim":16,
	"rms_norm_eps":1e-6,
	"rope_theta":10000,
	"tie_word_embeddings":true
}`

func auditConfigWithoutEOS(t *testing.T) Config {
	t.Helper()
	var cfg Config
	if err := json.Unmarshal([]byte(auditMissingEOSConfigJSON), &cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestAuditMissingEOSDoesNotDefaultToTokenZero(t *testing.T) {
	for _, tc := range []struct {
		name string
		data string
	}{
		{"minimal", `{}`},
		{"generation_config", auditMissingEOSConfigJSON},
		{"nested", `{"text_config":{"model_type":"llama"}}`},
		{"null_text_config", `{"text_config":null}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal([]byte(tc.data), &cfg); err != nil {
				t.Fatal(err)
			}
			if cfg.IsEOS(0) || cfg.EOSTokenID != -1 || len(cfg.EOSTokenIDs) != 0 {
				t.Fatalf("absent eos_token_id must use the no-early-stop convention: EOSTokenID=%d EOSTokenIDs=%v IsEOS(0)=%v", cfg.EOSTokenID, cfg.EOSTokenIDs, cfg.IsEOS(0))
			}
		})
	}
}

// This exercises the real CPU Prefill/Step/Generate path with a tiny in-memory
// checkpoint. Zero projections and embeddings force greedy decode to select 0;
// unit norm weights keep the normalization path well-conditioned.
// fak-test:runtime fast est=20ms lane=default
func TestAuditMissingEOSFalseStopsGeneration(t *testing.T) {
	cfg := auditConfigWithoutEOS(t)
	m := NewSynthetic(cfg)
	for name, meta := range m.manifest {
		raw := m.raw[meta.Offset : meta.Offset+meta.Nbytes]
		fill := uint32(0)
		if strings.HasSuffix(name, "layernorm.weight") || name == "model.norm.weight" {
			fill = math.Float32bits(1)
		}
		for off := 0; off < len(raw); off += 4 {
			binary.LittleEndian.PutUint32(raw[off:], fill)
		}
	}
	scalarZero, listZero, fixedLength := cfg, cfg, cfg
	scalarZero.EOSTokenID = 0
	listZero.EOSTokenIDs = []int{2, 0}
	fixedLength.EOSTokenID = -1
	for _, tc := range []struct {
		name string
		cfg  Config
		want []int
	}{
		{"absent_eos", cfg, []int{0, 0, 0}},
		{"explicit_scalar_zero", scalarZero, []int{0}},
		{"explicit_list_zero", listZero, []int{0}},
		{"fixed_length_sentinel", fixedLength, []int{0, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m.Cfg = tc.cfg
			session := m.NewSession()
			defer session.Close()
			got := session.Generate([]int{1}, 3)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("all-zero logits: Generate = %v, want %v (EOSTokenID=%d EOSTokenIDs=%v)", got, tc.want, m.Cfg.EOSTokenID, m.Cfg.EOSTokenIDs)
			}
		})
	}
}

func TestAuditMissingEOSPreservesExplicitAndNestedEOS(t *testing.T) {
	for _, tc := range []struct {
		name string
		data string
		id   int
		ids  []int
	}{
		{"scalar_zero", `{"eos_token_id":0}`, 0, []int{0}},
		{"scalar_nonzero", `{"eos_token_id":2}`, 2, []int{2}},
		{"list", `{"eos_token_id":[2,0,3]}`, 2, []int{2, 0, 3}},
		{"explicit_null", `{"eos_token_id":null}`, 0, nil},
		{"explicit_empty_list", `{"eos_token_id":[]}`, 0, nil},
		{"nested_scalar", `{"text_config":{"eos_token_id":2}}`, 2, []int{2}},
		{"nested_list", `{"text_config":{"eos_token_id":[2,3]}}`, 2, []int{2, 3}},
		{"wrapper_scalar_overrides_nested_list", `{"text_config":{"eos_token_id":[2,3]},"eos_token_id":0}`, 0, []int{0}},
		{"wrapper_list_overrides_nested_scalar", `{"text_config":{"eos_token_id":2},"eos_token_id":[3,0]}`, 3, []int{3, 0}},
		// Explicit null has historically left the nested auxiliary EOS value intact.
		{"wrapper_null_keeps_nested", `{"text_config":{"eos_token_id":[2,3]},"eos_token_id":null}`, 2, []int{2, 3}},
		{"wrapper_empty_list_clears_nested", `{"text_config":{"eos_token_id":[2,3]},"eos_token_id":[]}`, 0, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal([]byte(tc.data), &cfg); err != nil {
				t.Fatal(err)
			}
			auditMissingEOSCheck(t, cfg, tc.id, tc.ids)
		})
	}
}

func TestAuditMissingEOSPreservesExplicitEOSOnReusedConfig(t *testing.T) {
	for _, tc := range []struct {
		name string
		data string
		id   int
		ids  []int
	}{
		{"scalar_replaces_list", `{"eos_token_id":0}`, 0, []int{0}},
		{"list_replaces_list", `{"eos_token_id":[0,1]}`, 0, []int{0, 1}},
		{"null_keeps_scalar_and_clears_list", `{"eos_token_id":null}`, 2, nil},
		{"empty_list_keeps_scalar", `{"eos_token_id":[]}`, 2, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{EOSTokenID: 2, EOSTokenIDs: []int{2, 3}}
			if err := json.Unmarshal([]byte(tc.data), &cfg); err != nil {
				t.Fatal(err)
			}
			auditMissingEOSCheck(t, cfg, tc.id, tc.ids)
		})
	}
}

func TestAuditMissingEOSRejectsMalformedEOS(t *testing.T) {
	for _, data := range []string{
		`{"eos_token_id":`,
		`{"eos_token_id":"bad"}`,
		`{"eos_token_id":[2,"bad"]}`,
		`{"text_config":{"eos_token_id":false}}`,
	} {
		t.Run(data, func(t *testing.T) {
			cfg := Config{EOSTokenID: 2, EOSTokenIDs: []int{2, 3}}
			if err := json.Unmarshal([]byte(data), &cfg); err == nil {
				t.Fatal("malformed EOS accepted")
			}
			// Do not require Config-wide rollback: JSON decoding can update other
			// fields before reporting an error. No partially parsed EOS is published.
			auditMissingEOSCheck(t, cfg, 2, []int{2, 3})
		})
	}
}

func auditMissingEOSCheck(t *testing.T, cfg Config, id int, ids []int) {
	t.Helper()
	if cfg.EOSTokenID != id || !slices.Equal(cfg.EOSTokenIDs, ids) {
		t.Fatalf("EOS = %d/%v, want %d/%v", cfg.EOSTokenID, cfg.EOSTokenIDs, id, ids)
	}
	for token := 0; token < 4; token++ {
		want := token == id
		if len(ids) > 0 {
			want = slices.Contains(ids, token)
		}
		if got := cfg.IsEOS(token); got != want {
			t.Fatalf("IsEOS(%d) = %v, want %v for EOS %d/%v", token, got, want, id, ids)
		}
	}
}
