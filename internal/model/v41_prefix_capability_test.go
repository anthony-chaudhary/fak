package model

import (
	"strings"
	"testing"
)

// DeepSeek V4.1 requires shared-KV, compressed-attention, and Engram state that
// KVCache and PrefixSnapshot do not yet represent. Until they carry that state,
// cloning KVCache alone is not a complete or safe prefix.
func TestV41KVPrefixReuseFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"wrapper identity", Config{ModelType: "deepseek_v41"}},
		{"text identity", Config{ModelType: "deepseek_v41_text"}},
		{"retained metadata", Config{DeepSeekV41: &DeepSeekV41Config{
			WrapperModelType: "deepseek_v41",
			TextModelType:    "deepseek_v41_text",
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.cfg.KVPrefixReuseSupported() {
				t.Fatal("V4.1 KV prefix reuse advertised before snapshots retain V4.1 session state")
			}
		})
	}
}

func TestV41PrefixCloneConstructorsRefuseByArchitecture(t *testing.T) {
	for _, modelType := range []string{"deepseek_v41", "deepseek_v41_text"} {
		t.Run(modelType+"/session", func(t *testing.T) {
			m := &Model{Cfg: Config{ModelType: modelType}}
			assertV41PrefixPanic(t, modelType, func() {
				m.SessionFromPrefix(NewKVCache(m.Cfg))
			})
		})
		t.Run(modelType+"/batch", func(t *testing.T) {
			m := &Model{Cfg: Config{ModelType: modelType}}
			assertV41PrefixPanic(t, modelType, func() {
				m.NewBatchFromPrefixReserve(NewKVCache(m.Cfg), 2, 4)
			})
		})
	}
}

func TestV41PrefixCapabilityGuardIsNarrow(t *testing.T) {
	for _, modelType := range []string{"llama", "deepseek_v4"} {
		cfg := Config{ModelType: modelType}
		if !cfg.KVPrefixReuseSupported() {
			t.Errorf("legacy cached architecture %q lost KV prefix reuse", modelType)
		}
	}
}

func assertV41PrefixPanic(t *testing.T, modelType string, call func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("prefix clone constructor accepted incomplete V4.1 session state")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("prefix clone constructor panicked with %T, want named string refusal", r)
		}
		want := strings.NewReplacer("_", "", "-", "").Replace(modelType)
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal %q does not name architecture %q", msg, want)
		}
	}()
	call()
}
