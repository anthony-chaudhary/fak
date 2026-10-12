package ggufload

import "testing"

// A deepseek41 GGUF carries its tokenizer's BOS id into the config so a raw
// completion can start with it; an undeclared BOS stays nil.
// fak-test:runtime fast est=10ms
func TestDeepSeek41GGUFCarriesBOSTokenID(t *testing.T) {
	f := &File{Metadata: ds41Meta("deepseek41")}
	f.Metadata["tokenizer.ggml.bos_token_id"] = Value{Type: TypeUint32, Value: uint32(0)}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.BOSTokenID == nil || *cfg.BOSTokenID != 0 {
		t.Fatalf("BOSTokenID = %v, want 0", cfg.BOSTokenID)
	}
	cfg, err = (&File{Metadata: ds41Meta("deepseek41")}).Config()
	if err != nil {
		t.Fatalf("Config without BOS: %v", err)
	}
	if cfg.BOSTokenID != nil {
		t.Fatalf("undeclared BOS became %d", *cfg.BOSTokenID)
	}
}
