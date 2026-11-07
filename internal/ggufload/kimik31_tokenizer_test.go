package ggufload

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Kimi K3.1 ("kimi-k31") tokenizer pin (ticket 1611-kimi-k31 leaf L3): the
// K3-inherited embedded BPE table must load through the same
// GGMLTokenizer -> tokenizer.FromGGML path every K3 file rides today, and a
// kimi-k31 header missing the required tokenizer.ggml.* keys refuses at the
// header naming the exact key.
//
// Synthetic fixtures only (no network, no real weights): the fixture table is
// a four-entry byte-level BPE shaped like the community K3/K3.1 converter's
// output (model "gpt2", pre "kimi-k2", CONTROL-marked special at id 2).
// Byte-parity against the HF reference tokenizer (PF-27 Unicode pin) stays
// gated on a staged real K3.1 GGUF; these tests pin the loader contract.

// TestKimiK31CompleteHeaderLoadsK3TokenizerPath is the done-condition witness:
// a complete kimi-k31 header admits AND routes its tokenizer through the K3
// loader path — File.Config records the pin, GGMLTokenizer extracts the table
// ok=true with the same family facts, and tokenizer.FromGGML builds a working
// tokenizer from exactly what the shared seam consumed.
func TestKimiK31CompleteHeaderLoadsK3TokenizerPath(t *testing.T) {
	f := &File{Metadata: synthKimiK31Meta("kimi-k31")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "kimik31" {
		t.Fatalf("ModelType = %q, want kimik31", cfg.ModelType)
	}
	pin := f.KimiK31Tokenizer
	if pin == nil {
		t.Fatal("KimiK31Tokenizer pin not recorded on admission")
	}
	if pin.Family != "kimi-k3" {
		t.Errorf("pin.Family = %q, want %q", pin.Family, "kimi-k3")
	}
	if pin.Model != "gpt2" || pin.Pre != "kimi-k2" {
		t.Errorf("pin hints = model:%q pre:%q, want gpt2/kimi-k2", pin.Model, pin.Pre)
	}
	if pin.VocabSize != 4 {
		t.Errorf("pin.VocabSize = %d, want 4", pin.VocabSize)
	}
	if pin.BOS != -1 || pin.EOS != 2 {
		t.Errorf("pin bounds = BOS:%d EOS:%d, want -1/2 (bos undeclared)", pin.BOS, pin.EOS)
	}
	if cfg.VocabSize != pin.VocabSize {
		t.Errorf("cfg.VocabSize = %d, pin.VocabSize = %d", cfg.VocabSize, pin.VocabSize)
	}
	if cfg.EOSTokenID != pin.EOS {
		t.Errorf("cfg.EOSTokenID = %d, pin.EOS = %d", cfg.EOSTokenID, pin.EOS)
	}

	// The shared loader path: the exact seam rawdecode/executor.go consumes.
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("GGMLTokenizer() ok=false: a complete kimi-k31 header must expose the embedded BPE table")
	}
	if len(gt.Tokens) != pin.VocabSize || len(gt.Merges) == 0 || len(gt.TokenTypes) != pin.VocabSize {
		t.Fatalf("extracted table = tokens:%d merges:%d types:%d, want 4/non-empty/4",
			len(gt.Tokens), len(gt.Merges), len(gt.TokenTypes))
	}
	if gt.Model != pin.Model || gt.Pre != pin.Pre || gt.BOS != pin.BOS || gt.EOS != pin.EOS {
		t.Errorf("extracted hints diverge from the pin: got model:%q pre:%q bos:%d eos:%d, want model:%q pre:%q bos:%d eos:%d",
			gt.Model, gt.Pre, gt.BOS, gt.EOS, pin.Model, pin.Pre, pin.BOS, pin.EOS)
	}

	// And the extracted table must build a working K3-path tokenizer whose
	// behavior matches the pinned facts: the BPE merge fires, the CONTROL
	// special matches literally, and the UNUSED pad slot is NOT special.
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	ids, err := tok.Encode("!!")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(ids) != 1 || ids[0] != 1 {
		t.Fatalf("Encode(%q) = %v, want [1] (the \"! !\" merge must fire inside one chunk)", "!!", ids)
	}
	back, err := tok.Decode([]int{1})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if back != "!!" {
		t.Errorf("Decode([1]) = %q, want %q", back, "!!")
	}
	specials := tok.SpecialTokens()
	if _, ok := specials[2]; !ok {
		t.Error("id 2 (<|end_of_msg|>) not marked special: the CONTROL token_type entry was lost")
	}
	if _, ok := specials[3]; ok {
		t.Error("id 3 ([PAD3]) marked special: the UNUSED token_type entry must stay ordinary")
	}
	sid, err := tok.Encode("<|end_of_msg|>")
	if err != nil {
		t.Fatalf("Encode(special): %v", err)
	}
	if len(sid) != 1 || sid[0] != 2 {
		t.Fatalf("Encode(<|end_of_msg|>) = %v, want [2] (literal special match)", sid)
	}
}

// TestKimiK31MissingTokenizerKeysRefused is the fail-closed witness: a
// kimi-k31-branded header missing any required tokenizer metadata key refuses
// at the header with a typed error that names the exact key — never a silent
// fall-through to an external tokenizer.json fallback.
func TestKimiK31MissingTokenizerKeysRefused(t *testing.T) {
	required := []string{
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.merges",
		"tokenizer.ggml.token_type",
	}
	for _, key := range required {
		meta := synthKimiK31Meta("kimi-k31")
		delete(meta, key)
		f := &File{Metadata: meta}
		_, err := f.Config()
		if err == nil {
			t.Errorf("%s absent: Config succeeded, want typed refusal", key)
			continue
		}
		if !strings.Contains(err.Error(), key) {
			t.Errorf("%s absent: refusal %q does not name the offending key", key, err)
		}
		if f.KimiK31Tokenizer != nil {
			t.Errorf("%s absent: pin recorded despite refusal", key)
		}
	}
}

// TestKimiK31MalformedTokenizerTableRefused pins the shape guards: present but
// unusable tables refuse naming the key — empty merges, a non-array vocab, a
// token_type array shorter than the vocab, and declared bounds pointing past
// the end of the embedded vocab.
func TestKimiK31MalformedTokenizerTableRefused(t *testing.T) {
	cases := []struct {
		name string
		edit func(map[string]Value)
		want string // substring the refusal must carry
	}{
		{
			name: "empty merges",
			edit: func(m map[string]Value) {
				m["tokenizer.ggml.merges"] = Value{Type: TypeArray, Value: []Value{}}
			},
			want: "tokenizer.ggml.merges",
		},
		{
			name: "non-array tokens",
			edit: func(m map[string]Value) {
				m["tokenizer.ggml.tokens"] = Value{Type: TypeString, Value: "not-an-array"}
			},
			want: "tokenizer.ggml.tokens",
		},
		{
			name: "short token_type",
			edit: func(m map[string]Value) {
				full, _ := m["tokenizer.ggml.token_type"]
				items := full.Value.([]Value)[:3]
				m["tokenizer.ggml.token_type"] = Value{Type: TypeArray, Value: items}
			},
			want: "tokenizer.ggml.token_type",
		},
		{
			name: "eos past vocab",
			edit: func(m map[string]Value) {
				m["tokenizer.ggml.eos_token_id"] = Value{Type: TypeUint32, Value: uint32(99)}
			},
			want: "tokenizer.ggml.eos_token_id",
		},
		{
			name: "bos past vocab",
			edit: func(m map[string]Value) {
				m["tokenizer.ggml.bos_token_id"] = Value{Type: TypeUint32, Value: uint32(4)}
			},
			want: "tokenizer.ggml.bos_token_id",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			meta := synthKimiK31Meta("kimi-k31")
			tc.edit(meta)
			f := &File{Metadata: meta}
			_, err := f.Config()
			if err == nil {
				t.Fatalf("Config succeeded, want typed refusal for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal %q does not name %q", err, tc.want)
			}
			if f.KimiK31Tokenizer != nil {
				t.Fatal("pin recorded despite refusal")
			}
		})
	}
}

// TestLegacyKimiArchTokenizerBehaviorUnchanged pins the sibling-leak from the
// tokenizer side: the pin must change NOTHING for the legacy K2/K3 arches.
// (a) A genuine K3-branded header with NO tokenizer keys at all still loads
// via the deepseek2 alias (no new global requirement); (b) with the embedded
// table it extracts arch-agnostic as before — Pre falls back to
// general.architecture when tokenizer.ggml.pre is absent — and no kimik31 pin
// is ever recorded on a non-kimik31 file.
func TestLegacyKimiArchTokenizerBehaviorUnchanged(t *testing.T) {
	// (a) No tokenizer keys at all: the legacy header keeps loading untouched.
	f := &File{Metadata: synthKimiMeta("kimi-k3")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config(kimi-k3, no tokenizer keys): %v", err)
	}
	if cfg.ModelType != "deepseek2" {
		t.Fatalf("ModelType = %q, want deepseek2 (legacy alias must be untouched)", cfg.ModelType)
	}
	if f.KimiK31Tokenizer != nil {
		t.Fatal("KimiK31Tokenizer pin recorded on a non-kimik31 file")
	}
	if _, ok := f.GGMLTokenizer(); ok {
		t.Fatal("GGMLTokenizer ok=true without any tokenizer.ggml.* keys")
	}

	// (b) With the embedded table (no .pre hint): extraction stays
	// arch-agnostic and the shared loader still builds from it.
	meta := synthKimiMeta("kimi-k3")
	for k, v := range kimiK31SynthTokenizerTable(false) {
		meta[k] = v
	}
	f2 := &File{Metadata: meta}
	cfg2, err := f2.Config()
	if err != nil {
		t.Fatalf("Config(kimi-k3, embedded table): %v", err)
	}
	if cfg2.ModelType != "deepseek2" {
		t.Fatalf("ModelType = %q, want deepseek2", cfg2.ModelType)
	}
	if f2.KimiK31Tokenizer != nil {
		t.Fatal("KimiK31Tokenizer pin recorded on a non-kimik31 file")
	}
	gt, ok := f2.GGMLTokenizer()
	if !ok {
		t.Fatal("GGMLTokenizer ok=false: the legacy extraction regressed")
	}
	if gt.Pre != "kimi-k3" {
		t.Errorf("gt.Pre = %q, want %q (legacy general.architecture fallback when .pre is absent)", gt.Pre, "kimi-k3")
	}
	if len(gt.Tokens) != 4 || len(gt.TokenTypes) != 4 {
		t.Fatalf("extracted table = tokens:%d types:%d, want 4/4", len(gt.Tokens), len(gt.TokenTypes))
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	ids, err := tok.Encode("!!")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(ids) != 1 || ids[0] != 1 {
		t.Fatalf("Encode(%q) = %v, want [1]", "!!", ids)
	}
}
