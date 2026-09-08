package qwen38quantrun

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const (
	// GGUFTokenizerIdentity names the byte-preserving canonicalization used by
	// PromptTokenPacket v2. It is deliberately an algorithm identity, not a model
	// repository label: the packet's artifact SHA binds the model bytes while the
	// digest below binds the tokenizer metadata embedded in those bytes.
	GGUFTokenizerIdentity = "gguf-tokenizer-identity/v1"

	ggufTokenizerDigestDomain = "fak.gguf-tokenizer-identity.v1"
	ggufTemplateDigestDomain  = "fak.gguf-chat-template-identity.v1"
	emptySHA256               = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// PromptPacketGGUFIdentity is the tokenizer/template identity derived from one
// parsed GGUF header. It does not replace ArtifactSHA256: these digests cover
// only the exact metadata that controls tokenization and chat rendering.
type PromptPacketGGUFIdentity struct {
	TokenizerIdentity string
	TokenizerDigest   string
	TemplateDigest    string
	Architecture      string
	TokenizerPre      string
	VocabSize         int
}

// DerivePromptPacketGGUFIdentity reads only the GGUF header and derives the
// identities required by a v2 prompt-token packet. The caller must separately
// bind the packet to the SHA-256 of the complete artifact.
func DerivePromptPacketGGUFIdentity(path string) (PromptPacketGGUFIdentity, error) {
	gg, err := ggufload.Open(path)
	if err != nil {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: %w", err)
	}
	return derivePromptPacketGGUFIdentity(gg)
}

func derivePromptPacketGGUFIdentity(gg *ggufload.File) (PromptPacketGGUFIdentity, error) {
	if gg == nil {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: nil GGUF header")
	}
	cfg, err := gg.Config()
	if err != nil {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: Qwen3.8 config: %w", err)
	}
	arch, ok := gg.String("general.architecture")
	if !ok || arch != "qwen35" || cfg.Name != "Qwen3.8-27B" || cfg.ModelType != "qwen35" {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: supported Qwen3.8-27B qwen35 architecture is required")
	}
	model, ok := gg.String("tokenizer.ggml.model")
	if !ok || model == "" {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: non-empty tokenizer.ggml.model is required")
	}
	pre, ok := gg.String("tokenizer.ggml.pre")
	if !ok || pre != "qwen35" {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: tokenizer.ggml.pre must be explicitly qwen35")
	}
	tokens, ok := gg.StringArray("tokenizer.ggml.tokens")
	if !ok || len(tokens) == 0 {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: non-empty tokenizer.ggml.tokens is required")
	}
	if cfg.VocabSize != len(tokens) {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: config vocabulary %d does not match tokenizer.ggml.tokens length %d", cfg.VocabSize, len(tokens))
	}
	merges, ok := gg.StringArray("tokenizer.ggml.merges")
	if !ok || len(merges) == 0 {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: non-empty tokenizer.ggml.merges is required")
	}
	tokenTypes, ok := gg.Int32Array("tokenizer.ggml.token_type")
	if !ok || len(tokenTypes) != len(tokens) {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: tokenizer.ggml.token_type length must equal vocabulary length")
	}
	for key := range gg.Metadata {
		if strings.HasPrefix(key, "tokenizer.ggml.") && strings.HasSuffix(key, "_token_id") {
			id, ok := gg.Uint64(key)
			if !ok || id >= uint64(len(tokens)) {
				return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: %s must be an in-vocabulary non-negative integer", key)
			}
		}
	}
	chatTemplate, ok := gg.String("tokenizer.chat_template")
	if !ok || chatTemplate == "" {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: non-empty string tokenizer.chat_template is required")
	}

	tokenizerKeys := []string{"general.architecture"}
	for key := range gg.Metadata {
		if strings.HasPrefix(key, "tokenizer.ggml.") {
			tokenizerKeys = append(tokenizerKeys, key)
		}
	}
	tokenizerDigest, err := digestGGUFMetadata(gg.Metadata, ggufTokenizerDigestDomain, tokenizerKeys)
	if err != nil {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: tokenizer metadata: %w", err)
	}
	templateDigest, err := digestGGUFMetadata(gg.Metadata, ggufTemplateDigestDomain, []string{"tokenizer.chat_template"})
	if err != nil {
		return PromptPacketGGUFIdentity{}, fmt.Errorf("derive prompt packet GGUF identity: chat template: %w", err)
	}
	return PromptPacketGGUFIdentity{
		TokenizerIdentity: GGUFTokenizerIdentity,
		TokenizerDigest:   tokenizerDigest,
		TemplateDigest:    templateDigest,
		Architecture:      arch,
		TokenizerPre:      pre,
		VocabSize:         len(tokens),
	}, nil
}

// ValidatePromptPacketGGUFIdentity is the strict comparison-eligibility gate.
// Generic v2 packets may carry synthetic identities for device-free tests, but
// a physical Qwen3.8 comparison must bind to identities derived from the exact
// GGUF header and keep every supplied prompt/stop token inside that vocabulary.
func ValidatePromptPacketGGUFIdentity(packet PromptTokenPacket, derived PromptPacketGGUFIdentity) error {
	if err := VerifyPromptPacket(packet); err != nil {
		return fmt.Errorf("prompt packet GGUF identity: invalid packet: %w", err)
	}
	if packet.Schema != PromptTokenPacketSchema {
		return fmt.Errorf("prompt packet GGUF identity: schema %q is not comparison eligible", packet.Schema)
	}
	if derived.TokenizerIdentity != GGUFTokenizerIdentity || derived.Architecture != "qwen35" || derived.TokenizerPre != "qwen35" || derived.VocabSize <= 0 {
		return fmt.Errorf("prompt packet GGUF identity: derived Qwen3.8 identity is incomplete")
	}
	if !validCanonicalSHA256(derived.TokenizerDigest) || derived.TokenizerDigest == emptySHA256 || !validCanonicalSHA256(derived.TemplateDigest) || derived.TemplateDigest == emptySHA256 {
		return fmt.Errorf("prompt packet GGUF identity: derived digests are not canonical non-empty SHA-256 values")
	}
	if packet.TokenizerIdentity != GGUFTokenizerIdentity {
		return fmt.Errorf("prompt packet GGUF identity: tokenizer_identity %q must be %q", packet.TokenizerIdentity, GGUFTokenizerIdentity)
	}
	if packet.TokenizerDigest != derived.TokenizerDigest {
		return fmt.Errorf("prompt packet GGUF identity: tokenizer digest does not match exact GGUF metadata")
	}
	if packet.TemplateDigest != derived.TemplateDigest {
		return fmt.Errorf("prompt packet GGUF identity: template digest does not match exact GGUF metadata")
	}
	if !validCanonicalSHA256(packet.ArtifactSHA256) || packet.ArtifactSHA256 == emptySHA256 {
		return fmt.Errorf("prompt packet GGUF identity: artifact digest is not a canonical non-empty SHA-256")
	}
	for _, field := range []struct {
		name string
		ids  []int
	}{
		{name: "prompt_token_ids", ids: packet.PromptTokenIDs},
		{name: "stop_token_ids", ids: packet.StopTokenIDs},
		{name: "generation_controls.stop_token_ids", ids: packet.GenerationControls.StopTokenIDs},
	} {
		for i, id := range field.ids {
			if id < 0 || id >= derived.VocabSize {
				return fmt.Errorf("prompt packet GGUF identity: %s[%d]=%d is outside vocabulary [0,%d)", field.name, i, id, derived.VocabSize)
			}
		}
	}
	return nil
}

// digestGGUFMetadata hashes a canonical TLV projection of GGUF metadata. Keys
// are sorted by their raw UTF-8 bytes (Go string order), lengths/counts are
// uint64 little-endian, type tags are uint32 little-endian GGUF ValueType tags,
// and scalar payloads retain their authored width and IEEE bit pattern.
func digestGGUFMetadata(metadata map[string]ggufload.Value, domain string, keys []string) (string, error) {
	ordered := append([]string(nil), keys...)
	sort.Strings(ordered)
	h := sha256.New()
	_, _ = h.Write([]byte(domain))
	_, _ = h.Write([]byte{0})
	writeUint64(h, uint64(len(ordered)))
	for _, key := range ordered {
		value, ok := metadata[key]
		if !ok {
			return "", fmt.Errorf("missing metadata key %q", key)
		}
		writeBytes(h, []byte(key))
		if err := writeGGUFValue(h, value, true); err != nil {
			return "", fmt.Errorf("metadata key %q: %w", key, err)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeGGUFValue(w hash.Hash, value ggufload.Value, withType bool) error {
	if withType {
		writeUint32(w, uint32(value.Type))
	}
	switch value.Type {
	case ggufload.TypeUint8:
		v, ok := value.Value.(uint8)
		if !ok {
			return fmt.Errorf("uint8 payload has type %T", value.Value)
		}
		_, _ = w.Write([]byte{v})
	case ggufload.TypeInt8:
		v, ok := value.Value.(int8)
		if !ok {
			return fmt.Errorf("int8 payload has type %T", value.Value)
		}
		_, _ = w.Write([]byte{byte(v)})
	case ggufload.TypeUint16:
		v, ok := value.Value.(uint16)
		if !ok {
			return fmt.Errorf("uint16 payload has type %T", value.Value)
		}
		var raw [2]byte
		binary.LittleEndian.PutUint16(raw[:], v)
		_, _ = w.Write(raw[:])
	case ggufload.TypeInt16:
		v, ok := value.Value.(int16)
		if !ok {
			return fmt.Errorf("int16 payload has type %T", value.Value)
		}
		var raw [2]byte
		binary.LittleEndian.PutUint16(raw[:], uint16(v))
		_, _ = w.Write(raw[:])
	case ggufload.TypeUint32:
		v, ok := value.Value.(uint32)
		if !ok {
			return fmt.Errorf("uint32 payload has type %T", value.Value)
		}
		writeUint32(w, v)
	case ggufload.TypeInt32:
		v, ok := value.Value.(int32)
		if !ok {
			return fmt.Errorf("int32 payload has type %T", value.Value)
		}
		writeUint32(w, uint32(v))
	case ggufload.TypeFloat32:
		v, ok := value.Value.(float32)
		if !ok {
			return fmt.Errorf("float32 payload has type %T", value.Value)
		}
		writeUint32(w, math.Float32bits(v))
	case ggufload.TypeBool:
		v, ok := value.Value.(bool)
		if !ok {
			return fmt.Errorf("bool payload has type %T", value.Value)
		}
		if v {
			_, _ = w.Write([]byte{1})
		} else {
			_, _ = w.Write([]byte{0})
		}
	case ggufload.TypeString:
		v, ok := value.Value.(string)
		if !ok {
			return fmt.Errorf("string payload has type %T", value.Value)
		}
		writeBytes(w, []byte(v))
	case ggufload.TypeArray:
		items, ok := value.Value.([]ggufload.Value)
		if !ok {
			return fmt.Errorf("array payload has type %T", value.Value)
		}
		if len(items) == 0 {
			return fmt.Errorf("empty array has no retained GGUF element type")
		}
		elemType := items[0].Type
		if elemType == ggufload.TypeArray {
			return fmt.Errorf("nested arrays are not canonical")
		}
		writeUint32(w, uint32(elemType))
		writeUint64(w, uint64(len(items)))
		for i, item := range items {
			if item.Type != elemType {
				return fmt.Errorf("heterogeneous array item %d has type %s, want %s", i, item.Type.String(), elemType.String())
			}
			if err := writeGGUFValue(w, item, false); err != nil {
				return fmt.Errorf("array item %d: %w", i, err)
			}
		}
	case ggufload.TypeUint64:
		v, ok := value.Value.(uint64)
		if !ok {
			return fmt.Errorf("uint64 payload has type %T", value.Value)
		}
		writeUint64(w, v)
	case ggufload.TypeInt64:
		v, ok := value.Value.(int64)
		if !ok {
			return fmt.Errorf("int64 payload has type %T", value.Value)
		}
		writeUint64(w, uint64(v))
	case ggufload.TypeFloat64:
		v, ok := value.Value.(float64)
		if !ok {
			return fmt.Errorf("float64 payload has type %T", value.Value)
		}
		writeUint64(w, math.Float64bits(v))
	default:
		return fmt.Errorf("unsupported GGUF value type %d", value.Type)
	}
	return nil
}

func writeBytes(w hash.Hash, value []byte) {
	writeUint64(w, uint64(len(value)))
	_, _ = w.Write(value)
}

func writeUint32(w hash.Hash, value uint32) {
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], value)
	_, _ = w.Write(raw[:])
}

func writeUint64(w hash.Hash, value uint64) {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], value)
	_, _ = w.Write(raw[:])
}
