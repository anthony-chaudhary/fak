package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"unicode/utf8"
)

// ErrNativeSPMProfile marks a recognized native profile that must not fall back
// to BPE, a mock planner, or a different normalization algorithm.
var ErrNativeSPMProfile = errors.New("tokenizer: invalid llama-spm-v1 profile")

// nativeSPMObject requires exactly one object, unique keys and no trailing input.
// It is used separately for the root and options so duplicates cannot override
// a previously validated trust or special-token policy.
func nativeSPMObject(b []byte, keys ...string) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	t, err := d.Token()
	if err != nil || t != json.Delim('{') {
		return nil, fmt.Errorf("%w: expected object", ErrNativeSPMProfile)
	}
	allowed := make(map[string]bool, len(keys))
	for _, k := range keys {
		allowed[k] = true
	}
	out := make(map[string]json.RawMessage, len(keys))
	for d.More() {
		t, err = d.Token()
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrNativeSPMProfile, err)
		}
		k, ok := t.(string)
		if !ok || !allowed[k] || out[k] != nil {
			return nil, fmt.Errorf("%w: unknown or duplicate field %q", ErrNativeSPMProfile, k)
		}
		var v json.RawMessage
		if err = d.Decode(&v); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrNativeSPMProfile, err)
		}
		if bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, fmt.Errorf("%w: null field %q", ErrNativeSPMProfile, k)
		}
		out[k] = v
	}
	if _, err = d.Token(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNativeSPMProfile, err)
	}
	if _, err = d.Token(); err != io.EOF {
		return nil, fmt.Errorf("%w: trailing input", ErrNativeSPMProfile)
	}
	for _, k := range keys {
		if out[k] == nil {
			return nil, fmt.Errorf("%w: missing %s", ErrNativeSPMProfile, k)
		}
	}
	return out, nil
}

// nativeSPMUnicode refuses encoding/json's replacement of malformed UTF-8 or
// unpaired UTF-16 escapes. Valid escaped pairs retain their exact code point.
func nativeSPMUnicode(b []byte) bool {
	if !utf8.Valid(b) {
		return false
	}
	for i := 0; i < len(b); i++ {
		if b[i] != '\\' {
			continue
		}
		i++
		if i >= len(b) {
			return false
		}
		if b[i] != 'u' {
			continue
		}
		if i+4 >= len(b) {
			return false
		}
		v, err := strconv.ParseUint(string(b[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		i += 4
		if v >= 0xDC00 && v <= 0xDFFF {
			return false
		}
		if v >= 0xD800 && v <= 0xDBFF {
			if i+6 >= len(b) || b[i+1] != '\\' || b[i+2] != 'u' {
				return false
			}
			low, err := strconv.ParseUint(string(b[i+3:i+7]), 16, 16)
			if err != nil || low < 0xDC00 || low > 0xDFFF {
				return false
			}
			i += 6
		}
	}
	return true
}

func parseNativeSPMJSON(b []byte) (*Tokenizer, error) {
	if !nativeSPMUnicode(b) {
		return nil, fmt.Errorf("%w: malformed Unicode", ErrNativeSPMProfile)
	}
	// This bound is for the explicit new profile, not legacy BPE files.
	if len(b) > 64<<20 {
		return nil, fmt.Errorf("%w: profile exceeds 64 MiB", ErrNativeSPMProfile)
	}
	root, err := nativeSPMObject(b, "fak_tokenizer_profile", "normalization", "tokens", "scores", "types", "options")
	if err != nil {
		return nil, err
	}
	var profile, norm string
	if json.Unmarshal(root["fak_tokenizer_profile"], &profile) != nil || profile != "llama-spm-v1" ||
		json.Unmarshal(root["normalization"], &norm) != nil || norm != "identity" {
		return nil, fmt.Errorf("%w: unsupported profile or normalization", ErrNativeSPMProfile)
	}
	var tokens []string
	var scoreValues []*float32
	var types []int32
	if json.Unmarshal(root["tokens"], &tokens) != nil || json.Unmarshal(root["scores"], &scoreValues) != nil || json.Unmarshal(root["types"], &types) != nil {
		return nil, fmt.Errorf("%w: invalid vocabulary arrays", ErrNativeSPMProfile)
	}
	scores := make([]float32, len(scoreValues))
	for i, v := range scoreValues {
		if v == nil {
			return nil, fmt.Errorf("%w: null score", ErrNativeSPMProfile)
		}
		scores[i] = *v
	}
	opts, err := nativeSPMObject(root["options"], "bos_id", "eos_id", "add_bos", "add_eos", "add_space_prefix")
	if err != nil {
		return nil, err
	}
	var o SPMOptions
	for k, dst := range map[string]any{"bos_id": &o.BOS, "eos_id": &o.EOS, "add_bos": &o.AddBOS, "add_eos": &o.AddEOS, "add_space_prefix": &o.AddSpacePrefix} {
		if err := json.Unmarshal(opts[k], dst); err != nil {
			return nil, fmt.Errorf("%w: invalid %s", ErrNativeSPMProfile, k)
		}
	}
	tok, err := FromGGMLSPM(tokens, scores, types, o)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNativeSPMProfile, err)
	}
	return tok, nil
}
