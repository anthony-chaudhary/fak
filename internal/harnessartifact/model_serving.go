package harnessartifact

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const LocalModelDeclarationSchema = "fak.harness.local-model-declaration.v1"

var (
	ErrInvalidModelDeclaration = errors.New("invalid local-model declaration")
	sha256Pattern              = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// CanonicalLocalModelDeclaration validates and emits stable declaration bytes.
// It performs no filesystem or network mutation and does not infer hardware.
func CanonicalLocalModelDeclaration(in harnesskit.LocalModelDeclaration) ([]byte, error) {
	in.Schema = strings.TrimSpace(in.Schema)
	in.ModelID = strings.TrimSpace(in.ModelID)
	in.GGUFPath = filepath.Clean(strings.TrimSpace(in.GGUFPath))
	in.GGUFSHA256 = strings.ToLower(strings.TrimSpace(in.GGUFSHA256))
	in.Runtime = strings.TrimSpace(in.Runtime)
	if in.Schema != LocalModelDeclarationSchema {
		return nil, fmt.Errorf("%w: schema must be %q", ErrInvalidModelDeclaration, LocalModelDeclarationSchema)
	}
	if in.ModelID == "" {
		return nil, fmt.Errorf("%w: model_id is required", ErrInvalidModelDeclaration)
	}
	if !filepath.IsAbs(in.GGUFPath) || strings.ToLower(filepath.Ext(in.GGUFPath)) != ".gguf" {
		return nil, fmt.Errorf("%w: gguf_path must be an absolute .gguf path", ErrInvalidModelDeclaration)
	}
	if !sha256Pattern.MatchString(in.GGUFSHA256) {
		return nil, fmt.Errorf("%w: gguf_sha256 must be 64 lowercase hexadecimal characters", ErrInvalidModelDeclaration)
	}
	if in.Runtime == "" {
		return nil, fmt.Errorf("%w: runtime is required", ErrInvalidModelDeclaration)
	}
	if in.ContextTokens <= 0 {
		return nil, fmt.Errorf("%w: context_tokens must be positive", ErrInvalidModelDeclaration)
	}
	seen := map[string]struct{}{}
	devices := in.RequiredDevices[:0]
	for _, device := range in.RequiredDevices {
		device = strings.TrimSpace(device)
		if device == "" {
			return nil, fmt.Errorf("%w: required_devices cannot contain an empty device", ErrInvalidModelDeclaration)
		}
		if _, ok := seen[device]; ok {
			continue
		}
		seen[device] = struct{}{}
		devices = append(devices, device)
	}
	sort.Strings(devices)
	in.RequiredDevices = devices
	return json.Marshal(in)
}

func LocalModelDeclarationDigest(canonical []byte) string {
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}
