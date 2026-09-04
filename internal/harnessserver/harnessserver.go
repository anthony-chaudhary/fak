// Package harnessserver binds a harness to an externally owned ready-server
// receipt without acquiring any server lifecycle authority.
package harnessserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

// BindingSchema identifies the versioned JSON schema for harness server bindings.
const BindingSchema = "fak.harness-server-binding/v1"

// ResolutionSchema identifies the versioned JSON schema for verified harness server resolutions.
const ResolutionSchema = "fak.harness-server-resolution/v1"

// BindingFileName is the canonical filename for the harness server binding document.
const BindingFileName = "server.binding.json"

const maxDocumentSize = 1 << 20

var (
	digestPattern     = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	capabilityPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,63}$`)
	jwtPattern        = regexp.MustCompile(`[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}`)
)

// Requirements are the explicit compatibility constraints selected by the
// harness owner. MinimumGeneration prevents importing an older ready instance.
type Requirements struct {
	ModelAlias           string   `json:"model_alias"`
	ProtocolFamily       string   `json:"protocol_family"`
	ProtocolRevision     string   `json:"protocol_revision"`
	RequiredCapabilities []string `json:"required_capabilities"`
	MinimumGeneration    uint64   `json:"minimum_generation"`
}

// Binding pins the exact external receipt bytes and generation accepted by a
// harness. The receipt remains outside the harness directory and externally owned.
type Binding struct {
	Schema            string       `json:"schema"`
	HarnessDirectory  string       `json:"harness_directory"`
	ReceiptPath       string       `json:"receipt_path"`
	ReceiptDigest     string       `json:"receipt_digest"`
	ReceiptGeneration uint64       `json:"receipt_generation"`
	Requirements      Requirements `json:"requirements"`
}

// Verified is the secret-free consumer view of a successfully revalidated receipt.
type Verified struct {
	Schema           string   `json:"schema"`
	ReceiptPath      string   `json:"receipt_path"`
	ReceiptDigest    string   `json:"receipt_digest"`
	Generation       uint64   `json:"generation"`
	BaseURL          string   `json:"base_url"`
	ModelAlias       string   `json:"model_alias"`
	ProtocolFamily   string   `json:"protocol_family"`
	ProtocolRevision string   `json:"protocol_revision"`
	Capabilities     []string `json:"capabilities"`
}

// WriteResult describes whether an immutable binding was created or already present.
type WriteResult struct {
	Path      string `json:"path"`
	Created   bool   `json:"created"`
	Preserved bool   `json:"preserved"`
}

// Import validates one external ready receipt and returns an immutable harness binding.
func Import(harnessDir, receiptPath string, requirements Requirements) (Binding, error) {
	harnessRoot, err := absolutePath("harness directory", harnessDir)
	if err != nil {
		return Binding{}, err
	}
	receiptFile, err := absolutePath("server receipt path", receiptPath)
	if err != nil {
		return Binding{}, err
	}
	if pathWithin(harnessRoot, receiptFile) {
		return Binding{}, errors.New("server receipt must remain outside the harness directory")
	}
	requirements, err = canonicalRequirements(requirements)
	if err != nil {
		return Binding{}, err
	}
	raw, receipt, err := readReceipt(receiptFile)
	if err != nil {
		return Binding{}, err
	}
	if err := checkCompatibility(requirements, receipt); err != nil {
		return Binding{}, err
	}
	return Binding{
		Schema:            BindingSchema,
		HarnessDirectory:  harnessRoot,
		ReceiptPath:       receiptFile,
		ReceiptDigest:     digestBytes(raw),
		ReceiptGeneration: receipt.Generation,
		Requirements:      requirements,
	}, nil
}

// VerifyFile strictly decodes a binding and revalidates its pinned external receipt.
func VerifyFile(path string) (Verified, error) {
	bindingPath, err := absolutePath("server binding path", path)
	if err != nil {
		return Verified{}, err
	}
	binding, err := ReadBinding(bindingPath)
	if err != nil {
		return Verified{}, err
	}
	if !samePath(filepath.Dir(bindingPath), binding.HarnessDirectory) {
		return Verified{}, errors.New("server binding is not stored in its declared harness directory")
	}
	if pathWithin(binding.HarnessDirectory, binding.ReceiptPath) {
		return Verified{}, errors.New("server receipt must remain outside the harness directory")
	}
	raw, err := readBounded(binding.ReceiptPath, "server receipt")
	if err != nil {
		return Verified{}, err
	}
	if digestBytes(raw) != binding.ReceiptDigest {
		return Verified{}, errors.New("server receipt changed since immutable import")
	}
	receipt, err := decodeReceipt(raw)
	if err != nil {
		return Verified{}, err
	}
	if receipt.Generation != binding.ReceiptGeneration {
		return Verified{}, errors.New("server receipt generation changed since immutable import")
	}
	if err := checkCompatibility(binding.Requirements, receipt); err != nil {
		return Verified{}, err
	}
	capabilities := append([]string(nil), receipt.Protocol.Capabilities...)
	sort.Strings(capabilities)
	return Verified{
		Schema:           ResolutionSchema,
		ReceiptPath:      binding.ReceiptPath,
		ReceiptDigest:    binding.ReceiptDigest,
		Generation:       receipt.Generation,
		BaseURL:          receipt.Endpoint.BaseURL,
		ModelAlias:       receipt.ModelAlias,
		ProtocolFamily:   receipt.Protocol.Family,
		ProtocolRevision: receipt.Protocol.Revision,
		Capabilities:     capabilities,
	}, nil
}

// ReadBinding strictly decodes and validates one harness server binding.
func ReadBinding(path string) (Binding, error) {
	raw, err := readBounded(path, "server binding")
	if err != nil {
		return Binding{}, err
	}
	var binding Binding
	if err := decodeStrict(raw, &binding); err != nil {
		return Binding{}, fmt.Errorf("decode server binding: %w", err)
	}
	return canonicalBinding(binding)
}

// WriteBinding atomically creates a binding and refuses to replace different bytes.
func WriteBinding(path string, binding Binding) (WriteResult, error) {
	bindingPath, err := absolutePath("server binding path", path)
	if err != nil {
		return WriteResult{}, err
	}
	binding, err = canonicalBinding(binding)
	if err != nil {
		return WriteResult{}, err
	}
	if !samePath(filepath.Dir(bindingPath), binding.HarnessDirectory) {
		return WriteResult{}, errors.New("server binding path must be inside its declared harness directory")
	}
	raw, err := encode(binding)
	if err != nil {
		return WriteResult{}, err
	}
	prior, readErr := os.ReadFile(bindingPath)
	switch {
	case readErr == nil && bytes.Equal(prior, raw):
		return WriteResult{Path: bindingPath, Preserved: true}, nil
	case readErr == nil:
		return WriteResult{}, errors.New("refusing to replace immutable server binding")
	case !errors.Is(readErr, os.ErrNotExist):
		return WriteResult{}, fmt.Errorf("read server binding: %w", readErr)
	}
	if err := os.MkdirAll(filepath.Dir(bindingPath), 0o755); err != nil {
		return WriteResult{}, fmt.Errorf("create harness directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(bindingPath), ".server-binding-*.tmp")
	if err != nil {
		return WriteResult{}, fmt.Errorf("create server binding temp file: %w", err)
	}
	tmpPath := tmp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = tmp.Close()
		}
		_ = os.Remove(tmpPath)
	}()
	if err := tmp.Chmod(0o644); err != nil {
		return WriteResult{}, fmt.Errorf("set server binding permissions: %w", err)
	}
	if _, err := tmp.Write(raw); err != nil {
		return WriteResult{}, fmt.Errorf("write server binding: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return WriteResult{}, fmt.Errorf("sync server binding: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return WriteResult{}, fmt.Errorf("close server binding: %w", err)
	}
	closed = true
	if err := os.Rename(tmpPath, bindingPath); err != nil {
		return WriteResult{}, fmt.Errorf("publish server binding: %w", err)
	}
	return WriteResult{Path: bindingPath, Created: true}, nil
}

func canonicalBinding(binding Binding) (Binding, error) {
	if binding.Schema != BindingSchema {
		return Binding{}, fmt.Errorf("server binding schema must be %q", BindingSchema)
	}
	if !isCleanAbsolute(binding.HarnessDirectory) {
		return Binding{}, errors.New("harness_directory must be an absolute clean path")
	}
	if !isCleanAbsolute(binding.ReceiptPath) {
		return Binding{}, errors.New("receipt_path must be an absolute clean path")
	}
	if !digestPattern.MatchString(binding.ReceiptDigest) {
		return Binding{}, errors.New("receipt_digest must be lowercase sha256:<64 hex>")
	}
	if binding.ReceiptGeneration == 0 {
		return Binding{}, errors.New("receipt_generation must be positive")
	}
	if pathWithin(binding.HarnessDirectory, binding.ReceiptPath) {
		return Binding{}, errors.New("server receipt must remain outside the harness directory")
	}
	requirements, err := canonicalRequirements(binding.Requirements)
	if err != nil {
		return Binding{}, err
	}
	binding.Requirements = requirements
	return binding, nil
}

func canonicalRequirements(requirements Requirements) (Requirements, error) {
	if err := validateSafeText("model alias", requirements.ModelAlias); err != nil {
		return Requirements{}, err
	}
	if requirements.ProtocolFamily != serverproduct.ProtocolOpenAIHTTP {
		return Requirements{}, fmt.Errorf("protocol family must be %q", serverproduct.ProtocolOpenAIHTTP)
	}
	if err := validateSafeText("protocol revision", requirements.ProtocolRevision); err != nil {
		return Requirements{}, err
	}
	if requirements.MinimumGeneration == 0 {
		return Requirements{}, errors.New("minimum generation must be positive")
	}
	if len(requirements.RequiredCapabilities) == 0 {
		return Requirements{}, errors.New("required capabilities must not be empty")
	}
	seen := make(map[string]struct{}, len(requirements.RequiredCapabilities))
	hasChat := false
	for _, capability := range requirements.RequiredCapabilities {
		if !capabilityPattern.MatchString(capability) {
			return Requirements{}, errors.New("required capabilities contain an invalid value")
		}
		if _, exists := seen[capability]; exists {
			return Requirements{}, errors.New("required capabilities contain a duplicate value")
		}
		seen[capability] = struct{}{}
		hasChat = hasChat || capability == "chat.completions"
	}
	if !hasChat {
		return Requirements{}, errors.New(`required capabilities must include "chat.completions"`)
	}
	requirements.RequiredCapabilities = append([]string(nil), requirements.RequiredCapabilities...)
	sort.Strings(requirements.RequiredCapabilities)
	return requirements, nil
}

func checkCompatibility(requirements Requirements, receipt serverproduct.ServerReceipt) error {
	if receipt.Generation < requirements.MinimumGeneration {
		return errors.New("server receipt generation is stale")
	}
	if receipt.ModelAlias != requirements.ModelAlias {
		return errors.New("server receipt model alias mismatch")
	}
	if receipt.Protocol.Family != requirements.ProtocolFamily {
		return errors.New("server receipt protocol family mismatch")
	}
	if receipt.Protocol.Revision != requirements.ProtocolRevision {
		return errors.New("server receipt protocol revision mismatch")
	}
	observed := make(map[string]struct{}, len(receipt.Protocol.Capabilities))
	for _, capability := range receipt.Protocol.Capabilities {
		observed[capability] = struct{}{}
	}
	for _, required := range requirements.RequiredCapabilities {
		if _, ok := observed[required]; !ok {
			return fmt.Errorf("server receipt is missing required capability %q", required)
		}
	}
	return nil
}

func readReceipt(path string) ([]byte, serverproduct.ServerReceipt, error) {
	raw, err := readBounded(path, "server receipt")
	if err != nil {
		return nil, serverproduct.ServerReceipt{}, err
	}
	receipt, err := decodeReceipt(raw)
	if err != nil {
		return nil, serverproduct.ServerReceipt{}, err
	}
	return raw, receipt, nil
}

func decodeReceipt(raw []byte) (serverproduct.ServerReceipt, error) {
	receipt, err := serverproduct.DecodeReceipt(raw)
	if err != nil {
		message := err.Error()
		if strings.Contains(message, "contains invalid capability") {
			message = "probed capabilities contain an invalid value"
		}
		return serverproduct.ServerReceipt{}, fmt.Errorf("server receipt is invalid: %s", message)
	}
	return receipt, nil
}

func readBounded(path, label string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s must be a regular file", label)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, maxDocumentSize+1))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	if len(raw) > maxDocumentSize {
		return nil, fmt.Errorf("%s exceeds %d bytes", label, maxDocumentSize)
	}
	return raw, nil
}

func decodeStrict(raw []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

func encode(value any) ([]byte, error) {
	var data bytes.Buffer
	encoder := json.NewEncoder(&data)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return data.Bytes(), nil
}

func absolutePath(label, path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("%s is required", label)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", label, err)
	}
	return filepath.Clean(abs), nil
}

func isCleanAbsolute(path string) bool {
	return path != "" && filepath.IsAbs(path) && filepath.Clean(path) == path
}

func pathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)))
}

func samePath(left, right string) bool {
	rel, err := filepath.Rel(left, right)
	return err == nil && rel == "."
}

func validateSafeText(label, value string) error {
	if value == "" {
		return fmt.Errorf("%s is required", label)
	}
	if strings.TrimSpace(value) != value || strings.IndexFunc(value, func(r rune) bool { return r < 0x20 || r == 0x7f }) >= 0 {
		return fmt.Errorf("%s contains boundary whitespace or control characters", label)
	}
	lower := strings.ToLower(value)
	for _, marker := range []string{"bearer ", "-----begin ", "api_key=", "apikey=", "token=", "secret="} {
		if strings.Contains(lower, marker) {
			return fmt.Errorf("%s appears to contain secret material", label)
		}
	}
	if strings.HasPrefix(lower, "sk-") || jwtPattern.MatchString(value) {
		return fmt.Errorf("%s appears to contain secret material", label)
	}
	return nil
}

func digestBytes(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}
