package harnessartifact

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const ModelLifecycleReceiptSchema = "fak-harness-model-lifecycle-receipt/1"

type LifecycleIdentity struct {
	ID     string `json:"id"`
	Digest string `json:"digest,omitempty"`
}

type ModelLifecycleReceipt struct {
	Schema        string            `json:"schema"`
	Declaration   LifecycleIdentity `json:"declaration"`
	Artifact      LifecycleIdentity `json:"artifact"`
	Runtime       LifecycleIdentity `json:"runtime"`
	Admission     LifecycleIdentity `json:"admission"`
	Process       LifecycleIdentity `json:"process"`
	Readiness     LifecycleIdentity `json:"readiness"`
	Stop          LifecycleIdentity `json:"stop"`
	HealthURL     string            `json:"health_url,omitempty"`
	CompletionURL string            `json:"completion_url,omitempty"`
	State         string            `json:"state"`
	PayloadSHA256 string            `json:"payload_sha256"`
}

type LifecycleDiagnostic struct {
	Code   string
	Detail string
}

func (d *LifecycleDiagnostic) Error() string { return d.Code + ": " + RedactLifecycleText(d.Detail) }

func NewModelLifecycleReceipt(receipt ModelLifecycleReceipt) (ModelLifecycleReceipt, error) {
	receipt.Schema = ModelLifecycleReceiptSchema
	receipt.PayloadSHA256 = ""
	if err := validateLifecycle(receipt); err != nil {
		return ModelLifecycleReceipt{}, err
	}
	digest, err := lifecycleDigest(receipt)
	if err != nil {
		return ModelLifecycleReceipt{}, err
	}
	receipt.PayloadSHA256 = digest
	return receipt, nil
}

func WriteModelLifecycleReceipt(path string, receipt ModelLifecycleReceipt) error {
	sealed, err := NewModelLifecycleReceipt(receipt)
	if err != nil {
		return err
	}
	raw, err := json.MarshalIndent(sealed, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".lifecycle-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(raw); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func ReadModelLifecycleReceipt(path string) (ModelLifecycleReceipt, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return ModelLifecycleReceipt{}, &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_UNREADABLE", Detail: err.Error()}
	}
	var receipt ModelLifecycleReceipt
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&receipt); err != nil {
		return ModelLifecycleReceipt{}, &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_INVALID", Detail: err.Error()}
	}
	if receipt.Schema != ModelLifecycleReceiptSchema {
		return ModelLifecycleReceipt{}, &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_STALE", Detail: fmt.Sprintf("schema %q is not %q", receipt.Schema, ModelLifecycleReceiptSchema)}
	}
	claimed := receipt.PayloadSHA256
	receipt.PayloadSHA256 = ""
	if err := validateLifecycle(receipt); err != nil {
		return ModelLifecycleReceipt{}, err
	}
	actual, err := lifecycleDigest(receipt)
	if err != nil {
		return ModelLifecycleReceipt{}, err
	}
	if claimed == "" || !strings.EqualFold(claimed, actual) {
		return ModelLifecycleReceipt{}, &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_TAMPERED", Detail: "payload digest mismatch"}
	}
	receipt.PayloadSHA256 = strings.ToLower(claimed)
	return receipt, nil
}

func CheckLifecycleDeclaration(receipt ModelLifecycleReceipt, expected string) error {
	if expected != "" && receipt.Declaration.ID != expected {
		return &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_STALE", Detail: fmt.Sprintf("declaration identity %q does not match expected %q", receipt.Declaration.ID, expected)}
	}
	return nil
}

func LifecycleDiagnosticCode(err error) string {
	var diagnostic *LifecycleDiagnostic
	if errors.As(err, &diagnostic) {
		return diagnostic.Code
	}
	return "LIFECYCLE_RECEIPT_INVALID"
}

func RedactLifecycleText(value string) string {
	fields := strings.Fields(value)
	for index, field := range fields {
		trimmed := strings.Trim(field, "\"'(),")
		parsed, err := url.Parse(trimmed)
		if err != nil || parsed.Scheme == "" || parsed.Host == "" {
			continue
		}
		parsed.User = nil
		parsed.RawQuery = ""
		parsed.Fragment = ""
		fields[index] = strings.Replace(field, trimmed, parsed.String(), 1)
	}
	return strings.Join(fields, " ")
}

func lifecycleDigest(receipt ModelLifecycleReceipt) (string, error) {
	raw, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func validateLifecycle(receipt ModelLifecycleReceipt) error {
	identities := []struct {
		name  string
		value LifecycleIdentity
	}{
		{"declaration", receipt.Declaration}, {"artifact", receipt.Artifact}, {"runtime", receipt.Runtime},
		{"admission", receipt.Admission}, {"process", receipt.Process}, {"readiness", receipt.Readiness}, {"stop", receipt.Stop},
	}
	for _, identity := range identities {
		if strings.TrimSpace(identity.value.ID) == "" {
			return &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_INVALID", Detail: identity.name + " identity is required"}
		}
	}
	if receipt.State != "ready" && receipt.State != "stopped" {
		return &LifecycleDiagnostic{Code: "LIFECYCLE_RECEIPT_INVALID", Detail: "state must be ready or stopped"}
	}
	return nil
}
