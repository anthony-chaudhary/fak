package serverproduct

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ReadyReceipt is the validated, immutable value accepted by the receipt
// writer. Its unexported state prevents a raw or failed-launch receipt from
// reaching the durable ready-receipt path.
type ReadyReceipt struct {
	receipt ServerReceipt
}

// NewReadyReceipt validates and binds a receipt to its authored spec.
func NewReadyReceipt(spec ServerSpec, receipt ServerReceipt) (ReadyReceipt, error) {
	receipt = canonicalReceipt(receipt)
	if err := CheckCompatibility(spec, receipt); err != nil {
		return ReadyReceipt{}, err
	}
	return ReadyReceipt{receipt: receipt}, nil
}

// Receipt returns a copy of the validated receipt.
func (ready ReadyReceipt) Receipt() ServerReceipt {
	return canonicalReceipt(ready.receipt)
}

// DecodeSpec strictly decodes and validates one schema-v1 spec.
func DecodeSpec(data []byte) (ServerSpec, error) {
	var spec ServerSpec
	if err := decodeStrict(data, &spec); err != nil {
		return ServerSpec{}, fmt.Errorf("decode server spec: %w", err)
	}
	if err := ValidateSpec(spec); err != nil {
		return ServerSpec{}, err
	}
	return canonicalSpec(spec), nil
}

// DecodeReceipt strictly decodes and validates receipt-internal invariants.
// Consumers should normally call DecodeReadyReceipt to bind it to a spec.
func DecodeReceipt(data []byte) (ServerReceipt, error) {
	var receipt ServerReceipt
	if err := decodeStrict(data, &receipt); err != nil {
		return ServerReceipt{}, fmt.Errorf("decode server receipt: %w", err)
	}
	if err := ValidateReceipt(receipt); err != nil {
		return ServerReceipt{}, err
	}
	return canonicalReceipt(receipt), nil
}

// DecodeReadyReceipt strictly decodes a receipt and binds it to spec.
func DecodeReadyReceipt(spec ServerSpec, data []byte) (ReadyReceipt, error) {
	receipt, err := DecodeReceipt(data)
	if err != nil {
		return ReadyReceipt{}, err
	}
	return NewReadyReceipt(spec, receipt)
}

// EncodeSpec returns deterministic, newline-terminated schema-v1 JSON.
func EncodeSpec(spec ServerSpec) ([]byte, error) {
	if err := ValidateSpec(spec); err != nil {
		return nil, err
	}
	return marshalCanonical(canonicalSpec(spec))
}

// EncodeReadyReceipt returns deterministic, newline-terminated schema-v1 JSON.
func EncodeReadyReceipt(ready ReadyReceipt) ([]byte, error) {
	receipt := canonicalReceipt(ready.receipt)
	if err := ValidateReceipt(receipt); err != nil {
		return nil, fmt.Errorf("invalid ready receipt: %w", err)
	}
	return marshalCanonical(receipt)
}

// WriteReadyReceipt validates, durably flushes, and atomically renames a
// same-directory temporary file. The final file is never exposed partially.
func WriteReadyReceipt(path string, ready ReadyReceipt) error {
	if path == "" || filepath.Clean(path) != path {
		return errors.New("receipt path must be nonempty and clean")
	}
	data, err := EncodeReadyReceipt(ready)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".server-receipt-*.tmp")
	if err != nil {
		return fmt.Errorf("create receipt temp file: %w", err)
	}
	tmpName := tmp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = tmp.Close()
		}
		_ = os.Remove(tmpName)
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return fmt.Errorf("set receipt permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write receipt temp file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync receipt temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close receipt temp file: %w", err)
	}
	closed = true
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("publish ready receipt: %w", err)
	}
	return nil
}

func decodeStrict(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
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

func marshalCanonical(value any) ([]byte, error) {
	var data bytes.Buffer
	encoder := json.NewEncoder(&data)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return data.Bytes(), nil
}
