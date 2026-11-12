package manifest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Load decodes exactly one serialized manifest from data, refusing anything
// the v1 file schema does not declare: bad JSON, a field outside the
// schema, or trailing data. Decoding enforces the closed validation rules
// too, so the returned manifest is ready to use.
//
// Faults split into two classes: a payload-shape fault (unparseable bytes,
// a foreign field, trailing data) reports ErrMalformedPayload, while a
// field that parses but violates the closed contract (for example a
// negative limit or a base URL without a host) reports the specific
// validation sentinel, matchable up the chain to ErrInvalidManifest.
func Load(data []byte) (*Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMalformedPayload, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("%w: trailing data after manifest", ErrMalformedPayload)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// LoadFile reads path and Loads it.
func LoadFile(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(data)
}

// MarshalJSON encodes the manifest in the operator-file format: the same
// strict v1 schema Load accepts, so a loaded manifest marshals back to bytes
// that load to an identical manifest. Encoding refuses a manifest that
// fails Validate.
func (m *Manifest) MarshalJSON() ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	type plain Manifest
	return json.Marshal(plain(*m))
}
