package keystore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/fsatomic"
)

// storeFile is the on-disk v1 schema: a version stamp and the key records.
// Decoding is strict (unknown fields refused) so a payload that tries to
// carry extra state alongside a record cannot be decoded into one.
type storeFile struct {
	Version string      `json:"version"`
	Keys    []KeyRecord `json:"keys"`
}

// Open opens the durable store at path and returns it ready for use.
//
// A MISSING FILE yields an EMPTY store and creates no file: the file comes
// into existence on the first mutation (Mint or Revoke). An existing file
// must be a well-formed store of this format — the v1 version stamp,
// records that decode strictly to the KeyRecord schema, and records that
// pass the closed record contract. Anything else is refused with
// ErrCorruptStore (payload-shape faults as ErrMalformedPayload, contract
// faults as ErrInvalidRecord) and the file is left untouched.
func Open(path string) (*Store, error) {
	if strings.TrimSpace(path) == "" {
		return nil, ErrEmptyPath
	}
	s := &Store{path: path, byHash: make(map[string]int)}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("keystore: read %q: %w", path, err)
	}
	sf, err := decodeStore(data)
	if err != nil {
		return nil, fmt.Errorf("keystore: open %q: %w", path, err)
	}
	for i, r := range sf.Keys {
		s.keys = append(s.keys, r)
		s.byHash[r.Hash] = i
	}
	return s, nil
}

// decodeStore decodes exactly one serialized store from data, refusing
// anything the v1 file schema does not declare: bad JSON, a field outside
// the schema, trailing data, a version stamp this package does not write,
// or a record that fails the closed record contract (including a duplicated
// hash). It mirrors the strict-decode posture of pkg/billing/pricebook.
func decodeStore(data []byte) (*storeFile, error) {
	var sf storeFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&sf); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMalformedPayload, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("%w: trailing data after store", ErrMalformedPayload)
	}
	if sf.Version != KeystoreVersionV1 {
		return nil, fmt.Errorf("%w: unsupported version %q", ErrMalformedPayload, sf.Version)
	}
	seen := make(map[string]struct{}, len(sf.Keys))
	for _, r := range sf.Keys {
		if err := r.Validate(); err != nil {
			return nil, err
		}
		if _, dup := seen[r.Hash]; dup {
			return nil, fmt.Errorf("%w: duplicate hash for prefix %s", ErrInvalidRecord, r.Prefix)
		}
		seen[r.Hash] = struct{}{}
	}
	return &sf, nil
}

// encodeStore serializes keys in the operator-file format: the same strict
// v1 schema decodeStore accepts, so a loaded store encodes back to bytes
// that decode to an identical store. Records are emitted in List order
// (CreatedAt, then Prefix), so the same logical state always yields the
// same bytes.
func encodeStore(keys []KeyRecord) ([]byte, error) {
	ordered := make([]KeyRecord, len(keys))
	copy(ordered, keys)
	sortByKeyOrder(ordered)
	sf := storeFile{Version: KeystoreVersionV1, Keys: ordered}
	b, err := json.MarshalIndent(&sf, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("keystore: encode store: %w", err)
	}
	return append(b, '\n'), nil
}

// persistLocked durably rewrites the store file to hold exactly keys,
// atomically through pkg/fsatomic: a sibling temporary file is written,
// synced, chmod'd, and renamed over the destination, so a crash or a failed
// write never leaves a torn file in place. Callers hold mu.
func (s *Store) persistLocked(keys []KeyRecord) error {
	if s.path == "" {
		return nil
	}
	data, err := encodeStore(keys)
	if err != nil {
		return err
	}
	if err := fsatomic.WriteFileAtomic(s.path, data, storeFilePerm); err != nil {
		return fmt.Errorf("keystore: write %q: %w", s.path, err)
	}
	return nil
}
