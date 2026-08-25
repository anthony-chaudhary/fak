package orchestration

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// EffectSuccessorStoreOutcome classifies durable admission results without
// turning missing or expired authority into an accepted admission.
type EffectSuccessorStoreOutcome string

const (
	EffectSuccessorStored    EffectSuccessorStoreOutcome = "STORED"
	EffectSuccessorDuplicate EffectSuccessorStoreOutcome = "DUPLICATE"
	EffectSuccessorMissing   EffectSuccessorStoreOutcome = "MISSING"
	EffectSuccessorStale     EffectSuccessorStoreOutcome = "STALE"
	EffectSuccessorMalformed EffectSuccessorStoreOutcome = "MALFORMED"
	EffectSuccessorConflict  EffectSuccessorStoreOutcome = "CONFLICT"
)

// EffectSuccessorStoreError reports a non-success store outcome.
type EffectSuccessorStoreError struct {
	Outcome EffectSuccessorStoreOutcome
	Detail  string
}

func (e *EffectSuccessorStoreError) Error() string {
	return fmt.Sprintf("%s: %s", e.Outcome, e.Detail)
}

// EffectSuccessorStore durably records the pre-effect receipt itself. It does
// not accept an effect observation and has no effect-execution surface.
type EffectSuccessorStore struct {
	dir    string
	maxAge time.Duration
	now    func() time.Time
}

// OpenEffectSuccessorStore opens an owner-only receipt directory. maxAge must
// be positive so a consumer can never accidentally treat admission as eternal.
func OpenEffectSuccessorStore(dir string, maxAge time.Duration) (*EffectSuccessorStore, error) {
	return openEffectSuccessorStore(dir, maxAge, time.Now)
}

func openEffectSuccessorStore(dir string, maxAge time.Duration, now func() time.Time) (*EffectSuccessorStore, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, errors.New("effect successor store directory is required")
	}
	if maxAge <= 0 {
		return nil, errors.New("effect successor admission max age must be positive")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("create effect successor store: %w", err)
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return nil, fmt.Errorf("secure effect successor store: %w", err)
	}
	return &EffectSuccessorStore{dir: dir, maxAge: maxAge, now: now}, nil
}

// Admit authoritatively evaluates a proposal and durably publishes the returned
// pre-effect receipt before returning it to the caller. It never executes an effect.
func (s *EffectSuccessorStore) Admit(proposal EffectSuccessorProposal) (EffectSuccessorAdmission, EffectSuccessorStoreOutcome, error) {
	admission, err := ProposeEffectSuccessor(proposal)
	if err != nil {
		return EffectSuccessorAdmission{}, EffectSuccessorMalformed, err
	}
	admission.Receipt.AdmittedAt = s.now().UTC().Format(time.RFC3339Nano)
	outcome, err := s.store(admission.Receipt)
	if err != nil {
		return EffectSuccessorAdmission{}, outcome, err
	}
	if outcome == EffectSuccessorDuplicate {
		data, readErr := os.ReadFile(s.receiptPath(admission.Receipt.ID))
		if readErr != nil {
			return EffectSuccessorAdmission{}, EffectSuccessorMalformed, readErr
		}
		admission.Receipt, readErr = decodeEffectSuccessorReceipt(data)
		if readErr != nil {
			return EffectSuccessorAdmission{}, EffectSuccessorMalformed, readErr
		}
	}
	return admission, outcome, nil
}

// store atomically creates the authoritative admission. It is deliberately
// unexported: receipt-shaped caller input is not admission authority.
func (s *EffectSuccessorStore) store(receipt EffectSuccessorReceipt) (EffectSuccessorStoreOutcome, error) {
	data, err := encodeEffectSuccessorReceipt(receipt)
	if err != nil {
		return EffectSuccessorMalformed, err
	}
	path := s.receiptPath(receipt.ID)
	if existing, readErr := os.ReadFile(path); readErr == nil {
		return compareEffectSuccessorReceipt(existing, data)
	} else if !errors.Is(readErr, os.ErrNotExist) {
		return EffectSuccessorMalformed, &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: readErr.Error()}
	}

	tmp, err := os.CreateTemp(s.dir, ".admission-*")
	if err != nil {
		return EffectSuccessorMalformed, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	ok := false
	defer func() {
		if !ok {
			tmp.Close()
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return EffectSuccessorMalformed, err
	}
	if _, err := tmp.Write(data); err != nil {
		return EffectSuccessorMalformed, err
	}
	if err := tmp.Sync(); err != nil {
		return EffectSuccessorMalformed, err
	}
	if err := tmp.Close(); err != nil {
		return EffectSuccessorMalformed, err
	}
	ok = true

	// Link is the atomic create-if-absent primitive: unlike Rename, it cannot
	// overwrite authority won by a concurrent writer.
	if err := os.Link(tmpName, path); err != nil {
		if existing, readErr := os.ReadFile(path); readErr == nil {
			return compareEffectSuccessorReceipt(existing, data)
		}
		return EffectSuccessorMalformed, err
	}
	if err := syncEffectSuccessorDir(s.dir); err != nil {
		return EffectSuccessorMalformed, err
	}
	return EffectSuccessorStored, nil
}

func syncEffectSuccessorDir(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

// Lookup strictly reads an unexpired authoritative admission for the admitted
// run, child (observer), and successor (node) identity.
func (s *EffectSuccessorStore) Lookup(runID, observerID, nodeID, receiptID string) (EffectSuccessorReceipt, EffectSuccessorStoreOutcome, error) {
	if strings.TrimSpace(runID) == "" || runID != strings.TrimSpace(runID) ||
		strings.TrimSpace(observerID) == "" || observerID != strings.TrimSpace(observerID) ||
		strings.TrimSpace(nodeID) == "" || nodeID != strings.TrimSpace(nodeID) || !validSuccessorStoreID(receiptID) {
		err := &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: "invalid receipt id"}
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, err
	}
	path := s.receiptPath(receiptID)
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return EffectSuccessorReceipt{}, EffectSuccessorMissing, nil
	}
	if err != nil {
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, err
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		err := &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: "admission file is not owner-only"}
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, err
	}
	receipt, err := decodeEffectSuccessorReceipt(data)
	if err != nil || receipt.ID != receiptID {
		if err == nil {
			err = errors.New("receipt id does not match durable key")
		}
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: err.Error()}
	}
	admittedAt, _ := time.Parse(time.RFC3339Nano, receipt.AdmittedAt)
	now := s.now()
	if admittedAt.After(now) {
		err := &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: "admission timestamp is in the future"}
		return EffectSuccessorReceipt{}, EffectSuccessorMalformed, err
	}
	if now.Sub(admittedAt) > s.maxAge {
		return EffectSuccessorReceipt{}, EffectSuccessorStale, nil
	}
	if receipt.RunID != runID || receipt.ObserverID != observerID || receipt.NodeID != nodeID {
		return EffectSuccessorReceipt{}, EffectSuccessorMissing, nil
	}
	return receipt, EffectSuccessorStored, nil
}

func (s *EffectSuccessorStore) receiptPath(receiptID string) string {
	return filepath.Join(s.dir, receiptID+".json")
}

func encodeEffectSuccessorReceipt(receipt EffectSuccessorReceipt) ([]byte, error) {
	if err := validateEffectSuccessorReceipt(receipt); err != nil {
		return nil, &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: err.Error()}
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		return nil, &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: err.Error()}
	}
	return append(data, '\n'), nil
}

func decodeEffectSuccessorReceipt(data []byte) (EffectSuccessorReceipt, error) {
	var receipt EffectSuccessorReceipt
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil {
		return receipt, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return receipt, err
	}
	return receipt, validateEffectSuccessorReceipt(receipt)
}

func validateEffectSuccessorReceipt(receipt EffectSuccessorReceipt) error {
	if receipt.Schema != EffectSuccessorReceiptSchema {
		return errors.New("unexpected effect successor receipt schema")
	}
	admittedAt, err := time.Parse(time.RFC3339Nano, receipt.AdmittedAt)
	if err != nil || admittedAt.Format(time.RFC3339Nano) != receipt.AdmittedAt {
		return errors.New("admitted_at is malformed")
	}
	for name, value := range map[string]string{
		"id": receipt.ID, "run_id": receipt.RunID, "node_id": receipt.NodeID, "observer_id": receipt.ObserverID,
		"observation_id": receipt.ObservationID, "snapshot_epoch": receipt.SnapshotEpoch,
		"envelope_digest": receipt.EnvelopeDigest, "lease_id": receipt.LeaseID,
	} {
		if strings.TrimSpace(value) == "" || value != strings.TrimSpace(value) {
			return fmt.Errorf("%s is malformed", name)
		}
	}
	if !validSuccessorStoreID(receipt.ID) || !strings.HasPrefix(receipt.NodeID, "effect-") ||
		!strings.HasPrefix(receipt.EnvelopeDigest, "sha256:") || len(receipt.EnvelopeDigest) != len("sha256:")+64 {
		return errors.New("receipt identity is malformed")
	}
	digest, err := digestValue(normalizeEffectEnvelope(receipt.Envelope))
	if err != nil || digest != receipt.EnvelopeDigest || len(receipt.Envelope.Tools) == 0 || len(receipt.Envelope.WriteSet) == 0 {
		return errors.New("receipt envelope is malformed")
	}
	if receipt.Budget.MaxWorkers < 0 || receipt.Budget.MaxTokens < 0 {
		return errors.New("receipt budget is malformed")
	}
	return nil
}

func validSuccessorStoreID(id string) bool {
	if !strings.HasPrefix(id, "effect-receipt-") || len(id) <= len("effect-receipt-") {
		return false
	}
	for _, r := range id {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' {
			return false
		}
	}
	return true
}

func compareEffectSuccessorReceipt(existing, proposed []byte) (EffectSuccessorStoreOutcome, error) {
	stored, err := decodeEffectSuccessorReceipt(existing)
	if err != nil {
		return EffectSuccessorMalformed, &EffectSuccessorStoreError{Outcome: EffectSuccessorMalformed, Detail: err.Error()}
	}
	canonical, err := encodeEffectSuccessorReceipt(stored)
	if err != nil {
		return EffectSuccessorMalformed, err
	}
	var candidate EffectSuccessorReceipt
	if err := json.Unmarshal(proposed, &candidate); err == nil {
		candidate.AdmittedAt = stored.AdmittedAt
		if sameAdmission, encodeErr := encodeEffectSuccessorReceipt(candidate); encodeErr == nil && bytes.Equal(canonical, sameAdmission) {
			return EffectSuccessorDuplicate, nil
		}
	}
	return EffectSuccessorConflict, &EffectSuccessorStoreError{Outcome: EffectSuccessorConflict, Detail: "receipt id already has a different admission"}
}
