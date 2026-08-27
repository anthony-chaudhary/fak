package model

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// Q4KResidencyReceiptSchema versions the per-model streamed Q4_K residency evidence.
const Q4KResidencyReceiptSchema = "fak-q4k-residency-receipt/v1"

const q4kResidencyUnset = "<unset>"

// Q4KResidencyCount is one mutually exclusive Q4_K upload outcome aggregate.
type Q4KResidencyCount struct {
	Tensors uint64 `json:"tensors"`
	Bytes   uint64 `json:"bytes"`
}

// Q4KResidencyReceipt records the once-only outcome of each streamed Q4_K tensor's first
// residency attempt. MappedDeclineCopiedUpload means a retained mapping was present but could
// not be aliased, so fak copied the tensor into model-owned memory and successfully uploaded it.
// UploadFailure is separate: a nil final upload never masquerades as a
// successful copied fallback. IntegritySHA256 binds every other field in this nested receipt.
type Q4KResidencyReceipt struct {
	Schema                    string            `json:"schema"`
	FAKGGUFMMap               string            `json:"fak_gguf_mmap"`
	MappedSuccess             Q4KResidencyCount `json:"mapped_success"`
	MappedDeclineCopiedUpload Q4KResidencyCount `json:"mapped_decline_copied_upload"`
	UploadFailure             Q4KResidencyCount `json:"upload_failure"`
	IntegritySHA256           string            `json:"integrity_sha256"`
}

type q4kResidencyOutcome uint8

const (
	q4kResidencyMappedSuccess q4kResidencyOutcome = iota + 1
	q4kResidencyMappedDeclineCopiedUpload
	q4kResidencyUploadFailure
)

type q4kResidencyState struct {
	mu      sync.Mutex
	receipt Q4KResidencyReceipt
	seen    map[string]struct{}
}

// q4kResidencyInitMu protects only lazy attachment of model-owned state. It retains no Model
// pointers; once attached, each model's independent state mutex owns capture and snapshots.
var q4kResidencyInitMu sync.Mutex

func q4kResidencyStateFor(m *Model) *q4kResidencyState {
	q4kResidencyInitMu.Lock()
	defer q4kResidencyInitMu.Unlock()
	if m.q4kResidency == nil {
		m.q4kResidency = &q4kResidencyState{}
	}
	return m.q4kResidency
}

func (s *q4kResidencyState) initializeLocked() {
	if s.seen != nil {
		return
	}
	control, ok := os.LookupEnv("FAK_GGUF_MMAP")
	if !ok {
		control = q4kResidencyUnset
	}
	s.receipt = Q4KResidencyReceipt{Schema: Q4KResidencyReceiptSchema, FAKGGUFMMap: control}
	s.seen = make(map[string]struct{})
}

// recordQ4KResidencyOutcome records only the first upload outcome for a model/name pair. The
// upload cache normally provides this once-only property; the independent seen set keeps the
// receipt stable even after handle teardown removes that cache.
func recordQ4KResidencyOutcome(m *Model, name string, bytes int, outcome q4kResidencyOutcome) {
	if m == nil || name == "" || bytes <= 0 {
		return
	}
	state := q4kResidencyStateFor(m)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.initializeLocked()
	if _, ok := state.seen[name]; ok {
		return
	}
	state.seen[name] = struct{}{}
	var count *Q4KResidencyCount
	switch outcome {
	case q4kResidencyMappedSuccess:
		count = &state.receipt.MappedSuccess
	case q4kResidencyMappedDeclineCopiedUpload:
		count = &state.receipt.MappedDeclineCopiedUpload
	case q4kResidencyUploadFailure:
		count = &state.receipt.UploadFailure
	default:
		delete(state.seen, name)
		return
	}
	count.Tensors++
	count.Bytes += uint64(bytes)
}

func q4kResidencyIntegrity(receipt Q4KResidencyReceipt) (string, error) {
	receipt.IntegritySHA256 = ""
	b, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum), nil
}

// Q4KResidencyReceipt returns an immutable value snapshot. Cached handle reuse and teardown do
// not mutate prior outcomes, and later snapshots remain byte-for-byte stable unless a new tensor
// performs its first upload attempt.
func (m *Model) Q4KResidencyReceipt() Q4KResidencyReceipt {
	state := q4kResidencyStateFor(m)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.initializeLocked()
	receipt := state.receipt
	receipt.IntegritySHA256, _ = q4kResidencyIntegrity(receipt)
	return receipt
}

// ValidateQ4KResidencyReceipt rejects malformed aggregates and any integrity-binding mismatch.
func ValidateQ4KResidencyReceipt(receipt Q4KResidencyReceipt) error {
	if receipt.Schema != Q4KResidencyReceiptSchema {
		return fmt.Errorf("unexpected Q4_K residency receipt schema %q", receipt.Schema)
	}
	for name, count := range map[string]Q4KResidencyCount{
		"mapped success":               receipt.MappedSuccess,
		"mapped-decline copied upload": receipt.MappedDeclineCopiedUpload,
		"upload failure":               receipt.UploadFailure,
	} {
		if (count.Tensors == 0) != (count.Bytes == 0) {
			return fmt.Errorf("Q4_K residency %s has contradictory tensors=%d bytes=%d", name, count.Tensors, count.Bytes)
		}
	}
	want, err := q4kResidencyIntegrity(receipt)
	if err != nil || want != receipt.IntegritySHA256 {
		return fmt.Errorf("Q4_K residency integrity digest mismatch")
	}
	return nil
}
