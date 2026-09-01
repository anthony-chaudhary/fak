package serverlifecycle

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
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

// ReadyExpectation pins the endpoint identity a read-only consumer intends to use.
// Generation is exact when nonzero; MinimumGeneration rejects older generations.
type ReadyExpectation struct {
	Generation           uint64
	MinimumGeneration    uint64
	ProcessID            int
	ProcessStartIdentity string
	ReceiptDigest        string
	ProtocolFamily       string
	ProtocolRevision     string
	Capabilities         []string
	BaseURL              string
	ModelAlias           string
}

// ReadyBinding is the validated immutable identity chain consumed by a client.
type ReadyBinding struct {
	Receipt       serverproduct.ServerReceipt
	ReceiptDigest string
	ReceiptBytes  []byte
}

// ConsumeReady validates the current READY lifecycle state without mutating files
// or signalling the managed process. It rereads both files after the live-process
// check so a concurrent restart cannot splice state from one generation to a
// receipt from another.
func ConsumeReady(dir string, want ReadyExpectation) (ReadyBinding, error) {
	if strings.TrimSpace(dir) == "" {
		return ReadyBinding{}, errors.New("lifecycle directory is required")
	}
	statePath := filepath.Join(dir, StateFilename)
	receiptPath := filepath.Join(dir, ReceiptFilename)
	stateRaw, err := os.ReadFile(statePath)
	if err != nil {
		return ReadyBinding{}, fmt.Errorf("read lifecycle state: %w", err)
	}
	state, err := readReadyConsumerState(stateRaw)
	if err != nil {
		return ReadyBinding{}, err
	}
	receiptRaw, err := os.ReadFile(receiptPath)
	if err != nil {
		return ReadyBinding{}, fmt.Errorf("read ready receipt: %w", err)
	}
	receipt, err := serverproduct.DecodeReceipt(receiptRaw)
	if err != nil {
		return ReadyBinding{}, fmt.Errorf("decode ready receipt: %w", err)
	}
	digest := receiptDigest(receiptRaw)
	if err := matchReadyConsumerState(state, receipt, digest, want); err != nil {
		return ReadyBinding{}, err
	}
	current, ok := processIdentity(state.ProcessID)
	if !ok {
		return ReadyBinding{}, errors.New("revalidate process start identity: process is not live")
	}
	if current != state.ProcessStartIdentity {
		return ReadyBinding{}, errors.New("live process start identity mismatch")
	}
	stateAfter, err := os.ReadFile(statePath)
	if err != nil {
		return ReadyBinding{}, fmt.Errorf("reread lifecycle state: %w", err)
	}
	receiptAfter, err := os.ReadFile(receiptPath)
	if err != nil {
		return ReadyBinding{}, fmt.Errorf("reread ready receipt: %w", err)
	}
	if !bytes.Equal(stateRaw, stateAfter) || !bytes.Equal(receiptRaw, receiptAfter) {
		return ReadyBinding{}, errors.New("lifecycle READY identity changed during validation")
	}
	return ReadyBinding{Receipt: receipt, ReceiptDigest: digest, ReceiptBytes: append([]byte(nil), receiptRaw...)}, nil
}

func readReadyConsumerState(raw []byte) (stateRecord, error) {
	var state stateRecord
	if err := decodeConsumerStrict(raw, &state); err != nil {
		return stateRecord{}, fmt.Errorf("decode lifecycle state: %w", err)
	}
	if state.Schema != stateSchema {
		return stateRecord{}, fmt.Errorf("lifecycle state schema must be %q", stateSchema)
	}
	if state.State != StateReady {
		return stateRecord{}, fmt.Errorf("lifecycle state = %q, want %q", state.State, StateReady)
	}
	if state.Generation == 0 || state.ProcessID <= 0 || state.ProcessStartIdentity == "" || state.InstanceID == "" || state.BaseURL == "" {
		return stateRecord{}, errors.New("ready lifecycle state identity is incomplete")
	}
	return state, nil
}

func matchReadyConsumerState(state stateRecord, receipt serverproduct.ServerReceipt, digest string, want ReadyExpectation) error {
	if receipt.Identity.InstanceID != state.InstanceID || receipt.Generation != state.Generation ||
		receipt.Ownership.InstanceID != state.InstanceID || receipt.Ownership.ProcessID != state.ProcessID ||
		receipt.Ownership.ProcessStartIdentity != state.ProcessStartIdentity || receipt.Endpoint.BaseURL != state.BaseURL {
		return errors.New("ready receipt identity does not match lifecycle state")
	}
	if want.Generation != 0 && state.Generation != want.Generation {
		return fmt.Errorf("generation = %d, want %d", state.Generation, want.Generation)
	}
	if want.MinimumGeneration != 0 && state.Generation < want.MinimumGeneration {
		return fmt.Errorf("generation = %d, want at least %d", state.Generation, want.MinimumGeneration)
	}
	if want.ProcessID != 0 && state.ProcessID != want.ProcessID {
		return fmt.Errorf("process id = %d, want %d", state.ProcessID, want.ProcessID)
	}
	if want.ProcessStartIdentity != "" && state.ProcessStartIdentity != want.ProcessStartIdentity {
		return errors.New("process start identity mismatch")
	}
	if want.ReceiptDigest != "" && digest != want.ReceiptDigest {
		return fmt.Errorf("receipt digest = %q, want %q", digest, want.ReceiptDigest)
	}
	if want.BaseURL != "" && receipt.Endpoint.BaseURL != want.BaseURL {
		return fmt.Errorf("base URL = %q, want %q", receipt.Endpoint.BaseURL, want.BaseURL)
	}
	if want.ModelAlias != "" && receipt.ModelAlias != want.ModelAlias {
		return fmt.Errorf("model alias = %q, want %q", receipt.ModelAlias, want.ModelAlias)
	}
	if want.ProtocolFamily != "" && receipt.Protocol.Family != want.ProtocolFamily {
		return fmt.Errorf("protocol family = %q, want %q", receipt.Protocol.Family, want.ProtocolFamily)
	}
	if want.ProtocolRevision != "" && receipt.Protocol.Revision != want.ProtocolRevision {
		return fmt.Errorf("protocol revision = %q, want %q", receipt.Protocol.Revision, want.ProtocolRevision)
	}
	if len(want.Capabilities) > 0 {
		got, expected := slices.Clone(receipt.Protocol.Capabilities), slices.Clone(want.Capabilities)
		slices.Sort(got)
		slices.Sort(expected)
		if !slices.Equal(got, expected) {
			return errors.New("protocol capabilities mismatch")
		}
	}
	return nil
}

// ProcessIdentity returns the kernel-observed start identity used by lifecycle receipts.
func ProcessIdentity(pid int) (string, bool) { return processIdentity(pid) }

func receiptDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func decodeConsumerStrict(raw []byte, target any) error {
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
