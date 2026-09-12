package hostdiag

import (
	"encoding/json"
	"errors"
	"io"
	"os"
)

// NativeForwardPath is the diagnostic vocabulary for identities emitted by
// agent.executionIdentity and model's Qwen35 GDN contracts. An unavailable
// execution identity stays empty; new paths require explicit diagnostic support.
type NativeForwardPath string

const (
	NativeForwardUnknown       NativeForwardPath = ""
	NativeForwardCPU           NativeForwardPath = "cpu/reference"
	NativeForwardGeneric       NativeForwardPath = "device/generic"
	NativeForwardMetal         NativeForwardPath = "metal/session-forward"
	NativeForwardMetalHybrid   NativeForwardPath = "metal/qwen35-hybrid-session-v1"
	NativeForwardCPUGDN        NativeForwardPath = "cpu/qwen35-gdn-reference"
	NativeForwardMetalSequence NativeForwardPath = "metal/qwen35-gdn-preprojected-sequence-v1"
	NativeForwardMetalDecode   NativeForwardPath = "fak-native/metal/qwen35-gdn-resident-decode-v1"
	NativeForwardGDNCapability NativeForwardPath = "qwen35/gdn-token-mixer-v1"
	NativeForwardCUDAGDN       NativeForwardPath = "cuda/qwen35-gdn-ssm-decode-v1"
	NativeForwardVulkanGDN     NativeForwardPath = "vulkan/qwen35-gdn-ssm-decode-v1"
)

// NativeFailureContext contains only execution facts supplied by the native owner.
// A nil retryability value means that retry safety is unknown.
type NativeFailureContext struct {
	Model           string            `json:"model,omitempty"`
	Backend         string            `json:"backend,omitempty"`
	ForwardPath     NativeForwardPath `json:"forward_path,omitempty"`
	Phase           string            `json:"phase"`
	Retryable       *bool             `json:"retryable,omitempty"`
	RecoveryAttempt int               `json:"recovery_attempt"`
	Outcome         string            `json:"outcome,omitempty"`
	HostSpill       bool              `json:"host_spill"`
	FallbackActive  bool              `json:"fallback_active"`
}

// NativeFailureRecord retains the operation cause separately from a subsequent
// persistence failure. A disk-full receipt write does not turn a decode EOF into OOM.
type NativeFailureRecord struct {
	Schema string `json:"schema"`
	Engine string `json:"engine"`
	NativeFailureContext
	Failure            *ResourceFailure `json:"failure"`
	PersistenceFailure *ResourceFailure `json:"persistence_failure,omitempty"`
	EvidenceState      EvidenceState    `json:"evidence_state,omitempty"`
}

func NewNativeFailure(err error, context NativeFailureContext) (NativeFailureRecord, error) {
	record := NativeFailureRecord{Schema: "fak.native-failure.v1", Engine: "fak-native", NativeFailureContext: context, Failure: ClassifyResourceFailure(err, context.Phase)}
	return record, validateNativeFailure(record)
}

type EvidenceState string

const (
	EvidenceDurable       EvidenceState = "durable"
	EvidenceEmergencyOnly EvidenceState = "emergency-only"
	EvidenceLost          EvidenceState = "lost"
)

// EvidencePersistence is the authoritative post-write result. A record cannot
// certify its own durability before the storage operation has completed.
type EvidencePersistence struct {
	State EvidenceState `json:"state"`
}

const NativeFailureMaxBytes = 64 << 10

// PersistNativeFailureFile appends and syncs one JSONL record. The owner supplies
// an existing private journal and serializes its writers. Journal creation and
// directory durability belong to that owner's initialization. A successful
// emergency write does not make an incomplete primary write durable.
func PersistNativeFailureFile(path string, record NativeFailureRecord, emergency io.Writer) (EvidencePersistence, error) {
	return PersistNativeFailure(record, func(data []byte) error {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
		if err != nil {
			return err
		}
		n, writeErr := file.Write(data)
		if writeErr == nil && n != len(data) {
			writeErr = io.ErrShortWrite
		}
		if writeErr == nil {
			writeErr = file.Sync()
		}
		return errors.Join(writeErr, file.Close())
	}, emergency)
}

// PersistNativeFailure tries one durable writer, then one bounded emergency write.
// persist must return nil only after durable storage (including sync) succeeds.
// The emergency channel is typically an already-open stderr or supervisor pipe;
// successful delivery there never establishes disk durability.
func PersistNativeFailure(record NativeFailureRecord, persist func([]byte) error, emergency io.Writer) (EvidencePersistence, error) {
	if err := validateNativeFailure(record); err != nil {
		return EvidencePersistence{EvidenceLost}, err
	}
	record.EvidenceState = ""
	record.PersistenceFailure = nil
	data, err := encodeNativeFailure(record)
	if err != nil {
		return EvidencePersistence{EvidenceLost}, err
	}
	if persist == nil {
		err = errors.New("native failure durable writer unavailable")
	} else {
		err = persist(data)
	}
	if err == nil {
		return EvidencePersistence{EvidenceDurable}, nil
	}
	persistErr := err
	record.PersistenceFailure = ClassifyResourceFailure(persistErr, "receipt_persist")
	record.EvidenceState = EvidenceEmergencyOnly
	data, err = encodeNativeFailure(record)
	if err != nil {
		return EvidencePersistence{EvidenceLost}, errors.Join(persistErr, err)
	}
	if emergency == nil {
		return EvidencePersistence{EvidenceLost}, persistErr
	}
	n, emergencyErr := emergency.Write(data)
	if emergencyErr == nil && n != len(data) {
		emergencyErr = io.ErrShortWrite
	}
	if emergencyErr != nil {
		return EvidencePersistence{EvidenceLost}, errors.Join(persistErr, emergencyErr)
	}
	return EvidencePersistence{EvidenceEmergencyOnly}, persistErr
}

func validateNativeFailure(record NativeFailureRecord) error {
	if record.Engine != "fak-native" || record.Schema != "fak.native-failure.v1" {
		return errors.New("native failure record has conflicting engine or schema")
	}
	switch record.Backend {
	case "", "cuda", "metal", "vulkan", "rocm", "cpu", "cpu-ref":
	default:
		return errors.New("native failure record has an unrecognized native backend")
	}
	// Validate execution facts together. The generic identities name a HAL
	// implementation/capability, so they require a concrete native backend but
	// do not themselves imply a GPU (cpu-ref is also a compute.Backend).
	compatible := false
	switch record.ForwardPath {
	case NativeForwardUnknown:
		compatible = true
	case NativeForwardCPU, NativeForwardCPUGDN:
		compatible = record.Backend == "cpu" || record.Backend == "cpu-ref"
	case NativeForwardMetal, NativeForwardMetalHybrid, NativeForwardMetalSequence, NativeForwardMetalDecode:
		compatible = record.Backend == "metal"
	case NativeForwardCUDAGDN:
		compatible = record.Backend == "cuda"
	case NativeForwardVulkanGDN:
		compatible = record.Backend == "vulkan"
	case NativeForwardGeneric, NativeForwardGDNCapability:
		compatible = record.Backend != ""
	}
	if !compatible {
		return errors.New("native failure record has an unknown or conflicting backend/forward-path pair")
	}
	switch record.Phase {
	case "artifact", "mmap", "model_residency", "kv", "workspace", "decode", "receipt_persist":
	default:
		return errors.New("native failure record requires a recognized phase")
	}
	return nil
}

func encodeNativeFailure(record NativeFailureRecord) ([]byte, error) {
	data, err := json.Marshal(record)
	if err != nil {
		return nil, err
	}
	if len(data) >= NativeFailureMaxBytes {
		return nil, errors.New("native failure record exceeds emergency size bound")
	}
	return append(data, '\n'), nil
}
