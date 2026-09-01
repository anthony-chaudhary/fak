package codexresume

import (
	"errors"
	"fmt"
	"os"
)

// WriterOwnershipVerdict is the closed ownership vocabulary for a Codex
// thread-writer resource.
type WriterOwnershipVerdict string

const (
	WriterOwnershipLiveOwner    WriterOwnershipVerdict = "live_owner"
	WriterOwnershipStaleResidue WriterOwnershipVerdict = "stale_residue"
	WriterOwnershipAbsent       WriterOwnershipVerdict = "absent"
	WriterOwnershipUnknown      WriterOwnershipVerdict = "unknown"
)

// WriterOwnership is a JSON-safe, read-only ownership receipt for one Codex
// thread writer lock. Compatibility fields remain alongside the typed resource.
type WriterOwnership struct {
	ThreadID         string                 `json:"thread_id"`
	LockPath         string                 `json:"lock_path"`
	Resource         *WriterResourceHandle  `json:"resource,omitempty"`
	LockPresent      bool                   `json:"lock_present"`
	Verdict          WriterOwnershipVerdict `json:"verdict"`
	PID              int                    `json:"pid,omitempty"`
	ProcessStartTime string                 `json:"process_start_time,omitempty"`
	ProcessImage     string                 `json:"process_image,omitempty"`
	EvidenceSource   string                 `json:"evidence_source"`
	HandleReceiptID  string                 `json:"handle_receipt_id,omitempty"`
	Detail           string                 `json:"detail,omitempty"`
}

type processOwner struct {
	pid        int
	startTime  string
	startToken uint64
	image      string
}

type ownershipWitness struct {
	source     string
	conclusive bool
	owners     []processOwner
}

type ownershipProbe interface {
	inspect(lockPath string) (ownershipWitness, error)
}

// InspectWriterOwnership returns a read-only ownership receipt using the native platform witness.
func InspectWriterOwnership(threadID, lockPath string) WriterOwnership {
	return inspectWriterOwnership(threadID, lockPath, nativeOwnershipProbe{})
}

func inspectWriterOwnership(threadID, lockPath string, probe ownershipProbe) WriterOwnership {
	thread, err := NewCodexThreadIdentity(threadID)
	if err != nil {
		return invalidWriterOwnership(threadID, lockPath, err)
	}
	resource, err := NewWriterResourceHandle(thread, lockPath)
	if err != nil {
		return invalidWriterOwnership(threadID, lockPath, err)
	}
	return inspectWriterResourceOwnership(resource, probe)
}

func inspectWriterResourceOwnership(resource WriterResourceHandle, probe ownershipProbe) WriterOwnership {
	result := WriterOwnership{
		ThreadID:       resource.Thread.ID,
		LockPath:       resource.LockPath,
		Verdict:        WriterOwnershipUnknown,
		EvidenceSource: "filesystem",
	}
	if err := resource.Validate(); err != nil {
		result.Detail = fmt.Sprintf("invalid writer resource handle: %v", err)
		return result
	}
	result.Resource = &resource
	if _, err := os.Stat(resource.LockPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			result.Verdict = WriterOwnershipAbsent
			result.Detail = "writer lock is absent"
			return result
		}
		result.Detail = fmt.Sprintf("inspect writer lock: %v", err)
		return result
	}
	result.LockPresent = true
	if probe == nil {
		probe = nativeOwnershipProbe{}
	}
	witness, err := probe.inspect(resource.LockPath)
	if witness.source != "" {
		result.EvidenceSource = witness.source
	}
	if err != nil {
		result.Detail = fmt.Sprintf("inspect writer ownership: %v", err)
		return result
	}
	if !witness.conclusive {
		result.Detail = "the platform could not conclusively inspect writer ownership"
		return result
	}
	if len(witness.owners) == 0 {
		result.Verdict = WriterOwnershipStaleResidue
		result.Detail = "the writer lock exists and a positive native witness found no owning process"
		return result
	}
	owner := witness.owners[0]
	if owner.pid <= 0 || owner.startToken == 0 {
		result.Detail = "the native ownership witness did not provide a stable PID and process start token"
		return result
	}
	result.Verdict = WriterOwnershipLiveOwner
	result.PID = owner.pid
	result.ProcessStartTime = owner.startTime
	result.ProcessImage = owner.image
	result.HandleReceiptID = ownershipReceipt(resource, owner)
	result.Detail = "a native resource witness identified a live writer process"
	return result
}

func invalidWriterOwnership(threadID, lockPath string, err error) WriterOwnership {
	return WriterOwnership{
		ThreadID:       threadID,
		LockPath:       lockPath,
		Verdict:        WriterOwnershipUnknown,
		EvidenceSource: "validation",
		Detail:         fmt.Sprintf("invalid writer resource identity: %v", err),
	}
}

func ownershipReceipt(resource WriterResourceHandle, owner processOwner) string {
	return fmt.Sprintf("writer-owner-v1:%s:%d:%016x", resource.ResourceID, owner.pid, owner.startToken)
}
