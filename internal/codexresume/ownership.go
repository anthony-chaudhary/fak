package codexresume

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
// thread writer lock. HandleReceiptID binds the PID to the witnessed process start
// time and resource identity; consumers must never compare PID alone.
type WriterOwnership struct {
	ThreadID         string                 `json:"thread_id"`
	LockPath         string                 `json:"lock_path"`
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
	result := WriterOwnership{
		ThreadID:       threadID,
		LockPath:       lockPath,
		Verdict:        WriterOwnershipUnknown,
		EvidenceSource: "filesystem",
	}
	if _, err := os.Stat(lockPath); err != nil {
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
	witness, err := probe.inspect(lockPath)
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
	result.Verdict = WriterOwnershipLiveOwner
	result.PID = owner.pid
	result.ProcessStartTime = owner.startTime
	result.ProcessImage = owner.image
	result.HandleReceiptID = ownershipReceipt(threadID, lockPath, owner)
	result.Detail = "a native resource witness identified a live writer process"
	return result
}

func ownershipReceipt(threadID, lockPath string, owner processOwner) string {
	resource := sha256.Sum256([]byte(strings.ToLower(filepath.Clean(lockPath))))
	return fmt.Sprintf("writer-v1:%x:%d:%016x", resource[:12], owner.pid, owner.startToken)
}
