package hostdiag

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ResourceClass is empty when the available evidence cannot attribute a cause.
type ResourceClass string

const (
	HostStorageFull     ResourceClass = "HOST_STORAGE_FULL"
	HostMemoryExhausted ResourceClass = "HOST_MEMORY_EXHAUSTED"
	HostFDLimit         ResourceClass = "HOST_FD_LIMIT"
	HostProcessLimit    ResourceClass = "HOST_PROCESS_LIMIT"
	DeviceUnavailable   ResourceClass = "DEVICE_UNAVAILABLE"
	DeviceOOM           ResourceClass = "DEVICE_OOM"
)

// ResourceFailure preserves the original error independently of its classification.
// Byte counts are absent unless the typed source actually reports them.
type ResourceFailure struct {
	Message          string           `json:"message"`
	MessageTruncated bool             `json:"message_truncated,omitempty"`
	Class            ResourceClass    `json:"class,omitempty"`
	Subcause         string           `json:"subcause,omitempty"`
	Source           string           `json:"source,omitempty"`
	Recovery         string           `json:"recovery,omitempty"`
	RequestedBytes   *int64           `json:"requested_bytes,omitempty"`
	AvailableBytes   *int64           `json:"available_bytes,omitempty"`
	OOMEvidence      *OOMKillEvidence `json:"oom_evidence,omitempty"`
	cause            error
}

func (f *ResourceFailure) Error() string {
	if f == nil {
		return "resource failure: cause unavailable"
	}
	if f.cause == nil {
		return f.Message
	}
	return f.cause.Error()
}

func (f *ResourceFailure) Unwrap() error {
	if f == nil {
		return nil
	}
	return f.cause
}

// ClassifyResourceFailure inspects typed causes, never error-message substrings.
// EAGAIN denotes process exhaustion only at a known process-spawn boundary.
func ClassifyResourceFailure(err error, operation string) *ResourceFailure {
	if err == nil {
		return nil
	}
	f := &ResourceFailure{cause: err, Message: err.Error()}
	if len(f.Message) > 2048 {
		f.Message = f.Message[:2048]
		f.MessageTruncated = true
	}
	set := func(class ResourceClass, subcause, source, recovery string) {
		f.Class, f.Subcause, f.Source, f.Recovery = class, subcause, source, recovery
	}
	var oom *attributedOOMKill
	var alloc *compute.DeviceAllocError
	var fault *compute.DeviceFaultError
	var fit *compute.FitError
	switch {
	case errors.As(err, &oom) && oom != nil:
		set(HostMemoryExhausted, "os_oom_kill", oom.evidence.Source, "reduce admitted host memory before restarting the native process")
		e := oom.evidence
		f.OOMEvidence = &e
	case errors.Is(err, syscall.ENOSPC):
		set(HostStorageFull, "ENOSPC", "os_errno", "free storage or select a writable evidence destination")
	case errors.Is(err, syscall.ENOMEM):
		set(HostMemoryExhausted, "ENOMEM", "os_errno", "reduce host memory demand and check address-space limits before retrying")
	case errors.Is(err, syscall.EMFILE):
		set(HostFDLimit, "EMFILE", "os_errno", "release process file descriptors before retrying")
	case errors.Is(err, syscall.ENFILE):
		set(HostFDLimit, "ENFILE", "os_errno", "restore system file-descriptor headroom before retrying")
	case operation == "spawn" && errors.Is(err, syscall.EAGAIN):
		set(HostProcessLimit, "EAGAIN", "os_errno", "restore process or thread headroom before retrying launch")
	case errors.As(err, &fault) && fault != nil:
		set(DeviceUnavailable, "device_fault_"+string(fault.Class), "compute.DeviceFaultError", "reconstruct and validate the device session through its fault latch")
	case errors.As(err, &alloc) && alloc != nil:
		set(DeviceOOM, "allocation_failed", "compute.DeviceAllocError", "reduce device residency or release idle allocations before a bounded retry")
		if alloc.Bytes >= 0 {
			n := int64(alloc.Bytes)
			f.RequestedBytes = &n
		}
	case errors.As(err, &fit) && fit != nil && fit.Verdict == compute.FitTooBig && fit.Want > fit.Avail && fit.Avail >= 0:
		class := DeviceOOM
		if fit.Scope == compute.MemoryScopeHost {
			class = HostMemoryExhausted
		}
		if fit.Scope != "" && fit.Scope != compute.MemoryScopeDevice && fit.Scope != compute.MemoryScopeHost {
			break
		}
		set(class, "admission_capacity", "compute.FitError", "reduce the requested memory envelope before admission")
		want, avail := fit.Want, fit.Avail
		f.RequestedBytes, f.AvailableBytes = &want, &avail
	}
	return f
}

// ProcessIdentity binds evidence to a process lifetime, including across reboots
// and PID reuse. StartBootTimeUS is the supervisor's observed process start in
// CLOCK_BOOTTIME microseconds. PID is the host-namespace PID. Journal receipt
// time cannot establish lifetime membership for a delayed kernel message.
type ProcessIdentity struct {
	PID             int    `json:"pid"`
	BootID          string `json:"boot_id"`
	StartBootTimeUS uint64 `json:"start_boottime_us"`
}

// OOMKillEvidence is an output receipt. Only the kernel-record parser can mint
// the internal attributed error; constructing this receipt grants no attribution.
type OOMKillEvidence struct {
	Process         ProcessIdentity `json:"process"`
	Source          string          `json:"source"`
	ArtifactSHA256  string          `json:"artifact_sha256"`
	EventBootTimeUS uint64          `json:"event_boottime_us"`
	Cursor          string          `json:"cursor,omitempty"`
	// Artifact contains the exact matched journal row. Keep diagnostic records
	// private: this source can contain operator metadata. Digest covers these bytes.
	Artifact string `json:"artifact"`
}

type attributedOOMKill struct {
	cause    error
	evidence OOMKillEvidence
}

func (e *attributedOOMKill) Error() string { return e.cause.Error() }
func (e *attributedOOMKill) Unwrap() error { return e.cause }

var kernelOOMVictim = regexp.MustCompile(`^(?:Out of memory|Memory cgroup out of memory): Killed process ([1-9][0-9]*) \([^\n]*\) total-vm:[0-9]+kB, anon-rss:[0-9]+kB, file-rss:[0-9]+kB, shmem-rss:[0-9]+kB`)

// CollectKernelOOM reads the local OS kernel journal through a fixed executable,
// with bounded output and runtime. The second error describes collection failure;
// it is separate from the child's cause and cannot classify that cause as OOM.
// The supervisor supplies process lifetime facts captured before and after wait.
func CollectKernelOOM(ctx context.Context, cause error, observed ProcessIdentity, exitBootTimeUS uint64) (error, error) {
	if runtime.GOOS != "linux" {
		return cause, errors.New("kernel OOM collector requires Linux journal support")
	}
	if cause == nil || observed.PID <= 0 || observed.StartBootTimeUS == 0 || exitBootTimeUS <= observed.StartBootTimeUS {
		return cause, errors.New("kernel OOM collector requires an observed process lifetime")
	}
	boot, err := hex.DecodeString(observed.BootID)
	if err != nil || len(boot) != 16 {
		return cause, errors.New("kernel OOM collector requires the journal boot ID")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/journalctl", "--kernel", "--boot="+observed.BootID, "--output=json", "--no-pager", "--lines=256", "--grep=Killed process")
	cmd.WaitDelay = time.Second
	output := &boundedKernelJournal{}
	cmd.Stdout, cmd.Stderr = output, io.Discard
	if err := cmd.Run(); err != nil {
		return cause, err
	}
	return attributeKernelOOM(cause, observed, exitBootTimeUS, output.buffer.Bytes()), nil
}

type boundedKernelJournal struct{ buffer bytes.Buffer }

func (b *boundedKernelJournal) Write(p []byte) (int, error) {
	if b.buffer.Len()+len(p) > 1<<20 {
		return 0, errors.New("kernel journal exceeds OOM evidence bound")
	}
	return b.buffer.Write(p)
}

// attributeKernelOOM matches a supervisor-captured kernel journal JSONL artifact
// to the exact observed process lifetime. Callers must obtain it from the trusted
// OS journal, never from model output, stderr, or a client request. The digest is
// computed from those bytes, not accepted as a caller-authored claim. Unavailable,
// oversized, malformed, or unrelated evidence leaves the original cause intact.
// The exit timestamp must use the same CLOCK_BOOTTIME domain as the start.
func attributeKernelOOM(cause error, observed ProcessIdentity, exitBootTimeUS uint64, artifact []byte) error {
	if cause == nil || observed.PID <= 0 || observed.BootID == "" || observed.StartBootTimeUS == 0 || exitBootTimeUS <= observed.StartBootTimeUS || len(artifact) == 0 || len(artifact) > 1<<20 {
		return cause
	}
	scan := bufio.NewScanner(bytes.NewReader(artifact))
	scan.Buffer(make([]byte, 4096), 64<<10)
	var matched uint64
	var matchedRow, matchedCursor string
	for scan.Scan() {
		if len(scan.Bytes()) > 8192 || !utf8.Valid(scan.Bytes()) {
			return cause
		}
		var row struct {
			Transport            string `json:"_TRANSPORT"`
			BootID               string `json:"_BOOT_ID"`
			SourceBootTime       string `json:"_SOURCE_BOOTTIME_TIMESTAMP"`
			LegacySourceBootTime string `json:"_SOURCE_MONOTONIC_TIMESTAMP"`
			Facility             string `json:"SYSLOG_FACILITY"`
			Message              string `json:"MESSAGE"`
			Cursor               string `json:"__CURSOR"`
		}
		if json.Unmarshal(scan.Bytes(), &row) != nil {
			return cause
		}
		if row.Transport != "kernel" || row.Facility != "0" || row.BootID != observed.BootID {
			continue
		}
		// journald historically gave the kernel's BOOTTIME timestamp a
		// MONOTONIC field name. Both source fields represent the kernel event;
		// the double-underscore journal receive timestamp never substitutes.
		stampText := row.SourceBootTime
		if stampText == "" {
			stampText = row.LegacySourceBootTime
		}
		if row.SourceBootTime != "" && row.LegacySourceBootTime != "" && row.SourceBootTime != row.LegacySourceBootTime {
			continue
		}
		stamp, err := strconv.ParseUint(stampText, 10, 64)
		if err != nil || stamp < observed.StartBootTimeUS || stamp > exitBootTimeUS {
			continue
		}
		victim := kernelOOMVictim.FindStringSubmatch(row.Message)
		if len(victim) != 2 {
			continue
		}
		pid, err := strconv.Atoi(victim[1])
		if err == nil && pid == observed.PID {
			matched = stamp
			matchedRow, matchedCursor = string(scan.Bytes()), row.Cursor
		}
	}
	if scan.Err() != nil || matched == 0 {
		return cause
	}
	digest := sha256.Sum256([]byte(matchedRow))
	return &attributedOOMKill{cause: cause, evidence: OOMKillEvidence{Process: observed, Source: "kernel_oom_victim", ArtifactSHA256: hex.EncodeToString(digest[:]), EventBootTimeUS: matched, Cursor: matchedCursor, Artifact: matchedRow}}
}
