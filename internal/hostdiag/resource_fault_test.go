package hostdiag

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestResourceFailureEvidence(t *testing.T) {
	if os.Getenv("FAK_HOSTDIAG_TEST_SELF_KILL") == "1" {
		process, err := os.FindProcess(os.Getpid())
		if err != nil {
			os.Exit(2)
		}
		_ = process.Kill()
		os.Exit(3)
	}
	for _, tt := range []struct {
		name  string
		err   error
		op    string
		class ResourceClass
	}{
		{"storage", syscall.ENOSPC, "receipt_persist", HostStorageFull},
		{"memory", syscall.ENOMEM, "mmap", HostMemoryExhausted},
		{"process_fds", syscall.EMFILE, "artifact", HostFDLimit},
		{"system_fds", syscall.ENFILE, "artifact", HostFDLimit},
		{"spawn", syscall.EAGAIN, "spawn", HostProcessLimit},
		{"read_again", syscall.EAGAIN, "artifact", ""},
		{"eof", io.EOF, "decode", ""},
		{"signal_text", errors.New("signal: killed (SIGKILL), status 137, OOM"), "decode", ""},
		{"allocation", &compute.DeviceAllocError{Bytes: 4096}, "kv", DeviceOOM},
		{"device_lost", &compute.DeviceFaultError{Class: compute.DeviceFaultContext}, "decode", DeviceUnavailable},
		{"host_fit", &compute.FitError{Verdict: compute.FitTooBig, Want: 20, Avail: 10, Scope: compute.MemoryScopeHost}, "model_residency", HostMemoryExhausted},
		{"device_fit", &compute.FitError{Verdict: compute.FitTooBig, Want: 20, Avail: 10, Scope: compute.MemoryScopeDevice}, "model_residency", DeviceOOM},
		{"invalid_fit_scope", &compute.FitError{Verdict: compute.FitTooBig, Want: 20, Avail: 10, Scope: "invalid"}, "model_residency", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := fmt.Errorf("native operation: %w", tt.err)
			got := ClassifyResourceFailure(wrapped, tt.op)
			if got.Class != tt.class || !errors.Is(got, tt.err) {
				t.Fatalf("class=%q want=%q; causal chain retained=%v", got.Class, tt.class, errors.Is(got, tt.err))
			}
			if tt.class != "" && (got.Subcause == "" || got.Source == "" || got.Recovery == "") {
				t.Fatalf("missing provenance/recovery: %+v", got)
			}
		})
	}

	process := ProcessIdentity{PID: 42, BootID: "fixture-boot", StartBootTimeUS: 100}
	fixture := []byte(`{"__CURSOR":"fixture-cursor","_TRANSPORT":"kernel","SYSLOG_FACILITY":"0","_BOOT_ID":"fixture-boot","_SOURCE_MONOTONIC_TIMESTAMP":"150","MESSAGE":"Out of memory: Killed process 42 (fixture) total-vm:4096kB, anon-rss:2048kB, file-rss:0kB, shmem-rss:0kB"}`)
	for _, mismatch := range []string{"", "pid", "boot", "start", "source", "artifact", "forged_claim", "after_exit", "receive_time", "userspace_kmsg"} {
		p, artifact, exitAt := process, fixture, uint64(200)
		switch mismatch {
		case "pid":
			p.PID++
		case "boot":
			p.BootID = "previous-boot"
		case "start":
			p.StartBootTimeUS = 151
		case "source":
			artifact = bytes.ReplaceAll(fixture, []byte(`"kernel"`), []byte(`"stdout"`))
		case "artifact":
			artifact = nil
		case "forged_claim":
			artifact, _ = json.Marshal(OOMKillEvidence{Process: process, Source: "kernel_oom_victim", ArtifactSHA256: strings.Repeat("a", 64)})
		case "after_exit":
			exitAt = 149
		case "receive_time":
			artifact = bytes.ReplaceAll(fixture, []byte("_SOURCE_MONOTONIC_TIMESTAMP"), []byte("__MONOTONIC_TIMESTAMP"))
		case "userspace_kmsg":
			artifact = bytes.ReplaceAll(fixture, []byte(`"SYSLOG_FACILITY":"0"`), []byte(`"SYSLOG_FACILITY":"1"`))
		}
		got := ClassifyResourceFailure(attributeKernelOOM(io.EOF, p, exitAt, artifact), "decode")
		want := ResourceClass("")
		if mismatch == "" {
			want = HostMemoryExhausted
		}
		if got.Class != want || !errors.Is(got, io.EOF) {
			t.Fatalf("attribution %q: %+v", mismatch, got)
		}
		if mismatch == "" {
			digest := sha256.Sum256(fixture)
			if got.OOMEvidence.ArtifactSHA256 != hex.EncodeToString(digest[:]) {
				t.Fatal("digest is not bound to captured artifact")
			}
		}
	}
	unrelated := bytes.ReplaceAll(fixture, []byte("process 42"), []byte("process 43"))
	query := append(append(append([]byte(nil), fixture...), '\n'), unrelated...)
	oom := ClassifyResourceFailure(attributeKernelOOM(io.EOF, process, 200, query), "decode")
	if oom.OOMEvidence == nil || oom.OOMEvidence.Artifact != string(fixture) || oom.OOMEvidence.Cursor != "fixture-cursor" {
		t.Fatalf("matched artifact not retained: %+v", oom)
	}
	digest := sha256.Sum256(fixture)
	if oom.OOMEvidence.ArtifactSHA256 != hex.EncodeToString(digest[:]) {
		t.Fatal("unrelated row changed evidence digest")
	}

	record, err := NewNativeFailure(io.EOF, NativeFailureContext{Phase: "decode", Model: "fixture", Backend: "vulkan"})
	if err != nil {
		t.Fatal(err)
	}
	var emergency bytes.Buffer
	result, err := PersistNativeFailure(record, func([]byte) error { return syscall.ENOSPC }, &emergency)
	if result.State != EvidenceEmergencyOnly || !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("persistence=%+v err=%v", result, err)
	}
	var decoded NativeFailureRecord
	if err := json.Unmarshal(emergency.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Engine != "fak-native" || decoded.EvidenceState != EvidenceEmergencyOnly || decoded.Failure.Class != "" || decoded.Failure.Message != "EOF" || decoded.PersistenceFailure.Class != HostStorageFull {
		t.Fatalf("lost or invented cause: %+v", decoded)
	}
	result, err = PersistNativeFailure(record, func([]byte) error { return syscall.ENOSPC }, shortEmergency{})
	if result.State != EvidenceLost || !errors.Is(err, io.ErrShortWrite) || !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("short emergency=%+v err=%v", result, err)
	}
	result, err = PersistNativeFailure(record, func(data []byte) error {
		if !bytes.Contains(data, []byte(`"engine":"fak-native"`)) {
			t.Fatal("engine absent")
		}
		return nil
	}, nil)
	if result.State != EvidenceDurable || err != nil {
		t.Fatalf("durable=%+v err=%v", result, err)
	}
	for _, conflict := range []string{"engine", "backend", "forward_path", "llama_cpp", "llama-cpp", "external/service", "empty_phase", "invalid_phase"} {
		bad := record
		switch conflict {
		case "engine":
			bad.Engine = "llama.cpp"
		case "backend":
			bad.Backend = "llama.cpp"
		case "forward_path":
			bad.ForwardPath = "llama.cpp-native"
		case "empty_phase":
			bad.Phase = ""
		case "invalid_phase":
			bad.Phase = "guess"
		default:
			bad.ForwardPath = NativeForwardPath(conflict)
		}
		result, err := PersistNativeFailure(bad, func([]byte) error { t.Fatal("persisted conflicting identity"); return nil }, &emergency)
		if err == nil || result.State != EvidenceLost {
			t.Fatalf("accepted identity conflict %s", conflict)
		}
	}
	if _, err := NewNativeFailure(io.EOF, NativeFailureContext{Backend: "llama.cpp"}); err == nil {
		t.Fatal("constructor accepted non-native identity")
	}
	for _, phase := range []string{"artifact", "mmap", "model_residency", "kv", "workspace", "decode", "receipt_persist"} {
		if _, err := NewNativeFailure(syscall.ENOMEM, NativeFailureContext{Phase: phase}); err != nil {
			t.Fatalf("valid phase %s: %v", phase, err)
		}
	}
	for _, pair := range []struct {
		backend string
		forward NativeForwardPath
		valid   bool
	}{
		{"cpu-ref", NativeForwardCPU, true}, {"cpu-ref", NativeForwardGeneric, true}, {"cuda", NativeForwardGeneric, true}, {"metal", NativeForwardMetal, true}, {"cpu-ref", NativeForwardCPUGDN, true},
		{"cpu-ref", NativeForwardCUDAGDN, false}, {"cuda", NativeForwardCPU, false}, {"", NativeForwardMetal, false}, {"", NativeForwardGeneric, false}, {"", NativeForwardGDNCapability, false},
		{"cuda", NativeForwardGDNCapability, true}, {"vulkan", NativeForwardVulkanGDN, true},
	} {
		_, err := NewNativeFailure(io.EOF, NativeFailureContext{Phase: "decode", Backend: pair.backend, ForwardPath: pair.forward})
		if (err == nil) != pair.valid {
			t.Fatalf("native pair %+v: %v", pair, err)
		}
	}
	emergency.Reset()
	oomRecord, err := NewNativeFailure(oom, NativeFailureContext{Phase: "decode", Backend: "cpu-ref", ForwardPath: "cpu/reference", HostSpill: true, FallbackActive: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err = PersistNativeFailure(oomRecord, func([]byte) error { return syscall.ENOSPC }, &emergency)
	if result.State != EvidenceEmergencyOnly || err == nil {
		t.Fatalf("OOM persistence=%+v err=%v", result, err)
	}
	if err := json.Unmarshal(emergency.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Engine != "fak-native" || !decoded.HostSpill || !decoded.FallbackActive || decoded.Failure.OOMEvidence.Artifact != string(fixture) {
		t.Fatalf("native OOM evidence did not survive: %+v", decoded)
	}
	emergency.Reset()
	result, err = PersistNativeFailure(record, nil, &emergency)
	if result.State != EvidenceEmergencyOnly || err == nil {
		t.Fatalf("nil primary=%+v err=%v", result, err)
	}
	result, err = PersistNativeFailure(record, func([]byte) error { return syscall.ENOSPC }, failedEmergency{})
	if result.State != EvidenceLost || !errors.Is(err, io.ErrClosedPipe) || !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("failed emergency=%+v err=%v", result, err)
	}
	oversize := record
	oversize.Model = strings.Repeat("x", NativeFailureMaxBytes)
	result, err = PersistNativeFailure(oversize, func([]byte) error { t.Fatal("oversize record written"); return nil }, &emergency)
	if result.State != EvidenceLost || err == nil {
		t.Fatalf("oversize=%+v err=%v", result, err)
	}
	journal := filepath.Join(t.TempDir(), "failure.jsonl")
	if err := os.WriteFile(journal, nil, 0600); err != nil {
		t.Fatal(err)
	}
	result, err = PersistNativeFailureFile(journal, record, nil)
	if result.State != EvidenceDurable || err != nil {
		t.Fatalf("file persistence=%+v err=%v", result, err)
	}
	data, err := os.ReadFile(journal)
	if err != nil || !bytes.Contains(data, []byte(`"engine":"fak-native"`)) {
		t.Fatalf("file readback: %s, %v", data, err)
	}
	if runtime.GOOS == "linux" {
		emergency.Reset()
		result, err = PersistNativeFailureFile("/dev/full", record, &emergency)
		if result.State != EvidenceEmergencyOnly || !errors.Is(err, syscall.ENOSPC) || !json.Valid(bytes.TrimSpace(emergency.Bytes())) {
			t.Fatalf("real ENOSPC=%+v err=%v evidence=%s", result, err, emergency.Bytes())
		}
		child := exec.Command(os.Args[0], "-test.run=^TestResourceFailureEvidence$")
		child.Env = append(os.Environ(), "FAK_HOSTDIAG_TEST_SELF_KILL=1")
		err := child.Run()
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ProcessState.Sys().(syscall.WaitStatus).Signal() != syscall.SIGKILL {
			t.Fatalf("self-kill fixture: %v", err)
		}
		if got := ClassifyResourceFailure(err, "decode"); got.Class != "" || !errors.Is(got, err) {
			t.Fatalf("SIGKILL attributed without evidence: %+v", got)
		}
	}
}

type shortEmergency struct{}

func (shortEmergency) Write(p []byte) (int, error) { return len(p) / 2, nil }

type failedEmergency struct{}

func (failedEmergency) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
