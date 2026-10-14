//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The subprocesses execute only this already-built test image. They do not
// import a model/backend, access a device, or create a network endpoint.
func TestMain(m *testing.M) {
	if len(os.Args) == 6 && strings.Join(os.Args[1:], " ") == "benchmarks run v41-ep-owner -- --supervised-owner" {
		if err := guardTestOwner(); err != nil {
			os.Exit(3)
		}
		os.Exit(0)
	}
	if len(os.Args) == 3 && os.Args[1] == "--guard-test-parent" {
		raw, err := os.ReadFile(os.Args[2])
		if err != nil {
			os.Exit(4)
		}
		var p guardPlan
		if json.Unmarshal(raw, &p) != nil {
			os.Exit(4)
		}
		result := guardRun(context.Background(), p, guardTestAdmission(p))
		_ = json.NewEncoder(os.Stdout).Encode(result)
		if result.Status != "complete" {
			os.Exit(5)
		}
		os.Exit(0)
	}
	if len(os.Args) == 3 && os.Args[1] == "--guard-test-reaper" {
		if err := guardTestReaper(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(6)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// fak-test:runtime medium est=7s lane=default
func TestV41EPOwnerGuardContract(t *testing.T) {
	t.Run("strict-plan-and-caps", func(t *testing.T) {
		p := guardTestPlan(t, "normal", "")
		for _, mode := range []string{"rank", "helper-pin", "image-pin", "request", "path", "term", "grace", "memory", "disk", "environment", "driver-input", "shader-input"} {
			q := p
			switch mode {
			case "rank":
				q.Rank = nil
			case "helper-pin":
				q.HelperSHA256 = strings.Repeat("a", 40)
			case "image-pin":
				q.OwnerSHA256 = ""
			case "request":
				q.RequestSHA256 = strings.Repeat("a", 64)
			case "path":
				q.OwnerPath = "/tmp/../owner"
			case "term":
				q.TerminateNS = guardMaxTerminate + 1
			case "grace":
				q.KillGraceNS = guardMaxKillGrace + 1
			case "memory":
				q.MinimumMemoryBytes = guardMemoryFloor - 1
			case "disk":
				q.MinimumDiskBytes = guardDiskFloor - 1
			case "environment":
				q.Environment = []string{"LANG=C", "LC_ALL=C", "TZ=UTC", "LD_PRELOAD=evil"}
			case "driver-input":
				q.Environment = []string{"LANG=C", "LC_ALL=C", "TZ=UTC", "VK_ICD_FILENAMES=/qualified-later.json"}
			case "shader-input":
				q.Environment = []string{"LANG=C", "LC_ALL=C", "TZ=UTC", "FAK_VULKAN_SPIRV=/qualified-later"}
			}
			if q.validate() == nil {
				t.Fatalf("accepted %s", mode)
			}
		}
		raw, _ := json.Marshal(p)
		var decoded guardPlan
		if err := guardStrict(raw, &decoded); err != nil {
			t.Fatal(err)
		}
		for _, malformed := range [][]byte{append(append([]byte(nil), raw...), ' '), bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":null`), 1), bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":0,"rank":0`), 1)} {
			if err := guardStrict(malformed, &decoded); err == nil {
				t.Fatal("accepted ambiguous/noncanonical plan")
			}
		}
	})
	t.Run("artifact-admission-is-independent-and-exact", func(t *testing.T) {
		p := guardTestPlan(t, "normal", "")
		for _, admission := range []guardArtifactAdmission{{}, {ownerSHA: p.OwnerSHA256, environmentSHA: guardEnvironmentSHA(p.Environment), evidenceSHA: strings.Repeat("a", 64), kind: "untrusted-plan-claim"}} {
			result := guardRun(context.Background(), p, admission)
			if result.PID != 0 || result.WaitJoined || result.ArtifactQualification != nil || result.NativeLoadingQualification != "unresolved" || !strings.Contains(result.Failure, "native launch refused") {
				t.Fatalf("unqualified launch not refused before Start: %+v", result)
			}
		}
		admission := guardTestAdmission(p)
		p.Environment = append(p.Environment, "CUDA_VISIBLE_DEVICES=0")
		if result := guardRun(context.Background(), p, admission); result.PID != 0 || !strings.Contains(result.Failure, "native launch refused") {
			t.Fatal("changed environment reused qualification")
		}
	})
	t.Run("startup-cap-is-independent-and-sticky", func(t *testing.T) {
		now := time.Now()
		d := newGuardDeadline(now.Add(-time.Millisecond), now.Add(time.Hour), now.Add(2*time.Hour), time.Hour)
		defer d.close()
		until := time.Now().Add(time.Second)
		for time.Now().Before(until) {
			d.mu.Lock()
			expired := d.startupExpired
			d.mu.Unlock()
			if expired {
				break
			}
			time.Sleep(time.Millisecond)
		}
		if d.admit() == nil {
			t.Fatal("late admission won")
		}
		_, err := d.snapshot()
		if err == nil || !strings.Contains(err.Error(), "startup/admission deadline exceeded") {
			t.Fatal("startup failure disappeared")
		}
	})
	t.Run("memory-observation-preserves-unknown-versus-zero", func(t *testing.T) {
		for _, raw := range []string{"", "MemTotal: 100 kB\n", "MemAvailable: 5 MB\n", "MemAvailable: 18446744073709551615 kB\n", "MemAvailable: 2 kB\nMemAvailable: 3 kB\n"} {
			if _, err := guardMemoryAvailable([]byte(raw)); err == nil {
				t.Fatal("missing or malformed memory accepted")
			}
		}
		if n, err := guardMemoryAvailable([]byte("MemAvailable: 0 kB\n")); err != nil || n != 0 {
			t.Fatal("observed zero became unavailable")
		}
		if n, err := guardMemoryAvailable([]byte("MemAvailable: 5 kB\n")); err != nil || n != 5120 {
			t.Fatal("memory unit conversion changed")
		}
	})
	t.Run("resource-monitor-refuses-before-start-and-joins", func(t *testing.T) {
		p := guardTestPlan(t, "normal", "")
		initial, err := guardResources(p.ResourcePath)
		if err != nil {
			t.Fatal(err)
		}
		p.ResourcePath = filepath.Join(p.ResourcePath, "missing")
		calls := 0
		m := newGuardResourceMonitor(p, initial, func(error) { calls++ })
		if m.failure() == nil {
			t.Fatal("unavailable observation was admitted")
		}
		samples, err := m.close()
		if err == nil || len(samples) != 2 || calls != 1 || samples[0].Available {
			t.Fatal("unjoined or ambiguous unavailable resource evidence")
		}
	})
	t.Run("actual-owned-completion-and-refusal", func(t *testing.T) {
		for _, mode := range []string{"normal", "wrong-challenge", "output-cap", "ignore-term"} {
			t.Run(mode, func(t *testing.T) {
				p := guardTestPlan(t, mode, "")
				if mode == "ignore-term" {
					p.TerminateNS = guardStartup + 250*time.Millisecond
				}
				result := guardRun(context.Background(), p, guardTestAdmission(p))
				if !result.WaitJoined || result.GroupReleased == nil || !*result.GroupReleased || result.ExitCode == nil {
					t.Fatalf("owned child not retired: %+v", result)
				}
				if (mode == "normal") != (result.Status == "complete") {
					t.Fatalf("mode=%s: %+v", mode, result)
				}
				if mode == "normal" && (result.AdmissionSHA256 == "" || result.ChildReceiptSHA256 == "" || len(result.ResourceSamples) < 2 || result.ArtifactQualification == nil || result.ArtifactQualification.Kind != "test-image-qualification" || result.NativeLoadingQualification != "unresolved") {
					t.Fatal("success lacks bound observations")
				}
				if mode == "wrong-challenge" && (result.AdmissionSHA256 != "" || !strings.Contains(result.Failure, "live child challenge disagrees")) {
					t.Fatal("mismatch admitted")
				}
				if mode == "output-cap" && (len(result.Stdout) != guardMaxOutput || !strings.Contains(result.Failure, "stdout cap exceeded")) {
					t.Fatal("unbounded output")
				}
				if mode == "ignore-term" {
					term, kill := false, false
					for _, event := range result.Signals {
						term = term || event.Signal == syscall.SIGTERM.String()
						kill = kill || event.Signal == syscall.SIGKILL.String()
					}
					if !term || !kill || result.ChildReceipt != nil || result.ExitCode == nil || *result.ExitCode != -1 || result.AdmissionSHA256 == "" || result.TerminatingSignal == nil || *result.TerminatingSignal != syscall.SIGKILL.String() {
						t.Fatal("hard-killed owner became completed work")
					}
				}
			})
		}
	})
	t.Run("held-image-and-fd-cap-refusal", func(t *testing.T) {
		p := guardTestPlan(t, "normal", "")
		if _, err := guardOpenImage(p.OwnerPath, strings.Repeat("0", 64), false); err == nil {
			t.Fatal("wrong image accepted")
		}
		if _, err := guardTerminal(-1); err == nil {
			t.Fatal("missing pidfd accepted")
		}
		if _, err := guardTerminal(1 << 20); err == nil {
			t.Fatal("unbounded pidfd accepted")
		}
		var frame bytes.Buffer
		if guardWriteFrame(&frame, 1, make([]byte, guardMaxControl+1), guardMaxControl) == nil {
			t.Fatal("oversized control accepted")
		}
		if err := guardWriteFrame(&frame, 1, []byte(`{}`), guardMaxControl); err != nil {
			t.Fatal(err)
		}
		if _, err := guardReadFrame(&frame, 2, guardMaxControl); err == nil {
			t.Fatal("wrong frame order accepted")
		}
	})
	t.Run("abrupt-supervisor-death-kills-and-reaps-owner", func(t *testing.T) {
		ready := filepath.Join(t.TempDir(), "owner-ready.json")
		p := guardTestPlan(t, "parent-death", ready)
		p.TerminateNS = guardStartup + 500*time.Millisecond
		planPath := filepath.Join(t.TempDir(), "plan.json")
		raw, _ := json.Marshal(p)
		if err := os.WriteFile(planPath, raw, 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), guardStartup+guardRetireGrace+time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, p.OwnerPath, "--guard-test-reaper", planPath)
		cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("parent-death witness: %v %s", err, output)
		}
		if string(output) != "OWNER_KILLED_AND_REAPED\n" {
			t.Fatalf("missing terminal witness: %s", output)
		}
	})
}

// Only the already-built, actual running test ELF can receive this admission.
// This issuer is absent from production and supplies no native driver evidence.
func guardTestAdmission(p guardPlan) guardArtifactAdmission {
	raw, err := guardReadFile("/proc/self/exe", guardMaxImage)
	if err != nil || guardSHA(raw) != p.OwnerSHA256 || p.validate() != nil {
		return guardArtifactAdmission{}
	}
	environment := guardEnvironmentSHA(p.Environment)
	evidence := guardSHA([]byte("test-image-qualification\x00" + guardSHA(raw) + "\x00" + environment))
	return guardArtifactAdmission{ownerSHA: guardSHA(raw), environmentSHA: environment, evidenceSHA: evidence, kind: "test-image-qualification"}
}

func guardTestPlan(t *testing.T, mode, ready string) guardPlan {
	t.Helper()
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := guardReadFile("/proc/self/exe", guardMaxImage)
	if err != nil {
		t.Fatal(err)
	}
	digest := guardSHA(raw)
	request, _ := json.Marshal(struct {
		Mode  string `json:"mode"`
		Ready string `json:"ready"`
	}{mode, ready})
	rank := 0
	return guardPlan{Schema: guardPlanSchema, HelperSHA256: digest, OwnerPath: path, OwnerSHA256: digest, Request: request,
		RequestSHA256: guardSHA(request), Rank: &rank, ContractID: "test-only", TerminateNS: 2 * time.Second, KillGraceNS: 100 * time.Millisecond,
		Environment: []string{"LANG=C", "LC_ALL=C", "TZ=UTC"}, ResourcePath: t.TempDir(), MinimumMemoryBytes: guardMemoryFloor, MinimumDiskBytes: guardDiskFloor}
}

func guardTestOwner() error {
	if err := syscall.SetNonblock(4, true); err != nil {
		return err
	}
	if err := syscall.SetNonblock(5, true); err != nil {
		return err
	}
	image, read, write := os.NewFile(3, "image"), os.NewFile(4, "control"), os.NewFile(5, "events")
	defer image.Close()
	defer read.Close()
	defer write.Close()
	for _, fd := range []int{3, 4, 5} {
		syscall.CloseOnExec(fd)
	}
	if err := read.SetReadDeadline(time.Now().Add(4 * time.Second)); err != nil {
		return err
	}
	helloRaw, err := guardReadFrame(read, 1, guardMaxControl)
	if err != nil {
		return err
	}
	var hello guardHello
	if err := guardStrict(helloRaw, &hello); err != nil {
		return err
	}
	request, err := guardReadFrame(read, 2, guardMaxRequest)
	if err != nil {
		return err
	}
	var fixture struct {
		Mode  string `json:"mode"`
		Ready string `json:"ready"`
	}
	if err := json.Unmarshal(request, &fixture); err != nil {
		return err
	}
	if fixture.Mode == "ignore-term" || fixture.Mode == "parent-death" {
		signal.Ignore(syscall.SIGTERM)
	}
	if fixture.Mode == "output-cap" {
		_, err := os.Stdout.Write(bytes.Repeat([]byte{'x'}, guardMaxOutput+1))
		if err != nil {
			return err
		}
	}
	var nonce [32]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return err
	}
	challenge := guardChallenge{Nonce: hex.EncodeToString(nonce[:]), ContractID: hello.ContractID, RequestSHA256: hello.RequestSHA256, Rank: hello.Rank,
		PID: os.Getpid(), ExecutableSHA256: hello.OwnerSHA256, LatestTerminateUnixNS: hello.TerminateUnixNS, KillGraceNS: hello.KillUnixNS - hello.TerminateUnixNS}
	if fixture.Mode == "wrong-challenge" {
		challenge.Rank = 1
	}
	raw, _ := json.Marshal(challenge)
	if err := guardWriteFrame(write, 3, raw, guardMaxControl); err != nil {
		return err
	}
	admissionRaw, err := guardReadFrame(read, 4, guardMaxControl)
	if err != nil {
		return err
	}
	var admission guardAdmission
	if err := guardStrict(admissionRaw, &admission); err != nil {
		return err
	}
	if fixture.Mode == "parent-death" {
		identity, err := guardObserveProcess(os.Getpid())
		if err != nil {
			return err
		}
		raw, _ := json.Marshal([]any{identity.pid, identity.ppid, identity.start})
		if err := os.WriteFile(fixture.Ready, raw, 0600); err != nil {
			return err
		}
	}
	if fixture.Mode == "ignore-term" || fixture.Mode == "parent-death" {
		for {
			time.Sleep(time.Hour)
		}
	}
	result := struct {
		Schema                   string `json:"schema"`
		Status                   string `json:"status"`
		Rank                     int    `json:"rank"`
		RequestSHA256            string `json:"request_sha256"`
		ExecutableSHA256         string `json:"executable_sha256"`
		SupervisorEvidenceSHA256 string `json:"supervisor_evidence_sha256"`
	}{"fak.v41-ep-owner.receipt/v1", "complete", hello.Rank, hello.RequestSHA256, hello.OwnerSHA256, admission.EvidenceSHA256}
	raw, _ = json.Marshal(result)
	return guardWriteFrame(write, 5, raw, guardMaxReceipt)
}

// This setting is confined to the disposable witness helper. It allows it to
// reap the intentionally orphaned child after abruptly killing the supervisor.
// Production guardRun never changes subreaper state or host configuration.
func guardTestReaper(planPath string) error {
	const prSetChildSubreaper = 36
	if _, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, prSetChildSubreaper, 1, 0, 0, 0, 0); errno != 0 {
		return errno
	}
	raw, err := os.ReadFile(planPath)
	if err != nil {
		return err
	}
	var p guardPlan
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	var fixture struct {
		Ready string `json:"ready"`
	}
	if err := json.Unmarshal(p.Request, &fixture); err != nil {
		return err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	parent := exec.Command(p.OwnerPath, "--guard-test-parent", planPath)
	parent.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	parent.Stdout, parent.Stderr = io.Discard, io.Discard
	if err := parent.Start(); err != nil {
		return err
	}
	parentJoined := false
	defer func() {
		if !parentJoined {
			_ = parent.Process.Kill()
			_ = parent.Wait()
		}
	}()
	var identity guardProcess
	until := time.Now().Add(guardStartup + 250*time.Millisecond)
	for time.Now().Before(until) {
		raw, err := os.ReadFile(fixture.Ready)
		if err == nil {
			var values []json.RawMessage
			if json.Unmarshal(raw, &values) != nil || len(values) != 3 {
				return errors.New("invalid ready witness")
			}
			var pid int
			if json.Unmarshal(values[0], &pid) != nil {
				return errors.New("invalid child PID")
			}
			identity, err = guardObserveProcess(pid)
			if err != nil {
				return err
			}
			if identity.ppid != parent.Process.Pid {
				return errors.New("wrong owned parent")
			}
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if identity.pid == 0 {
		return errors.New("owner never reached admission")
	}
	owner, err := os.FindProcess(identity.pid)
	if err != nil {
		return err
	}
	defer owner.Release()
	defer func() {
		if !parentJoined {
			_ = parent.Process.Kill()
			_ = parent.Wait()
			parentJoined = true
		}
		_ = owner.Kill()
		var status syscall.WaitStatus
		_, _ = syscall.Wait4(identity.pid, &status, 0, nil)
	}()
	if err := parent.Process.Kill(); err != nil {
		return err
	}
	_ = parent.Wait()
	parentJoined = true
	until = time.Now().Add(guardRetireGrace)
	for time.Now().Before(until) {
		var status syscall.WaitStatus
		pid, err := syscall.Wait4(identity.pid, &status, syscall.WNOHANG, nil)
		if err != nil {
			return err
		}
		if pid == identity.pid {
			if !status.Signaled() || status.Signal() != syscall.SIGKILL {
				return errors.New("owner did not die by parent-death SIGKILL")
			}
			if err := syscall.Kill(-identity.pgid, 0); !errors.Is(err, syscall.ESRCH) {
				return errors.New("owner group not released")
			}
			fmt.Println("OWNER_KILLED_AND_REAPED")
			return nil
		}
		time.Sleep(5 * time.Millisecond)
	}
	return errors.New("owner survived abrupt supervisor death")
}
