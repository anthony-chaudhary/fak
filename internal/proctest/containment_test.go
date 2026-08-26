// Package proctest contains black-box process supervision contracts.
package proctest

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

const helperEnv = "FAK_PROCTEST_HELPER"

type processIdentity struct {
	PID        int    `json:"pid"`
	StartToken string `json:"start_token"`
	OwnerID    string `json:"owner_id"`
}

type containmentReceipt struct {
	OwnerID             string            `json:"owner_id"`
	RootPID             int               `json:"root_pid"`
	RootIdentity        processIdentity   `json:"root_identity"`
	Victim              processIdentity   `json:"victim"`
	Survivors           []processIdentity `json:"survivors"`
	TerminationReason   string            `json:"termination_reason"`
	ContainmentBoundary string            `json:"containment_boundary"`
	OSMechanism         string            `json:"os_mechanism"`
	CheckpointBefore    int               `json:"checkpoint_before"`
	CheckpointAfter     int               `json:"checkpoint_after"`
}

func TestMain(m *testing.M) {
	if mode := os.Getenv(helperEnv); mode != "" {
		os.Exit(runHelper(mode))
	}
	os.Exit(m.Run())
}

func runHelper(mode string) int {
	switch mode {
	case "heartbeat":
		path := os.Getenv("FAK_PROCTEST_CHECKPOINT")
		for n := 1; ; n++ {
			if err := os.WriteFile(path, []byte(strconv.Itoa(n)), 0o600); err != nil {
				return 3
			}
			time.Sleep(20 * time.Millisecond)
		}
	case "wait":
		select {}
	case "exit-nonzero":
		return 23
	case "coordinator-crash", "gateway-crash":
		return 31
	default:
		return 2
	}
}

// TestIndependentOwnerSurvivesCrash proves fault containment by post-fault
// progress, rather than by the weaker observation that a sibling PID exists.
func TestIndependentOwnerSurvivesCrash(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mode   string
		forced bool
		reason string
	}{
		{name: "worker exits nonzero", mode: "exit-nonzero", reason: "exit_nonzero"},
		{name: "worker is forcibly killed", mode: "wait", forced: true, reason: "forced_kill"},
		{name: "coordinator crashes", mode: "coordinator-crash", reason: "coordinator_exit"},
		{name: "shared gateway crashes", mode: "gateway-crash", reason: "shared_service_exit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runContainmentCase(t, tc.mode, tc.forced, tc.reason)
		})
	}
}

func runContainmentCase(t *testing.T, victimMode string, forced bool, reason string) {
	t.Helper()
	dir := t.TempDir()
	checkpoint := filepath.Join(dir, "survivor.checkpoint")
	receiptPath := filepath.Join(dir, "containment-receipt.json")

	survivorToken := fmt.Sprintf("survivor-%d", time.Now().UnixNano())
	survivor := helperCommand("heartbeat")
	survivor.Env = append(survivor.Env, "FAK_PROCTEST_CHECKPOINT="+checkpoint)
	if err := survivor.Start(); err != nil {
		t.Fatalf("start survivor: %v", err)
	}
	t.Cleanup(func() { stopProcess(survivor) })
	before := awaitCheckpoint(t, checkpoint, 2, 5*time.Second)

	victimToken := fmt.Sprintf("victim-%d", time.Now().UnixNano())
	victim := helperCommand(victimMode)
	if err := victim.Start(); err != nil {
		t.Fatalf("start victim: %v", err)
	}
	victimPID := victim.Process.Pid
	if forced {
		if err := victim.Process.Kill(); err != nil {
			t.Fatalf("kill victim: %v", err)
		}
	}
	if err := victim.Wait(); err == nil {
		t.Fatal("crash injection unexpectedly exited successfully")
	}

	after := awaitCheckpoint(t, checkpoint, before+1, 5*time.Second)
	receipt := containmentReceipt{
		OwnerID:             "victim-owner",
		RootPID:             victimPID,
		RootIdentity:        processIdentity{PID: victimPID, StartToken: victimToken, OwnerID: "victim-owner"},
		Victim:              processIdentity{PID: victimPID, StartToken: victimToken, OwnerID: "victim-owner"},
		Survivors:           []processIdentity{{PID: survivor.Process.Pid, StartToken: survivorToken, OwnerID: "survivor-owner"}},
		TerminationReason:   reason,
		ContainmentBoundary: "independent_owner",
		OSMechanism:         osMechanism(),
		CheckpointBefore:    before,
		CheckpointAfter:     after,
	}
	if err := persistReceipt(receiptPath, receipt); err != nil {
		t.Fatalf("persist receipt after victim exit: %v", err)
	}

	got, err := readReceipt(receiptPath)
	if err != nil {
		t.Fatalf("read persisted receipt: %v", err)
	}
	if err := got.validate(); err != nil {
		t.Fatalf("invalid receipt: %v", err)
	}
	if got.CheckpointAfter <= got.CheckpointBefore {
		t.Fatalf("survivor made no post-fault progress: before=%d after=%d", got.CheckpointBefore, got.CheckpointAfter)
	}
}

func TestReceiptRejectsPIDReuseAmbiguity(t *testing.T) {
	r := containmentReceipt{
		OwnerID: "owner-a", RootPID: 42,
		RootIdentity:      processIdentity{PID: 42, StartToken: "start-a", OwnerID: "owner-a"},
		Victim:            processIdentity{PID: 42, StartToken: "start-b", OwnerID: "owner-a"},
		Survivors:         []processIdentity{{PID: 43, StartToken: "start-c", OwnerID: "owner-b"}},
		TerminationReason: "forced_kill", ContainmentBoundary: "independent_owner",
		OSMechanism: osMechanism(), CheckpointBefore: 1, CheckpointAfter: 2,
	}
	if err := r.validate(); !errors.Is(err, errPIDIdentityAmbiguous) {
		t.Fatalf("validate error = %v, want %v", err, errPIDIdentityAmbiguous)
	}
}

var errPIDIdentityAmbiguous = errors.New("pid identity is ambiguous")

func (r containmentReceipt) validate() error {
	if strings.TrimSpace(r.OwnerID) == "" || r.RootPID <= 0 || strings.TrimSpace(r.TerminationReason) == "" || strings.TrimSpace(r.ContainmentBoundary) == "" || strings.TrimSpace(r.OSMechanism) == "" {
		return errors.New("receipt is missing required containment metadata")
	}
	if r.RootIdentity.PID != r.RootPID || r.RootIdentity.StartToken == "" || r.Victim.StartToken == "" || r.RootIdentity.OwnerID != r.OwnerID || r.Victim.OwnerID != r.OwnerID {
		return errPIDIdentityAmbiguous
	}
	if r.Victim.PID == r.RootIdentity.PID && r.Victim.StartToken != r.RootIdentity.StartToken {
		return errPIDIdentityAmbiguous
	}
	if len(r.Survivors) == 0 || r.CheckpointAfter <= r.CheckpointBefore {
		return errors.New("receipt has no witnessed survivor progress")
	}
	for _, survivor := range r.Survivors {
		if survivor.PID <= 0 || survivor.StartToken == "" || survivor.OwnerID == "" {
			return errPIDIdentityAmbiguous
		}
		if survivor.OwnerID == r.OwnerID {
			return errors.New("independent survivor shares victim owner")
		}
		if survivor.PID == r.Victim.PID && survivor.StartToken != r.Victim.StartToken {
			return errPIDIdentityAmbiguous
		}
	}
	return nil
}

func helperCommand(mode string) *exec.Cmd {
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), helperEnv+"="+mode)
	return cmd
}

func stopProcess(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
}

func awaitCheckpoint(t *testing.T, path string, minimum int, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(path)
		if err == nil {
			n, convErr := strconv.Atoi(string(data))
			if convErr == nil && n >= minimum {
				return n
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("checkpoint %q did not reach %d within %s", path, minimum, timeout)
	return 0
}

func persistReceipt(path string, receipt containmentReceipt) error {
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func readReceipt(path string) (containmentReceipt, error) {
	var receipt containmentReceipt
	data, err := os.ReadFile(path)
	if err != nil {
		return receipt, err
	}
	err = json.Unmarshal(data, &receipt)
	return receipt, err
}

func osMechanism() string {
	if runtime.GOOS == "windows" {
		return "windows_process_handle"
	}
	return "posix_signal"
}
