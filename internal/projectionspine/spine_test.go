package projectionspine

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/supervisionpolicy"
)

const helperRoleEnv = "FAK_PROJECTION_SPINE_HELPER"

type childReport struct {
	Address string   `json:"address,omitempty"`
	PID     int      `json:"pid"`
	State   Snapshot `json:"state"`
}

func TestProjectionReplacementPreservesAuthority(t *testing.T) {
	if role := os.Getenv(helperRoleEnv); role != "" {
		runHelper(role)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	authority := startAuthority(t, ctx)
	defer authority.stop(t)
	original := startProjection(t, ctx, authority.report.Address)

	if original.report.State.AuthorityPID != authority.cmd.Process.Pid {
		t.Fatalf("projection observed authority PID %d, want %d", original.report.State.AuthorityPID, authority.cmd.Process.Pid)
	}

	// The failure is confined to the disposable projection. Kill and Wait are
	// both required so the dead process cannot leak into the replacement proof.
	original.killAndWait(t)
	now := time.Unix(1_800_000_000, 0)
	budget := supervisionpolicy.Budget{MaxRestarts: 2, Window: time.Minute}
	decision := DecideProjectionFailure(original.report.State, "terminal-view", 1, nil, now, budget)
	if decision.Action != supervisionpolicy.ActionReattach {
		t.Fatalf("failure action = %v, want ActionReattach", decision.Action)
	}

	var replacement *projectionChild
	if decision.Action == supervisionpolicy.ActionReattach {
		replacement = startProjection(t, ctx, authority.report.Address)
	}
	if replacement == nil {
		t.Fatal("reattach decision did not launch replacement projection")
	}
	defer replacement.killAndWait(t)

	if replacement.report.PID == original.report.PID {
		t.Fatalf("replacement reused projection PID %d", replacement.report.PID)
	}
	before, after := original.report.State, replacement.report.State
	if after.AuthorityPID != before.AuthorityPID || after.SessionID != before.SessionID || after.WriterEpoch != before.WriterEpoch {
		t.Fatalf("durable identity changed across projection replacement: before=%+v after=%+v", before, after)
	}
	if after.TranscriptMarker != before.TranscriptMarker || after.TranscriptMarker == "" {
		t.Fatalf("replacement did not resume transcript marker: before=%q after=%q", before.TranscriptMarker, after.TranscriptMarker)
	}

	first, err := ExecuteEffect(ctx, authority.report.Address, after.SessionID, after.WriterEpoch, "book:42")
	if err != nil {
		t.Fatal(err)
	}
	second, err := ExecuteEffect(ctx, authority.report.Address, after.SessionID, after.WriterEpoch, "book:42")
	if err != nil {
		t.Fatal(err)
	}
	if !first.Executed || second.Executed || first.Count != 1 || second.Count != 1 {
		t.Fatalf("effect was not exactly-once: first=%+v second=%+v", first, second)
	}
	if _, err := ExecuteEffect(ctx, authority.report.Address, after.SessionID, after.WriterEpoch+1, "book:43"); err == nil {
		t.Fatal("stale/foreign writer epoch crossed the authority fence")
	}
}

func TestProjectionRestartBudgetEscalatesWithoutLaunch(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	state := Snapshot{SessionID: "session-stable", WriterEpoch: 7}
	budget := supervisionpolicy.Budget{MaxRestarts: 2, Window: time.Minute}
	var failures []time.Time
	launches := 0

	for attempt := 0; attempt < 3; attempt++ {
		decision := DecideProjectionFailure(state, "terminal-view", 1, failures, now, budget)
		if attempt < 2 {
			if decision.Action != supervisionpolicy.ActionReattach {
				t.Fatalf("attempt %d action = %v, want ActionReattach", attempt+1, decision.Action)
			}
			if decision.Action == supervisionpolicy.ActionReattach {
				launches++
			}
			failures = append(failures, now)
			continue
		}
		if decision.Action != supervisionpolicy.ActionEscalate {
			t.Fatalf("attempt 3 action = %v, want ActionEscalate", decision.Action)
		}
		if decision.Action == supervisionpolicy.ActionReattach {
			launches++
		}
	}
	if launches != 2 {
		t.Fatalf("launched %d projections, want exactly 2 before escalation", launches)
	}
}

func runHelper(role string) {
	switch role {
	case "authority":
		runAuthorityHelper()
	case "projection":
		runProjectionHelper()
	default:
		fmt.Fprintln(os.Stderr, "unknown projection-spine helper role")
		os.Exit(2)
	}
}

func runAuthorityHelper() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	authority, err := NewAuthority(os.Getpid(), "logical-session-8912", 19, "transcript:turn-41")
	if err != nil {
		panic(err)
	}
	report := childReport{Address: listener.Addr().String(), PID: os.Getpid(), State: authority.Snapshot()}
	if err := json.NewEncoder(os.Stdout).Encode(report); err != nil {
		panic(err)
	}
	serveDone := make(chan error, 1)
	go func() { serveDone <- authority.Serve(listener) }()
	_, _ = bufio.NewReader(os.Stdin).ReadBytes(0)
	_ = listener.Close()
	<-serveDone
}

func runProjectionHelper() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	projection, err := Attach(ctx, os.Getenv("FAK_PROJECTION_SPINE_ADDRESS"))
	if err != nil {
		panic(err)
	}
	defer projection.Close()
	if err := json.NewEncoder(os.Stdout).Encode(childReport{PID: os.Getpid(), State: projection.Snapshot}); err != nil {
		panic(err)
	}
	if err := projection.Wait(ctx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		panic(err)
	}
}

type authorityChild struct {
	cmd    *exec.Cmd
	stdin  ioWriteCloser
	report childReport
}

type ioWriteCloser interface {
	Write([]byte) (int, error)
	Close() error
}

func startAuthority(t *testing.T, ctx context.Context) *authorityChild {
	t.Helper()
	cmd := helperCommand(ctx, "authority")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var report childReport
	if err := json.NewDecoder(stdout).Decode(&report); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("read authority readiness: %v", err)
	}
	if report.PID != cmd.Process.Pid || report.Address == "" {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("invalid authority report: %+v process=%d", report, cmd.Process.Pid)
	}
	return &authorityChild{cmd: cmd, stdin: stdin, report: report}
}

func (c *authorityChild) stop(t *testing.T) {
	t.Helper()
	_ = c.stdin.Close()
	waitWithFallback(t, c.cmd, "authority")
}

type projectionChild struct {
	cmd    *exec.Cmd
	report childReport
	waited bool
}

func startProjection(t *testing.T, ctx context.Context, address string) *projectionChild {
	t.Helper()
	cmd := helperCommand(ctx, "projection")
	cmd.Env = append(cmd.Env, "FAK_PROJECTION_SPINE_ADDRESS="+address)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var report childReport
	if err := json.NewDecoder(stdout).Decode(&report); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("read projection snapshot: %v", err)
	}
	if report.PID != cmd.Process.Pid {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("projection report PID %d, process PID %d", report.PID, cmd.Process.Pid)
	}
	return &projectionChild{cmd: cmd, report: report}
}

func (c *projectionChild) killAndWait(t *testing.T) {
	t.Helper()
	if c == nil || c.waited {
		return
	}
	c.waited = true
	if err := c.cmd.Process.Kill(); err != nil {
		t.Fatalf("kill projection %d: %v", c.cmd.Process.Pid, err)
	}
	done := make(chan error, 1)
	go func() { done <- c.cmd.Wait() }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatalf("projection %d exited successfully after Process.Kill", c.cmd.Process.Pid)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for killed projection %d", c.cmd.Process.Pid)
	}
}

func helperCommand(ctx context.Context, role string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestProjectionReplacementPreservesAuthority$")
	cmd.Env = append(os.Environ(), helperRoleEnv+"="+role, "FAK_HELPER_NONCE="+strconv.FormatInt(time.Now().UnixNano(), 10))
	return cmd
}

func waitWithFallback(t *testing.T, cmd *exec.Cmd, name string) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("%s cleanup: %v", name, err)
		}
	case <-time.After(2 * time.Second):
		_ = cmd.Process.Kill()
		<-done
		t.Errorf("%s required forced cleanup", name)
	}
}
