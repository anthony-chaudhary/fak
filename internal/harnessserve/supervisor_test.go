package harnessserve

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

var fakeRuntimePath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fak-harnessserve-runtime-")
	if err != nil {
		panic(err)
	}
	name := "fakeruntime"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	fakeRuntimePath = filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", fakeRuntimePath, "./testdata/fakeruntime")
	if output, err := cmd.CombinedOutput(); err != nil {
		panic("build fake runtime: " + err.Error() + ": " + string(output))
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

func TestSupervisorLaunchReadinessOneTokenProbeAndGracefulStop(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "lifecycle.txt")
	s := &Supervisor{}
	receipt, err := s.Launch(context.Background(), testPlan("normal", marker))
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Ownership.PID <= 0 || len(receipt.Ownership.Token) < 32 || receipt.Ownership.StartIdentity == "" {
		t.Fatalf("ownership receipt is not PID/start/token bound: %+v", receipt.Ownership)
	}
	argvDigest, err := hex.DecodeString(receipt.ArgvSHA256)
	if err != nil || len(argvDigest) != sha256.Size || receipt.ArgvSHA256 != hex.EncodeToString(argvDigest) {
		t.Fatalf("argv digest is not canonical SHA-256: %q err=%v", receipt.ArgvSHA256, err)
	}
	if receipt.Probe.HTTPStatus != http.StatusOK || receipt.Probe.CompletionTokens != 1 {
		t.Fatalf("launch receipt = %+v", receipt)
	}
	endpoint, err := url.Parse(receipt.Endpoint)
	if err != nil {
		t.Fatal(err)
	}
	host, _, err := net.SplitHostPort(endpoint.Host)
	if err != nil || !net.ParseIP(host).IsLoopback() {
		t.Fatalf("endpoint escaped loopback: %q err=%v", receipt.Endpoint, err)
	}

	// Independent readback: do not trust Launch's readiness claim.
	response, err := (&http.Client{Timeout: time.Second}).Get(receipt.Endpoint + "/health")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(io.LimitReader(response.Body, 128))
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK || !strings.Contains(string(body), `"ready"`) {
		t.Fatalf("independent health = status %d body %q", response.StatusCode, body)
	}

	stopped, err := s.Stop(context.Background(), receipt.Ownership)
	if err != nil {
		t.Fatal(err)
	}
	if !stopped.GracefulAttempted || stopped.Escalated {
		t.Fatalf("graceful stop receipt = %+v", stopped)
	}
	if got := strings.TrimSpace(readFileEventually(t, marker)); got != "graceful" {
		t.Fatalf("runtime lifecycle marker = %q, want graceful", got)
	}
}

func TestSupervisorReadinessTimeoutIsBoundedAndReapsChild(t *testing.T) {
	plan := testPlan("never-ready", filepath.Join(t.TempDir(), "lifecycle.txt"))
	plan.StartupTimeout = 120 * time.Millisecond
	s := &Supervisor{}
	started := time.Now()
	_, err := s.Launch(context.Background(), plan)
	if refusalCode(err) != RefusalReadinessTimeout {
		t.Fatalf("Launch error = %v, want %s", err, RefusalReadinessTimeout)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("readiness refusal exceeded bound: %s", elapsed)
	}
	s.mu.Lock()
	owned := s.owned
	s.mu.Unlock()
	if owned != nil {
		t.Fatal("timed-out launch retained process ownership")
	}
}

func TestSupervisorRejectsUnboundedOrMalformedCompletionProbe(t *testing.T) {
	for _, tc := range []struct {
		name string
		plan Plan
	}{
		{name: "two token response", plan: testPlan("bad-probe", filepath.Join(t.TempDir(), "bad.txt"))},
		{name: "probe timeout", plan: func() Plan {
			p := testPlan("normal", filepath.Join(t.TempDir(), "slow.txt"))
			p.Args = append(p.Args, "--probe-delay", "250ms")
			p.ProbeTimeout = 40 * time.Millisecond
			return p
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &Supervisor{}
			_, err := s.Launch(context.Background(), tc.plan)
			if refusalCode(err) != RefusalProbeFailed {
				t.Fatalf("Launch error = %v, want %s", err, RefusalProbeFailed)
			}
			s.mu.Lock()
			owned := s.owned
			s.mu.Unlock()
			if owned != nil {
				t.Fatal("failed protocol probe retained process ownership")
			}
		})
	}
}

func TestSupervisorEscalatesAfterGracefulStopBound(t *testing.T) {
	s := &Supervisor{}
	plan := testPlan("ignore-shutdown", filepath.Join(t.TempDir(), "ignored.txt"))
	plan.GracefulStopTimeout = 60 * time.Millisecond
	receipt, err := s.Launch(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	stopped, err := s.Stop(context.Background(), receipt.Ownership)
	if err != nil {
		t.Fatal(err)
	}
	if !stopped.GracefulAttempted || !stopped.Escalated {
		t.Fatalf("stop receipt = %+v, want graceful attempt plus escalation", stopped)
	}
	if processalive.Check(receipt.Ownership.PID) {
		t.Fatalf("escalated child pid %d remains alive", receipt.Ownership.PID)
	}
}

func TestSupervisorRefusesStaleOwnershipWithoutTouchingSentinel(t *testing.T) {
	s := &Supervisor{}
	receipt, err := s.Launch(context.Background(), testPlan("normal", filepath.Join(t.TempDir(), "managed.txt")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = s.Stop(context.Background(), receipt.Ownership) })
	wrongToken := receipt.Ownership
	wrongToken.Token = strings.Repeat("0", len(wrongToken.Token))
	if _, err := s.Stop(context.Background(), wrongToken); refusalCode(err) != RefusalStaleOwnership {
		t.Fatalf("wrong-token stop error = %v, want %s", err, RefusalStaleOwnership)
	}

	sentinelMarker := filepath.Join(t.TempDir(), "sentinel.txt")
	sentinel := exec.Command(fakeRuntimePath, "--mode", "sentinel", "--marker", sentinelMarker)
	if err := sentinel.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = sentinel.Process.Kill()
		_ = sentinel.Wait()
	})
	_ = readFileEventually(t, sentinelMarker)

	stale := receipt.Ownership
	stale.PID = sentinel.Process.Pid
	stale.StartIdentity = "stale-process-start"
	_, err = s.Stop(context.Background(), stale)
	if refusalCode(err) != RefusalStaleOwnership {
		t.Fatalf("stale stop error = %v, want %s", err, RefusalStaleOwnership)
	}
	if !processalive.Check(sentinel.Process.Pid) {
		t.Fatal("stale ownership refusal touched unrelated sentinel")
	}
	response, err := (&http.Client{Timeout: time.Second}).Get(receipt.Endpoint + "/health")
	if err != nil {
		t.Fatalf("owned runtime was touched by stale stop: %v", err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("owned runtime health after stale stop = %d", response.StatusCode)
	}
}

func TestSupervisorRejectsRemoteOrImplicitLaunchShapes(t *testing.T) {
	cases := []struct {
		name string
		edit func(*Plan)
	}{
		{"relative executable", func(p *Plan) { p.Executable = filepath.Base(fakeRuntimePath) }},
		{"implicit address", func(p *Plan) { p.Args = []string{"--address", "127.0.0.1:8080"} }},
		{"remote completion URL", func(p *Plan) { p.CompletionPath = "https://example.com/v1/completions" }},
		{"control character argv", func(p *Plan) { p.Args = append(p.Args, "bad\narg") }},
		{"ownership token injection", func(p *Plan) { p.Env = []string{ownershipTokenEnv + "=forged"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := testPlan("normal", filepath.Join(t.TempDir(), "invalid.txt"))
			tc.edit(&plan)
			_, err := (&Supervisor{}).Launch(context.Background(), plan)
			if refusalCode(err) != RefusalInvalidPlan {
				t.Fatalf("Launch error = %v, want %s", err, RefusalInvalidPlan)
			}
		})
	}
}

func testPlan(mode, marker string) Plan {
	return Plan{
		Executable: fakeRuntimePath,
		Args: []string{
			"--address", addressPlaceholder,
			"--mode", mode,
			"--ready-delay", "30ms",
			"--marker", marker,
		},
		Model:                "fake-model",
		HealthPath:           "/health",
		CompletionPath:       "/v1/completions",
		GracefulShutdownPath: "/shutdown",
		StartupTimeout:       2 * time.Second,
		ProbeTimeout:         time.Second,
		PollInterval:         10 * time.Millisecond,
		GracefulStopTimeout:  300 * time.Millisecond,
		KillTimeout:          time.Second,
	}
}

func refusalCode(err error) string {
	var refusal *Refusal
	if errors.As(err, &refusal) {
		return refusal.Code
	}
	return ""
}

func readFileEventually(t *testing.T, path string) string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			return string(data)
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("file %s was not written", path)
	return ""
}
