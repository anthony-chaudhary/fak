package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const controlIngressLiveHelperEnv = "FAK_TEST_CONTROL_INGRESS_LIVE_HELPER"

const (
	controlIngressDeliveryModeEnv   = "FAK_TEST_CONTROL_DELIVERY_MODE"
	controlIngressDeliveryMarkerEnv = "FAK_TEST_CONTROL_DELIVERY_MARKER"
)

// TestControlIngressLiveServeHelper enters the same cmdServe function selected by
// `fak serve`. Keeping the server in a child process makes startup refusal and
// crash/restart observable without mutating cmdServe's os.Exit contract.
func TestControlIngressLiveServeHelper(t *testing.T) {
	if os.Getenv(controlIngressLiveHelperEnv) != "1" {
		return
	}
	sep := -1
	for i, arg := range os.Args {
		if arg == "--" {
			sep = i
			break
		}
	}
	if sep < 0 {
		t.Fatal("helper argv has no -- separator")
	}
	switch os.Getenv(controlIngressDeliveryModeEnv) {
	case "block":
		serveControlDirectiveDeliver = func(_ context.Context, directive gateway.ControlDirective) error {
			if err := os.WriteFile(os.Getenv(controlIngressDeliveryMarkerEnv), []byte(directive.ID), 0o600); err != nil {
				return err
			}
			select {}
		}
	case "replay":
		serveControlDirectiveDeliver = func(_ context.Context, directive gateway.ControlDirective) error {
			return os.WriteFile(os.Getenv(controlIngressDeliveryMarkerEnv), []byte(directive.ID), 0o600)
		}
	}
	cmdServe(os.Args[sep+1:])
}

type liveServeProcess struct {
	cmd    *exec.Cmd
	output bytes.Buffer
	base   string
}

func reserveLoopbackAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	return addr
}

func startLiveServe(t *testing.T, args []string, env ...string) *liveServeProcess {
	t.Helper()
	addr := reserveLoopbackAddr(t)
	argv := append([]string{"-test.run=^TestControlIngressLiveServeHelper$", "--", "--addr", addr}, args...)
	p := &liveServeProcess{base: "http://" + addr}
	p.cmd = exec.Command(os.Args[0], argv...)
	p.cmd.Env = append(os.Environ(), append([]string{controlIngressLiveHelperEnv + "=1"}, env...)...)
	p.cmd.Stdout, p.cmd.Stderr = &p.output, &p.output
	if err := p.cmd.Start(); err != nil {
		t.Fatalf("start fak serve helper: %v", err)
	}
	t.Cleanup(func() { p.crashAndWait() })
	return p
}

func (p *liveServeProcess) crashAndWait() {
	if p == nil || p.cmd == nil || p.cmd.Process == nil || p.cmd.ProcessState != nil {
		return
	}
	_ = p.cmd.Process.Kill()
	_ = p.cmd.Wait()
}

func waitLiveReady(t *testing.T, client *http.Client, p *liveServeProcess, key string) {
	t.Helper()
	base := p.base
	deadline := time.Now().Add(12 * time.Second)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequest(http.MethodGet, base+"/readyz", nil)
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := client.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	p.crashAndWait()
	t.Fatalf("fak serve did not become ready at %s\n%s", base, p.output.String())
}

func liveControlRequest(t *testing.T, client *http.Client, method, url, key string, body any) (int, gateway.ControlReceipt, []byte) {
	t.Helper()
	var r io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		r = bytes.NewReader(raw)
	}
	req, err := http.NewRequest(method, url, r)
	if err != nil {
		t.Fatal(err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var receipt gateway.ControlReceipt
	_ = json.Unmarshal(raw, &receipt)
	return resp.StatusCode, receipt, raw
}

func preprovisionLiveJournalOnWindows(t *testing.T, path string) {
	t.Helper()
	if runtime.GOOS != "windows" {
		return
	}
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("preprovision Windows control journal: %v", err)
	}
}

func waitForMarker(t *testing.T, path, want string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil && string(raw) == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	raw, err := os.ReadFile(path)
	t.Fatalf("delivery marker %q never became %q: got=%q err=%v", path, want, raw, err)
}

func TestControlIngressLiveServeReplaysAcceptedDirectiveAfterCrashBeforeDelivery(t *testing.T) {
	if testing.Short() {
		t.Skip("process-level fak serve witness")
	}
	dir := t.TempDir()
	journal := filepath.Join(dir, "control.jsonl")
	preprovisionLiveJournalOnWindows(t, journal)
	enteredMarker := filepath.Join(dir, "delivery-entered")
	replayMarker := filepath.Join(dir, "delivery-replayed")
	const key = "crash-replay-control-secret"
	const id = "cancel-crash-before-delivery"
	args := []string{"--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY"}
	baseEnv := []string{serveControlIngressJournalEnv + "=" + journal, "FAK_TEST_CONTROL_KEY=" + key}
	client := &http.Client{Timeout: 3 * time.Second}

	blockedEnv := append(append([]string{}, baseEnv...),
		controlIngressDeliveryModeEnv+"=block", controlIngressDeliveryMarkerEnv+"="+enteredMarker)
	p := startLiveServe(t, args, blockedEnv...)
	waitLiveReady(t, client, p, key)
	directive := gateway.ControlDirective{ID: id, Target: "crash-replay-target", Action: "cancel", Generation: 1}
	status, accepted, raw := liveControlRequest(t, client, http.MethodPost, p.base+serveControlDirectivePath, key, directive)
	if status != http.StatusAccepted || accepted.State != "accepted" || accepted.ID != id {
		t.Fatalf("POST before blocked delivery = status %d receipt %+v body=%s", status, accepted, raw)
	}
	waitForMarker(t, enteredMarker, id)
	status, beforeCrash, raw := liveControlRequest(t, client, http.MethodGet, p.base+serveControlDirectivePath+"/"+id, key, nil)
	if status != http.StatusOK || beforeCrash.State != "accepted" {
		t.Fatalf("receipt advanced while delivery callback was blocked: status %d receipt %+v body=%s", status, beforeCrash, raw)
	}
	p.crashAndWait()

	replayEnv := append(append([]string{}, baseEnv...),
		controlIngressDeliveryModeEnv+"=replay", controlIngressDeliveryMarkerEnv+"="+replayMarker)
	p2 := startLiveServe(t, args, replayEnv...)
	waitLiveReady(t, client, p2, key)
	waitForMarker(t, replayMarker, id)
	deadline := time.Now().Add(5 * time.Second)
	var recovered gateway.ControlReceipt
	for time.Now().Before(deadline) {
		status, recovered, raw = liveControlRequest(t, client, http.MethodGet, p2.base+serveControlDirectivePath+"/"+id, key, nil)
		if status == http.StatusOK && recovered.State == "delivered" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if status != http.StatusOK || recovered.State != "delivered" || recovered.ID != id || recovered.Digest != accepted.Digest || !recovered.AcceptedAt.Equal(accepted.AcceptedAt) {
		t.Fatalf("restart did not replay the accepted directive to delivered: status %d before=%+v after=%+v body=%s", status, accepted, recovered, raw)
	}
}

func TestControlIngressLiveServeCancelSurvivesCrashRestart(t *testing.T) {
	if testing.Short() {
		t.Skip("process-level fak serve witness")
	}
	entered := make(chan struct{}, 1)
	releaseUpstream := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		entered <- struct{}{}
		select {
		case <-r.Context().Done():
		case <-releaseUpstream:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"released"},"finish_reason":"stop"}]}`)
		}
	}))
	t.Cleanup(func() {
		close(releaseUpstream)
		upstream.CloseClientConnections()
		upstream.Close()
	})

	dir := t.TempDir()
	journal := filepath.Join(dir, "control.jsonl")
	preprovisionLiveJournalOnWindows(t, journal)
	const key = "live-control-secret"
	const trace = "live-control-held-native-turn"
	const id = "cancel-live-1"
	env := []string{
		serveControlIngressJournalEnv + "=" + journal,
		"FAK_TEST_CONTROL_KEY=" + key,
		"FAK_HTTP_WRITE_TIMEOUT_S=0",
	}
	args := []string{"--native", "--base-url", upstream.URL, "--model", "test-model", "--require-key-env", "FAK_TEST_CONTROL_KEY"}
	client := &http.Client{Timeout: 5 * time.Second}
	p := startLiveServe(t, args, env...)
	waitLiveReady(t, client, p, key)

	requestBody := `{"model":"test-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hold this native turn"}],"tools":[{"name":"search","description":"test tool","input_schema":{"type":"object","properties":{"query":{"type":"string"}}}}]}`
	req, _ := http.NewRequest(http.MethodPost, p.base+"/v1/messages", strings.NewReader(requestBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("X-Trace-Id", trace)
	type turnResult struct {
		status int
		body   []byte
		err    error
	}
	turnDone := make(chan turnResult, 1)
	go func() {
		resp, err := client.Do(req)
		if err != nil {
			turnDone <- turnResult{err: err}
			return
		}
		defer resp.Body.Close()
		raw, readErr := io.ReadAll(resp.Body)
		turnDone <- turnResult{status: resp.StatusCode, body: raw, err: readErr}
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("real native serve turn never reached the held upstream")
	}

	directive := gateway.ControlDirective{ID: id, Target: trace, Action: "cancel", Generation: 1}
	status, accepted, raw := liveControlRequest(t, client, http.MethodPost, p.base+serveControlDirectivePath, key, directive)
	if status != http.StatusAccepted || accepted.State != "accepted" || accepted.ID != id {
		t.Fatalf("POST durable cancel = status %d receipt %+v body=%s", status, accepted, raw)
	}

	itemURL := p.base + serveControlDirectivePath + "/" + id
	deadline := time.Now().Add(5 * time.Second)
	var readback gateway.ControlReceipt
	for time.Now().Before(deadline) {
		status, readback, raw = liveControlRequest(t, client, http.MethodGet, itemURL, key, nil)
		if status == http.StatusOK && readback.State == "delivered" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if status != http.StatusOK || readback.State != "delivered" {
		t.Fatalf("GET durable cancel never observed delivery: status %d receipt %+v body=%s", status, readback, raw)
	}
	select {
	case result := <-turnDone:
		if result.err != nil {
			t.Fatalf("held native turn after cancel: %v", result.err)
		}
		if bytes.Contains(result.body, []byte("event: tool_started")) {
			t.Fatalf("native turn emitted tool_started after terminating receipt: %s", result.body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("held native turn did not observe the terminating signal")
	}

	// Kill rather than gracefully stop: the restart must recover solely from the
	// provisioned, synced journal, as a production crash would.
	p.crashAndWait()
	p2 := startLiveServe(t, args, env...)
	waitLiveReady(t, client, p2, key)
	status, recovered, raw := liveControlRequest(t, client, http.MethodGet, p2.base+serveControlDirectivePath+"/"+id, key, nil)
	if status != http.StatusOK || recovered.ID != id || recovered.Digest != accepted.Digest || recovered.Sequence < accepted.Sequence || !recovered.AcceptedAt.Equal(accepted.AcceptedAt) {
		t.Fatalf("restart did not recover the same durable receipt: status %d before=%+v after=%+v body=%s", status, accepted, recovered, raw)
	}
	status, replayed, raw := liveControlRequest(t, client, http.MethodPost, p2.base+serveControlDirectivePath, key, directive)
	if status != http.StatusAccepted || replayed.Digest != recovered.Digest || replayed.Sequence != recovered.Sequence {
		t.Fatalf("same-ID replay after restart = status %d receipt %+v body=%s", status, replayed, raw)
	}
}

func TestControlIngressLiveServeUnavailableCases(t *testing.T) {
	if testing.Short() {
		t.Skip("process-level fak serve witness")
	}
	client := &http.Client{Timeout: 3 * time.Second}
	t.Run("authenticated serve without explicit control opt-in", func(t *testing.T) {
		const key = "no-control-opt-in-secret"
		p := startLiveServe(t, []string{"--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY"},
			"FAK_TEST_CONTROL_KEY="+key, serveControlIngressJournalEnv+"=")
		waitLiveReady(t, client, p, key)
		status, receipt, raw := liveControlRequest(t, client, http.MethodPost, p.base+serveControlDirectivePath, key,
			gateway.ControlDirective{ID: "not-opted-in", Target: "turn", Action: "cancel", Generation: 1})
		if status != http.StatusServiceUnavailable || receipt.State != "unavailable" {
			t.Fatalf("control route without explicit opt-in = status %d receipt %+v body=%s", status, receipt, raw)
		}
	})

	t.Run("known unsupported action", func(t *testing.T) {
		journal := filepath.Join(t.TempDir(), "control.jsonl")
		preprovisionLiveJournalOnWindows(t, journal)
		const key = "unsupported-control-secret"
		p := startLiveServe(t,
			[]string{"--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY"},
			serveControlIngressJournalEnv+"="+journal, "FAK_TEST_CONTROL_KEY="+key)
		waitLiveReady(t, client, p, key)
		status, receipt, raw := liveControlRequest(t, client, http.MethodPost, p.base+serveControlDirectivePath, key,
			gateway.ControlDirective{ID: "pause-unsupported", Target: "turn", Action: "pause", Generation: 1})
		if status != http.StatusServiceUnavailable || receipt.State != "unavailable" || receipt.Reason != "unsupported_action" {
			t.Fatalf("unsupported action = status %d receipt %+v body=%s", status, receipt, raw)
		}
	})

	t.Run("no credential door", func(t *testing.T) {
		journal := filepath.Join(t.TempDir(), "must-not-open.jsonl")
		p := startLiveServe(t, []string{"--mock"}, serveControlIngressJournalEnv+"="+journal)
		waitLiveReady(t, client, p, "")
		status, receipt, raw := liveControlRequest(t, client, http.MethodPost, p.base+serveControlDirectivePath, "",
			gateway.ControlDirective{ID: "no-door", Target: "turn", Action: "cancel", Generation: 1})
		if status != http.StatusServiceUnavailable || receipt.State != "unavailable" {
			t.Fatalf("control route without credential door = status %d receipt %+v body=%s", status, receipt, raw)
		}
		if _, err := os.Stat(journal); !os.IsNotExist(err) {
			t.Fatalf("no-door serve opened a control journal: err=%v", err)
		}
	})
}

func TestControlIngressLiveServeProvisionFailureRefusesStartup(t *testing.T) {
	if testing.Short() {
		t.Skip("process-level fak serve witness")
	}
	if runtime.GOOS == "windows" {
		t.Run("explicit missing journal fails closed", func(t *testing.T) {
			missing := filepath.Join(t.TempDir(), "missing-control.jsonl")
			cmd := exec.Command(os.Args[0], "-test.run=^TestControlIngressLiveServeHelper$", "--",
				"--addr", reserveLoopbackAddr(t), "--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY")
			cmd.Env = append(os.Environ(), controlIngressLiveHelperEnv+"=1", "FAK_TEST_CONTROL_KEY=bootstrap-secret",
				serveControlIngressJournalEnv+"="+missing)
			raw, err := cmd.CombinedOutput()
			if err == nil || !bytes.Contains(raw, []byte(windowsJournalProvisionReason)) {
				t.Fatalf("Windows missing explicit journal did not fail closed with %s: err=%v output=%s", windowsJournalProvisionReason, err, raw)
			}
		})
	}
	t.Run("configured nonregular journal", func(t *testing.T) {
		nonregular := filepath.Join(t.TempDir(), "journal-directory")
		if err := os.Mkdir(nonregular, 0o700); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(os.Args[0], "-test.run=^TestControlIngressLiveServeHelper$", "--",
			"--addr", reserveLoopbackAddr(t), "--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY")
		cmd.Env = append(os.Environ(), controlIngressLiveHelperEnv+"=1", "FAK_TEST_CONTROL_KEY=bootstrap-secret",
			serveControlIngressJournalEnv+"="+nonregular)
		raw, err := cmd.CombinedOutput()
		if err == nil || !bytes.Contains(raw, []byte("regular non-symlink file")) {
			t.Fatalf("configured directory journal did not fail closed as nonregular: err=%v output=%s", err, raw)
		}
	})
	root := t.TempDir()
	blocker := filepath.Join(root, "not-a-directory")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	addr := reserveLoopbackAddr(t)
	cmd := exec.Command(os.Args[0], "-test.run=^TestControlIngressLiveServeHelper$", "--",
		"--addr", addr, "--mock", "--require-key-env", "FAK_TEST_CONTROL_KEY")
	cmd.Env = append(os.Environ(), controlIngressLiveHelperEnv+"=1", "FAK_TEST_CONTROL_KEY=bootstrap-secret",
		serveControlIngressJournalEnv+"="+filepath.Join(blocker, "control.jsonl"))
	raw, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("fak serve started despite an unprovisionable control journal: %s", raw)
	}
	if !bytes.Contains(raw, []byte("control ingress")) {
		t.Fatalf("startup refusal did not identify control ingress provisioning: %s", raw)
	}
	conn, dialErr := net.DialTimeout("tcp", addr, 150*time.Millisecond)
	if dialErr == nil {
		_ = conn.Close()
		t.Fatal("listener bound despite control journal provisioning failure")
	}
}

func TestProvisionServeControlJournalRefusesSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		// Windows symlink creation requires Developer Mode or an elevated token. Run
		// the assertion when the host supports it; the startup process witness above
		// remains mandatory on every platform.
		t.Log("Windows symlink assertion runs when the current token permits symlink creation")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "unrelated")
	if err := os.WriteFile(target, []byte("do-not-touch"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "control.jsonl")
	if err := os.Symlink(target, link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink privilege unavailable: %v", err)
		}
		t.Fatal(err)
	}
	if err := provisionServeControlJournal(link); err == nil {
		t.Fatal("provisioning followed a symlink journal path; startup must refuse path substitution")
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "do-not-touch" {
		t.Fatalf("symlink target changed during refused provisioning: %q", got)
	}
}
