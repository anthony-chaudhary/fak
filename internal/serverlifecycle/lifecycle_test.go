package serverlifecycle_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
)

func TestServerLifecyclePublicCLISpine(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and exercises the public fak binary")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	fakBin := buildGoBinary(t, root, "fak", "./cmd/fak")
	fixture := buildFixtureServer(t, root)
	modelBytes := []byte("fixture gguf bytes")
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, modelBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(modelBytes)
	events := filepath.Join(t.TempDir(), "events.log")
	dir := filepath.Join(t.TempDir(), "local-code")
	port := freePort(t)

	initResult, _, err := runCLI(fakBin, nil, "server", "init",
		"--dir", dir,
		"--name", "local-code",
		"--model", modelPath,
		"--sha256", hex.EncodeToString(digest[:]),
		"--executable", fixture,
		"--port", fmt.Sprint(port),
		"--json",
	)
	if err != nil || initResult.State != serverlifecycle.StateConfigured {
		t.Fatalf("server init: result=%+v err=%v", initResult, err)
	}
	if _, err := os.Stat(events); !os.IsNotExist(err) {
		t.Fatalf("init launched the fixture: stat error=%v", err)
	}
	assertStatus(t, fakBin, dir, serverlifecycle.StateConfigured)

	up := exec.Command(fakBin, "server", "up", "--dir", dir, "--readiness-timeout", "5s", "--probe-interval", "25ms", "--json")
	up.Env = append(os.Environ(), "FAK_SERVER_FIXTURE_EVENTS="+events, "FAK_SERVER_FIXTURE_READY_DELAY_MS=800")
	var upStdout, upStderr bytes.Buffer
	up.Stdout, up.Stderr = &upStdout, &upStderr
	if err := up.Start(); err != nil {
		t.Fatal(err)
	}
	var readyResult serverlifecycle.Result
	t.Cleanup(func() {
		if readyResult.ProcessID > 0 {
			if process, err := os.FindProcess(readyResult.ProcessID); err == nil {
				_ = process.Kill()
			}
		}
	})
	waitForEvent(t, events, "health:not-ready", 10*time.Second)
	if _, err := os.Stat(filepath.Join(dir, serverlifecycle.ReceiptFilename)); !os.IsNotExist(err) {
		t.Fatalf("ready receipt existed before readiness: %v", err)
	}
	assertStatus(t, fakBin, dir, serverlifecycle.StateStarting)

	contended, _, err := runCLI(fakBin, []string{"FAK_SERVER_FIXTURE_EVENTS=" + events}, "server", "up", "--dir", dir, "--readiness-timeout", "2s", "--json")
	if err == nil || !contended.Refused || contended.Reason != serverlifecycle.ReasonInstanceLocked {
		t.Fatalf("concurrent up = %+v err=%v, want typed refusal", contended, err)
	}
	if err := up.Wait(); err != nil {
		t.Fatalf("first up failed: %v\nstdout=%s\nstderr=%s", err, upStdout.String(), upStderr.String())
	}
	if err := json.Unmarshal(upStdout.Bytes(), &readyResult); err != nil {
		t.Fatalf("decode up result: %v\n%s", err, upStdout.String())
	}
	if readyResult.State != serverlifecycle.StateReady || !readyResult.Evidence.ProtocolReady {
		t.Fatalf("up result = %+v", readyResult)
	}
	eventText := readText(t, events)
	assertOrdered(t, eventText, "health:not-ready", "health:ready", "models", "chat")
	if _, err := os.Stat(filepath.Join(dir, serverlifecycle.ReceiptFilename)); err != nil {
		t.Fatalf("receipt missing after readiness: %v", err)
	}
	assertStatus(t, fakBin, dir, serverlifecycle.StateReady)

	statePath := filepath.Join(dir, serverlifecycle.StateFilename)
	receiptPath := filepath.Join(dir, serverlifecycle.ReceiptFilename)
	originalIdentity := mutateProcessIdentity(t, statePath, receiptPath, "-mismatched")
	refused, _, err := runCLI(fakBin, nil, "server", "down", "--dir", dir, "--stop-timeout", "2s", "--json")
	if err == nil || !refused.Refused || refused.Reason != serverlifecycle.ReasonProcessIdentityMismatch {
		t.Fatalf("mismatched down = %+v err=%v", refused, err)
	}
	response, err := http.Get(readyResult.BaseURL + "/health")
	if err != nil {
		t.Fatalf("identity refusal signaled the fixture: %v", err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("fixture status after refused down = %d", response.StatusCode)
	}
	assertStatus(t, fakBin, dir, serverlifecycle.StateStale)
	restoreProcessIdentity(t, statePath, receiptPath, originalIdentity)
	assertStatus(t, fakBin, dir, serverlifecycle.StateReady)

	down, _, err := runCLI(fakBin, nil, "server", "down", "--dir", dir, "--stop-timeout", "5s", "--json")
	if err != nil || down.State != serverlifecycle.StateStopped {
		t.Fatalf("down = %+v err=%v", down, err)
	}
	readyResult.ProcessID = 0
	downAgain, _, err := runCLI(fakBin, nil, "server", "down", "--dir", dir, "--json")
	if err != nil || downAgain.State != serverlifecycle.StateStopped {
		t.Fatalf("idempotent down = %+v err=%v", downAgain, err)
	}
	assertStatus(t, fakBin, dir, serverlifecycle.StateStopped)

	failedDir := filepath.Join(t.TempDir(), "failed")
	badDigest := strings.Repeat("0", 64)
	if _, _, err := runCLI(fakBin, nil, "server", "init", "--dir", failedDir, "--name", "failed", "--model", modelPath, "--sha256", badDigest, "--executable", fixture, "--port", fmt.Sprint(freePort(t)), "--json"); err != nil {
		t.Fatalf("init failed-state fixture: %v", err)
	}
	failed, _, err := runCLI(fakBin, nil, "server", "up", "--dir", failedDir, "--readiness-timeout", "1s", "--json")
	if err == nil || failed.State != serverlifecycle.StateFailed {
		t.Fatalf("failed up = %+v err=%v", failed, err)
	}
	assertStatus(t, fakBin, failedDir, serverlifecycle.StateFailed)
}

func buildFixtureServer(t *testing.T, root string) string {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "main.go")
	program := `package main
import (
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "strconv"
    "sync"
    "time"
)
var mu sync.Mutex
func event(value string) {
    path := os.Getenv("FAK_SERVER_FIXTURE_EVENTS")
    if path == "" { return }
    mu.Lock()
    defer mu.Unlock()
    f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
    if err != nil { panic(err) }
    defer f.Close()
    fmt.Fprintln(f, value)
}
func main() {
    if len(os.Args) == 2 && os.Args[1] == "--version" {
        fmt.Println("llama.cpp version: fixture-b8157")
        return
    }
    values := map[string]string{}
    for i := 1; i+1 < len(os.Args); i += 2 { values[os.Args[i]] = os.Args[i+1] }
    port := values["--port"]
    alias := values["--alias"]
    delayMS, _ := strconv.Atoi(os.Getenv("FAK_SERVER_FIXTURE_READY_DELAY_MS"))
    started := time.Now()
    http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
        if time.Since(started) < time.Duration(delayMS)*time.Millisecond {
            event("health:not-ready")
            w.WriteHeader(http.StatusServiceUnavailable)
            fmt.Fprint(w, ` + "`" + `{"status":"loading"}` + "`" + `)
            return
        }
        event("health:ready")
        json.NewEncoder(w).Encode(map[string]string{"status":"ok"})
    })
    http.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
        event("models")
        json.NewEncoder(w).Encode(map[string]any{"object":"list","data":[]map[string]string{{"id":alias}}})
    })
    http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
        event("chat")
        json.NewEncoder(w).Encode(map[string]any{"object":"chat.completion","model":alias,"choices":[]map[string]any{{"message":map[string]string{"role":"assistant","content":"OK"}}}})
    })
    event("listen")
    if err := http.ListenAndServe("127.0.0.1:"+port, nil); err != nil { panic(err) }
}
`
	if err := os.WriteFile(source, []byte(program), 0o600); err != nil {
		t.Fatal(err)
	}
	return buildGoBinary(t, root, "llama-server", source)
}

func buildGoBinary(t *testing.T, root, name string, target ...string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(t.TempDir(), name)
	args := append([]string{"build", "-o", path}, target...)
	cmd := exec.Command("go", args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", name, err, output)
	}
	return path
}

func runCLI(binary string, extraEnv []string, argv ...string) (serverlifecycle.Result, string, error) {
	cmd := exec.Command(binary, argv...)
	cmd.Env = append(os.Environ(), extraEnv...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	var result serverlifecycle.Result
	if decodeErr := json.Unmarshal(stdout.Bytes(), &result); decodeErr != nil {
		return result, stderr.String(), fmt.Errorf("decode CLI result: %w (run error: %v, stdout: %s, stderr: %s)", decodeErr, err, stdout.String(), stderr.String())
	}
	return result, stderr.String(), err
}

func assertStatus(t *testing.T, binary, dir string, want serverlifecycle.State) serverlifecycle.Result {
	t.Helper()
	result, stderr, err := runCLI(binary, nil, "server", "status", "--dir", dir, "--probe-timeout", "2s", "--json")
	if err != nil || result.State != want {
		t.Fatalf("status = %+v err=%v stderr=%s, want %s", result, err, stderr, want)
	}
	return result
}

func waitForEvent(t *testing.T, path, event string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if raw, err := os.ReadFile(path); err == nil && strings.Contains(string(raw), event) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("event %q not observed in %s", event, readText(t, path))
}

func assertOrdered(t *testing.T, text string, values ...string) {
	t.Helper()
	position := -1
	for _, value := range values {
		next := strings.Index(text[position+1:], value)
		if next < 0 {
			t.Fatalf("event %q missing from %q", value, text)
		}
		position += next + 1
	}
}

func readText(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ""
		}
		t.Fatal(err)
	}
	return string(raw)
}

func mutateProcessIdentity(t *testing.T, statePath, receiptPath, suffix string) string {
	t.Helper()
	state := readObject(t, statePath)
	original := state["process_start_identity"].(string)
	state["process_start_identity"] = original + suffix
	writeObject(t, statePath, state)
	receipt := readObject(t, receiptPath)
	receipt["ownership"].(map[string]any)["process_start_identity"] = original + suffix
	writeObject(t, receiptPath, receipt)
	return original
}

func restoreProcessIdentity(t *testing.T, statePath, receiptPath, identity string) {
	t.Helper()
	state := readObject(t, statePath)
	state["process_start_identity"] = identity
	writeObject(t, statePath, state)
	receipt := readObject(t, receiptPath)
	receipt["ownership"].(map[string]any)["process_start_identity"] = identity
	writeObject(t, receiptPath, receipt)
}

func readObject(t *testing.T, path string) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal([]byte(readText(t, path)), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func writeObject(t *testing.T, path string, value map[string]any) {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}
