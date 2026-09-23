package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// isolatePiHome points PI_CODING_AGENT_DIR at a throwaway directory for the duration of
// the test. runPi defaults `--settings-path`/`--config-path` to the REAL user Pi config
// (~/.pi/agent/{settings,models}.json), so a test that omits those flags rewrites the
// operator's live harness default — the exact drift class that let a bare `pi` launch
// stop resolving to the live fak router. Every test that calls runPi must call this
// first. TestMain in guard_login_e2e_test.go also sets it package-wide as a backstop.
func isolatePiHome(t *testing.T) {
	t.Helper()
	t.Setenv("PI_CODING_AGENT_DIR", t.TempDir())
}

func TestBuildPiLaunchArgv(t *testing.T) {
	isolatePiHome(t)
	opts := piLaunchOptions{
		command:     "pi",
		provider:    "fak",
		model:       "qwen38:27b-q4",
		probePrompt: "Explain Metal GPU acceleration",
		thinking:    "high",
		tools:       "read,bash,edit,write",
		passthrough: []string{"--no-session"},
	}

	argv := buildPiLaunchArgv(opts)
	want := []string{
		"pi",
		"--provider", "fak",
		"--model", "qwen38:27b-q4",
		"-p", "Explain Metal GPU acceleration",
		"--thinking", "high",
		"--tools", "read,bash,edit,write",
		"--no-session",
	}

	if len(argv) != len(want) {
		t.Fatalf("buildPiLaunchArgv length = %d, want %d: %v", len(argv), len(want), argv)
	}
	for i := range argv {
		if argv[i] != want[i] {
			t.Errorf("argv[%d] = %q, want %q", i, argv[i], want[i])
		}
	}
}

func TestRunPiDryRun(t *testing.T) {
	isolatePiHome(t)
	var stdout, stderr bytes.Buffer
	// Pin a dead addr so the dry-run never adopts a dev-box `fak serve` on the
	// default port (a live gateway on ::1/127.0.0.1 would resolve the base URL
	// and flip this test's expectation).
	code := runPi(&stdout, &stderr, []string{"--dry-run", "--addr", "127.0.0.1:65531", "--model", "qwen38:27b-q4", "--check-backend=false"})
	if code != 0 {
		t.Fatalf("runPi --dry-run returned %d, want 0; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	errOut := stderr.String()

	if !strings.Contains(out, "pi --provider fak --model qwen38:27b-q4") {
		t.Errorf("expected stdout to contain launch command, got: %s", out)
	}
	if !strings.Contains(errOut, "backend     = http://127.0.0.1:65531/v1 (raw without guard)") {
		t.Errorf("expected stderr to contain raw backend note, got: %s", errOut)
	}
	if !strings.Contains(errOut, "provider    = fak") {
		t.Errorf("expected stderr to contain provider fak, got: %s", errOut)
	}
}

func TestRunPiDryRunDoesNotMutateExistingConfig(t *testing.T) {
	isolatePiHome(t)
	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")

	modelsBefore := []byte("{\n  \"providers\": [],\n  \"sentinel\": \"models-original\"\n}\n")
	settingsBefore := []byte("{\n  \"defaultProvider\": \"operator\",\n  \"defaultModel\": \"operator-model\",\n  \"compaction\": {\"reserveTokens\": 777, \"keepRecentTokens\": 555},\n  \"sentinel\": \"settings-original\"\n}\n")
	if err := os.WriteFile(modelsPath, modelsBefore, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, settingsBefore, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--dry-run",
		"--check-backend=false",
		"--addr", "127.0.0.1:65531",
		"--model", "qwen38:27b-q4",
		"--config-path", modelsPath,
		"--settings-path", settingsPath,
	})
	if code != 0 {
		t.Fatalf("runPi --dry-run returned %d, want 0; stderr: %s", code, stderr.String())
	}

	modelsAfter, err := os.ReadFile(modelsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(modelsAfter, modelsBefore) {
		t.Errorf("ordinary fak pi dry-run mutated models.json\nbefore: %s\nafter:  %s", modelsBefore, modelsAfter)
	}
	settingsAfter, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(settingsAfter, settingsBefore) {
		t.Errorf("ordinary fak pi dry-run mutated settings.json\nbefore: %s\nafter:  %s", settingsBefore, settingsAfter)
	}
}

func TestRunPiFreshConfigUsesSessionExtensionWithoutPersistentWrites(t *testing.T) {
	isolatePiHome(t)
	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")

	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()
	var extensionPath string
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		for i := 1; i+1 < len(argv); i++ {
			if argv[i] == "-e" {
				extensionPath = argv[i+1]
				break
			}
		}
		if extensionPath == "" {
			t.Errorf("Pi launch argv omitted session extension: %v", argv)
			return 0
		}
		raw, err := os.ReadFile(extensionPath)
		if err != nil {
			t.Errorf("read live Pi session extension: %v", err)
			return 0
		}
		if !strings.Contains(string(raw), `registerProvider("fak"`) {
			t.Errorf("session extension does not register provider fak:\n%s", raw)
		}
		budget := projectassets.PiModelContextBudget("qwen38:27b-q4", 80000)
		if want := fmt.Sprintf("contextWindow: %d", budget.ResidentTarget); !strings.Contains(string(raw), want) {
			t.Errorf("session extension omitted explicit --window budget %q:\n%s", want, raw)
		}
		return 0
	}

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--check-backend=false",
		"--addr", "127.0.0.1:65531",
		"--model", "qwen38:27b-q4",
		"--window", "80000",
		"--config-path", modelsPath,
		"--settings-path", settingsPath,
	})
	if code != 0 {
		t.Fatalf("runPi returned %d, want 0; stderr: %s", code, stderr.String())
	}
	if extensionPath == "" {
		t.Fatal("Pi session extension was not observed")
	}
	if _, err := os.Stat(extensionPath); !os.IsNotExist(err) {
		t.Errorf("Pi session extension was not cleaned up: %v", err)
	}
	for _, path := range []string{modelsPath, settingsPath} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("ordinary Pi launch created persistent config %s: %v", path, err)
		}
	}
}

// TestRunPiWritesOnlyUnderIsolatedHome is the hermeticity regression: a launcher turn that
// explicitly requests safe settings but omits --settings-path/--config-path must confine its writes to PI_CODING_AGENT_DIR and
// never touch the real ~/.pi/agent. This is the drift class that silently rewrote the
// operator's harness defaultModel (and pointed models.json at a dead test port) on every
// `go test ./cmd/fak/` run, so a bare `pi` launch regressed away from the live fak router.
func TestRunPiWritesOnlyUnderIsolatedHome(t *testing.T) {
	isolated := t.TempDir()
	t.Setenv("PI_CODING_AGENT_DIR", isolated)

	realHome, err := os.UserHomeDir()
	if err != nil || realHome == "" {
		t.Skip("no resolvable user home to guard")
	}
	realSettings := filepath.Join(realHome, ".pi", "agent", "settings.json")
	realModels := filepath.Join(realHome, ".pi", "agent", "models.json")
	beforeSettings, settingsErr := os.ReadFile(realSettings)
	beforeModels, modelsErr := os.ReadFile(realModels)

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--dry-run",
		"--addr", "127.0.0.1:65531",
		"--model", "custom-model",
		"--check-backend=false",
		"--safe-settings=true",
		"--quiet",
	})
	if code != 0 {
		t.Fatalf("runPi returned %d, want 0; stderr: %s", code, stderr.String())
	}

	if _, err := os.Stat(filepath.Join(isolated, "settings.json")); err != nil {
		t.Errorf("expected settings.json under PI_CODING_AGENT_DIR: %v", err)
	}
	if settingsErr == nil {
		afterSettings, err := os.ReadFile(realSettings)
		if err != nil {
			t.Fatalf("real settings.json vanished: %v", err)
		}
		if string(beforeSettings) != string(afterSettings) {
			t.Errorf("runPi rewrote the REAL ~/.pi/agent/settings.json despite PI_CODING_AGENT_DIR isolation")
		}
	}
	if modelsErr == nil {
		afterModels, err := os.ReadFile(realModels)
		if err != nil {
			t.Fatalf("real models.json vanished: %v", err)
		}
		if string(beforeModels) != string(afterModels) {
			t.Errorf("runPi rewrote the REAL ~/.pi/agent/models.json despite PI_CODING_AGENT_DIR isolation")
		}
	}
}

func TestRunPiConfigSubcommand(t *testing.T) {
	isolatePiHome(t)
	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{"config", "--addr", "127.0.0.1:8080", "--model", "qwen38:27b-q4"})
	if code != 0 {
		t.Fatalf("runPi config returned %d, want 0; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "http://127.0.0.1:8080/v1") {
		t.Errorf("expected baseURL in output: %s", out)
	}
	if !strings.Contains(out, "qwen38:27b-q4") {
		t.Errorf("expected model in output: %s", out)
	}
	if !strings.Contains(out, `"openai-completions"`) {
		t.Errorf("expected openai-completions in output: %s", out)
	}
}

func TestRunPiConfigSubcommandWrite(t *testing.T) {
	isolatePiHome(t)
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models.json")

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"config",
		"--write",
		"--path", targetPath,
		"--addr", "127.0.0.1:9000",
		"--model", "qwen38:27b-q4",
	})
	if code != 0 {
		t.Fatalf("runPi config --write returned %d, want 0; stderr: %s", code, stderr.String())
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read written file %s: %v", targetPath, err)
	}

	content := string(data)
	if !strings.Contains(content, "http://127.0.0.1:9000/v1") {
		t.Errorf("expected baseURL in written file: %s", content)
	}
	if !strings.Contains(content, "qwen38:27b-q4") {
		t.Errorf("expected model in written file: %s", content)
	}
}

func TestProbePiBackend(t *testing.T) {
	// 1. Healthz endpoint returns model
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"qwen38:27b-q4","engine":"inkernel"}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	model, ok, resolved := probePiBackend(ts.URL+"/v1", time.Second)
	if !ok || model != "qwen38:27b-q4" {
		t.Errorf("probePiBackend healthz = (%q, %v), want (\"qwen38:27b-q4\", true)", model, ok)
	}
	// The origin that answered is reported unchanged when the literal works.
	if resolved != ts.URL+"/v1" {
		t.Errorf("probePiBackend resolved = %q, want the supplied base URL %q", resolved, ts.URL+"/v1")
	}

	// 2. Unreachable endpoint returns false
	_, unreachOk, unreachResolved := probePiBackend("http://127.0.0.1:65530/v1", 100*time.Millisecond)
	if unreachOk {
		t.Errorf("probePiBackend unreachable want false, got true")
	}
	if unreachResolved != "http://127.0.0.1:65530/v1" {
		t.Errorf("probePiBackend unreachable resolved = %q, want the supplied base URL", unreachResolved)
	}
}

// TestProbePiBackendLoopbackFallback proves the family fallback resolves the
// origin that answered: an IPv6-only loopback listener is reachable via ::1 but
// refused on the 127.0.0.1 literal, so the probe must report the "localhost"
// alias as the resolved base URL. Skipped where the loopbacks dual-stack.
func TestProbePiBackendLoopbackFallback(t *testing.T) {
	ln, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Skipf("no IPv6 loopback on this host: %v", err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"qwen38:27b-q4"}`))
			return
		}
		http.NotFound(w, r)
	})}
	go srv.Serve(ln)
	defer srv.Close()
	_, port, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatalf("split listener addr: %v", err)
	}
	literal := "http://127.0.0.1:" + port + "/v1"
	// If the IPv4 literal already reaches the IPv6 listener the split is absent.
	// Probe the SINGLE origin (not the fallback wrapper) so this guard cannot
	// succeed via the very fallback it is testing for.
	if _, ok := probePiBackendOnce(&http.Client{Timeout: 500 * time.Millisecond}, literal); ok {
		t.Skip("loopback is dual-stacked on this host; fallback unobservable")
	}
	model, ok, resolved := probePiBackend(literal, 2*time.Second)
	if !ok || model != "qwen38:27b-q4" {
		t.Fatalf("probePiBackend fallback = (%q, %v), want the served model via localhost", model, ok)
	}
	if resolved != "http://localhost:"+port+"/v1" {
		t.Fatalf("probePiBackend resolved = %q, want the localhost alias", resolved)
	}
}

// TestRunPiAutoDetectPreservesPinnedDefault is the launcher-level regression: a backend
// whose /healthz reports a routing-mode local engine label (a nemotron-class id) must NOT
// overwrite a deliberately pinned defaultModel, while the provider is still repointed to
// fak so a bare launch reaches the router. On a routing-mode router /healthz names the
// local planner engine, not the routed model set; adopting it makes the routing ladder
// unreachable and silently changes which model answers.
func TestRunPiAutoDetectPreservesPinnedDefault(t *testing.T) {
	isolatePiHome(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"nemotron-3-super-120b-a12b","engine":"inkernel"}`))
		case "/models", "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"data":[{"id":"nemotron-3-super-120b-a12b","context_length":131072}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")
	// The operator's deliberate default survives the launcher auto-detect.
	seed := `{"defaultProvider":"fak","defaultModel":"deepseek-ai/DeepSeek-V4.1-Flash"}`
	if err := os.WriteFile(settingsPath, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int { return 0 }

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--config-path", modelsPath,
		"--settings-path", settingsPath,
		"--base-url", ts.URL + "/v1",
		"--quiet",
	})
	if code != 0 {
		t.Fatalf("runPi returned %d, want 0; stderr: %s", code, stderr.String())
	}

	var settings map[string]interface{}
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse settings.json: %v", err)
	}
	if settings["defaultModel"] != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("defaultModel = %v, want the pinned default preserved (auto-detect must not clobber)", settings["defaultModel"])
	}
	if settings["defaultProvider"] != "fak" {
		t.Errorf("defaultProvider = %v, want fak", settings["defaultProvider"])
	}
}

func TestRunPiMockExecution(t *testing.T) {
	isolatePiHome(t)
	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()

	var capturedArgv []string
	var capturedEnv []string
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		capturedArgv = argv
		capturedEnv = env
		return 0
	}

	tmp := t.TempDir()
	configPath := filepath.Join(tmp, "models.json")

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--check-backend=false",
		"--config-path", configPath,
		"--write-config=true",
		"--model", "qwen38:27b-q4",
		"--probe", "hello from pi",
	})
	if code != 0 {
		t.Fatalf("runPi mock execution returned %d, want 0; stderr: %s", code, stderr.String())
	}

	// Check argv
	if len(capturedArgv) == 0 || capturedArgv[0] != "pi" ||
		!containsArgPair(capturedArgv, "--provider", "fak") ||
		!containsArgPair(capturedArgv, "--model", "qwen38:27b-q4") ||
		!containsArgPair(capturedArgv, "-p", "hello from pi") {
		t.Errorf("unexpected captured argv: %v", capturedArgv)
	}

	// Check config file was created
	if _, err := os.Stat(configPath); err != nil {
		t.Errorf("expected config file %s to be created: %v", configPath, err)
	}

	// Check env contains PI_CODING_AGENT_DIR
	foundEnv := false
	for _, e := range capturedEnv {
		if strings.HasPrefix(e, "PI_CODING_AGENT_DIR=") {
			foundEnv = true
			break
		}
	}
	if !foundEnv {
		t.Errorf("expected PI_CODING_AGENT_DIR in env: %v", capturedEnv)
	}
}

// TestProbePiBackendWithWindow witnesses that the backend probe reports the served model's
// advertised context_length, which is what the safe Pi budget is derived from.
func TestProbePiBackendWithWindow(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models", "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"data":[{"id":"qwen38:27b-q4","context_length":131072}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	model, ok, window, _ := probePiBackendWithWindow(ts.URL+"/v1", time.Second)
	if !ok || model != "qwen38:27b-q4" {
		t.Fatalf("probe window = (%q, %v), want (qwen38:27b-q4, true)", model, ok)
	}
	if window != 131072 {
		t.Fatalf("window = %d, want 131072", window)
	}
}

// TestRunPiReplacesNonAdvertisedDefault is the launcher-level regression for the
// self-perpetuating placeholder default: a settings.json defaultModel the backend does
// NOT advertise (`custom-model`) must be corrected to an id the catalog actually lists,
// instead of being honored verbatim and re-written on every launch.
func TestRunPiReplacesNonAdvertisedDefault(t *testing.T) {
	isolatePiHome(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"deepseek-ai/DeepSeek-V4.1-Flash","engine":"router"}`))
		case "/models", "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash","context_length":1000000},{"id":"Qwen3.8-27B-UD-Q2_K_XL","context_length":131072}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")
	// The stale placeholder that must NOT be re-confirmed.
	seed := `{"defaultProvider":"fak","defaultModel":"custom-model"}`
	if err := os.WriteFile(settingsPath, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()
	var capturedArgv []string
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		capturedArgv = argv
		return 0
	}

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--config-path", modelsPath,
		"--settings-path", settingsPath,
		"--base-url", ts.URL + "/v1",
		"--write-config=true",
		"--quiet",
	})
	if code != 0 {
		t.Fatalf("runPi returned %d, want 0; stderr: %s", code, stderr.String())
	}

	var settings map[string]interface{}
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse settings.json: %v", err)
	}
	if settings["defaultModel"] == "custom-model" {
		t.Errorf("defaultModel still %q; a non-advertised placeholder must be replaced", settings["defaultModel"])
	}
	if settings["defaultModel"] != "deepseek-ai/DeepSeek-V4.1-Flash" {
		t.Errorf("defaultModel = %v, want the backend-advertised id", settings["defaultModel"])
	}
	// The rebound model must also reach the launch argv (not just the file).
	if !containsArgPair(capturedArgv, "--model", "deepseek-ai/DeepSeek-V4.1-Flash") {
		t.Errorf("launch argv did not rebind the model: %v", capturedArgv)
	}
}

// TestRunPiPreservesAdvertisedDefault: the counterpart guard — a default the backend DOES
// advertise is a real operator pin and must survive untouched.
func TestRunPiPreservesAdvertisedDefault(t *testing.T) {
	isolatePiHome(t)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"deepseek-ai/DeepSeek-V4.1-Flash","engine":"router"}`))
		case "/models", "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash","context_length":1000000},{"id":"Qwen3.8-27B-UD-Q2_K_XL","context_length":131072}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")
	seed := `{"defaultProvider":"fak","defaultModel":"Qwen3.8-27B-UD-Q2_K_XL"}`
	if err := os.WriteFile(settingsPath, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int { return 0 }

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--config-path", modelsPath,
		"--settings-path", settingsPath,
		"--base-url", ts.URL + "/v1",
		"--quiet",
	})
	if code != 0 {
		t.Fatalf("runPi returned %d, want 0; stderr: %s", code, stderr.String())
	}

	var settings map[string]interface{}
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse settings.json: %v", err)
	}
	if settings["defaultModel"] != "Qwen3.8-27B-UD-Q2_K_XL" {
		t.Errorf("defaultModel = %v, want the advertised default preserved", settings["defaultModel"])
	}
}

// TestRunPiConfigWriteSafeContext is the end-to-end witness for the goal: `fak pi config --write`
// must write a SAFE resident context target (at most half the served window) and a safe Pi
// compaction block, never the raw hard cap.
func TestRunPiConfigWriteSafeContext(t *testing.T) {
	isolatePiHome(t)
	tmp := t.TempDir()
	modelsPath := filepath.Join(tmp, "models.json")
	settingsPath := filepath.Join(tmp, "settings.json")

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"config",
		"--write",
		"--path", modelsPath,
		"--settings-path", settingsPath,
		"--addr", "127.0.0.1:9000",
		"--model", "qwen38:27b-q4",
		"--window", "131072",
	})
	if code != 0 {
		t.Fatalf("runPi config --write returned %d, want 0; stderr: %s", code, stderr.String())
	}

	var models map[string]interface{}
	modelsData, err := os.ReadFile(modelsPath)
	if err != nil {
		t.Fatalf("read models.json: %v", err)
	}
	if err := json.Unmarshal(modelsData, &models); err != nil {
		t.Fatalf("parse models.json: %v", err)
	}
	model := models["providers"].(map[string]interface{})["fak"].(map[string]interface{})["models"].([]interface{})[0].(map[string]interface{})
	cw, _ := model["contextWindow"].(float64)
	if cw != 65536 {
		t.Fatalf("contextWindow = %v, want 65536 (safe 50%% of 131072)", cw)
	}
	if cw == 131072 {
		t.Fatal("models.json advertises the raw served window (cap-is-not-target violation)")
	}

	var settings map[string]interface{}
	settingsData, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if err := json.Unmarshal(settingsData, &settings); err != nil {
		t.Fatalf("parse settings.json: %v", err)
	}
	block, ok := settings["compaction"].(map[string]interface{})
	if !ok {
		t.Fatalf("settings.json missing compaction block: %v", settings)
	}
	if enabled, _ := block["enabled"].(bool); !enabled {
		t.Fatal("compaction.enabled should be true")
	}
	if _, hasReserve := block["reserveTokens"]; !hasReserve {
		t.Fatal("compaction.reserveTokens missing")
	}
	if _, hasKeep := block["keepRecentTokens"]; !hasKeep {
		t.Fatal("compaction.keepRecentTokens missing")
	}
}

// TestPiLaunchSkillWiring is the witness for fak#13462: the launcher must
// explicitly pass the discovered project skill pack to Pi via repeatable
// --skill flags, so the pack survives launches whose working directory is not
// the project root (Pi only auto-discovers .agents/skills from cwd).
func TestPiLaunchSkillWiring(t *testing.T) {
	root := t.TempDir()
	agentsSkills := filepath.Join(root, ".agents", "skills")
	if err := os.MkdirAll(filepath.Join(agentsSkills, "goal"), 0o755); err != nil {
		t.Fatalf("mkdir .agents/skills: %v", err)
	}
	claudeSkills := filepath.Join(root, ".claude", "skills")
	if err := os.MkdirAll(filepath.Join(claudeSkills, "fak-flow"), 0o755); err != nil {
		t.Fatalf("mkdir .claude/skills: %v", err)
	}

	// Discovery walks up from a nested working directory, so launching from a
	// subdirectory of the project still resolves the pack.
	nested := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("mkdir nested: %v", err)
	}
	got := discoverPiSkillPack(nested)
	wantRoots := []string{agentsSkills, claudeSkills}
	for _, want := range wantRoots {
		if !strings.Contains(got, want) {
			t.Errorf("discoverPiSkillPack(%q) = %q, want it to contain %q", nested, got, want)
		}
	}

	argv := buildPiLaunchArgv(piLaunchOptions{command: "pi", provider: "fak", model: "m", skillPack: got})
	want := []string{
		"pi", "--provider", "fak", "--model", "m",
		"--skill", agentsSkills,
		"--skill", claudeSkills,
	}
	if len(argv) != len(want) {
		t.Fatalf("argv length = %d, want %d: %v", len(argv), len(want), argv)
	}
	for i := range argv {
		if argv[i] != want[i] {
			t.Errorf("argv[%d] = %q, want %q", i, argv[i], want[i])
		}
	}

	// No pack discovered -> no --skill flags (never emit an empty path).
	if argv := buildPiLaunchArgv(piLaunchOptions{command: "pi"}); len(argv) != 1 {
		t.Errorf("argv without a skill pack = %v, want just the command", argv)
	}

	// An explicit override wins over filesystem discovery.
	t.Setenv("FAK_PI_SKILLS", filepath.Join(root, "custom"))
	if got := discoverPiSkillPack(nested); got != filepath.Join(root, "custom") {
		t.Errorf("FAK_PI_SKILLS override = %q, want the custom root", got)
	}
}

// TestPiHarnessRegisteredInManifest is the witness for the second half of
// fak#13462: the project-assets sync must reconcile a "pi" harness entry the
// same way it does for codex/opencode, and its absence must be an unexplained gap.
func TestPiHarnessRegisteredInManifest(t *testing.T) {
	root := t.TempDir()
	manifest := `{
  "schema": "fak-project-assets/1",
  "skills": {"canonical_root": ".claude/skills", "codex_root": ".agents/skills", "include": ["SKILL.md"], "exclude": []},
  "memories": {"canonical_root": ".claude/memory", "include": ["*.md"], "exclude": []},
  "goal_prompts": {"canonical_root": ".claude/goal-prompts", "include": ["*.md"], "exclude": []},
  "harnesses": {}
}`
	if err := os.MkdirAll(filepath.Join(root, ".claude"), 0o755); err != nil {
		t.Fatalf("mkdir .claude: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, ".claude", "project-assets.json"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	receipt, err := projectassets.Build(root, false)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if _, ok := receipt.Harnesses["pi"]; !ok {
		t.Fatalf("receipt has no pi harness; harnesses = %v", keys(receipt.Harnesses))
	}
	// The pi receipt is derived, not manifest-declared, so it must reconcile cleanly
	// (no stale adapters) exactly like the codex/opencode receipts it mirrors.
	if got := receipt.Harnesses["pi"]; len(got.Stale) != 0 {
		t.Errorf("pi harness has stale entries %v; it must track the generated .agents/skills adapters", got.Stale)
	}
	pi, opencode := receipt.Harnesses["pi"], receipt.Harnesses["opencode"]
	if len(pi.Imported) != len(opencode.Imported) {
		t.Errorf("pi imported %d entries, opencode imported %d; pi must mirror the same adapter set", len(pi.Imported), len(opencode.Imported))
	}
}

func keys(m map[string]projectassets.HarnessReceipt) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
