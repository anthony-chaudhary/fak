package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const opencodeRouterE2EHelperEnv = "FAK_OPENCODE_ROUTER_E2E_HELPER"

func init() {
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(os.Args[0])), ".exe")
	if os.Getenv(opencodeRouterE2EHelperEnv) == "1" && name == "opencode" {
		runOpencodeRouterE2EHelper()
	}
}

func runOpencodeRouterE2EHelper() {
	fail := func(format string, args ...any) {
		_, _ = fmt.Fprintf(os.Stderr, "opencode fixture: "+format+"\n", args...)
		os.Exit(2)
	}
	args := os.Args[1:]
	boundary := slices.Index(args, "--")
	preBoundary := args
	if boundary >= 0 {
		preBoundary = args[:boundary]
	}
	if len(preBoundary) < 4 || preBoundary[0] != "run" || preBoundary[1] != "--standalone" {
		fail("OpenCode run parser rejected argv shape %q", args)
	}
	if slices.Contains(preBoundary, "--dir") {
		fail("launcher-only --dir reached OpenCode argv %q", args)
	}
	wantModel := "fak-router/" + os.Getenv("FAK_OPENCODE_ROUTER_E2E_MODEL")
	for i := 0; i+1 < len(preBoundary); i++ {
		if preBoundary[i] == "--model" && preBoundary[i+1] != wantModel {
			fail("model pin %q, want %q in %q", preBoundary[i+1], wantModel, args)
		}
	}
	if boundary < 0 || !slices.Equal(args[boundary+1:], []string{"--model", "opaque-prompt-text"}) {
		fail("opaque prompt suffix changed in %q", args)
	}
	wd, err := os.Getwd()
	if err != nil || filepath.Clean(wd) != filepath.Clean(os.Getenv("FAK_OPENCODE_ROUTER_E2E_WORKSPACE")) {
		fail("working directory = %q (%v), want %q", wd, err, os.Getenv("FAK_OPENCODE_ROUTER_E2E_WORKSPACE"))
	}
	var config struct {
		Provider map[string]struct {
			Options map[string]string `json:"options"`
			Models  map[string]any    `json:"models"`
		} `json:"provider"`
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(os.Getenv("OPENCODE_CONFIG_CONTENT")), &config); err != nil {
		fail("invalid OPENCODE_CONFIG_CONTENT: %v", err)
	}
	provider, ok := config.Provider[guardOpenCodeRouterProviderID]
	if !ok || config.Model != wantModel || provider.Options["baseURL"] != "{env:OPENAI_BASE_URL}" || provider.Options["apiKey"] != "{env:OPENAI_API_KEY}" {
		fail("router provider config missing or invalid: model=%q provider=%+v", config.Model, provider)
	}
	if _, ok := provider.Models[os.Getenv("FAK_OPENCODE_ROUTER_E2E_MODEL")]; !ok {
		fail("router model absent from config: %+v", provider.Models)
	}
	body := strings.NewReader(`{"model":"` + os.Getenv("FAK_OPENCODE_ROUTER_E2E_MODEL") + `","messages":[{"role":"user","content":"probe"}]}`)
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(os.Getenv("OPENAI_BASE_URL"), "/")+"/chat/completions", body)
	if err != nil {
		fail("build chat request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fail("guard chat request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(resp.Body)
		fail("guard chat status %d: %s", resp.StatusCode, payload)
	}
	os.Exit(0)
}

// OpenCode v2 binds a flag to the command it follows: `opencode --model m run`
// exits "Unrecognized flag: --model in command opencode", and `run` has no
// --pure or --dir. The child argv must lead with `run` so the guard's
// --standalone insertion applies.
func TestOpencodeLauncherChildArgvIsOpenCodeV2RunShape(t *testing.T) {
	for _, tc := range []struct {
		name           string
		opts           opencodeLaunchOptions
		wantChild      []string
		wantStandalone []string
	}{
		{
			name:           "probe",
			opts:           opencodeLaunchOptions{model: "m1", probePrompt: "do it", auto: true},
			wantChild:      []string{"opencode", "run", "--model", "fak/m1", "--format", "json", "--auto", "do it"},
			wantStandalone: []string{"opencode", "run", "--standalone", "--model", "fak/m1", "--format", "json", "--auto", "do it"},
		},
		{
			name:           "passthrough run",
			opts:           opencodeLaunchOptions{model: "fak/m1", passthrough: []string{"run", "status"}},
			wantChild:      []string{"opencode", "run", "--model", "fak/m1", "status"},
			wantStandalone: []string{"opencode", "run", "--standalone", "--model", "fak/m1", "status"},
		},
		{
			name:           "interactive",
			opts:           opencodeLaunchOptions{model: "m1", passthrough: []string{"--continue"}},
			wantChild:      []string{"opencode", "--continue"},
			wantStandalone: []string{"opencode", "--continue"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := buildOpencodeLaunchArgv("fak", tc.opts)
			sep := slices.Index(argv, "--")
			if sep < 0 {
				t.Fatalf("guard argv has no child boundary: %q", argv)
			}
			child := argv[sep+1:]
			if !slices.Equal(child, tc.wantChild) {
				t.Fatalf("guarded child argv = %q, want %q", child, tc.wantChild)
			}
			if direct := buildOpencodeDirectArgv(tc.opts); !slices.Equal(direct, tc.wantChild) {
				t.Fatalf("direct argv = %q, want %q", direct, tc.wantChild)
			}
			if got := guardOpenCodeStandaloneCommand(child); !slices.Equal(got, tc.wantStandalone) {
				t.Fatalf("guard standalone command = %q, want %q", got, tc.wantStandalone)
			}
		})
	}
}

// fak-test:runtime fast est=20ms
func TestRunOpencodeGuardBindsRegisteredWorkspaceRouter(t *testing.T) {
	const (
		baseURL   = "http://127.0.0.1:19123/v1"
		modelID   = "deepseek-ai/DeepSeek-V4.1-Flash"
		targetKey = "fixture-target-router-key"
		cwdKey    = "fixture-cwd-router-key"
	)
	workingDir := t.TempDir()
	workspaceDir := t.TempDir()
	t.Chdir(workingDir)
	writeConfig := func(dir, key string) {
		t.Helper()
		config := `{"provider":{"fak-router":{"options":{"baseURL":"` + baseURL + `","apiKey":"` + key + `"}}}}`
		if err := os.WriteFile(filepath.Join(dir, "opencode.json"), []byte(config), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeConfig(workingDir, cwdKey)
	writeConfig(workspaceDir, targetKey)
	t.Setenv("APPDATA", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	t.Setenv("FAK_AGENT_ROUTER_ORIGIN", "http://127.0.0.1:19124")
	t.Setenv("FAK_GATEWAY_KEY", "")

	originalRun := opencodeLaunchRun
	t.Cleanup(func() { opencodeLaunchRun = originalRun })
	var capturedArgv, capturedEnv []string
	opencodeLaunchRun = func(_, _ io.Writer, argv, env []string) int {
		capturedArgv = append([]string(nil), argv...)
		capturedEnv = append([]string(nil), env...)
		return 0
	}

	var stderr bytes.Buffer
	code := runOpencode(io.Discard, &stderr, []string{
		"--guard", "--quiet", "--split", "off",
		"--dir", workspaceDir,
		"--base-url", baseURL, "--model", modelID,
		"--", "run",
		"--model", "fak/escape", "check the repo",
		"--", "--model", "prompt-only",
	})
	if code != 0 {
		t.Fatalf("runOpencode exit = %d, stderr = %q", code, stderr.String())
	}
	outerBoundary := slices.Index(capturedArgv, "--")
	if outerBoundary < 0 || outerBoundary+1 >= len(capturedArgv) || capturedArgv[outerBoundary+1] != "opencode" {
		t.Fatalf("guard argv has no OpenCode child boundary: %q", capturedArgv)
	}
	outer := capturedArgv[:outerBoundary]
	if !argvHasPair(outer, "--model", modelID) {
		t.Fatalf("guard upstream argv does not keep raw model %q: %q", modelID, outer)
	}
	if !argvHasPair(outer, "--api-key-env", "FAK_GATEWAY_KEY") {
		t.Fatalf("guard argv does not name the session credential env: %q", outer)
	}
	if strings.Contains(strings.Join(capturedArgv, " "), targetKey) || strings.Contains(stderr.String(), targetKey) {
		t.Fatalf("router credential leaked into argv or stderr: argv=%q stderr=%q", capturedArgv, stderr.String())
	}
	gotKey := ""
	for _, entry := range capturedEnv {
		if strings.HasPrefix(entry, "FAK_GATEWAY_KEY=") {
			gotKey = strings.TrimPrefix(entry, "FAK_GATEWAY_KEY=")
		}
	}
	if gotKey != targetKey {
		t.Fatalf("session router key = %q, want launch --dir workspace key", gotKey)
	}

	child := capturedArgv[outerBoundary+2:]
	promptBoundary := slices.Index(child, "--")
	if promptBoundary < 0 {
		t.Fatalf("OpenCode child lost opaque prompt boundary: %q", child)
	}
	wantChildModel := "fak-router/" + modelID
	modelPins := 0
	for i := 0; i+1 < promptBoundary; i++ {
		if child[i] == "--model" {
			modelPins++
			if child[i+1] != wantChildModel {
				t.Fatalf("pre-prompt child model pin = %q, want %q in %q", child[i+1], wantChildModel, child)
			}
		}
	}
	if modelPins != 2 {
		t.Fatalf("pre-prompt child model pin count = %d, want 2 in %q", modelPins, child)
	}
	if got := child[promptBoundary+1:]; !slices.Equal(got, []string{"--model", "prompt-only"}) {
		t.Fatalf("opaque prompt suffix = %q, want unchanged model-shaped text", got)
	}
}

// fak-test:runtime slow est=5s
func TestRunOpencodeGuardRouterWorkspaceEndToEnd(t *testing.T) {
	const (
		modelID   = "fixture-router-model"
		targetKey = "fixture-workspace-router-key"
	)
	var mu sync.Mutex
	var handlerErr error
	chatCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || r.URL.Path == "/v1/models" || r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"context_length":8192,"max_output_tokens":1024}]}`, modelID)
			return
		}
		if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var request struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		chatCalls++
		if got := r.Header.Get("Authorization"); got != "Bearer "+targetKey {
			handlerErr = fmt.Errorf("upstream authorization = %q", got)
			http.Error(w, handlerErr.Error(), http.StatusUnauthorized)
			return
		}
		if request.Model != modelID {
			handlerErr = fmt.Errorf("upstream model = %q, want raw %q", request.Model, modelID)
			http.Error(w, handlerErr.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-fixture","object":"chat.completion","model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, modelID)
	}))
	defer upstream.Close()

	workspace := t.TempDir()
	config := `{"provider":{"fak-router":{"options":{"baseURL":"` + upstream.URL + `/v1","apiKey":"` + targetKey + `"}}}}`
	if err := os.WriteFile(filepath.Join(workspace, "opencode.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	binDir := t.TempDir()
	opencodeName := "opencode"
	if runtime.GOOS == "windows" {
		opencodeName += ".exe"
	}
	fakeOpencode := filepath.Join(binDir, opencodeName)
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(testBinary)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fakeOpencode, data, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("APPDATA", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	t.Setenv("FAK_AGENT_ROUTER_ORIGIN", "http://127.0.0.1:1")
	t.Setenv("FAK_GATEWAY_KEY", "")
	t.Setenv(opencodeRouterE2EHelperEnv, "1")
	t.Setenv("FAK_OPENCODE_ROUTER_E2E_MODEL", modelID)
	t.Setenv("FAK_OPENCODE_ROUTER_E2E_WORKSPACE", workspace)

	opts := opencodeLaunchOptions{
		guard: true, splitMode: "off", splitWhere: "bottom", splitInterval: 2 * time.Second,
		baseURL: upstream.URL + "/v1", model: modelID, quiet: true, noAudit: true,
		workspaceDir: workspace,
		passthrough:  []string{"run", "--model", "fak/escape", "probe", "--", "--model", "opaque-prompt-text"},
	}
	opts, _, err = prepareOpencodeRouterCredential(opts, os.Environ())
	if err != nil {
		t.Fatal(err)
	}
	guardArgv := buildOpencodeLaunchArgv(tuiExecutable(), opts)
	t.Setenv(guardE2EHelperEnv, strings.Join(guardArgv[2:], " "))

	for _, tc := range []struct {
		name    string
		dirArgs []string
		cwd     string
	}{
		{name: "default current workspace", cwd: workspace},
		{name: "explicit equals form", dirArgs: []string{"--dir=" + workspace}, cwd: t.TempDir()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Chdir(tc.cwd)
			if tc.cwd != workspace {
				decoy := `{"provider":{"fak-router":{"options":{"baseURL":"http://127.0.0.1:1/v1","apiKey":"decoy-cwd-key"}}}}`
				if err := os.WriteFile(filepath.Join(tc.cwd, "opencode.json"), []byte(decoy), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			args := []string{"--guard", "--quiet", "--no-audit", "--split", "off"}
			args = append(args, tc.dirArgs...)
			args = append(args,
				"--base-url", upstream.URL+"/v1", "--model", modelID,
				"--", "run", "--model", "fak/escape", "probe", "--", "--model", "opaque-prompt-text",
			)
			var output bytes.Buffer
			if code := runOpencode(&output, &output, args); code != 0 {
				t.Fatalf("actual launcher/guard/OpenCode fixture exit = %d\n%s", code, output.String())
			}
		})
	}
	mu.Lock()
	defer mu.Unlock()
	if handlerErr != nil || chatCalls != 2 {
		t.Fatalf("router upstream witness calls=%d err=%v", chatCalls, handlerErr)
	}
}

// fak-test:runtime fast est=5ms
func TestOpencodeRegisteredRouterSelectionIsExact(t *testing.T) {
	t.Chdir(t.TempDir())
	t.Setenv("APPDATA", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	t.Setenv("FAK_AGENT_ROUTER_ORIGIN", "http://127.0.0.1:19123")
	if !isRegisteredFakRouterURL("http://127.0.0.1:19123/v1") {
		t.Fatal("nondefault registered router port was not recognized")
	}
	for _, lookalike := range []string{
		"http://127.0.0.1:18101/v1",
		"http://127.0.0.1:19123.evil/v1",
		"http://127.0.0.1:19123/v1/models",
		"http://user@127.0.0.1:19123/v1",
		"http://127.0.0.1:19123/v1?next=router",
	} {
		if isRegisteredFakRouterURL(lookalike) {
			t.Fatalf("lookalike router %q was accepted", lookalike)
		}
	}

	for _, tc := range []struct {
		name    string
		opts    opencodeLaunchOptions
		wantPin string
	}{
		{"registered", opencodeLaunchOptions{guard: true, model: "m1", baseURL: "http://127.0.0.1:19123/v1", passthrough: []string{"run"}}, "fak-router/m1"},
		{"direct registered origin stays legacy", opencodeLaunchOptions{model: "m1", baseURL: "http://127.0.0.1:19123/v1", passthrough: []string{"run"}}, "fak/m1"},
		{"custom", opencodeLaunchOptions{model: "m1", baseURL: "https://provider.example/v1", passthrough: []string{"run"}}, "fak/m1"},
		{"custom strips router lookalike", opencodeLaunchOptions{model: "fak-router/m1", baseURL: "https://provider.example/v1", passthrough: []string{"run"}}, "fak/m1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			child := opencodeChildArgs(tc.opts)
			if !argvHasPair(child, "--model", tc.wantPin) {
				t.Fatalf("child argv = %q, want model %q", child, tc.wantPin)
			}
		})
	}
}

// fak-test:runtime fast est=5ms
func TestPrepareOpencodeRouterCredentialValidatesExplicitEnv(t *testing.T) {
	base := opencodeLaunchOptions{guard: true, baseURL: "https://provider.example/v1", model: "m1"}
	for _, name := range []string{"1INVALID", "HAS-DASH", "MISSING_KEY"} {
		opts := base
		opts.apiKeyEnv = name
		if _, _, err := prepareOpencodeRouterCredential(opts, []string{"MISSING_KEY="}); err == nil {
			t.Fatalf("--api-key-env %q was not refused", name)
		}
	}

	const explicitKey = "fixture-explicit-key"
	opts := base
	opts.apiKeyEnv = "CUSTOM_ROUTER_KEY"
	got, env, err := prepareOpencodeRouterCredential(opts, []string{"CUSTOM_ROUTER_KEY=" + explicitKey})
	if err != nil {
		t.Fatal(err)
	}
	if got.apiKeyEnv != opts.apiKeyEnv || !slices.Equal(env, []string{"CUSTOM_ROUTER_KEY=" + explicitKey}) {
		t.Fatalf("explicit credential precedence changed opts/env: opts=%+v env=%q", got, env)
	}
	argv := buildOpencodeLaunchArgv("fak", got)
	if !argvHasPair(argv, "--api-key-env", "CUSTOM_ROUTER_KEY") || strings.Contains(strings.Join(argv, " "), explicitKey) {
		t.Fatalf("explicit credential was omitted or leaked in argv: %q", argv)
	}
}

func TestOpencodeWorkspaceDirStopsAtPromptBoundary(t *testing.T) {
	for _, flagArgs := range [][]string{{"--dir", t.TempDir()}, {"--dir=" + t.TempDir()}} {
		workspace := strings.TrimPrefix(flagArgs[len(flagArgs)-1], "--dir=")
		originalRun := opencodeLaunchRun
		var childArgv, childEnv []string
		opencodeLaunchRun = func(_, _ io.Writer, argv, env []string) int {
			childArgv = append([]string(nil), argv...)
			childEnv = append([]string(nil), env...)
			return 0
		}
		args := append([]string{"--quiet", "--split", "off"}, flagArgs...)
		args = append(args, "--", "run", "prompt", "--", "--dir", "opaque")
		code := runOpencode(io.Discard, io.Discard, args)
		opencodeLaunchRun = originalRun
		if code != 0 {
			t.Fatalf("runOpencode(%q) exit = %d", flagArgs, code)
		}
		if got := opencodeEnvValue(childEnv, "FAK_OPENCODE_LAUNCH_WORKING_DIR"); filepath.Clean(got) != filepath.Clean(workspace) {
			t.Fatalf("runOpencode(%q) working dir = %q, want %q", flagArgs, got, workspace)
		}
		if slices.Contains(childArgv, "--dir") && !slices.Equal(childArgv[len(childArgv)-3:], []string{"--", "--dir", "opaque"}) {
			t.Fatalf("launcher --dir leaked into OpenCode argv: %q", childArgv)
		}
	}
}
