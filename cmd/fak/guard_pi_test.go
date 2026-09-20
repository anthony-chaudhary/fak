package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type fakePiObservation struct {
	Extension string `json:"extension"`
	Endpoint  string `json:"endpoint"`
	Model     string `json:"model"`
	ToolCall  bool   `json:"tool_call"`
	Ready     bool   `json:"ready"`
}

const fakePiChildSource = `package main
import (
 "bufio"
 "bytes"
 "encoding/json"
 "io"
 "net/http"
 "os"
 "strings"
 "time"
)
type observation struct { Extension string ` + "`json:\"extension\"`" + `; Endpoint string ` + "`json:\"endpoint\"`" + `; Model string ` + "`json:\"model\"`" + `; ToolCall bool ` + "`json:\"tool_call\"`" + `; Ready bool ` + "`json:\"ready\"`" + ` }
func field(src, key string) string {
 i := strings.Index(src, key+":")
 if i < 0 { return "" }
 rest := strings.TrimSpace(src[i+len(key)+1:])
 var value string
 if json.NewDecoder(strings.NewReader(rest)).Decode(&value) != nil { return "" }
 return value
}
func write(path string, o observation) { b, _ := json.Marshal(o); _ = os.WriteFile(path, b, 0600) }
func main() {
 var ext, provider, model string
 for i := 1; i < len(os.Args); i++ {
  switch os.Args[i] {
  case "-e", "--extension": i++; if i < len(os.Args) { ext = os.Args[i] }
  case "--provider": i++; if i < len(os.Args) { provider = os.Args[i] }
  case "--model": i++; if i < len(os.Args) { model = os.Args[i] }
  }
 }
 obs := observation{Extension: ext, Model: model}
 out := os.Getenv("FAK_PI_WIRE_OBSERVATION")
 write(out, obs)
 raw, err := os.ReadFile(ext)
 if err != nil { os.Exit(40) }
 src := string(raw)
 base := os.Getenv("FAK_PI_DIRECT_UPSTREAM")
 wirePath, toolNeedle := "/chat/completions", "tool_calls"
 if provider == "fak" && strings.Contains(src, "registerProvider(\"fak\"") && field(src, "api") == "openai-completions" && field(src, "id") == model {
  base = field(src, "baseUrl")
 } else if provider == "anthropic" && strings.Contains(src, "registerProvider(\"anthropic\"") {
  base, wirePath, toolNeedle = field(src, "baseUrl"), "/v1/messages", "tool_use"
 }
 obs.Endpoint = strings.TrimRight(base, "/") + wirePath
 body, _ := json.Marshal(map[string]any{"model": model, "stream": true, "messages": []map[string]string{{"role":"user", "content":"call inspect"}}, "tools": []map[string]any{{"type":"function", "function":map[string]any{"name":"inspect", "parameters":map[string]string{"type":"object"}}}}})
 req, _ := http.NewRequest(http.MethodPost, obs.Endpoint, bytes.NewReader(body))
 req.Header.Set("Content-Type", "application/json")
 resp, err := http.DefaultClient.Do(req)
 if err != nil { os.Exit(41) }
 scan := bufio.NewScanner(io.LimitReader(resp.Body, 1<<20))
 for scan.Scan() { if strings.Contains(scan.Text(), toolNeedle) { obs.ToolCall = true } }
 _ = resp.Body.Close()
 obs.Ready = true
 write(out, obs)
 switch os.Getenv("FAK_PI_WIRE_MODE") {
 case "failure": os.Exit(17)
 case "cancellation": for { time.Sleep(time.Hour) }
 }
}
`

func buildFakePiChild(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "main.go")
	if err := os.WriteFile(source, []byte(fakePiChildSource), 0o600); err != nil {
		t.Fatal(err)
	}
	name := "pi"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", binary, source)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fake Pi: %v\n%s", err, out)
	}
	return binary
}

// TestGuardPiWire drives the real guard lifecycle around a deterministic Pi consumer. The
// child structurally consumes the installed provider/model, sends a streamed tool-call turn
// through the guard-owned gateway, and uses a sentinel only if the extension is ineffective.
// Production — not this test — must remove its session extension on every terminal path.
func TestGuardPiWire(t *testing.T) {
	pi := buildFakePiChild(t)
	for _, tc := range []struct {
		name          string
		mode          string
		childProvider string
		guardProvider string
	}{
		{name: "anthropic-origin-success", childProvider: "anthropic", guardProvider: "anthropic"},
		{name: "openai-compatible-v1-success", childProvider: "fak", guardProvider: "openai"},
		{name: "failure", mode: "failure", childProvider: "fak", guardProvider: "openai"},
		{name: "cancellation", mode: "cancellation", childProvider: "fak", guardProvider: "openai"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var gatewayHits, directHits atomic.Int32
			var gotModel atomic.Value
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gatewayHits.Add(1)
				var body struct {
					Model string `json:"model"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				gotModel.Store(body.Model)
				w.Header().Set("Content-Type", "text/event-stream")
				if tc.guardProvider == "anthropic" {
					_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"fixture\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu-1\",\"name\":\"inspect\",\"input\":{}}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
				} else {
					_, _ = io.WriteString(w, "data: {\"id\":\"pi-wire\",\"object\":\"chat.completion.chunk\",\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"inspect\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n")
					_, _ = io.WriteString(w, "data: {\"id\":\"pi-wire\",\"object\":\"chat.completion.chunk\",\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
				}
			}))
			defer upstream.Close()
			direct := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				directHits.Add(1)
				http.Error(w, "direct upstream must not be reached", http.StatusTeapot)
			}))
			defer direct.Close()

			root := t.TempDir()
			observation := filepath.Join(root, "observation.json")
			workspace := filepath.Join(root, "workspace")
			if err := os.MkdirAll(filepath.Join(workspace, ".git"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(workspace, "dos.toml"), []byte("[lanes]\n[lanes.trees]\ncmd = [\"cmd/**\"]\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			baseURL := upstream.URL + "/v1"
			if tc.guardProvider == "anthropic" {
				baseURL = upstream.URL
			}
			args := []string{"--quiet", "--split", "off", "--provider", tc.guardProvider, "--base-url", baseURL, "--api-key-env", "FAK_PI_TEST_KEY", "--model", "fixture"}
			if tc.mode == "cancellation" {
				args = append(args, "--max-duration", "400ms", "--soft-deadline-lead", "0", "--commit-grace-period", "0", "--child-stop-grace", "20ms")
			}
			args = append(args, "--", pi, "--provider", tc.childProvider, "--model", "fixture")
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0])
			cmd.Dir = workspace
			cmd.Env = append(os.Environ(),
				guardE2EHelperEnv+"="+strings.Join(args, " "),
				"FAK_PI_TEST_KEY=test-only", "FAK_PI_WIRE_MODE="+tc.mode,
				"FAK_PI_WIRE_OBSERVATION="+observation, "FAK_PI_DIRECT_UPSTREAM="+direct.URL+"/v1",
				"FAK_FLEET_BUS="+filepath.Join(root, "fleet-bus"), "TMP="+root, "TEMP="+root,
			)
			var output bytes.Buffer
			cmd.Stdout, cmd.Stderr = &output, &output
			err := cmd.Run()
			if ctx.Err() != nil {
				t.Fatalf("guard lifecycle timed out: %v\n%s", ctx.Err(), output.String())
			}
			if strings.HasSuffix(tc.name, "success") && err != nil {
				t.Fatalf("guard success: %v\n%s", err, output.String())
			}
			if tc.name == "failure" && err == nil {
				t.Fatalf("guard %s unexpectedly succeeded", tc.mode)
			}

			var got fakePiObservation
			raw, readErr := os.ReadFile(observation)
			if readErr != nil || json.Unmarshal(raw, &got) != nil {
				t.Fatalf("fake Pi observation: %v: %s\n%s", readErr, raw, output.String())
			}
			wantToolCall := tc.childProvider == "fak"
			if gatewayHits.Load() < 1 || directHits.Load() != 0 || gotModel.Load() != "fixture" || (wantToolCall && !got.ToolCall) || !got.Ready {
				t.Fatalf("wire not qualified: gateway=%d direct=%d request_model=%v observation=%+v\n%s", gatewayHits.Load(), directHits.Load(), gotModel.Load(), got, output.String())
			}
			if _, statErr := os.Stat(got.Extension); !errors.Is(statErr, os.ErrNotExist) {
				t.Fatalf("production %s lifecycle left Pi extension %q: %v", tc.name, got.Extension, statErr)
			}
		})
	}

	for _, command := range [][]string{{"pi", "--provider", "gemini", "--model", "gemini-test"}, {"pi", "--provider", "fak", "--model", "fixture", "-e", "override.ts"}} {
		dir := t.TempDir()
		if _, _, err := installGuardPiExtensionAt(command, "http://127.0.0.1:4567", dir); err == nil {
			t.Fatalf("unsupported Pi wire admitted: %v", command)
		}
		if _, err := os.Stat(filepath.Join(dir, guardPiExtensionFileName)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("refused Pi wire left extension behind: %v", err)
		}
	}
}

// TestGuardPiExtensionInstallsProviderRepoint is the acceptance witness: wrapping `pi` writes a
// session-scoped extension that registers the anthropic provider at the gateway origin and
// prepends `-e <path>` before the user args, so Pi routes through the kernel.
func TestGuardPiExtensionInstallsProviderRepoint(t *testing.T) {
	dir := t.TempDir()
	command, install, err := installGuardPiExtensionAt(
		[]string{"pi", "-p", "hello"},
		"http://127.0.0.1:4567",
		dir,
	)
	if err != nil {
		t.Fatalf("install pi extension: %v", err)
	}
	if !install.Applied {
		t.Fatalf("pi extension not applied: %+v", install)
	}
	if got, want := install.BaseURL, "http://127.0.0.1:4567"; got != want {
		t.Fatalf("install.BaseURL = %q, want %q (bare origin; Pi appends /v1/messages)", got, want)
	}
	// -e <path> must sit immediately after the executable, before the prompt args.
	if got, want := command[1], "-e"; got != want {
		t.Fatalf("command missing -e flag after executable: %v", command)
	}
	if got, want := command[2], install.ExtensionPath; got != want {
		t.Fatalf("extension path = %q, want %q", got, want)
	}
	if got, want := strings.Join(command[3:], "\x00"), strings.Join([]string{"-p", "hello"}, "\x00"); got != want {
		t.Fatalf("user args changed or -e was appended after prompt args: %v", command)
	}

	data, err := os.ReadFile(install.ExtensionPath)
	if err != nil {
		t.Fatalf("read pi extension: %v", err)
	}
	src := string(data)
	// The module must register the anthropic provider at the gateway origin with NO models
	// (so Pi keeps every Claude model it knows and swaps only the endpoint).
	for _, want := range []string{
		"export default function (pi)",
		`pi.registerProvider("anthropic"`,
		`baseUrl: "http://127.0.0.1:4567"`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("extension missing %q:\n%s", want, src)
		}
	}
	if strings.Contains(src, "models") {
		t.Fatalf("extension should NOT set models (a baseUrl-only override preserves Pi's model list):\n%s", src)
	}
}

// TestGuardPiExtensionInjectsAcrossInvocationForms proves `-e <path>` lands immediately after
// the executable and before every user arg regardless of how Pi is named — a launcher-suffixed
// `pi.exe` or an absolute path resolves through the same profile gate as a bare `pi`, so the
// repoint is not silently skipped for a path-qualified invocation (the detection test covers
// these names; this locks the actual `-e` injection + ordering for them).
func TestGuardPiExtensionInjectsAcrossInvocationForms(t *testing.T) {
	for _, tc := range []struct {
		name    string
		command []string
		rest    []string
	}{
		{name: "exe-suffix", command: []string{"pi.exe", "chat", "hi"}, rest: []string{"chat", "hi"}},
		{name: "absolute-path", command: []string{"/usr/local/bin/pi", "-p", "hello"}, rest: []string{"-p", "hello"}},
		{name: "no-user-args", command: []string{"pi"}, rest: []string{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			command, install, err := installGuardPiExtensionAt(tc.command, "http://127.0.0.1:4567", t.TempDir())
			if err != nil {
				t.Fatalf("install pi extension: %v", err)
			}
			if !install.Applied {
				t.Fatalf("pi extension not applied for %v: %+v", tc.command, install)
			}
			if got := command[0]; got != tc.command[0] {
				t.Fatalf("executable changed: %q -> %q", tc.command[0], got)
			}
			if command[1] != "-e" || command[2] != install.ExtensionPath {
				t.Fatalf("-e <path> not injected right after executable: %v", command)
			}
			if got, want := strings.Join(command[3:], "\x00"), strings.Join(tc.rest, "\x00"); got != want {
				t.Fatalf("user args changed or -e appended after them: %v", command)
			}
		})
	}
}

// TestGuardPiExtensionSkipsOffAndNonPi proves the install is inert for --pi-extension=false, a
// non-Pi child, and an empty command — the command is returned byte-identical, nothing written.
func TestGuardPiExtensionSkipsOffAndNonPi(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
		command []string
	}{
		{name: "off", enabled: false, command: []string{"pi"}},
		{name: "non-pi", enabled: true, command: []string{"claude"}},
		{name: "non-pi-codex", enabled: true, command: []string{"codex"}},
		{name: "empty", enabled: true, command: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var (
				command []string
				install guardPiInstall
				err     error
			)
			if !tc.enabled {
				command, install, err = installGuardPiExtension(tc.command, false, "http://127.0.0.1:4567")
			} else {
				command, install, err = installGuardPiExtensionAt(tc.command, "http://127.0.0.1:4567", t.TempDir())
			}
			if err != nil {
				t.Fatalf("install pi extension: %v", err)
			}
			if install.Applied {
				t.Fatalf("pi extension applied unexpectedly: %+v", install)
			}
			if strings.Join(command, "\x00") != strings.Join(tc.command, "\x00") {
				t.Fatalf("command changed: %v -> %v", tc.command, command)
			}
		})
	}
}

// TestGuardIsPiMatchesProfileRegistry proves the gate is driven by the profile registry
// (RepointExtension), so exactly the pi profile takes the extension repoint and no other
// wrapped agent does.
func TestGuardIsPiMatchesProfileRegistry(t *testing.T) {
	for _, tc := range []struct {
		command string
		want    bool
	}{
		{"pi", true},
		{"pi.exe", true},
		{"/usr/local/bin/pi", true},
		{"claude", false},
		{"codex", false},
		{"opencode", false},
		{"vim", false},
		{"", false},
	} {
		if got := guardIsPi(tc.command); got != tc.want {
			t.Fatalf("guardIsPi(%q) = %v, want %v", tc.command, got, tc.want)
		}
	}
}

// TestGuardPiBaseURLTrimsTrailingSlash proves the base URL handed to Pi is the bare origin with
// any trailing slash trimmed — Pi appends /v1/messages itself, exactly like ANTHROPIC_BASE_URL.
func TestGuardPiBaseURLTrimsTrailingSlash(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"http://127.0.0.1:4567", "http://127.0.0.1:4567"},
		{"http://127.0.0.1:4567/", "http://127.0.0.1:4567"},
		{"  http://127.0.0.1:4567  ", "http://127.0.0.1:4567"},
	} {
		if got := guardPiBaseURL(tc.in); got != tc.want {
			t.Fatalf("guardPiBaseURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
