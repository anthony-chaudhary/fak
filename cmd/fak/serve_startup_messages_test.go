package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestServeStartupNoticeInventory(t *testing.T) {
	tests := []struct {
		file         string
		legacyWrite  string
		retainedFile string
		kind         string
	}{
		{file: "serve.go", legacyWrite: "writeServeDurabilityBanner(bannerOut, durability)", retainedFile: "serve_durability.go", kind: "session-durability"},
		{file: "serve_stages.go", legacyWrite: "writeServeBackendForwardPreflight(os.Stderr, pf)", retainedFile: "serve_backend_preflight.go", kind: "backend-forward"},
		{file: "serve_stages.go", legacyWrite: "fmt.Fprintf(os.Stderr, \"fak serve: --gguf %s", kind: "model-alias"},
		{file: "serve_stages.go", legacyWrite: "fmt.Printf(\"fak: in-kernel chat decode", kind: "compute-backend"},
		{file: "serve_stages.go", legacyWrite: "fmt.Println(\"fak: in-kernel chat decode", kind: "compute-backend"},
		{file: "serve_stages.go", legacyWrite: "fmt.Printf(\"fak: CUDA-graph decode replay enabled", kind: "cuda-graph"},
		{file: "serve_stages.go", legacyWrite: "fmt.Printf(\"fak: expert-parallel rank %d/%d loads experts", kind: "shard-residency"},
		{file: "serve_stages.go", legacyWrite: "fmt.Printf(\"fak: expert-parallel rank %d/%d: device-NCCL process group unavailable", kind: "collective-fallback"},
		{file: "serve_stages.go", legacyWrite: "fmt.Printf(\"fak: expert-parallel rank %d/%d joined the process group", kind: "process-group"},
		{file: "serve_stages.go", legacyWrite: "fmt.Fprintf(os.Stderr, \"fak serve: registered %d peer-DRAM lender", kind: "remote-dram"},
		{file: "serve_stages.go", legacyWrite: "fmt.Fprintln(os.Stderr, \"fak serve: native complete-prefix L3", kind: "prefix-store"},
		{file: "serve_ep_coord.go", legacyWrite: "fmt.Printf(\"fak: expert-parallel rank 0/%d owns tokenization", kind: "decode-topology"},
		{file: "serve_ep_coord.go", legacyWrite: "fmt.Printf(\"fak: expert-parallel rank %d/%d parks", kind: "decode-topology"},
		{file: "serve_load_helpers.go", legacyWrite: "fmt.Fprintf(os.Stderr, \"fak serve: safetensors directory has no usable tokenizer.json", kind: "tokenizer-fallback"},
		{file: "serve_load_helpers.go", legacyWrite: "fmt.Fprintf(os.Stderr, \"fak serve: --gguf set without --tokenizer", kind: "tokenizer-fallback"},
	}

	for _, test := range tests {
		t.Run(test.file+"/"+test.legacyWrite, func(t *testing.T) {
			body, err := os.ReadFile(test.file)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(body), test.legacyWrite) {
				t.Fatalf("ordinary startup notice still writes directly: %s", test.legacyWrite)
			}
			retainedFile := test.retainedFile
			if retainedFile == "" {
				retainedFile = test.file
			}
			retained, err := os.ReadFile(retainedFile)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(retained), `"`+test.kind+`"`) {
				t.Fatalf("ordinary startup notice was removed without retained kind %q", test.kind)
			}
		})
	}
}

func TestServeTokenizerFallbackIsRetained(t *testing.T) {
	tests := []struct {
		name string
		path func(*testing.T) string
		want string
	}{
		{
			name: "safetensors-directory",
			path: func(t *testing.T) string { return t.TempDir() },
			want: "no usable tokenizer.json",
		},
		{
			name: "gguf-without-embedded-bpe",
			path: func(t *testing.T) string { return filepath.Join(t.TempDir(), "missing.gguf") },
			want: "no usable embedded BPE tokenizer",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var messages []gateway.StartupMessage
			if tok, loaded := resolveServeTokenizer("", test.path(t), func(message gateway.StartupMessage) {
				messages = append(messages, message)
			}); tok != nil || loaded {
				t.Fatalf("fallback resolved tokenizer=%v loaded=%v", tok, loaded)
			}
			if len(messages) != 1 {
				t.Fatalf("fallback messages=%+v, want one", messages)
			}
			message := messages[0]
			if message.Source != "model-load" || message.Kind != "tokenizer-fallback" || message.Level != "warning" || !strings.Contains(message.Text, test.want) {
				t.Fatalf("fallback message=%+v, want model-load/tokenizer-fallback/warning containing %q", message, test.want)
			}
		})
	}
}

func TestAttendedSuccessfulServeLaunchWriterStaysQuiet(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and launches the real fak binary")
	}

	root := repoRootForDoctorTest(t)
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	exe := filepath.Join(t.TempDir(), "fak-startup-writer"+suffix)
	build := exec.Command("go", "build", "-o", exe, "./cmd/fak")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build real fak binary: %v\n%s", err, out)
	}

	addr := reserveServeStartupAddr(t)
	state := t.TempDir()
	cmd := exec.Command(exe,
		"serve",
		"--addr", addr,
		"--base-url", "http://127.0.0.1:1",
		"--provider", "anthropic",
		"--session-state", "off",
		"--cuda-graph",
	)
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"FAK_SESSION_REGISTRY="+filepath.Join(state, "session-registry.json"),
		"HOME="+state,
		"USERPROFILE="+state,
		"XDG_CONFIG_HOME="+filepath.Join(state, ".config"),
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})

	vars := readServeStartupVars(t, "http://"+addr+"/debug/vars", 12*time.Second)
	if runtime.GOOS == "windows" {
		_ = cmd.Process.Kill()
	} else if err := cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatalf("interrupt serve: %v", err)
	}
	if err := cmd.Wait(); err != nil && runtime.GOOS != "windows" {
		t.Fatalf("serve did not stop cleanly: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	if got := stdout.String(); got != "" {
		t.Fatalf("successful attended launch wrote ordinary stdout chatter:\n%s", got)
	}
	const readyMarker = "fak gateway listening on http://"
	stderrText := stderr.String()
	readyAt := strings.Index(stderrText, readyMarker)
	if readyAt < 0 {
		t.Fatalf("successful attended launch never emitted the post-ready marker:\n%s", stderrText)
	}
	if got := serveStartupChatterLines(stderrText[:readyAt]); len(got) != 0 {
		t.Fatalf("successful attended launch wrote ordinary pre-ready stderr chatter:\n%s", strings.Join(got, "\n"))
	}

	want := map[string]bool{"cuda-graph": false, "session-durability": false}
	for _, message := range vars.Startup.Messages {
		if _, ok := want[message.Kind]; ok {
			want[message.Kind] = true
		}
	}
	for kind, found := range want {
		if !found {
			t.Errorf("startup.messages missing kind %q: %+v", kind, vars.Startup.Messages)
		}
	}
}

func serveStartupChatterLines(text string) []string {
	var chatter []string
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if _, err := time.Parse("2006/01/02 15:04:05", line); err == nil {
			continue
		}
		chatter = append(chatter, line)
	}
	return chatter
}

func reserveServeStartupAddr(t *testing.T) string {
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

func readServeStartupVars(t *testing.T, url string, timeout time.Duration) struct {
	Startup struct {
		Messages []gateway.StartupMessage `json:"messages"`
	} `json:"startup"`
} {
	t.Helper()
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 250 * time.Millisecond}
	for time.Now().Before(deadline) {
		response, err := client.Get(url)
		if err == nil {
			body, readErr := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if readErr == nil && response.StatusCode == http.StatusOK {
				var vars struct {
					Startup struct {
						Messages []gateway.StartupMessage `json:"messages"`
					} `json:"startup"`
				}
				if err := json.Unmarshal(body, &vars); err != nil {
					t.Fatalf("decode /debug/vars: %v\n%s", err, body)
				}
				return vars
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("serve did not become ready at %s", url)
	return struct {
		Startup struct {
			Messages []gateway.StartupMessage `json:"messages"`
		} `json:"startup"`
	}{}
}
