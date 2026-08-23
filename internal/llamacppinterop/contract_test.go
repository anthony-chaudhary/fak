package llamacppinterop

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

type fakeRunner struct {
	out     string
	help    string
	devices string
	err     error
	got     [][]string
}

func (f *fakeRunner) Output(_ context.Context, n string, a ...string) ([]byte, error) {
	f.got = append(f.got, append([]string{n}, a...))
	if f.err != nil {
		return nil, f.err
	}
	if len(a) > 0 && a[0] == "--help" {
		return []byte(f.help), nil
	}
	if len(a) > 0 && a[0] == "--list-devices" {
		return []byte(f.devices), nil
	}
	return []byte(f.out), nil
}

func TestDiscoverAndPlan(t *testing.T) {
	f := &fakeRunner{out: "llama-server version: 0.0.6123 (commit 8144f31)", help: "--spec-type none,draft-mtp", devices: "CUDA0: NVIDIA A100"}
	r := Discover(context.Background(), f, "llama-server")
	if r.Outcome != OutcomeDelegate || r.Capability.Version != "0.0.6123" || r.Capability.Commit != "8144f31" || !r.Capability.DraftMTP || !r.Capability.CUDA {
		t.Fatalf("%+v got=%v", r, f.got)
	}
	d := quantmeta.Descriptor{Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}, Extra: map[string]json.RawMessage{"gguf_architecture": json.RawMessage(`"qwen35"`)}}
	p := PlanQwen38MTP(r.Capability, "tiny.gguf", d, 18080, 4096)
	if p.Outcome != OutcomeDelegate || !containsPair(p.Argv, "--spec-type", "draft-mtp") || !containsPair(p.Argv, "--host", "127.0.0.1") || !containsPair(p.Argv, "-b", "4096") || !containsPair(p.Argv, "-ub", "1024") {
		t.Fatalf("%+v", p)
	}
}
func TestWitnessedQwen38MTP(t *testing.T) {
	cap := Capability{Commit: "8144f3192e5a3131cd043f284525e6ceebf82d0f", Server: true, DraftMTP: true, CUDA: true}
	if !WitnessedQwen38MTP(cap) {
		t.Fatal("measured runtime was not admitted")
	}
	cap.Commit = "deadbeef"
	if WitnessedQwen38MTP(cap) {
		t.Fatal("unmeasured runtime was admitted")
	}
}

func TestDiscoverFailsClosed(t *testing.T) {
	if r := Discover(context.Background(), &fakeRunner{err: errors.New("missing")}, "llama-cli"); r.Outcome != OutcomeRefuse {
		t.Fatalf("%+v", r)
	}
	if r := Discover(context.Background(), &fakeRunner{out: "unknown"}, "llama-cli"); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
func TestPlanRejectsNonGGUFAndUnsupportedMTP(t *testing.T) {
	cap := Capability{Binary: "llama-server", Version: "1", Server: true, DraftMTP: true, CUDA: true}
	if r := Plan(cap, "m.bin", quantmeta.Descriptor{}); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
	d := quantmeta.Descriptor{Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}, Extra: map[string]json.RawMessage{"gguf_architecture": json.RawMessage(`"llama"`)}}
	if r := PlanQwen38MTP(cap, "m.gguf", d, 1, 1); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
	d.Extra["gguf_architecture"] = json.RawMessage(`"qwen35"`)
	cap.CUDA = false
	if r := PlanQwen38MTP(cap, "m.gguf", d, 1, 1); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
	cap.CUDA = true
	cap.DraftMTP = false
	if r := PlanQwen38MTP(cap, "m.gguf", d, 1, 1); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
func TestHealth(t *testing.T) {
	for _, tc := range []struct {
		name string
		code int
		body string
		want Outcome
	}{{"ready", 200, `{"status":"ready"}`, OutcomeDelegate}, {"down", 503, `{}`, OutcomeRefuse}, {"unknown", 200, `{"status":"loading"}`, OutcomeAbstain}} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(tc.code); _, _ = w.Write([]byte(tc.body)) }))
			defer s.Close()
			if r := CheckHealth(context.Background(), s.Client(), s.URL); r.Outcome != tc.want {
				t.Fatalf("%+v", r)
			}
		})
	}
}
func TestProcessStartAndStop(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "fake-server")
	body := `#!/bin/sh
port=""
while [ "$#" -gt 0 ]; do if [ "$1" = "--port" ]; then port="$2"; shift; fi; shift; done
python3 - "$port" <<'PY'
import http.server,json,sys
class H(http.server.BaseHTTPRequestHandler):
 def do_GET(self): self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(b'{"status":"ok"}')
 def log_message(self,*a): pass
http.server.HTTPServer(("127.0.0.1",int(sys.argv[1])),H).serve_forever()
PY
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	port := freePort(t)
	p, err := Start(context.Background(), []string{script, "--port", port}, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	pid := p.PID()
	if pid == 0 || p.BaseURL() != "http://127.0.0.1:"+port+"/v1" {
		t.Fatalf("pid=%d url=%s", pid, p.BaseURL())
	}
	if err := p.Stop(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Fatal(err)
	}
}
func containsPair(v []string, a, b string) bool {
	for i := 0; i+1 < len(v); i++ {
		if v[i] == a && v[i+1] == b {
			return true
		}
	}
	return false
}
func freePort(t *testing.T) string {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer s.Close()
	return strings.TrimPrefix(s.URL, "http://127.0.0.1:")
}

var _ = reflect.DeepEqual
