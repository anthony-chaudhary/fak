package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime slow est=5s
func TestPiRouterAuthConsumesProtectedOriginCredential(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Fatal("Pi's existing Node runtime is required for provider-consumption witness")
	}
	const secret = "independent-pi-origin-key-13499"
	for _, tc := range []struct {
		name                           string
		proof, workspace, childFailure bool
		wantAuthorized                 bool
	}{
		{name: "proved-selected-origin", proof: true, wantAuthorized: true},
		{name: "proved-child-failure-cleanup", proof: true, childFailure: true, wantAuthorized: true},
		{name: "unproved-workspace-exact-origin", workspace: true},
		{name: "workspace-different-selected-origin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolatePiHome(t)
			t.Setenv("FAK_GATEWAY_KEY", secret)
			// Exercise workspace resolution itself; a nonempty implicit env key
			// would short-circuit before the matching workspace fallback.
			if !tc.proof {
				t.Setenv("FAK_GATEWAY_KEY", "")
			}
			t.Setenv("OPENAI_API_KEY", "unrelated-provider-secret")
			var mu sync.Mutex
			attempts, authorized := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/healthz":
					if tc.proof {
						nonce, e := base64.StdEncoding.DecodeString(r.Header.Get(routerAuthChallengeHeader))
						if e == nil && len(nonce) == 32 {
							mac := hmac.New(sha256.New, []byte(secret))
							mac.Write([]byte(routerAuthProofDomain))
							mac.Write(nonce)
							w.Header().Set(routerAuthProofHeader, base64.StdEncoding.EncodeToString(mac.Sum(nil)))
						}
					}
					w.Header().Set("Content-Type", "application/json")
					io.WriteString(w, `{"ok":true,"model":"independent-pi-model"}`)
				case "/v1/chat/completions":
					mu.Lock()
					attempts++
					ok := r.Header.Get("Authorization") == "Bearer "+secret
					if ok {
						authorized++
					}
					mu.Unlock()
					if !ok {
						http.Error(w, "unauthorized", http.StatusUnauthorized)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			if tc.workspace {
				writeRouterConfig(t, server.URL+"/v1", secret)
			} else if !tc.proof {
				writeRouterConfig(t, "http://127.0.0.1:65530/v1", secret)
			}
			old := piLaunchRun
			t.Cleanup(func() { piLaunchRun = old })
			var extension string
			var childOutput bytes.Buffer
			piLaunchRun = func(_ io.Writer, _ io.Writer, argv, env []string) int {
				for _, a := range argv {
					if strings.Contains(a, secret) || strings.Contains(a, "unrelated-provider-secret") {
						t.Error("credential leaked into child argv")
					}
				}
				for i := 0; i+1 < len(argv); i++ {
					if argv[i] == "-e" {
						extension = argv[i+1]
						break
					}
				}
				if extension == "" {
					t.Error("native launcher omitted provider extension")
					return 70
				}
				info, e := os.Stat(extension)
				if e != nil {
					t.Error("live extension inaccessible")
					return 70
				}
				if info.Mode().Perm() != 0600 {
					t.Errorf("credential-bearing extension permissions=%o want0600", info.Mode().Perm())
				}
				// Execute the exact generated provider registration in a real child, then
				// consume its bearer through an HTTP boundary. No authored provider bypass.
				script := `import fs from 'node:fs';
let registered;
const source=fs.readFileSync(process.env.FAK_TEST_PI_EXTENSION,'utf8');
const extension=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
extension.default({registerProvider:(name,config)=>{if(name==='fak')registered=config;}});
if(!registered)process.exit(70);
let key=registered.apiKey;
if(typeof key==='string'&&key.startsWith('$'))key=process.env[key.slice(1)]||'';
const response=await fetch(registered.baseUrl+'/chat/completions',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+(key||'')},body:JSON.stringify({model:registered.models[0].id,messages:[{role:'user',content:'test'}],max_tokens:1})});
console.log('provider_http_status='+response.status);
process.exit(response.status===200?(process.env.FAK_TEST_PI_FAIL==='1'?23:0):24);`
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				cmd := exec.CommandContext(ctx, node, "--input-type=module", "-e", script)
				cmd.Env = append(env, "FAK_TEST_PI_EXTENSION="+extension, fmt.Sprintf("FAK_TEST_PI_FAIL=%d", boolToIntPiAuth(tc.childFailure)))
				cmd.Stdout = &childOutput
				cmd.Stderr = &childOutput
				e = cmd.Run()
				if e == nil {
					return 0
				}
				if x, ok := e.(*exec.ExitError); ok {
					return x.ExitCode()
				}
				t.Error("provider child failed to execute")
				return 70
			}
			dir := t.TempDir()
			models := filepath.Join(dir, "models.json")
			settings := filepath.Join(dir, "settings.json")
			args := []string{"--check-backend=false", "--base-url", server.URL + "/v1", "--model", "independent-pi-model", "--window", "8192", "--config-path", models, "--settings-path", settings, "--probe", "test"}
			var stdout, stderr bytes.Buffer
			rc := runPi(&stdout, &stderr, args)
			mu.Lock()
			gotAttempts, gotAuthorized := attempts, authorized
			mu.Unlock()
			if gotAttempts != 1 {
				t.Errorf("actual provider HTTP attempts=%d want1", gotAttempts)
			}
			if tc.wantAuthorized {
				if gotAuthorized != 1 {
					t.Error("proved selected-origin credential did not authorize actual provider request")
				}
				want := 0
				if tc.childFailure {
					want = 23
				}
				if rc != want {
					t.Errorf("launcher exit=%d want%d", rc, want)
				}
			} else {
				if gotAuthorized != 0 {
					t.Error("credential crossed unproved or different selected origin")
				}
				if rc == 0 {
					t.Error("unproved protected origin unexpectedly completed")
				}
			}
			for _, raw := range []string{stdout.String(), stderr.String(), childOutput.String()} {
				if strings.Contains(raw, secret) || strings.Contains(raw, "unrelated-provider-secret") {
					t.Error("credential leaked into launcher/child output")
				}
			}
			if extension == "" {
				t.Error("extension was never consumed")
			} else if _, e := os.Stat(extension); !os.IsNotExist(e) {
				t.Error("temporary provider extension survived child terminal")
			}
			for _, path := range []string{models, settings} {
				if _, e := os.Stat(path); !os.IsNotExist(e) {
					t.Error("ordinary launch changed persistent isolated config")
				}
			}
			stdout.Reset()
			stderr.Reset()
			if rc := runPi(&stdout, &stderr, append([]string{"--dry-run"}, args...)); rc != 0 {
				t.Errorf("dry-run exit=%d", rc)
			}
			if strings.Contains(stdout.String()+stderr.String(), secret) {
				t.Error("dry-run disclosed credential")
			}
		})
	}
}

func boolToIntPiAuth(v bool) int {
	if v {
		return 1
	}
	return 0
}
