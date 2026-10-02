package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
)

const opsRunAuthTestKeyEnv = "FAK_OPS_PREFLIGHT_TEST_KEY"
const opsRunAuthTestRole = "isolated Go reexec for preflight auth tests"

// Native keeps its real subprocess construction. Only an exact disposable role
// file, delivered through the existing platform-temp environment, selects this
// fake native child. Pi/OpenCode use their existing executor seams to reexec the
// same bounded child after the real CLI preflight. No guard or probe is bypassed.
func init() {
	if len(os.Args) > 1 && os.Args[1] == "chat" && opsRunAuthChildRoot() != "" {
		if err := opsRunAuthRecordChild(); err != nil {
			os.Exit(91)
		}
		writeOpsNativeFixtureReceipt([]byte(`{"schema":"fak.agent.native.v1","status":"completed","metrics":{"arm":"fak"}}`))
		os.Exit(0)
	}
}

func opsRunAuthChildRoot() string {
	root := os.Getenv("TMPDIR")
	b, err := os.ReadFile(filepath.Join(root, "auth-child-role"))
	if err != nil || string(b) != opsRunAuthTestRole || !filepath.IsAbs(root) {
		return ""
	}
	return root
}

func opsRunAuthRecordChild() error {
	root := opsRunAuthChildRoot()
	if root == "" {
		return fmt.Errorf("missing isolated child role")
	}
	digest := sha256.Sum256([]byte(os.Getenv(opsRunAuthTestKeyEnv)))
	f, err := os.OpenFile(filepath.Join(root, "child-starts"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintln(f, hex.EncodeToString(digest[:]))
	return err
}

// fak-test:runtime fast est=1s lane=default
func TestOpsRunPreflightAuthChild(t *testing.T) {
	if opsRunAuthChildRoot() == "" {
		return
	}
	if err := opsRunAuthRecordChild(); err != nil {
		os.Exit(92)
	}
	fmt.Println(`{"type":"assistant","message":"fixture completed"}`)
	os.Exit(0)
}

type opsRunAuthFixture struct {
	root string
}

func newOpsRunAuthFixture(t *testing.T) opsRunAuthFixture {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "auth-child-role"), []byte(opsRunAuthTestRole), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TMPDIR", root)
	t.Setenv(opsNativeReceiptStoreEnv, filepath.Join(root, "native-evidence"))
	oldOpenCode, oldPi := opsRunExecute, opsPiRunExecute
	t.Cleanup(func() { opsRunExecute, opsPiRunExecute = oldOpenCode, oldPi })
	execute := func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, _ []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		// The real CLI composes this guarded argv; the fixture substitutes only
		// executable consumption. Credential values must remain off argv.
		for _, arg := range argv {
			if key := os.Getenv(opsRunAuthTestKeyEnv); key != "" && strings.Contains(arg, key) {
				t.Error("selected credential leaked onto child argv")
				return 1, false, false, nil
			}
		}
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunPreflightAuthChild$")
		cmd.Env, cmd.Stdout, cmd.Stderr = env, stdout, stderr
		if err := cmd.Run(); err != nil {
			return 1, false, false, nil
		}
		return 0, true, false, nil
	}
	opsRunExecute, opsPiRunExecute = execute, execute
	return opsRunAuthFixture{root: root}
}

func (f opsRunAuthFixture) childDigests(t *testing.T) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(f.root, "child-starts"))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Fields(string(b))
}

func (f opsRunAuthFixture) run(t *testing.T, harness, baseURL string, explicit bool) (int, opsRunReceipt, string) {
	t.Helper()
	dir := t.TempDir()
	prompt, receiptPath := filepath.Join(dir, "prompt.txt"), filepath.Join(dir, "receipt.json")
	if err := os.WriteFile(prompt, []byte("inspect isolated fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	provider := "openai"
	if harness == "pi" {
		provider = "fak"
	}
	args := []string{"--harness", harness, "--workspace", dir, "--prompt-file", prompt, "--receipt", receiptPath,
		"--model", "fixture", "--provider", provider, "--base-url", baseURL + "/v1", "--timeout", "5s"}
	if explicit {
		args = append(args, "--api-key-env", opsRunAuthTestKeyEnv)
	}
	if harness == "pi" {
		args = append(args, "--pi-bin", os.Args[0])
	} else if harness == "opencode" {
		args = append(args, "--opencode-bin", os.Args[0])
	}
	var out, errs bytes.Buffer
	code := runOpsRun(&out, &errs, args)
	b, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("CLI did not persist its preflight result: exit=%d error=%v", code, err)
	}
	var receipt opsRunReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		t.Fatal(err)
	}
	return code, receipt, out.String() + errs.String() + string(b)
}

type opsRunAuthGateway struct {
	*httptest.Server
	mu      sync.Mutex
	headers []string
}

func newOpsRunAuthGateway(t *testing.T, accepted ...string) *opsRunAuthGateway {
	t.Helper()
	g := &opsRunAuthGateway{}
	g.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		g.mu.Lock()
		g.headers = append(g.headers, header)
		g.mu.Unlock()
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Error("preflight did not use the loopback chat route")
		}
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error("malformed preflight request")
		}
		allowed := false
		for _, key := range accepted {
			allowed = allowed || header == "Bearer "+key
		}
		if !allowed {
			// Echo the synthetic credential to exercise response redaction without
			// any real secret or provider request.
			http.Error(w, "refused "+header, http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"model\":%q,\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"fixture\",\"type\":\"function\",\"function\":{\"name\":\"fak_inference_preflight\",\"arguments\":\"{\\\"ok\\\":true}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", body.Model)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(g.Close)
	return g
}

func (g *opsRunAuthGateway) observed() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.headers...)
}

func opsRunAuthNoLeak(t *testing.T, material string, secrets ...string) {
	t.Helper()
	for _, secret := range secrets {
		if secret != "" && strings.Contains(material, secret) {
			t.Error("synthetic credential leaked into output or durable receipt")
		}
		if secret != "" {
			digest := sha256.Sum256([]byte(secret))
			if strings.Contains(material, hex.EncodeToString(digest[:])) {
				t.Error("raw credential fingerprint leaked into output or durable receipt")
			}
		}
	}
}

// fak-test:runtime integration est=3s lane=default
func TestOpsRunPreflightAuthExplicitCredential(t *testing.T) {
	for _, harness := range []string{"pi", "native", "opencode"} {
		t.Run(harness, func(t *testing.T) {
			f := newOpsRunAuthFixture(t)
			key := "fixture-selected-auth-token"
			t.Setenv(opsRunAuthTestKeyEnv, key)
			gateway := newOpsRunAuthGateway(t, key)
			code, receipt, material := f.run(t, harness, gateway.URL, true)
			opsRunAuthNoLeak(t, material, key)
			headers := gateway.observed()
			bound := len(headers) == 1 && headers[0] == "Bearer "+key
			children := f.childDigests(t)
			t.Logf("actual CLI arm=%s exit=%d probes=%d credential_bound=%t children=%d preflight=%+v", harness, code, len(headers), bound, len(children), receipt.InferencePreflight)
			if code != 0 || !bound || receipt.InferencePreflight == nil || receipt.InferencePreflight.Status != "qualified" {
				t.Fatalf("explicit credential failed to qualify the authenticated route for %s", harness)
			}
			digest := sha256.Sum256([]byte(key))
			if len(children) != 1 || children[0] != hex.EncodeToString(digest[:]) {
				t.Fatal("qualified CLI did not launch exactly one isolated child with the selected credential")
			}
		})
	}
}

// fak-test:runtime integration est=3s lane=default
func TestOpsRunPreflightAuthCredentialCacheIsolation(t *testing.T) {
	f := newOpsRunAuthFixture(t)
	good, rotated, bad := "fixture-good-auth-token", "fixture-rotated-auth-token", "fixture-invalid-auth-token"
	gateway := newOpsRunAuthGateway(t, good, rotated)
	for _, step := range []struct {
		name, key string
		explicit  bool
		ok        bool
		probes    int
		children  int
	}{
		{"anonymous", good, false, false, 1, 0},
		{"bad", bad, true, false, 2, 0},
		{"good_after_failures", good, true, true, 3, 1},
		{"good_cache", good, true, true, 3, 2},
		{"bad_cannot_borrow_success", bad, true, false, 3, 2},
		{"rotated_value_same_env_name", rotated, true, true, 4, 3},
		{"anonymous_cannot_borrow_success", good, false, false, 4, 3},
	} {
		t.Run(step.name, func(t *testing.T) {
			t.Setenv(opsRunAuthTestKeyEnv, step.key)
			code, receipt, material := f.run(t, "opencode", gateway.URL, step.explicit)
			opsRunAuthNoLeak(t, material, good, rotated, bad)
			if (code == 0) != step.ok || receipt.InferencePreflight == nil {
				t.Fatalf("credential-scoped qualification mismatch: exit=%d preflight=%+v", code, receipt.InferencePreflight)
			}
			if len(gateway.observed()) != step.probes || len(f.childDigests(t)) != step.children {
				t.Fatalf("credential cache crossed an authorization boundary: probes=%d children=%d", len(gateway.observed()), len(f.childDigests(t)))
			}
			if step.ok {
				if receipt.InferencePreflight.Status != "qualified" {
					t.Fatal("successful launch lacks qualified preflight")
				}
			}
		})
	}
	for _, header := range gateway.observed() {
		if header != "" && header != "Bearer "+good && header != "Bearer "+rotated && header != "Bearer "+bad {
			t.Error("unexpected ambient authorization was forwarded")
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestOpsRunPreflightAuthRefusals(t *testing.T) {
	for _, harness := range []string{"pi", "native", "opencode"} {
		t.Run(harness, func(t *testing.T) {
			f := newOpsRunAuthFixture(t)
			key := "fixture-refused-auth-token"
			t.Setenv(opsRunAuthTestKeyEnv, key)
			t.Setenv("OPENAI_API_KEY", "fixture-ambient-must-not-be-selected")
			gateway := newOpsRunAuthGateway(t)
			for _, named := range []bool{true, false} {
				code, receipt, material := f.run(t, harness, gateway.URL, named)
				opsRunAuthNoLeak(t, material, key, os.Getenv("OPENAI_API_KEY"))
				if code == 0 || len(f.childDigests(t)) != 0 || receipt.InferencePreflight == nil || receipt.InferencePreflight.Status != "failed" || receipt.InferencePreflight.Reason != "http_status_401" {
					t.Fatalf("unauthorized route launched or lost its typed refusal: exit=%d preflight=%+v", code, receipt.InferencePreflight)
				}
			}
			headers := gateway.observed()
			if len(headers) != 2 || headers[0] != "Bearer "+key || headers[1] != "" {
				t.Error("explicit and anonymous probes did not preserve distinct auth contexts")
			}
			for _, value := range []string{"", "   ", "unset"} {
				if value == "unset" {
					if err := os.Unsetenv(opsRunAuthTestKeyEnv); err != nil {
						t.Fatal(err)
					}
				} else {
					t.Setenv(opsRunAuthTestKeyEnv, value)
				}
				code, _, material := f.run(t, harness, gateway.URL, true)
				opsRunAuthNoLeak(t, material, key)
				if code == 0 || len(f.childDigests(t)) != 0 {
					t.Error("empty explicitly selected credential launched a child")
				}
			}
			if len(gateway.observed()) != len(headers) {
				t.Error("missing explicitly selected credential issued an HTTP probe")
			}
		})
	}
}
