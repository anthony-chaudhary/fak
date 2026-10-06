package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

const coordinatorTestKey = "coordinator-test-secret"

const coordinatorTestSource = "refs/fak/locks/* on the coordinator clone (single-arbiter fenced write, serialized through the gateway)"

func configureLeaseCoordinator(t *testing.T, url string) {
	t.Helper()
	keyFile := filepath.Join(t.TempDir(), "coordinator.key")
	if err := os.WriteFile(keyFile, []byte(coordinatorTestKey+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_LEASE_COORDINATOR_URL", url)
	t.Setenv("FAK_LEASE_COORDINATOR_KEY_FILE", keyFile)
}

func coordinatorFallbackRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	dir := t.TempDir()
	cmd := exec.Command("git", "init", "-q")
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	return dir
}

func runCoordinatorLeaseref(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runLeaseref(&stdout, &stderr, args)
	return code, stdout.String(), stderr.String()
}

func serveCoordinatorAuthProof(t *testing.T, w http.ResponseWriter, r *http.Request) bool {
	t.Helper()
	if r.URL.Path != "/healthz" && r.URL.Path != "/v1/fak/key-proof" {
		return false
	}
	if r.Method != http.MethodGet || len(r.Header.Values("Authorization")) != 0 {
		t.Errorf("proof request must be an unauthenticated GET: method=%s Authorization=%q", r.Method, r.Header.Values("Authorization"))
		http.Error(w, "invalid proof request", http.StatusBadRequest)
		return true
	}
	nonce, err := base64.StdEncoding.DecodeString(r.Header.Get("X-Fak-Auth-Challenge"))
	if err != nil || len(nonce) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST router auth challenge nonce is a fixed 32-byte protocol width
		http.Error(w, "bad challenge", http.StatusBadRequest)
		return true
	}
	mac := hmac.New(sha256.New, []byte(coordinatorTestKey))
	_, _ = mac.Write([]byte("fak-health-v1\x00"))
	_, _ = mac.Write(nonce)
	w.Header().Set("X-Fak-Auth-Proof", base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	w.WriteHeader(http.StatusOK)
	return true
}

func requireCoordinatorLeaseWrite(t *testing.T, w http.ResponseWriter, r *http.Request, op string) bool {
	t.Helper()
	if r.Method != http.MethodPost || r.URL.Path != "/v1/leases/"+op || r.Header.Get("Authorization") != "Bearer "+coordinatorTestKey {
		t.Errorf("expected authenticated lease POST for %s: method=%s path=%s Authorization=%q", op, r.Method, r.URL.Path, r.Header.Get("Authorization"))
		http.Error(w, "invalid lease request", http.StatusBadRequest)
		return false
	}
	return true
}

func TestLeaserefCoordinatorAcquireRenewReleaseUsesAuthenticatedAuthority(t *testing.T) {
	type observed struct {
		path string
		req  gateway.LeaseWriteRequest
	}
	var calls []observed
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveCoordinatorAuthProof(t, w, r) {
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+coordinatorTestKey {
			t.Errorf("Authorization = %q", got)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		var req gateway.LeaseWriteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		calls = append(calls, observed{path: r.URL.Path, req: req})
		op := strings.TrimPrefix(r.URL.Path, "/v1/leases/")
		result := gateway.LeaseWriteResult{
			OK: true, Op: op, ID: req.ID,
			Holder: req.Holder, Generation: 17, CurrentGeneration: 17, TreeGlobs: req.TreeGlobs,
			ObservedUnix: 1, Source: coordinatorTestSource,
		}
		if op == "release" {
			result.Generation = 0
			result.TreeGlobs = nil
		}
		_ = json.NewEncoder(w).Encode(result)
	}))
	defer server.Close()
	configureLeaseCoordinator(t, server.URL)

	assertVerdictOnly := func(op, stdout string) {
		t.Helper()
		var got fencedResult
		if err := json.Unmarshal([]byte(stdout), &got); err != nil {
			t.Fatalf("%s decode stdout: %v; stdout=%q", op, err, stdout)
		}
		if !got.Verdict.OK || got.Verdict.Presented != 17 || got.Verdict.Current != 17 {
			t.Fatalf("%s verdict = %+v, want usable fencing token 17", op, got.Verdict)
		}
		if got.Record != nil {
			t.Fatalf("%s fabricated local record from partial coordinator wire: %+v", op, got.Record)
		}
	}
	code, stdout, stderr := runCoordinatorLeaseref(t, "acquire", "--id", "coord-lane", "--holder", "worker-a", "--ttl", "300", "--tree", "cmd/fak/**")
	if code != 0 {
		t.Fatalf("acquire exit=%d stderr=%q", code, stderr)
	}
	assertVerdictOnly("acquire", stdout)
	code, stdout, stderr = runCoordinatorLeaseref(t, "renew", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17", "--ttl", "600")
	if code != 0 {
		t.Fatalf("renew exit=%d stderr=%q", code, stderr)
	}
	assertVerdictOnly("renew", stdout)
	if code, _, stderr := runCoordinatorLeaseref(t, "release", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17"); code != 0 {
		t.Fatalf("release exit=%d stderr=%q", code, stderr)
	}

	wantPaths := []string{"/v1/leases/acquire", "/v1/leases/renew", "/v1/leases/release"}
	if len(calls) != len(wantPaths) {
		t.Fatalf("coordinator calls=%+v", calls)
	}
	for i, want := range wantPaths {
		if calls[i].path != want || calls[i].req.ID != "coord-lane" || calls[i].req.Holder != "worker-a" {
			t.Fatalf("call %d = %+v, want path=%s", i, calls[i], want)
		}
	}
	if calls[0].req.TTLSeconds != 300 || len(calls[0].req.TreeGlobs) != 1 || calls[0].req.TreeGlobs[0] != "cmd/fak/**" {
		t.Fatalf("acquire request = %+v", calls[0].req)
	}
	if calls[1].req.Generation != 17 || calls[1].req.TTLSeconds != 600 || calls[2].req.Generation != 17 {
		t.Fatalf("fencing tokens lost: renew=%+v release=%+v", calls[1].req, calls[2].req)
	}
}

func TestLeaserefCoordinatorRefusalAndServerErrorKeepDistinctExits(t *testing.T) {
	for _, tc := range []struct {
		name       string
		status     int
		body       string
		wantExit   int
		wantStdout string
	}{
		{"refused", http.StatusOK, `{"ok":false,"reason":"LEASE_HELD","op":"acquire","id":"coord-lane","current_generation":9,"holder":"peer","observed_unix":1,"source":"refs/fak/locks/* on the coordinator clone (single-arbiter fenced write, serialized through the gateway)"}`, leaserefRefused, "LEASE_HELD"},
		{"server error", http.StatusInternalServerError, `{"error":"unavailable"}`, 1, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var leaseWrites atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if serveCoordinatorAuthProof(t, w, r) {
					return
				}
				if !requireCoordinatorLeaseWrite(t, w, r, "acquire") {
					return
				}
				leaseWrites.Add(1)
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			configureLeaseCoordinator(t, server.URL)
			code, stdout, _ := runCoordinatorLeaseref(t, "acquire", "--id", "coord-lane", "--holder", "worker-a", "--ttl", "300", "--tree", "cmd/fak/**")
			if got := leaseWrites.Load(); got != 1 {
				t.Fatalf("lease write calls=%d, want 1; proof setup failure must not satisfy the exit assertion", got)
			}
			if code != tc.wantExit || (tc.wantStdout != "" && !strings.Contains(stdout, tc.wantStdout)) {
				t.Fatalf("exit=%d stdout=%q, want exit=%d containing %q", code, stdout, tc.wantExit, tc.wantStdout)
			}
		})
	}
}

func TestLeaserefCoordinatorAuthOrTransportFailureNeverFallsBackLocal(t *testing.T) {
	for _, tc := range []struct {
		name string
		url  func(t *testing.T) string
	}{
		{"authentication", func(t *testing.T) string {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "unauthorized", http.StatusUnauthorized) }))
			t.Cleanup(s.Close)
			return s.URL
		}},
		{"transport", func(t *testing.T) string {
			s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
			url := s.URL
			s.Close()
			return url
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := coordinatorFallbackRepo(t)
			t.Chdir(dir)
			configureLeaseCoordinator(t, tc.url(t))
			code, _, _ := runCoordinatorLeaseref(t, "acquire", "--id", "must-not-land-local", "--holder", "worker-a", "--ttl", "300", "--tree", "cmd/fak/**")
			if code != 1 {
				t.Fatalf("coordinator failure exit=%d, want infrastructure exit 1", code)
			}
			if _, ok, err := leaseref.NewInDir(dir).Get(context.Background(), "must-not-land-local"); err != nil || ok {
				t.Fatalf("local fallback mutated store: present=%v err=%v", ok, err)
			}
		})
	}
}

func TestLeaserefCoordinatorRejectsLocalOnlyForceAndFence(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		serveCoordinatorAuthProof(t, w, r)
	}))
	defer server.Close()
	configureLeaseCoordinator(t, server.URL)

	for _, args := range [][]string{
		{"release", "--id", "coord-lane", "--force"},
		{"fence", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17"},
	} {
		if code, _, _ := runCoordinatorLeaseref(t, args...); code != 2 {
			t.Fatalf("runLeaseref(%v) exit=%d, want usage refusal 2", args, code)
		}
	}
	if calls != 0 {
		t.Fatalf("local-only operations reached coordinator %d time(s)", calls)
	}
}

func TestLeaserefCoordinatorRejectsExplicitDirBeforeHTTP(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		serveCoordinatorAuthProof(t, w, r)
	}))
	defer server.Close()
	configureLeaseCoordinator(t, server.URL)

	for _, args := range [][]string{
		{"acquire", "--id", "coord-lane", "--holder", "worker-a", "--ttl", "300", "--tree", "cmd/fak/**", "--dir", t.TempDir()},
		{"renew", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17", "--dir", t.TempDir()},
		{"release", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17", "--dir", t.TempDir()},
	} {
		if code, _, _ := runCoordinatorLeaseref(t, args...); code != 2 {
			t.Fatalf("runLeaseref(%v) exit=%d, want usage refusal 2", args, code)
		}
	}
	if calls != 0 {
		t.Fatalf("explicit --dir reached coordinator %d time(s)", calls)
	}
}

func TestLeaserefCoordinatorRejectsMalformedAuthorityVerdicts(t *testing.T) {
	validAcquire := gateway.LeaseWriteResult{
		OK: true, Op: "acquire", ID: "coord-lane", Holder: "worker-a", Generation: 17,
		CurrentGeneration: 17, TreeGlobs: []string{"cmd/fak/**"}, ObservedUnix: 1, Source: coordinatorTestSource,
	}
	validRenew := gateway.LeaseWriteResult{
		OK: true, Op: "renew", ID: "coord-lane", Holder: "worker-a", Generation: 17,
		CurrentGeneration: 17, TreeGlobs: []string{"cmd/fak/**"}, ObservedUnix: 1, Source: coordinatorTestSource,
	}
	validRelease := gateway.LeaseWriteResult{
		OK: true, Op: "release", ID: "coord-lane", Holder: "worker-a",
		CurrentGeneration: 17, ObservedUnix: 1, Source: coordinatorTestSource,
	}
	encode := func(v gateway.LeaseWriteResult) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	acquireArgs := []string{"acquire", "--id", "coord-lane", "--holder", "worker-a", "--ttl", "300", "--tree", "cmd/fak/**"}
	renewArgs := []string{"renew", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17", "--ttl", "600"}
	releaseArgs := []string{"release", "--id", "coord-lane", "--holder", "worker-a", "--generation", "17"}
	for _, tc := range []struct {
		name string
		args []string
		body string
	}{
		{"acquire wrong holder", acquireArgs, func() string { v := validAcquire; v.Holder = "peer"; return encode(v) }()},
		{"acquire empty holder", acquireArgs, func() string { v := validAcquire; v.Holder = ""; return encode(v) }()},
		{"acquire wrong tree", acquireArgs, func() string { v := validAcquire; v.TreeGlobs = []string{"internal/**"}; return encode(v) }()},
		{"acquire empty tree", acquireArgs, func() string { v := validAcquire; v.TreeGlobs = nil; return encode(v) }()},
		{"accepted reason", acquireArgs, func() string { v := validAcquire; v.Reason = "LEASE_HELD"; return encode(v) }()},
		{"acquire current generation mismatch", acquireArgs, func() string { v := validAcquire; v.CurrentGeneration = 16; return encode(v) }()},
		{"renew unrelated generation", renewArgs, func() string { v := validRenew; v.Generation, v.CurrentGeneration = 99, 99; return encode(v) }()},
		{"renew current generation mismatch", renewArgs, func() string { v := validRenew; v.CurrentGeneration = 16; return encode(v) }()},
		{"release wrong holder", releaseArgs, func() string { v := validRelease; v.Holder = "peer"; return encode(v) }()},
		{"release nonzero generation", releaseArgs, func() string { v := validRelease; v.Generation = 17; return encode(v) }()},
		{"release wrong current generation", releaseArgs, func() string { v := validRelease; v.CurrentGeneration = 999; return encode(v) }()},
		{"release carries tree", releaseArgs, func() string { v := validRelease; v.TreeGlobs = []string{"cmd/fak/**"}; return encode(v) }()},
		{"trailing JSON", acquireArgs, encode(validAcquire) + ` {"ok":true}`},
		{"oversize response", acquireArgs, encode(validAcquire) + strings.Repeat(" ", 70<<10)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var leaseWrites atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if serveCoordinatorAuthProof(t, w, r) {
					return
				}
				if !requireCoordinatorLeaseWrite(t, w, r, tc.args[0]) {
					return
				}
				leaseWrites.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			configureLeaseCoordinator(t, server.URL)
			code, stdout, stderr := runCoordinatorLeaseref(t, tc.args...)
			if got := leaseWrites.Load(); got != 1 {
				t.Fatalf("lease write calls=%d, want 1; proof setup failure must not satisfy the exit assertion", got)
			}
			if code != 1 {
				t.Fatalf("malformed authority verdict accepted: exit=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
		})
	}
}
