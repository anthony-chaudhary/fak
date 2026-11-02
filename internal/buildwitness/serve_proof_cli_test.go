package buildwitness

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// Bound subprocess diagnostics without altering the child exit status.
type serveProofCLIBuffer struct {
	buf       bytes.Buffer
	truncated bool
}

func (b *serveProofCLIBuffer) Len() int       { return b.buf.Len() }
func (b *serveProofCLIBuffer) String() string { return b.buf.String() }
func (b *serveProofCLIBuffer) Bytes() []byte  { return b.buf.Bytes() }

func (b *serveProofCLIBuffer) Write(p []byte) (int, error) {
	n := len(p)
	left := (1 << 20) - b.Len()
	if len(p) > left {
		b.truncated = true
		p = p[:left]
	}
	_, _ = b.buf.Write(p)
	return n, nil
}

// fak-test:justify why=regression when=changed:cmd/fak/serve_proof.go,cmd/fak/doctor_serve.go,cmd/fak/main.go
// fak-test:runtime integration est=5m lane=default
// TestServeProofCLIIntegration builds the actual production CLI, not cmd/fak's
// thousands of unrelated tests. Native validation runs this test against the
// complete immutable feature candidate. It uses only pre-existing fixture APIs,
// so a parent diagnostic can distinguish missing behavior from setup failure.
func TestServeProofCLIIntegration(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("CLI setup failed [setup failed]: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("CLI setup failed [setup failed]: %v", err)
	}
	binary := filepath.Join(t.TempDir(), "fak-feature-contract")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	buildCtx, buildCancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer buildCancel()
	build := exec.CommandContext(buildCtx, "go", "build", "-o", binary, "./cmd/fak")
	build.Dir = root
	procguard.ConfigureProcessTreeCancel(build)
	build.WaitDelay = 10 * time.Second
	var buildOutput serveProofCLIBuffer
	build.Stdout = &buildOutput
	build.Stderr = &buildOutput
	if err := build.Run(); err != nil {
		t.Fatalf("CLI compiler error (not behavioral RED): %v\n%s", err, buildOutput.String())
	}
	if buildOutput.truncated {
		t.Fatal("CLI compiler error: diagnostic output exceeded bound; proof incomplete")
	}
	t.Log("actual production CLI build passed")
	home := t.TempDir()
	runtimeEnv := []string{"PATH=" + os.Getenv("PATH"), "HOME=" + home, "USERPROFILE=" + home, "XDG_CONFIG_HOME=" + home, "XDG_CACHE_HOME=" + home, "TMPDIR=" + home, "TMP=" + home, "TEMP=" + home}
	for _, key := range []string{"GOMAXPROCS", "GOMEMLIMIT", "GOGC"} {
		if value, ok := os.LookupEnv(key); ok {
			runtimeEnv = append(runtimeEnv, key+"="+value)
		}
	}
	run := func(args ...string) ([]byte, int) {
		t.Helper()
		ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, args...)
		procguard.ConfigureProcessTreeCancel(cmd)
		cmd.WaitDelay = 10 * time.Second
		cmd.Dir = home
		cmd.Env = runtimeEnv
		var out, stderr serveProofCLIBuffer
		cmd.Stdout = &out
		cmd.Stderr = &stderr
		err := cmd.Run()
		if ctx.Err() != nil {
			t.Fatalf("CLI setup failed [setup failed]: command timeout %v", args)
		}
		code := 0
		if err != nil {
			if e, ok := err.(*exec.ExitError); ok {
				code = e.ExitCode()
			} else {
				t.Fatalf("CLI setup failed [setup failed]: %v", err)
			}
		}
		if code < 0 || out.truncated || stderr.truncated {
			t.Fatalf("CLI setup failed [setup failed]: exit=%d bounded-output-truncated=%v stderr=%s", code, out.truncated || stderr.truncated, stderr.String())
		}
		for _, marker := range []string{"panic:", "fatal error:", "runtime:", "error while loading shared libraries", "exec format error"} {
			if strings.Contains(stderr.String(), marker) {
				t.Fatalf("CLI setup failed [setup failed]: exit=%d stderr=%s", code, stderr.String())
			}
		}
		t.Logf("actual CLI %v exit=%d stderr=%s", args, code, stderr.String())
		return out.Bytes(), code
	}
	collector, err := gateway.NewFeatureProofCollector(4)
	if err != nil {
		t.Fatal(err)
	}
	if err := collector.RecordRewrite(gateway.FeatureCompactHistory, []byte("prefix-cut-suffix"), []byte("prefix-suffix"), gateway.FeatureRewriteWitness{PrefixBytes: 7, SuffixBytes: 6, ShedBytes: 4}); err != nil {
		t.Fatal(err)
	}
	catalog := gateway.FeatureCatalog{Schema: gateway.FeatureSchema, Features: []gateway.FeatureStatus{
		{Feature: gateway.FeatureCompactHistory, State: gateway.FeatureConfiguredActive},
		{Feature: gateway.FeatureElideResults, State: gateway.FeatureConfiguredStandby},
		{Feature: gateway.FeatureVDSO, State: gateway.FeatureDisabled},
		{Feature: gateway.FeatureNativeModel, State: gateway.FeatureRefusedUnavailable},
	}}
	var catalogRequests, proofRequests, badRequests atomic.Int32
	var unsupported atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.Header.Get("Authorization") != "" {
			badRequests.Add(1)
			w.WriteHeader(405)
			return
		}
		switch r.URL.Path {
		case "/v1/fak/features":
			catalogRequests.Add(1)
			_ = json.NewEncoder(w).Encode(catalog)
		case "/v1/fak/features/proof":
			proofRequests.Add(1)
			if unsupported.Load() {
				w.WriteHeader(404)
				return
			}
			_ = json.NewEncoder(w).Encode(collector.Snapshot())
		default:
			badRequests.Add(1)
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()
	type row struct {
		Feature string `json:"feature"`
		Verdict string `json:"verdict"`
		Receipt *struct {
			SavedBytes *int `json:"saved_bytes"`
		} `json:"receipt"`
	}
	type proof struct {
		Code string `json:"code"`
		Rows []row  `json:"rows"`
	}
	type doctor struct {
		Rows []struct {
			Check string `json:"check"`
		} `json:"rows"`
		FeatureProof *proof `json:"feature_proof"`
	}
	// The existing command must work on both refs, before testing the new contract.
	baseline, code := run("doctor", "serve", "--json")
	var baselineReport doctor
	if code != 0 || json.Unmarshal(baseline, &baselineReport) != nil || len(baselineReport.Rows) == 0 || baselineReport.FeatureProof != nil {
		t.Fatalf("existing doctor setup failed [setup failed]: exit=%d output=%s", code, baseline)
	}
	if catalogRequests.Load() != 0 || proofRequests.Load() != 0 {
		t.Fatal("default doctor unexpectedly contacted fixture")
	}
	assertProof := func(p *proof) {
		t.Helper()
		if p == nil || p.Code != "FEATURE_PROOF_OBSERVED" {
			t.Fatalf("missing typed feature observation: %+v", p)
		}
		want := map[string]string{"compact_history": "VERIFIED_ACTIVE", "elide_results": "ENABLED_IDLE", "vdso": "DISABLED", "native_model": "DEGRADED_OR_BLOCKED"}
		if len(p.Rows) != len(want) {
			t.Fatalf("feature row count=%d want=%d", len(p.Rows), len(want))
		}
		for _, r := range p.Rows {
			expected, known := want[r.Feature]
			if !known || expected != r.Verdict {
				t.Fatalf("unexpected/duplicate feature %s=%s", r.Feature, r.Verdict)
			}
			delete(want, r.Feature)
			if r.Feature == "compact_history" && (r.Receipt == nil || r.Receipt.SavedBytes == nil || *r.Receipt.SavedBytes != 4) {
				t.Fatalf("actual byte-saving evidence missing: %+v", r)
			}
		}
		if len(want) != 0 {
			t.Fatalf("missing feature verdicts: %v", want)
		}
	}
	observed, code := run("doctor", "serve", "--json", "--verify-features", "--addr", srv.URL, "--feature-timeout", "2s")
	var d doctor
	// Parent's unsupported flags are not counted merely as a nonzero exit. The
	// failed contract is absence of both exact endpoint reads and typed results.
	if catalogRequests.Load() != 1 || proofRequests.Load() != 1 || badRequests.Load() != 0 || json.Unmarshal(observed, &d) != nil || d.FeatureProof == nil {
		t.Fatalf("feature consumer behavior missing: catalog=%d proof=%d bad=%d exit=%d output=%s", catalogRequests.Load(), proofRequests.Load(), badRequests.Load(), code, observed)
	}
	if code != 1 {
		t.Fatalf("degraded feature must affect doctor exit: %d", code)
	}
	assertProof(d.FeatureProof)
	// Only a successful doctor consumer reaches the new serve subcommand. This
	// prevents an old CLI from accidentally treating 'proof' as server startup.
	raw, code := run("serve", "proof", "--json", "--addr", srv.URL, "--timeout", "2s")
	var p proof
	if code != 1 || json.Unmarshal(raw, &p) != nil {
		t.Fatalf("serve proof contract: exit=%d output=%s", code, raw)
	}
	assertProof(&p)
	table, code := run("serve", "proof", "--addr", srv.URL, "--timeout", "2s")
	if code != 1 {
		t.Fatalf("serve proof table exit=%d", code)
	}
	for _, label := range []string{"VERIFIED_ACTIVE", "ENABLED_IDLE", "DISABLED", "DEGRADED_OR_BLOCKED"} {
		if !strings.Contains(string(table), label) {
			t.Fatalf("table missing %s: %s", label, table)
		}
	}
	unsupported.Store(true)
	old, code := run("doctor", "serve", "--json", "--verify-features", "--addr", srv.URL, "--feature-timeout", "2s")
	var oldDoctor doctor
	if code != 0 || json.Unmarshal(old, &oldDoctor) != nil || oldDoctor.FeatureProof == nil || oldDoctor.FeatureProof.Code != "FEATURE_PROOF_UNSUPPORTED" {
		t.Fatalf("404 fallback: exit=%d output=%s", code, old)
	}
	hardware := false
	for _, r := range oldDoctor.Rows {
		hardware = hardware || r.Check == "cpu-isa"
	}
	if !hardware {
		t.Fatal("404 discarded hardware doctor")
	}
	before := catalogRequests.Load() + proofRequests.Load()
	plain, code := run("doctor", "serve", "--json")
	if code != 0 || !json.Valid(plain) || catalogRequests.Load()+proofRequests.Load() != before {
		t.Fatal("default doctor gained network dependency")
	}
	if catalogRequests.Load() != 4 || proofRequests.Load() != 4 || badRequests.Load() != 0 {
		t.Fatal(fmt.Sprintf("unexpected endpoint consumption: catalog=%d proof=%d bad=%d", catalogRequests.Load(), proofRequests.Load(), badRequests.Load()))
	}
}
