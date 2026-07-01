package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestBuiltinComputeTargetsResolveAndValidate proves the four built-ins exist,
// each validates, and resolve is name-keyed (unknown -> not found).
func TestBuiltinComputeTargetsResolveAndValidate(t *testing.T) {
	reg, err := loadComputeTargets("")
	if err != nil {
		t.Fatalf("loadComputeTargets(\"\"): %v", err)
	}
	for _, name := range []string{"mac", "gcp", "local", "anthropic"} {
		tgt, ok := reg.resolve(name)
		if !ok {
			t.Fatalf("built-in target %q not resolvable", name)
		}
		if err := tgt.validate(); err != nil {
			t.Errorf("built-in %q does not validate: %v", name, err)
		}
	}
	if _, ok := reg.resolve("nope"); ok {
		t.Errorf("resolve(nope) unexpectedly found a target")
	}
	// resolve is case-insensitive on the canonical names.
	if _, ok := reg.resolve("MAC"); !ok {
		t.Errorf("resolve(MAC) should match the mac built-in case-insensitively")
	}
}

func TestComputeTargetValidateRejectsBad(t *testing.T) {
	cases := []struct {
		name string
		tgt  computeTarget
	}{
		{"empty name", computeTarget{Kind: targetGatewayURL, GatewayURL: "http://h:1", Locality: localityRemote}},
		{"unknown kind", computeTarget{Name: "x", Kind: "weird", Locality: localityRemote}},
		{"gateway-url without url", computeTarget{Name: "x", Kind: targetGatewayURL, Locality: localityRemote}},
		{"local-spawn without spec", computeTarget{Name: "x", Kind: targetLocalSpawn, Locality: localityLocal}},
		{"bad url scheme", computeTarget{Name: "x", Kind: targetGatewayURL, GatewayURL: "ftp://h", Locality: localityRemote}},
		{"unknown locality", computeTarget{Name: "x", Kind: targetGatewayURL, GatewayURL: "http://h:1", Locality: "elsewhere"}},
		{"cred env is a pasted secret", computeTarget{Name: "x", Kind: targetGatewayURL, GatewayURL: "http://h:1", Locality: localityRemote, CredEnv: "sk-ant-abc123"}},
	}
	for _, c := range cases {
		if err := c.tgt.validate(); err == nil {
			t.Errorf("validate(%s) = nil, want error", c.name)
		}
	}
}

func TestLoadComputeTargetsUserFileAdditive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "targets.json")
	body := `[{"name":"work","kind":"gateway-url","gateway_url":"http://10.0.0.5:8080","model":"qwen3.6","locality":"remote","healthz_path":"/healthz","cred_env":"WORK_GATEWAY_KEY"}]`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	reg, err := loadComputeTargets(path)
	if err != nil {
		t.Fatalf("loadComputeTargets(user file): %v", err)
	}
	// All four built-ins survive...
	for _, name := range []string{"mac", "gcp", "local", "anthropic"} {
		if _, ok := reg.resolve(name); !ok {
			t.Errorf("built-in %q lost after additive load", name)
		}
	}
	// ...and the user target is added on top.
	got, ok := reg.resolve("work")
	if !ok {
		t.Fatalf("user target %q not added", "work")
	}
	if got.GatewayURL != "http://10.0.0.5:8080" || got.CredEnv != "WORK_GATEWAY_KEY" {
		t.Errorf("user target loaded wrong: %+v", got)
	}
	if len(reg.all()) != 5 {
		t.Errorf("registry has %d targets, want 5 (4 built-in + 1 user)", len(reg.all()))
	}
}

func TestLoadComputeTargetsMissingFileIsBuiltinsOnly(t *testing.T) {
	reg, err := loadComputeTargets(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("a missing user file must be tolerated, got: %v", err)
	}
	if len(reg.all()) != 4 {
		t.Errorf("missing file should yield the 4 built-ins, got %d", len(reg.all()))
	}
}

func TestLoadComputeTargetsDuplicateFailsLoud(t *testing.T) {
	dir := t.TempDir()
	// Collision with a built-in name.
	collide := filepath.Join(dir, "collide.json")
	if err := os.WriteFile(collide, []byte(`[{"name":"mac","kind":"gateway-url","gateway_url":"http://h:1","locality":"remote"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadComputeTargets(collide); err == nil {
		t.Errorf("a user target colliding with built-in 'mac' must fail loud")
	}
	// Duplicate within the file itself.
	dup := filepath.Join(dir, "dup.json")
	if err := os.WriteFile(dup, []byte(`[{"name":"a","kind":"gateway-url","gateway_url":"http://h:1","locality":"remote"},{"name":"a","kind":"gateway-url","gateway_url":"http://h:2","locality":"remote"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadComputeTargets(dup); err == nil {
		t.Errorf("a duplicate target within the file must fail loud")
	}
}

func TestLoadComputeTargetsMalformedFailsLoud(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{not json`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadComputeTargets(bad); err == nil {
		t.Errorf("malformed JSON must fail loud")
	}
	// A syntactically valid file with an invalid target also fails loud.
	invalid := filepath.Join(dir, "invalid.json")
	if err := os.WriteFile(invalid, []byte(`[{"name":"x","kind":"gateway-url","locality":"remote"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadComputeTargets(invalid); err == nil {
		t.Errorf("a gateway-url target with no gateway_url must fail loud")
	}
}

// TestComputeTargetDumpHasNoSecret asserts the --json dump carries the credential
// env-var NAME but never the secret value it points at.
func TestComputeTargetDumpHasNoSecret(t *testing.T) {
	const secret = "sk-ant-shouldNeverAppearInAnyDump-000"
	t.Setenv("FAK_GATEWAY_KEY", secret)
	t.Setenv("ANTHROPIC_API_KEY", secret)

	reg, err := loadComputeTargets("")
	if err != nil {
		t.Fatal(err)
	}
	// listing probes; point nowhere reachable so probes return down quickly — the
	// dump must still never contain the secret, only the env-var name.
	rep := reg.listing(context.Background(), &http.Client{Timeout: 200 * time.Millisecond}, 200*time.Millisecond)
	blob, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	dump := string(blob)
	if strings.Contains(dump, secret) {
		t.Fatalf("registry dump leaked a secret value: %s", dump)
	}
	if !strings.Contains(dump, "FAK_GATEWAY_KEY") || !strings.Contains(dump, "ANTHROPIC_API_KEY") {
		t.Errorf("registry dump should carry the cred env-var NAMES; got: %s", dump)
	}
}

// TestProbeReflectsRealHealth proves the probe reads a REAL response: a live 200
// is up, a 500 is down (not a phantom up), an unreachable host is down, and a
// target with no healthz path is n/a.
func TestProbeReflectsRealHealth(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer up.Close()
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer down.Close()

	hc := &http.Client{Timeout: 2 * time.Second}
	ctx := context.Background()

	upTgt := computeTarget{Name: "up", Kind: targetGatewayURL, GatewayURL: up.URL, Locality: localityLocal, HealthzPath: "/healthz"}
	if got := upTgt.probe(ctx, hc); got.State != "up" {
		t.Errorf("live 200 target: state=%q detail=%q, want up", got.State, got.Detail)
	}

	downTgt := computeTarget{Name: "down", Kind: targetGatewayURL, GatewayURL: down.URL, Locality: localityLocal, HealthzPath: "/healthz"}
	if got := downTgt.probe(ctx, hc); got.State != "down" {
		t.Errorf("500 target: state=%q, want down (never a phantom up)", got.State)
	}

	// Closed port: unreachable -> down.
	deadTgt := computeTarget{Name: "dead", Kind: targetGatewayURL, GatewayURL: "http://127.0.0.1:1", Locality: localityLocal, HealthzPath: "/healthz"}
	if got := deadTgt.probe(ctx, hc); got.State != "down" {
		t.Errorf("unreachable target: state=%q, want down", got.State)
	}

	// No healthz path -> n/a (the real Anthropic API case).
	naTgt := computeTarget{Name: "na", Kind: targetProviderProxy, GatewayURL: "https://api.anthropic.com", Locality: localityRemote}
	if got := naTgt.probe(ctx, hc); got.State != "n/a" {
		t.Errorf("no-healthz target: state=%q, want n/a", got.State)
	}
}

// TestRenderComputeTargetTableKeepsHealthColumnCompact proves a verbose down
// detail (the Windows "connectex … actively refused it" dial error) never lands in
// the aligned HEALTH column — where it would pad every row and shove `up`/MODEL
// off-screen — but is still surfaced, verbatim, in the health-notes block.
func TestRenderComputeTargetTableKeepsHealthColumnCompact(t *testing.T) {
	longDetail := `Get "http://127.0.0.1:8200/health": dial tcp 127.0.0.1:8200: ` +
		`connectex: No connection could be made because the target machine actively refused it.`
	rep := targetListReport{
		Schema: computeTargetListSchema,
		Targets: []targetListing{
			{Name: "mac", Kind: targetGatewayURL, GatewayURL: "http://mac:8080", Locality: localityRemote,
				Model: "qwen3.6-27b", Health: targetHealth{State: "up"}},
			{Name: "gcp", Kind: targetGatewayURL, GatewayURL: "http://127.0.0.1:8200/v1", Locality: localityRemote,
				Model: "glm-5.2", Health: targetHealth{State: "down", Detail: longDetail}},
		},
	}
	var buf strings.Builder
	renderComputeTargetTable(&buf, rep)
	out := buf.String()

	// The table body (everything before the notes block) must NOT carry the long
	// dial error — that is what blew out the column width.
	table := out
	if i := strings.Index(out, "health notes:"); i >= 0 {
		table = out[:i]
	}
	if strings.Contains(table, "actively refused") {
		t.Errorf("HEALTH column leaked the verbose dial error into the aligned table:\n%s", table)
	}

	// The mac row's `up` must sit close to its columns, not be shoved right by a
	// sibling row's long detail: the mac line stays comfortably narrow.
	for _, line := range strings.Split(table, "\n") {
		if strings.HasPrefix(line, "mac") && len(line) > 100 {
			t.Errorf("mac row is %d cols wide — a down row's detail is still bleeding into alignment:\n%q", len(line), line)
		}
	}

	// The reason is not dropped: it appears once, verbatim, in the notes block.
	if !strings.Contains(out, "health notes:") || !strings.Contains(out, longDetail) {
		t.Errorf("down target's detail must survive in the health-notes block; got:\n%s", out)
	}
}

// TestHealthzURLUsesOrigin proves the probe targets /healthz at the gateway ORIGIN,
// not appended to a /v1 base (the gcp glm case).
func TestHealthzURLUsesOrigin(t *testing.T) {
	tgt := computeTarget{Kind: targetGatewayURL, GatewayURL: "http://127.0.0.1:8200/v1", HealthzPath: "/health"}
	got, ok := tgt.healthzURL()
	if !ok {
		t.Fatal("healthzURL should resolve for a gateway-url target")
	}
	if got != "http://127.0.0.1:8200/health" {
		t.Errorf("healthzURL = %q, want origin-joined http://127.0.0.1:8200/health", got)
	}
}
