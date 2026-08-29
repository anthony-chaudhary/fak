package nativeperfcoverage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type acceptingChecker struct {
	calls int
}

func (c *acceptingChecker) Check(_ context.Context, expr string) error {
	c.calls++
	if strings.TrimSpace(expr) == "" {
		return errors.New("empty expression")
	}
	return nil
}

func TestFullMatrix(t *testing.T) {
	checker := &acceptingChecker{}
	matrix, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: checker})
	if err != nil {
		t.Fatal(err)
	}
	if len(matrix.Dashboards) != 4 {
		t.Fatalf("dashboards=%d, want 4", len(matrix.Dashboards))
	}
	var targets, annotations, variables int
	for _, dashboard := range matrix.Dashboards {
		for _, panel := range dashboard.Panels {
			targets += len(panel.Queries)
		}
		for _, query := range dashboard.Queries {
			switch query.Kind {
			case PanelTarget:
				t.Fatalf("panel target %q escaped dashboard.Panels", query.Name)
			case Annotation:
				annotations++
			case Variable:
				variables++
			}
		}
	}
	if targets != 50 || annotations != 1 || variables != 7 {
		t.Fatalf("inventory targets=%d annotations=%d variables=%d, want 50/1/7", targets, annotations, variables)
	}
	if checker.calls != targets+annotations+variables {
		t.Fatalf("checked queries=%d, want %d", checker.calls, targets+annotations+variables)
	}
	text := matrix.Text()
	if !strings.Contains(text, "DASHBOARD\tfak-native-kernel-performance") || !strings.Contains(text, "PANEL\tfak-native-slo\t14") {
		t.Fatalf("matrix omits complete endpoints:\n%s", text)
	}
	checker2 := &acceptingChecker{}
	again, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: checker2})
	if err != nil {
		t.Fatal(err)
	}
	if text != again.Text() {
		t.Fatal("matrix is not deterministic")
	}
	t.Log("deterministic native dashboard coverage matrix:\n" + text)
}

func TestDefaultPromtoolChecksEveryQuery(t *testing.T) {
	path, err := exec.LookPath("promtool")
	if err != nil {
		t.Skip("promtool not installed")
	}
	if _, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: PromtoolChecker{Path: path}}); err != nil {
		t.Fatal(err)
	}
}

func TestPromtoolEvaluatesRateAndHistogramAgainstControlledData(t *testing.T) {
	path, err := exec.LookPath("promtool")
	if err != nil {
		t.Skip("promtool not installed")
	}
	if err := EvaluateControlledPromQL(context.Background(), path); err != nil {
		t.Fatal(err)
	}
}

func TestPromtoolEvaluatesEveryExtractedQueryAgainstControlledData(t *testing.T) {
	path, err := exec.LookPath("promtool")
	if err != nil {
		t.Skip("promtool not installed")
	}
	matrix, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: &acceptingChecker{}})
	if err != nil {
		t.Fatal(err)
	}
	if err := EvaluateMatrixControlledPromQL(context.Background(), repoRoot(t), path, matrix); err != nil {
		t.Fatal(err)
	}
}

func TestMutationRenamedMetric(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, Specs()[0].Dashboard)
	replaceFile(t, path, "fak_native_receipt_requests_total{", "fak_native_receipt_requests_renamed{")
	_, err := Validate(context.Background(), Config{Root: root, Checker: &acceptingChecker{}})
	assertErrorContains(t, err, "unknown or renamed metric")
}

func TestMutationMissingSupervisedJob(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, "tools/grafana/prometheus.yml")
	replaceFile(t, path, "job_name: fak_gateway", "job_name: renamed_gateway")
	_, err := Validate(context.Background(), Config{Root: root, Checker: &acceptingChecker{}})
	assertErrorContains(t, err, `required supervised Prometheus job "fak_gateway" is absent`)
}

func TestMutationQueryFailure(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, Specs()[0].Dashboard)
	replaceFile(t, path, "sum(rate(fak_native_receipt_requests_total{", "unsupported(sum(rate(fak_native_receipt_requests_total{")
	checker := QueryCheckerFunc(func(_ context.Context, expr string) error {
		if strings.Contains(expr, "unsupported(") {
			return errors.New("unsupported expression")
		}
		return nil
	})
	_, err := Validate(context.Background(), Config{Root: root, Checker: checker})
	assertErrorContains(t, err, "PromQL query error: unsupported expression")
}

func TestMutationZeroCoercion(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, Specs()[0].Dashboard)
	replaceFile(t, path, "vector(-1)", "vector(0)")
	_, err := Validate(context.Background(), Config{Root: root, Checker: &acceptingChecker{}})
	assertErrorContains(t, err, "zero coercion is forbidden")
}

func TestMutationUnavailableRenderedDishonestly(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, Specs()[3].Dashboard)
	var dashboard map[string]any
	readJSONTest(t, path, &dashboard)
	panels := dashboard["panels"].([]any)
	panel := panels[12].(map[string]any) // panel 13 uses fixture-absent fak_native_slo_ratio.
	field := panel["fieldConfig"].(map[string]any)
	defaults := field["defaults"].(map[string]any)
	defaults["noValue"] = "0"
	writeJSONTest(t, path, dashboard)
	fixturePath := filepath.Join(root, Specs()[3].Fixture)
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var kept []string
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "fak_native_slo_ratio{") {
			kept = append(kept, line)
		}
	}
	if err := os.WriteFile(fixturePath, []byte(strings.Join(kept, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Validate(context.Background(), Config{Root: root, Checker: &acceptingChecker{}})
	assertErrorContains(t, err, "missing fixture series would render dishonestly")
}

func TestFixtureBoundedCorrelation(t *testing.T) {
	root := fixtureRepo(t)
	path := filepath.Join(root, Specs()[2].Fixture)
	replaceFile(t, path, "npc1_0123456789abcdef0123456789abcdef", "private-request-path-123")
	_, err := Validate(context.Background(), Config{Root: root, Checker: &acceptingChecker{}})
	assertErrorContains(t, err, "unbounded correlation_key")
}

func TestLiveUnavailableWitnessIsPending(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tools/grafana/provisioning/witnesses/fak-native-qwen38-live-proof-unavailable.json"))
	if err != nil {
		t.Fatal(err)
	}
	state, err := ValidateLiveEvidence(raw, LiveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if state != LivePending {
		t.Fatalf("state=%q, want %q", state, LivePending)
	}
}

func TestLiveReceiptSuccess(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 20, 0, 0, time.UTC)
	raw := successfulLiveProof(t, now.Add(-2*time.Minute), NativeEngine)
	state, err := ValidateLiveEvidence(raw, LiveOptions{Now: now, MaxAge: 5 * time.Minute, RequiredJobs: []string{"native-serve"}})
	if err != nil {
		t.Fatal(err)
	}
	if state != LiveProven {
		t.Fatalf("state=%q, want %q", state, LiveProven)
	}
}

func TestMutationStaleLiveReceipt(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 20, 0, 0, time.UTC)
	raw := successfulLiveProof(t, now.Add(-16*time.Minute), NativeEngine)
	_, err := ValidateLiveEvidence(raw, LiveOptions{Now: now, MaxAge: 15 * time.Minute})
	assertErrorContains(t, err, "live proof is stale")
}

func TestMutationWrongEngine(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 20, 0, 0, time.UTC)
	raw := successfulLiveProof(t, now.Add(-time.Minute), "llama.cpp")
	_, err := ValidateLiveEvidence(raw, LiveOptions{Now: now})
	assertErrorContains(t, err, "want exact fak-native")
}

func TestMutationWrongModel(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 20, 0, 0, time.UTC)
	var proof map[string]any
	if err := json.Unmarshal(successfulLiveProof(t, now.Add(-time.Minute), NativeEngine), &proof); err != nil {
		t.Fatal(err)
	}
	proof["model"].(map[string]any)["family"] = "Qwen3.6"
	raw, err := json.Marshal(proof)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ValidateLiveEvidence(raw, LiveOptions{Now: now})
	assertErrorContains(t, err, "want exact Qwen3.8")
}

func TestMutationMissingReceiptJob(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 20, 0, 0, time.UTC)
	raw := successfulLiveProof(t, now.Add(-time.Minute), NativeEngine)
	_, err := ValidateLiveEvidence(raw, LiveOptions{Now: now, RequiredJobs: []string{"missing-job"}})
	assertErrorContains(t, err, "required supervised job")
}

func TestAggregatePNGsAreDeterministicAndDistinct(t *testing.T) {
	matrix, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: &acceptingChecker{}})
	if err != nil {
		t.Fatal(err)
	}
	var populated, populatedAgain, unavailable bytes.Buffer
	if err := RenderAggregatePNG(&populated, matrix, RenderPopulated); err != nil {
		t.Fatal(err)
	}
	if err := RenderAggregatePNG(&populatedAgain, matrix, RenderPopulated); err != nil {
		t.Fatal(err)
	}
	if err := RenderAggregatePNG(&unavailable, matrix, RenderUnavailable); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(populated.Bytes(), populatedAgain.Bytes()) {
		t.Fatal("populated PNG is not byte-for-byte deterministic")
	}
	if bytes.Equal(populated.Bytes(), unavailable.Bytes()) {
		t.Fatal("populated and unavailable PNGs are identical")
	}
	for name, raw := range map[string][]byte{"populated": populated.Bytes(), "unavailable": unavailable.Bytes()} {
		image, err := png.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("decode %s PNG: %v", name, err)
		}
		if image.Bounds().Dx() != 1600 || image.Bounds().Dy() < 1000 {
			t.Fatalf("%s bounds=%v, want complete aggregate", name, image.Bounds())
		}
	}
}

func TestAggregatePNGCommittedWitnesses(t *testing.T) {
	matrix, err := Validate(context.Background(), Config{Root: repoRoot(t), Checker: &acceptingChecker{}})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		mode RenderMode
		path string
	}{
		{RenderPopulated, "tools/grafana/provisioning/witnesses/fak-native-panel-coverage-populated.png"},
		{RenderUnavailable, "tools/grafana/provisioning/witnesses/fak-native-panel-coverage-unavailable.png"},
	}
	for _, test := range cases {
		var rendered bytes.Buffer
		if err := RenderAggregatePNG(&rendered, matrix, test.mode); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(repoRoot(t), test.path)
		if os.Getenv("UPDATE_NATIVEPERF_WITNESSES") == "1" {
			if err := os.WriteFile(path, rendered.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		committed, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v (regenerate with UPDATE_NATIVEPERF_WITNESSES=1)", test.path, err)
		}
		if !bytes.Equal(committed, rendered.Bytes()) {
			t.Fatalf("%s is stale (regenerate with UPDATE_NATIVEPERF_WITNESSES=1)", test.path)
		}
	}
}

func TestVisualWitnessManifestBindsCurrentInputs(t *testing.T) {
	type manifestDashboard struct {
		UID             string `json:"uid"`
		DashboardSHA256 string `json:"dashboard_sha256"`
		ContractSHA256  string `json:"contract_sha256"`
		FixtureSHA256   string `json:"fixture_sha256"`
	}
	type manifestRender struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
	}
	var manifest struct {
		Schema      string              `json:"schema"`
		Dashboards  []manifestDashboard `json:"dashboards"`
		Renders     []manifestRender    `json:"renders"`
		LiveReceipt struct {
			State                 LiveState `json:"state"`
			LiveExecutionObtained bool      `json:"live_execution_obtained"`
		} `json:"live_receipt"`
	}
	path := filepath.Join(repoRoot(t), "tools/grafana/provisioning/witnesses/fak-native-panel-coverage-manifest.json")
	readJSONTest(t, path, &manifest)
	if manifest.Schema != "fak-native-panel-coverage-witness/v1" {
		t.Fatalf("manifest schema=%q", manifest.Schema)
	}
	if len(manifest.Dashboards) != len(Specs()) {
		t.Fatalf("manifest dashboards=%d, want %d", len(manifest.Dashboards), len(Specs()))
	}
	byUID := make(map[string]manifestDashboard, len(manifest.Dashboards))
	for _, dashboard := range manifest.Dashboards {
		byUID[dashboard.UID] = dashboard
	}
	for _, spec := range Specs() {
		entry, ok := byUID[spec.UID]
		if !ok {
			t.Fatalf("manifest omits %s", spec.UID)
		}
		for file, want := range map[string]string{
			spec.Dashboard: entry.DashboardSHA256,
			spec.Contract:  entry.ContractSHA256,
			spec.Fixture:   entry.FixtureSHA256,
		} {
			raw, err := os.ReadFile(filepath.Join(repoRoot(t), file))
			if err != nil {
				t.Fatal(err)
			}
			got := fmt.Sprintf("%x", sha256.Sum256(raw))
			if got != want {
				t.Fatalf("manifest digest for %s is stale: got %s want %s", file, got, want)
			}
		}
	}
	if len(manifest.Renders) != 2 {
		t.Fatalf("manifest renders=%d, want 2", len(manifest.Renders))
	}
	for _, render := range manifest.Renders {
		raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tools/grafana/provisioning/witnesses", render.Path))
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != render.SHA256 {
			t.Fatalf("render %s digest=%s, want %s", render.Path, got, render.SHA256)
		}
		config, err := png.DecodeConfig(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		if config.Width != render.Width || config.Height != render.Height {
			t.Fatalf("render %s dimensions=%dx%d, want %dx%d", render.Path, config.Width, config.Height, render.Width, render.Height)
		}
	}
	if manifest.LiveReceipt.State != LivePending || manifest.LiveReceipt.LiveExecutionObtained {
		t.Fatalf("manifest live receipt must remain honest pending evidence: %+v", manifest.LiveReceipt)
	}
}

func successfulLiveProof(t *testing.T, completed time.Time, engine string) []byte {
	t.Helper()
	identity := map[string]any{
		"engine": engine, "runtime_engine": "inkernel", "planner": "inkernel", "model_owner": "fak",
		"fallback_count": 0, "fallback_active": false, "llama_cpp_used": false,
	}
	actual := make(map[string]any, len(identity)+4)
	for key, value := range identity {
		actual[key] = value
	}
	actual["model"] = "Qwen3.8-27B"
	actual["completed"] = true
	actual["output_tokens"] = 8
	actual["correlation_key"] = "npc1_0123456789abcdef0123456789abcdef"
	proof := map[string]any{
		"schema": PublicLiveProofSchema, "status": "success", "captured_at_utc": completed.Format(time.RFC3339),
		"completed_at_utc": completed.Format(time.RFC3339), "live_execution_obtained": true,
		"raw_logs_committed": false, "private_identifiers_committed": false,
		"model":              map[string]any{"alias": "qwen38:27b", "family": Qwen38Prefix},
		"required_execution": identity, "observed_execution": actual,
		"jobs": []map[string]any{{"name": "native-serve", "status": "succeeded"}},
	}
	raw, err := json.Marshal(proof)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func fixtureRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	sources := []string{"tools/grafana/prometheus.yml"}
	for _, spec := range Specs() {
		sources = append(sources, spec.Dashboard, spec.Contract, spec.Fixture)
	}
	for _, source := range sources {
		raw, err := os.ReadFile(filepath.Join(repoRoot(t), source))
		if err != nil {
			t.Fatal(err)
		}
		destination := filepath.Join(root, source)
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(destination, raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func replaceFile(t *testing.T, path, old, replacement string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(old)) {
		t.Fatalf("%s does not contain %q", path, old)
	}
	raw = bytes.Replace(raw, []byte(old), []byte(replacement), 1)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func readJSONTest(t *testing.T, path string, dst any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		t.Fatal(err)
	}
}

func writeJSONTest(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func assertErrorContains(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("error=%v, want substring %q", err, want)
	}
}

func ExampleMatrix_Text() {
	fmt.Println("Run: go test ./internal/nativeperfcoverage -run TestFullMatrix -v")
	// Output: Run: go test ./internal/nativeperfcoverage -run TestFullMatrix -v
}
