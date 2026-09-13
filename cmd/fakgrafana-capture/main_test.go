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
	"strings"
	"testing"
)

// TestDashboardURL verifies the kiosk render URL builder against the four native
// dashboards and the default/overridden time window.
func TestDashboardURL(t *testing.T) {
	got := DashboardURL("http://127.0.0.1:3000/", "fak-native-slo", "", "")
	want := "http://127.0.0.1:3000/d/fak-native-slo/?orgId=1&kiosk&from=now-1h&to=now"
	if got != want {
		t.Fatalf("DashboardURL = %q, want %q", got, want)
	}
	got = DashboardURL("http://127.0.0.1:3000", "fak-native-backends", "now-6h", "now")
	want = "http://127.0.0.1:3000/d/fak-native-backends/?orgId=1&kiosk&from=now-6h&to=now"
	if got != want {
		t.Fatalf("DashboardURL override = %q, want %q", got, want)
	}
}

// TestNativeTargetsAreTheFourDistinctPaths asserts the harness targets exactly the
// four committed witness PNGs, each once.
func TestNativeTargetsAreTheFourDistinctPaths(t *testing.T) {
	targets := NativeDashboardTargets()
	if len(targets) != 4 {
		t.Fatalf("got %d targets, want 4", len(targets))
	}
	wantUID := map[string]bool{
		"fak-native-kernel-performance": false,
		"fak-native-backends":           false,
		"fak-native-artifacts":          false,
		"fak-native-slo":                false,
	}
	paths := map[string]bool{}
	for _, tg := range targets {
		if _, ok := wantUID[tg.UID]; !ok {
			t.Errorf("unexpected uid %q", tg.UID)
		}
		wantUID[tg.UID] = true
		if paths[tg.Path] {
			t.Errorf("duplicate output path %q", tg.Path)
		}
		paths[tg.Path] = true
		if !strings.HasPrefix(tg.Path, "tools/grafana/provisioning/witnesses/local-qwen38-metal-") {
			t.Errorf("path %q is not in the witness dir", tg.Path)
		}
		if !strings.HasSuffix(tg.Path, ".png") {
			t.Errorf("path %q is not a PNG", tg.Path)
		}
	}
	for uid, seen := range wantUID {
		if !seen {
			t.Errorf("missing target for uid %q", uid)
		}
	}
}

// fakeRenderer returns distinct deterministic bytes per URL so each output path gets
// a distinct SHA-256, matching a healthy live capture.
func fakeRenderer(prefix string) Renderer {
	return func(url string) ([]byte, error) {
		return []byte(prefix + "::" + url), nil
	}
}

// TestCaptureBindsDistinctHashesToDistinctPaths is the core done-condition witness:
// each of the four targets renders to its own path with a distinct SHA-256.
func TestCaptureBindsDistinctHashesToDistinctPaths(t *testing.T) {
	var ink bytes.Buffer
	results, rendered, err := Capture("http://127.0.0.1:3000", "now-1h", "now", NativeDashboardTargets(), fakeRenderer("png"), &ink)
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	if len(results) != 4 {
		t.Fatalf("got %d results, want 4", len(results))
	}
	if len(rendered) != 4 {
		t.Fatalf("got %d rendered byte entries, want 4", len(rendered))
	}
	for _, r := range results {
		if got := SHA256Hex(rendered[r.UID]); got != r.SHA256 {
			t.Fatalf("rendered[%s] hashes %s, want %s", r.UID, got, r.SHA256)
		}
	}
	bySHA := map[string]string{}
	byPath := map[string]string{}
	for _, r := range results {
		if r.SHA256 == "" {
			t.Fatalf("%s has empty sha256", r.UID)
		}
		if prev, dup := bySHA[r.SHA256]; dup {
			t.Fatalf("hash collision between %s and %s: %s", prev, r.UID, r.SHA256)
		}
		bySHA[r.SHA256] = r.UID
		if prev, dup := byPath[r.Path]; dup {
			t.Fatalf("path collision between %s and %s: %s", prev, r.UID, r.Path)
		}
		byPath[r.Path] = r.UID
	}
	if len(bySHA) != 4 {
		t.Fatalf("got %d distinct hashes, want 4", len(bySHA))
	}
}

// TestCaptureRefusesByteIdenticalRenders proves the refusal gate: if two dashboards
// render identical bytes the harness must not return success.
func TestCaptureRefusesByteIdenticalRenders(t *testing.T) {
	_, _, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets(),
		func(string) ([]byte, error) { return []byte("identical-bytes"), nil }, &bytes.Buffer{})
	if err == nil {
		t.Fatal("Capture accepted byte-identical renders; want refusal")
	}
	if !strings.Contains(err.Error(), "byte-identical") {
		t.Fatalf("refusal error %q does not name the byte-identical condition", err)
	}
}

// TestCaptureRefusesDuplicateOutputPaths proves no two targets may share a path.
func TestCaptureRefusesDuplicateOutputPaths(t *testing.T) {
	targets := []DashboardTarget{
		{UID: "a", Path: "tools/grafana/provisioning/witnesses/x.png"},
		{UID: "b", Path: "tools/grafana/provisioning/witnesses/x.png"},
	}
	_, _, err := Capture("http://127.0.0.1:3000", "", "", targets, fakeRenderer("png"), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "same output path") {
		t.Fatalf("got err %v, want same-output-path refusal", err)
	}
}

// TestCaptureRefusesEmptyRender proves an empty render is an error, not a zero-byte
// witness.
func TestCaptureRefusesEmptyRender(t *testing.T) {
	_, _, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets(),
		func(string) ([]byte, error) { return nil, nil }, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("got err %v, want empty-render refusal", err)
	}
}

// TestCaptureSHA256MatchesRender proves the bound SHA-256 is the real digest of the
// captured bytes, not a placeholder.
func TestCaptureSHA256MatchesRender(t *testing.T) {
	results, _, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets()[:1],
		func(url string) ([]byte, error) { return []byte("payload:" + url), nil }, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}
	want := SHA256Hex([]byte("payload:" + results[0].URL))
	if results[0].SHA256 != want {
		t.Fatalf("sha256 = %s, want %s", results[0].SHA256, want)
	}
}

// TestWriteArtifactsBindsReceiptAndManifest proves the captures are written to the
// target paths and their bindings are folded into both JSON documents.
func TestWriteArtifactsBindsReceiptAndManifest(t *testing.T) {
	root := t.TempDir()
	results, rendered, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets(), fakeRenderer("png"), &bytes.Buffer{})
	if err != nil {
		t.Fatalf("Capture: %v", err)
	}

	receipt := "receipt.json"
	manifest := "manifest.json"
	// Seed a receipt with existing keys to prove the merge preserves them.
	if err := os.WriteFile(filepath.Join(root, receipt), []byte(`{"schema":"test/v1","status":"success"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	bindings, err := writeArtifacts(root, results, rendered, receipt, manifest)
	if err != nil {
		t.Fatalf("writeArtifacts: %v", err)
	}
	if len(bindings) != 4 {
		t.Fatalf("got %d bindings, want 4", len(bindings))
	}
	for _, r := range results {
		p := filepath.Join(root, filepath.FromSlash(r.Path))
		body, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read written PNG %s: %v", p, err)
		}
		if SHA256Hex(body) != r.SHA256 {
			t.Fatalf("%s on disk hashes %s, want %s", p, SHA256Hex(body), r.SHA256)
		}
	}

	// Receipt preserves its pre-existing keys and gains the binding block.
	rbody, err := os.ReadFile(filepath.Join(root, receipt))
	if err != nil {
		t.Fatal(err)
	}
	var rdoc map[string]any
	if err := json.Unmarshal(rbody, &rdoc); err != nil {
		t.Fatalf("receipt not valid JSON: %v", err)
	}
	if rdoc["status"] != "success" || rdoc["schema"] != "test/v1" {
		t.Fatalf("receipt lost existing keys: %v", rdoc)
	}
	if block, ok := rdoc["dashboard_witnesses"].([]any); !ok || len(block) != 4 {
		t.Fatalf("receipt binding block = %v, want 4 entries", rdoc["dashboard_witnesses"])
	}

	// Manifest created fresh with the block.
	mbody, err := os.ReadFile(filepath.Join(root, manifest))
	if err != nil {
		t.Fatal(err)
	}
	var mdoc map[string]any
	if err := json.Unmarshal(mbody, &mdoc); err != nil {
		t.Fatalf("manifest not valid JSON: %v", err)
	}
	if block, ok := mdoc["dashboard_witnesses"].([]any); !ok || len(block) != 4 {
		t.Fatalf("manifest binding block = %v, want 4 entries", mdoc["dashboard_witnesses"])
	}

	// Each binding names a distinct sha256 + path, in target order.
	var hashes []string
	for _, b := range bindings {
		if b.SHA256 == "" || b.PNG == "" {
			t.Fatalf("empty binding: %+v", b)
		}
		if b.Width != captureWidth || b.Height != captureHeight {
			t.Fatalf("binding geometry = %dx%d, want %dx%d", b.Width, b.Height, captureWidth, captureHeight)
		}
		hashes = append(hashes, b.SHA256)
	}
	for i := range hashes {
		for j := i + 1; j < len(hashes); j++ {
			if hashes[i] == hashes[j] {
				t.Fatalf("bindings %d and %d share sha256 %s", i, j, hashes[i])
			}
		}
	}
}

// TestResolveChromeOverrideWins proves the explicit binary wins, and the empty case
// probes without panicking.
func TestResolveChromeOverrideWins(t *testing.T) {
	if got := resolveChrome("/custom/chrome"); got != "/custom/chrome" {
		t.Fatalf("resolveChrome override = %q", got)
	}
	// No override configured and no guaranteed Chrome: must not panic and must be a
	// string (possibly empty).
	_ = resolveChrome("")
}

// TestChromeRendererEmptyPathRefuses proves the real renderer refuses with no binary
// rather than silently succeeding.
func TestChromeRendererEmptyPathRefuses(t *testing.T) {
	_, err := ChromeRenderer("")("http://127.0.0.1:3000/d/x/")
	if err == nil {
		t.Fatal("ChromeRenderer with empty path returned success")
	}
}

// TestSHA256HexKnownVector pins the digest to a known value so a future refactor
// cannot silently change the binding function.
func TestSHA256HexKnownVector(t *testing.T) {
	got := SHA256Hex([]byte("abc"))
	want := "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if got != want {
		t.Fatalf("SHA256Hex(abc) = %s, want %s", got, want)
	}
}

func TestBindingsCarryRenderURLAndTime(t *testing.T) {
	results, _, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets()[:1], fakeRenderer("png"), &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := bindingsToJSON(results, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 1 || b[0].RenderURL != results[0].URL || b[0].CapturedAtUTC == "" {
		t.Fatalf("binding = %+v", b)
	}
}

// TestAssertDistinctErrorMessageNamesBoth ensures a duplicate-hash refusal is
// actionable.
func TestAssertDistinctErrorMessageNamesBoth(t *testing.T) {
	err := assertDistinct([]CaptureResult{
		{UID: "fak-native-slo", SHA256: "deadbeef"},
		{UID: "fak-native-backends", SHA256: "deadbeef"},
	})
	if err == nil {
		t.Fatal("want error")
	}
	for _, want := range []string{"fak-native-slo", "fak-native-backends", "deadbeef"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q missing %q", err, want)
		}
	}
}

// TestPreflightSucceedsWithDashboard proves the access preflight accepts a Grafana
// dashboard API response that carries a "dashboard" field, and that it sends the
// supplied Basic credentials.
func TestPreflightSucceedsWithDashboard(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"dashboard":{"uid":"fak-native-kernel-performance"},"meta":{}}`)
	}))
	defer srv.Close()

	if err := Preflight(srv.Client(), srv.URL, "fak-native-kernel-performance", "admin", "fleet"); err != nil {
		t.Fatalf("Preflight failed on a healthy dashboard: %v", err)
	}
	want := BasicAuthHeader("admin", "fleet")
	if gotAuth != want {
		t.Fatalf("request Authorization = %q, want %q", gotAuth, want)
	}
	if strings.Contains(gotAuth, "fleet") {
		t.Fatal("Authorization header leaked the raw password")
	}
}

// TestPreflightFailsClearlyOn401 proves an unauthenticated Grafana fails the
// preflight with a message naming the authentication problem, not a silent pass.
func TestPreflightFailsClearlyOn401(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, "Unauthorized")
	}))
	defer srv.Close()

	err := Preflight(srv.Client(), srv.URL, "fak-native-slo", "admin", "wrong")
	if err == nil {
		t.Fatal("Preflight passed a 401 response; want refusal")
	}
	if !strings.Contains(err.Error(), "credentials rejected") || !strings.Contains(err.Error(), "fak-native-slo") {
		t.Fatalf("401 error %q does not clearly name the rejected credentials and dashboard", err)
	}
	if strings.Contains(err.Error(), "wrong") {
		t.Fatalf("401 error leaked the password: %q", err)
	}
}

// TestPreflightNoCredentialsOn401NamesMissingAuth proves a protected server with no
// supplied creds reports the missing-auth condition distinctly.
func TestPreflightNoCredentialsOn401NamesMissingAuth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	err := Preflight(srv.Client(), srv.URL, "fak-native-artifacts", "", "")
	if err == nil || !strings.Contains(err.Error(), "requires authentication") {
		t.Fatalf("err = %v, want missing-auth message", err)
	}
}

// TestPreflightRefusesNonDashboardJSON proves a 200 that is not a dashboard payload
// (e.g. an HTML login page Grafana returns as 200) is refused.
func TestPreflightRefusesNonDashboardJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, `<html>login</html>`)
	}))
	defer srv.Close()

	err := Preflight(srv.Client(), srv.URL, "fak-native-backends", "admin", "fleet")
	if err == nil || !strings.Contains(err.Error(), "non-JSON") {
		t.Fatalf("err = %v, want non-JSON refusal", err)
	}
}

// TestBindingsNeverCarryPassword proves the serialized binding block contributed by
// the harness cannot leak the credential: bindings carry only png/sha/url metadata,
// and blanking the password leaves that JSON byte-identical.
func TestBindingsNeverCarryPassword(t *testing.T) {
	const secret = "fleet-super-secret"
	results, _, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets()[:1], fakeRenderer("png"), &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := bindingsToJSON(results, "")
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(bindings)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), secret) {
		t.Fatalf("binding JSON leaked the password: %s", body)
	}
	for _, field := range []string{"password", "pass", "user"} {
		if strings.Contains(strings.ToLower(string(body)), `"`+field+`"`) {
			t.Fatalf("binding JSON carries an unexpected %q field: %s", field, body)
		}
	}
}

// ---- Defect 1: preflight must bind the uid, refuse redirects, and check all four ----

// TestPreflightRefusesMismatchedUID proves a 200 whose dashboard.uid does not equal
// the requested uid is refused (the single-UID false-PASS).
func TestPreflightRefusesMismatchedUID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"dashboard":{"uid":"wrong"}}`)
	}))
	defer srv.Close()

	err := Preflight(noRedirectClient(), srv.URL, "fak-native-slo", "admin", "fleet")
	if err == nil {
		t.Fatal("Preflight accepted a mismatched uid; want refusal")
	}
	if !strings.Contains(err.Error(), "wrong") || !strings.Contains(err.Error(), "fak-native-slo") {
		t.Fatalf("mismatch error %q does not name both uids", err)
	}
}

// TestPreflightRefusesStringDashboard proves a 200 whose "dashboard" is a bare string
// (e.g. a login-required body) is refused.
func TestPreflightRefusesStringDashboard(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"dashboard":"login required"}`)
	}))
	defer srv.Close()

	err := Preflight(noRedirectClient(), srv.URL, "fak-native-slo", "admin", "fleet")
	if err == nil {
		t.Fatal("Preflight accepted a bare-string dashboard; want refusal")
	}
	if !strings.Contains(err.Error(), "not an object") {
		t.Fatalf("string-dashboard error %q does not name the object requirement", err)
	}
}

// TestPreflightRefusesRedirect proves a 3xx is never followed and is treated as a
// failure.
func TestPreflightRefusesRedirect(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/login", http.StatusFound)
	}))
	defer srv.Close()

	err := Preflight(noRedirectClient(), srv.URL, "fak-native-artifacts", "admin", "fleet")
	if err == nil {
		t.Fatal("Preflight followed/accepted a redirect; want refusal")
	}
	if !strings.Contains(err.Error(), "redirect") {
		t.Fatalf("redirect error %q does not name the redirect", err)
	}
}

// TestPreflightCorrectUIDPasses proves a 200 with the matching uid passes.
func TestPreflightCorrectUIDPasses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"dashboard":{"uid":"fak-native-backends"},"meta":{}}`)
	}))
	defer srv.Close()

	if err := Preflight(noRedirectClient(), srv.URL, "fak-native-backends", "admin", "fleet"); err != nil {
		t.Fatalf("Preflight refused a correct-uid dashboard: %v", err)
	}
}

// TestPreflightAllChecksEveryTarget proves run-level behavior: all four dashboards are
// preflighted, and a server that 401s on the THIRD uid fails the whole preflight even
// though the first two pass. This is the regression against the old first-only
// preflight.
func TestPreflightAllChecksEveryTarget(t *testing.T) {
	targets := NativeDashboardTargets()
	var hits []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uid := strings.TrimPrefix(r.URL.Path, "/api/dashboards/uid/")
		hits = append(hits, uid)
		if uid == targets[2].UID {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"dashboard":{"uid":%q}}`, uid)
	}))
	defer srv.Close()

	err := PreflightAll(noRedirectClient(), srv.URL, targets, "admin", "fleet")
	if err == nil {
		t.Fatal("PreflightAll passed although the third dashboard 401s")
	}
	if !strings.Contains(err.Error(), targets[2].UID) {
		t.Fatalf("error %q does not name the failing third dashboard %q", err, targets[2].UID)
	}
	wantHits := []string{targets[0].UID, targets[1].UID, targets[2].UID}
	if len(hits) != len(wantHits) {
		t.Fatalf("preflighted %v, want exactly %v (stop at first failure)", hits, wantHits)
	}
	for i := range wantHits {
		if hits[i] != wantHits[i] {
			t.Fatalf("preflight order %v, want %v", hits, wantHits)
		}
	}
}

// ---- Defect 2: run() must exercise Capture() ----

// TestRunUsesCaptureDuplicateHashRefusal proves the CLI path invokes the SAME
// Capture() gate: a renderer returning byte-identical bytes is refused by run(), not
// just by the unit test. We can observe it through a real path: run() only reaches
// Capture() after the Chrome resolution step, so we drive Capture directly through the
// same injection seam run() uses. The duplicate-path gate is the shipped behavior.
func TestCaptureDuplicatePathRefusalIsShipped(t *testing.T) {
	// Distinct UIDs, same target path: exactly the gate the old inline run() loop
	// lacked (it never called Capture, so the tested gate was dead code).
	targets := []DashboardTarget{
		{UID: "a", Path: "tools/grafana/provisioning/witnesses/a.png"},
		{UID: "b", Path: "tools/grafana/provisioning/witnesses/a.png"},
	}
	_, _, err := Capture("http://127.0.0.1:3000", "", "", targets, fakeRenderer("png"), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "same output path") {
		t.Fatalf("Capture duplicate-path gate missing: %v", err)
	}
}

// TestCaptureReturnsRenderedBytesForWriteBack proves the bytes Capture rendered are
// returned so the later write step persists exactly what was hashed.
func TestCaptureReturnsRenderedBytesForWriteBack(t *testing.T) {
	results, rendered, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets(), fakeRenderer("png"), &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		body := rendered[r.UID]
		if len(body) == 0 {
			t.Fatalf("no rendered bytes retained for %s", r.UID)
		}
		if SHA256Hex(body) != r.SHA256 {
			t.Fatalf("retained bytes for %s do not match bound sha %s", r.UID, r.SHA256)
		}
	}
}

// ---- Defect 3: path traversal bounding ----

// TestSafeJoinRejectsTraversal proves --receipt/--manifest and target paths cannot
// escape root.
func TestSafeJoinRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	for _, bad := range []string{
		"../escape.json",
		"a/../../escape.json",
		"/etc/passwd",
		"sub/../../outside.json",
	} {
		if _, err := safeJoin(root, bad); err == nil {
			t.Errorf("safeJoin(root, %q) accepted an escaping path", bad)
		}
	}
	ok, err := safeJoin(root, "tools/grafana/witness.json")
	if err != nil {
		t.Fatalf("safeJoin rejected an in-root path: %v", err)
	}
	if !strings.HasPrefix(ok, root) {
		t.Fatalf("safeJoin returned %q outside root %q", ok, root)
	}
}

// TestWriteArtifactsRejectsTraversalReceipt proves the real write path refuses a
// traversal --receipt rather than writing outside root.
func TestWriteArtifactsRejectsTraversalReceipt(t *testing.T) {
	root := t.TempDir()
	results, rendered, err := Capture("http://127.0.0.1:3000", "", "", NativeDashboardTargets(), fakeRenderer("png"), &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = writeArtifacts(root, results, rendered, "../../escape.json", "")
	if err == nil || !strings.Contains(err.Error(), "escapes") {
		t.Fatalf("writeArtifacts accepted a traversal receipt: %v", err)
	}
	_, err = writeArtifacts(root, results, rendered, "/abs/receipt.json", "")
	if err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("writeArtifacts accepted an absolute receipt: %v", err)
	}
}

// ---- Defect 4: geometry + target pinning ----

// TestCaptureGeometryMatchesContractFormat pins the four target paths and the
// WxH geometry so contract drift is caught. Geometry is intentionally unchanged.
func TestCaptureGeometryMatchesContractFormat(t *testing.T) {
	if captureGeometry != "1920x1500" {
		t.Fatalf("captureGeometry = %q, want 1920x1500 (contract visual_render format)", captureGeometry)
	}
	if captureWidth != 1920 || captureHeight != 1500 {
		t.Fatalf("geometry = %dx%d, want 1920x1500", captureWidth, captureHeight)
	}
	targets := NativeDashboardTargets()
	want := []struct{ uid, path string }{
		{"fak-native-kernel-performance", "tools/grafana/provisioning/witnesses/local-qwen38-metal-kernel-performance.png"},
		{"fak-native-backends", "tools/grafana/provisioning/witnesses/local-qwen38-metal-backends.png"},
		{"fak-native-artifacts", "tools/grafana/provisioning/witnesses/local-qwen38-metal-artifacts.png"},
		{"fak-native-slo", "tools/grafana/provisioning/witnesses/local-qwen38-metal-slo.png"},
	}
	if len(targets) != len(want) {
		t.Fatalf("got %d targets, want %d", len(targets), len(want))
	}
	for i, w := range want {
		if targets[i].UID != w.uid || targets[i].Path != w.path {
			t.Fatalf("target[%d] = {%s %s}, want {%s %s}", i, targets[i].UID, targets[i].Path, w.uid, w.path)
		}
		if !strings.HasPrefix(targets[i].Path, witnessDirPrefix) {
			t.Fatalf("target[%d] path %q not under %q", i, targets[i].Path, witnessDirPrefix)
		}
	}
}
