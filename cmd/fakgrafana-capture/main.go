// Command fakgrafana-capture captures ONE full-page PNG per fak native-performance
// dashboard from a LIVE Grafana and binds each capture to a live receipt + manifest
// with a distinct SHA-256.
//
// It is the public, reproducible capture harness for the four native-performance
// dashboards. Against a locally provisioned Grafana (default http://127.0.0.1:3000,
// basic auth admin/fleet) it renders each dashboard in kiosk mode with headless
// Chrome at the geometry declared by the dashboard contract
// (tools/grafana/provisioning/contracts/fak-native-*.json -> visual_render
// "...headless Chrome at 1920x1500") and writes the resulting PNG to its committed
// witness path.
//
//	fakgrafana-capture \
//	  --base http://127.0.0.1:3000 --user admin --pass fleet \
//	  --receipt tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json \
//	  --manifest tools/grafana/provisioning/witnesses/fak-native-panel-coverage-manifest.json
//
// The capture MECHANISM is headless Chrome's CLI (--headless=new --screenshot). No
// shell/Python is involved: the program resolves a Chrome binary and execs it
// directly. If Chrome is unavailable the run refuses rather than fabricating a capture.
//
// Done condition (issue #10008): four distinct output paths each carry a PNG whose
// SHA-256 is distinct. The harness refuses to write when any two renders are
// byte-identical, since identical bytes across dashboards is not four captures.
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// captureWidth, captureHeight is the render geometry named by the dashboard
// contracts' "visual_render" field ("headless Chrome at 1920x1500"). Keep these in
// sync with tools/grafana/provisioning/contracts/fak-native-*.json.
const (
	captureWidth  = 1920
	captureHeight = 1500
)

// captureGeometry is the single canonical rendering of the geometry, matching the
// dashboard contracts' visual_render FORMAT (WxH). Geometry is intentionally
// unchanged; this constant exists so a test can pin it against contract drift.
var captureGeometry = fmt.Sprintf("%dx%d", captureWidth, captureHeight)

// witnessDirPrefix is the committed directory every target PNG must live under.
const witnessDirPrefix = "tools/grafana/provisioning/witnesses/local-qwen38-metal-"

// DashboardTarget binds one dashboard UID to the exact committed PNG witness path it
// must be captured into. Paths are repo-relative.
type DashboardTarget struct {
	UID  string
	SHA  string
	Path string
}

// NativeDashboardTargets is the canonical four-dashboard capture set from issue
// #10008, in stable order.
func NativeDashboardTargets() []DashboardTarget {
	return []DashboardTarget{
		{
			UID:  "fak-native-kernel-performance",
			Path: "tools/grafana/provisioning/witnesses/local-qwen38-metal-kernel-performance.png",
		},
		{
			UID:  "fak-native-backends",
			Path: "tools/grafana/provisioning/witnesses/local-qwen38-metal-backends.png",
		},
		{
			UID:  "fak-native-artifacts",
			Path: "tools/grafana/provisioning/witnesses/local-qwen38-metal-artifacts.png",
		},
		{
			UID:  "fak-native-slo",
			Path: "tools/grafana/provisioning/witnesses/local-qwen38-metal-slo.png",
		},
	}
}

// DashboardURL builds the kiosk render URL for a dashboard UID against a base URL.
// The trailing slash + query matches the Grafana kiosk idiom used for witness
// captures: orgId=1, kiosk, a fixed lookback of one hour.
func DashboardURL(base, uid string, from, to string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if uid == "" {
		return base
	}
	if from == "" {
		from = "now-1h"
	}
	if to == "" {
		to = "now"
	}
	return fmt.Sprintf("%s/d/%s/?orgId=1&kiosk&from=%s&to=%s", base, uid, from, to)
}

// BasicAuthHeader returns the Authorization header value for user/pass, or "" when
// either is empty. The password is never logged; this value goes only onto the HTTP
// request (and never into a render URL, receipt, or manifest).
func BasicAuthHeader(user, pass string) string {
	if user == "" || pass == "" {
		return ""
	}
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(user+":"+pass))
}

// DashboardAPIPath is the Grafana dashboard-by-uid endpoint the preflight reads.
func DashboardAPIPath(base, uid string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	return fmt.Sprintf("%s/api/dashboards/uid/%s", base, url.PathEscape(uid))
}

// noRedirectClient returns an http.Client that does NOT follow redirects. A 3xx
// response is handed back to Preflight, which treats it as a failure: a redirect to
// a login page is exactly the silent-capture trap the preflight exists to catch.
func noRedirectClient() *http.Client {
	return &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

// Preflight verifies Grafana is reachable and the supplied credentials (when any)
// can actually read the requested dashboard. It fails loudly rather than letting
// headless Chrome silently capture a login page.
//
// Guards:
//   - redirects are never followed; any 3xx is a failure.
//   - a non-200 (notably 401/403) is a failure naming the auth mode.
//   - the body must be a JSON object whose "dashboard" is an OBJECT carrying a "uid"
//     EQUAL to the requested uid. A bare-string dashboard, a missing uid, or a uid
//     mismatch (single-UID false-PASS) is refused.
//
// A nil client uses noRedirectClient(). The returned error never contains the password.
func Preflight(client *http.Client, base, uid, user, pass string) error {
	if client == nil {
		client = noRedirectClient()
	}
	req, err := http.NewRequest(http.MethodGet, DashboardAPIPath(base, uid), nil)
	if err != nil {
		return fmt.Errorf("preflight build request for %s: %w", uid, err)
	}
	req.Header.Set("Accept", "application/json")
	if auth := BasicAuthHeader(user, pass); auth != "" {
		req.Header.Set("Authorization", auth)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("preflight read dashboard %s: %w", uid, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("preflight read dashboard %s body: %w", uid, err)
	}
	if resp.StatusCode >= 300 && resp.StatusCode < 400 {
		loc := resp.Header.Get("Location")
		return fmt.Errorf("preflight: dashboard %s returned redirect HTTP %d (Location: %q); refusing to capture a redirected/login page", uid, resp.StatusCode, loc)
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		if user == "" && pass == "" {
			return fmt.Errorf("preflight: dashboard %s requires authentication but no --user/--pass supplied (HTTP %d)", uid, resp.StatusCode)
		}
		return fmt.Errorf("preflight: credentials rejected reading dashboard %s (HTTP %d); refusing to capture a login page", uid, resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("preflight: dashboard %s returned HTTP %d, want 200", uid, resp.StatusCode)
	}
	var doc struct {
		Dashboard json.RawMessage `json:"dashboard"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return fmt.Errorf("preflight: dashboard %s returned non-JSON body: %w", uid, err)
	}
	if len(doc.Dashboard) == 0 {
		return fmt.Errorf("preflight: dashboard %s response has no \"dashboard\" field; credentials may be valid but access was denied", uid)
	}
	// "dashboard" must be an object with a uid equal to the requested uid. This is
	// what rejects a bare-string dashboard (e.g. {"dashboard":"login required"}) and
	// a wrong-single-uid body (e.g. {"dashboard":{"uid":"wrong"}}).
	var dash struct {
		UID string `json:"uid"`
	}
	if err := json.Unmarshal(doc.Dashboard, &dash); err != nil {
		return fmt.Errorf("preflight: dashboard %s response \"dashboard\" is not an object with a uid: %w", uid, err)
	}
	if dash.UID == "" {
		return fmt.Errorf("preflight: dashboard %s response \"dashboard\" has no uid", uid)
	}
	if dash.UID != uid {
		return fmt.Errorf("preflight: dashboard %s response carries dashboard.uid %q, want %q; refusing a mismatched capture", uid, dash.UID, uid)
	}
	return nil
}

// PreflightAll preflights every target in order, failing loudly (and stopping) on the
// first unreachable/unauthorized/mismatched dashboard. A single-UID preflight is not
// enough: the harness captures four dashboards, so all four must be proven readable.
func PreflightAll(client *http.Client, base string, targets []DashboardTarget, user, pass string) error {
	if len(targets) == 0 {
		return errors.New("preflight: no dashboard targets configured")
	}
	if client == nil {
		client = noRedirectClient()
	}
	for _, t := range targets {
		if err := Preflight(client, base, t.UID, user, pass); err != nil {
			return err
		}
	}
	return nil
}

// Renderer produces the bytes of a full-page screenshot for the given URL. It is the
// seam a test injects to prove the harness logic with no live Grafana or Chrome.
type Renderer func(url string) ([]byte, error)

// ChromeRenderer returns a Renderer that execs a headless Chrome binary directly
// (no shell) and returns the screenshot file's bytes.
func ChromeRenderer(chromePath string) Renderer {
	return func(url string) ([]byte, error) {
		if strings.TrimSpace(chromePath) == "" {
			return nil, errors.New("chrome path is empty")
		}
		tmp, err := os.CreateTemp("", "fakgrafana-capture-*.png")
		if err != nil {
			return nil, fmt.Errorf("create temp screenshot: %w", err)
		}
		out := tmp.Name()
		tmp.Close()
		defer os.Remove(out)

		cmd := exec.Command(chromePath,
			"--headless=new",
			"--screenshot="+out,
			fmt.Sprintf("--window-size=%d,%d", captureWidth, captureHeight),
			"--hide-scrollbars",
			"--virtual-time-budget=15000",
			"--no-sandbox",
			"--disable-gpu",
			url,
		)
		var stderr strings.Builder
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("chrome render %s: %w: %s", url, err, strings.TrimSpace(stderr.String()))
		}
		body, err := os.ReadFile(out)
		if err != nil {
			return nil, fmt.Errorf("read screenshot %s: %w", out, err)
		}
		if len(body) == 0 {
			return nil, fmt.Errorf("chrome produced an empty screenshot for %s", url)
		}
		return body, nil
	}
}

// SHA256Hex returns the lowercase hex SHA-256 of b.
func SHA256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// CaptureResult binds one dashboard's target path to the SHA-256 of its captured PNG.
type CaptureResult struct {
	UID    string `json:"uid"`
	URL    string `json:"url"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int    `json:"bytes"`
}

// Capture renders every target once, checks the four hashes are distinct (the issue's
// done condition), refuses byte-identical renders, and returns both the per-dashboard
// results and the raw rendered bytes keyed by uid so the caller can write them after
// the refusal gate has passed.
//
// ink is where human-readable progress is written. A capture set that would carry a
// duplicate hash is refused before any file is written.
func Capture(base, from, to string, targets []DashboardTarget, render Renderer, ink io.Writer) ([]CaptureResult, map[string][]byte, error) {
	if len(targets) == 0 {
		return nil, nil, errors.New("no dashboard targets configured")
	}
	results := make([]CaptureResult, 0, len(targets))
	rendered := make(map[string][]byte, len(targets))
	seenPath := map[string]string{}
	for _, t := range targets {
		if t.Path == "" {
			return nil, nil, fmt.Errorf("dashboard %s has no output path", t.UID)
		}
		if prev, dup := seenPath[t.Path]; dup {
			return nil, nil, fmt.Errorf("dashboards %s and %s target the same output path %s", prev, t.UID, t.Path)
		}
		seenPath[t.Path] = t.UID

		url := DashboardURL(base, t.UID, from, to)
		body, err := render(url)
		if err != nil {
			return nil, nil, fmt.Errorf("render %s: %w", t.UID, err)
		}
		if len(body) == 0 {
			return nil, nil, fmt.Errorf("render %s produced empty bytes", t.UID)
		}
		rendered[t.UID] = body
		results = append(results, CaptureResult{
			UID:    t.UID,
			URL:    url,
			Path:   t.Path,
			SHA256: SHA256Hex(body),
			Bytes:  len(body),
		})
		fmt.Fprintf(ink, "rendered %-32s -> %s (%d bytes, sha256=%s)\n", t.UID, t.Path, len(body), results[len(results)-1].SHA256)
	}

	if err := assertDistinct(results); err != nil {
		return nil, nil, err
	}
	return results, rendered, nil
}

// assertDistinct enforces the issue's done condition: four distinct PNG paths must
// carry four distinct SHA-256 values. A duplicate hash means two dashboards rendered
// the same bytes, which is not four real captures.
func assertDistinct(results []CaptureResult) error {
	bySHA := map[string]string{}
	for _, r := range results {
		if prev, dup := bySHA[r.SHA256]; dup {
			return fmt.Errorf("capture refused: %s and %s produced byte-identical renders (sha256=%s); distinct captures required", prev, r.UID, r.SHA256)
		}
		bySHA[r.SHA256] = r.UID
	}
	return nil
}

// WitnessBinding is the per-dashboard binding appended to the receipt and manifest.
type WitnessBinding struct {
	UID           string `json:"uid"`
	PNG           string `json:"png"`
	SHA256        string `json:"sha256"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
	CapturedAtUTC string `json:"captured_at_utc"`
	RenderURL     string `json:"render_url"`
}

// bindingsToJSONL renders the bindings as an indented JSON array. Keeping the
// encoding in one place lets both the receipt and the manifest embed the identical
// binding block.
func bindingsToJSON(results []CaptureResult, base string) ([]WitnessBinding, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	out := make([]WitnessBinding, 0, len(results))
	for _, r := range results {
		out = append(out, WitnessBinding{
			UID:           r.UID,
			PNG:           r.Path,
			SHA256:        r.SHA256,
			Width:         captureWidth,
			Height:        captureHeight,
			CapturedAtUTC: now,
			RenderURL:     r.URL,
		})
	}
	return out, nil
}

// writeJSONFile writes v as indented JSON atomically (temp file + rename) so a
// reader never observes a half-written receipt or manifest.
func writeJSONFile(path string, v any) error {
	body, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	body = append(body, '\n')
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".fakgrafana-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp beside %s: %w", path, err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(body); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("write temp for %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("close temp for %s: %w", path, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("rename temp into %s: %w", path, err)
	}
	return nil
}

// safeJoin joins a repo-relative user-supplied path onto root and refuses any path
// that is absolute or that escapes root once cleaned. This bounds --receipt and
// --manifest against traversal ("../../etc/passwd", "/absolute/path").
func safeJoin(root, rel string) (string, error) {
	if rel == "" {
		return "", errors.New("path is empty")
	}
	if filepath.IsAbs(filepath.FromSlash(rel)) {
		return "", fmt.Errorf("path %q is absolute; must be repo-relative", rel)
	}
	for _, part := range strings.Split(filepath.ToSlash(rel), "/") {
		if part == ".." {
			return "", fmt.Errorf("path %q escapes the repository root", rel)
		}
	}
	cleanRoot, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve root %q: %w", root, err)
	}
	joined := filepath.Join(cleanRoot, filepath.FromSlash(rel))
	relToRoot, err := filepath.Rel(cleanRoot, joined)
	if err != nil {
		return "", fmt.Errorf("resolve path %q under root: %w", rel, err)
	}
	if relToRoot == ".." || strings.HasPrefix(relToRoot, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes the repository root", rel)
	}
	return joined, nil
}

// writeArtifacts writes each PNG to its committed path and folds the binding block
// into both the receipt JSON and the manifest JSON.
func writeArtifacts(root string, results []CaptureResult, rendered map[string][]byte, receiptPath, manifestPath string) ([]WitnessBinding, error) {
	bindings, err := bindingsToJSON(results, "")
	if err != nil {
		return nil, err
	}
	for _, r := range results {
		body := rendered[r.UID]
		path, err := safeJoin(root, r.Path)
		if err != nil {
			return nil, err
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, fmt.Errorf("mkdir for %s: %w", path, err)
		}
		if err := os.WriteFile(path, body, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
	}

	if receiptPath != "" {
		path, err := safeJoin(root, receiptPath)
		if err != nil {
			return nil, err
		}
		if err := mergeBindingBlock(path, "dashboard_witnesses", bindings); err != nil {
			return nil, err
		}
	}
	if manifestPath != "" {
		path, err := safeJoin(root, manifestPath)
		if err != nil {
			return nil, err
		}
		if err := mergeBindingBlock(path, "dashboard_witnesses", bindings); err != nil {
			return nil, err
		}
	}
	return bindings, nil
}

// mergeBindingBlock loads a JSON object document (receipt or manifest), sets the
// named key to the binding block, and rewrites it. An absent file is created with
// just the block so the harness can run before the receipt exists.
func mergeBindingBlock(path, key string, bindings []WitnessBinding) error {
	doc := map[string]any{}
	body, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(body, &doc); err != nil {
			return fmt.Errorf("parse %s: %w", path, err)
		}
	case errors.Is(err, os.ErrNotExist):
		// start from an empty document
	default:
		return fmt.Errorf("read %s: %w", path, err)
	}
	doc[key] = bindings
	return writeJSONFile(path, doc)
}

// resolveChrome finds a headless-capable Chrome/Chromium binary. The flag/env
// override wins; otherwise sane darwin defaults are probed, then PATH names.
func resolveChrome(override string) string {
	if override = strings.TrimSpace(override); override != "" {
		return override
	}
	for _, env := range []string{"FAK_CHROME_BIN", "CHROME_BIN", "GOOGLE_CHROME_SHIM"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	candidates := []string{}
	if runtime.GOOS == "darwin" {
		candidates = append(candidates,
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Chromium.app/Contents/MacOS/Chromium",
			"/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
		)
	}
	candidates = append(candidates, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")
	for _, c := range candidates {
		if filepath.IsAbs(c) {
			if info, err := os.Stat(c); err == nil && !info.IsDir() {
				return c
			}
			continue
		}
		if found, err := exec.LookPath(c); err == nil {
			return found
		}
	}
	return ""
}

func run(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fakgrafana-capture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", "http://127.0.0.1:3000", "live Grafana base URL")
	user := fs.String("user", "admin", "Grafana basic-auth user used for the access preflight (never embedded in the render URL or written to disk)")
	pass := fs.String("pass", "fleet", "Grafana basic-auth password used for the access preflight (never logged or written to disk)")
	root := fs.String("root", ".", "repository root the witness paths are relative to")
	receipt := fs.String("receipt", "tools/grafana/provisioning/witnesses/local-qwen38-metal-live-proof.json", "receipt JSON to bind the captures into")
	manifest := fs.String("manifest", "tools/grafana/provisioning/witnesses/fak-native-panel-coverage-manifest.json", "manifest JSON to bind the captures into")
	chrome := fs.String("chrome", "", "path to a headless Chrome/Chromium binary (default: $FAK_CHROME_BIN, darwin app paths, then PATH)")
	from := fs.String("from", "now-1h", "render window start")
	to := fs.String("to", "now", "render window end")
	noAuth := fs.Bool("no-auth", false, "skip the credential access preflight (for an anonymous-access Grafana)")
	dryRun := fs.Bool("dry-run", false, "render to temp paths only; do not write witnesses, receipt, or manifest")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// Authenticate with Go's http.Client before launching Chrome. Chrome's CLI has
	// neither a basic-auth flag nor a way to inject a session cookie, so the session
	// cannot be established through the renderer. Instead the preflight proves the
	// supplied credentials can read EVERY dashboard in the capture set; if any one
	// cannot be read we refuse loudly rather than let Chrome capture a login page.
	// The password stays in memory here and is never logged or written to the
	// receipt/manifest.
	targets := NativeDashboardTargets()
	if !*noAuth {
		if err := PreflightAll(nil, *base, targets, *user, *pass); err != nil {
			fmt.Fprintf(stderr, "fakgrafana-capture: %v\n", err)
			return 1
		}
	}

	chromeBin := resolveChrome(*chrome)
	if chromeBin == "" {
		fmt.Fprintln(stderr, "fakgrafana-capture: no headless Chrome found; pass --chrome or set FAK_CHROME_BIN")
		return 2
	}

	render := ChromeRenderer(chromeBin)

	// Render everything first, keep the bytes in memory, and only then write. This is
	// what makes the distinct-hash refusal a true pre-write gate. run() calls the SAME
	// Capture() the tests exercise, so the tested duplicate-path/duplicate-hash gates
	// are in the shipped binary.
	results, rendered, err := Capture(*base, *from, *to, targets, render, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fakgrafana-capture: %v\n", err)
		return 1
	}

	if *dryRun {
		for _, r := range results {
			fmt.Fprintf(stdout, "dry-run %-32s -> %s (%d bytes, sha256=%s)\n", r.UID, r.Path, r.Bytes, r.SHA256)
		}
		return 0
	}

	bindings, err := writeArtifacts(*root, results, rendered, *receipt, *manifest)
	if err != nil {
		fmt.Fprintf(stderr, "fakgrafana-capture: %v\n", err)
		return 1
	}
	for _, b := range bindings {
		fmt.Fprintf(stdout, "captured %-32s -> %s sha256=%s\n", b.UID, b.PNG, b.SHA256)
	}
	fmt.Fprintf(stdout, "fakgrafana-capture: bound %d distinct witnesses\n", len(bindings))
	return 0
}

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}
