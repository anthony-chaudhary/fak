package hooks

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const demoLiveLinksGoodHTML = `
<!-- The demo host is plain HTTP. -->
<link rel="canonical" href="https://anthony-chaudhary.github.io/fak/demos.html">
<meta property="og:url" content="https://anthony-chaudhary.github.io/fak/demos.html">
<meta property="og:image" content="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png">
<a class="card" href="run-the-demos.html">guard</a>
<a class="card" href="http://136.111.250.205:8150/">turntax</a>
<a class="card" href="http://136.111.250.205:8153/">ctx</a>
<a class="card" href="http://136.111.250.205/demorace/">race</a>
<a href="http://136.111.250.205/">hub</a>
`

func writeTestSocialPreview(t *testing.T, root string, data []byte) {
	t.Helper()
	p := filepath.Join(root, "visuals", "social-preview.png")
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTestDemosDoc(t *testing.T, root string, content string) {
	t.Helper()
	p := filepath.Join(root, "docs", "demos.html")
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func validTestPNGData() []byte {
	var buf bytes.Buffer
	buf.WriteString(demoLiveLinksPNGMagic)
	buf.Write(bytes.Repeat([]byte("x"), 2048))
	return buf.Bytes()
}

func TestDemoLiveLinks_LiveTreeClean(t *testing.T) {
	root := repoRoot(t)
	defects := demoLiveLinksDefects(root)
	if len(defects) != 0 {
		t.Fatalf("expected live docs/demos.html to be clean, got defects: %v", defects)
	}

	tree, err := ReadTrackedTree(root)
	if err != nil {
		t.Fatal(err)
	}
	findings, err := gateDemoLiveLinksTree(tree)
	if err != nil {
		t.Fatalf("gateDemoLiveLinksTree errored: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings on live tree, got: %v", findings)
	}
}

func TestDemoLiveLinks_StaticAuditAcceptsCurrentShape(t *testing.T) {
	defects := staticDemoLinksAudit(demoLiveLinksGoodHTML, demoLiveLinksHost)
	if len(defects) != 0 {
		t.Fatalf("expected clean static audit, got: %v", defects)
	}
}

func TestDemoLiveLinks_StaticAuditRejectsStaleGuardHostedLink(t *testing.T) {
	bad := strings.Replace(demoLiveLinksGoodHTML, `href="run-the-demos.html"`, `href="http://136.111.250.205/guarddemo/"`, 1)
	defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
	foundStale := false
	foundGuard := false
	for _, d := range defects {
		if strings.Contains(d, "stale hosted demo link") {
			foundStale = true
		}
		if strings.Contains(d, "for guarddemo") {
			foundGuard = true
		}
	}
	if !foundStale || !foundGuard {
		t.Fatalf("expected stale guarddemo defect, got: %v", defects)
	}
}

func TestDemoLiveLinks_KnownStaleMatchIdentifiesDemo(t *testing.T) {
	prefix, demo, ok := knownStaleDemoMatch("http://136.111.250.205/guarddemo/")
	if !ok || prefix != "http://136.111.250.205/guarddemo/" || demo != "guarddemo" {
		t.Errorf("guarddemo match failed: got (%q, %q, %v)", prefix, demo, ok)
	}

	prefix, demo, ok = knownStaleDemoMatch("http://136.111.250.205/unsee/api/events")
	if !ok || prefix != "http://136.111.250.205/unsee/" || demo != "unseedemo" {
		t.Errorf("unseedemo match failed: got (%q, %q, %v)", prefix, demo, ok)
	}

	_, _, ok = knownStaleDemoMatch("http://136.111.250.205:8150/")
	if ok {
		t.Errorf("unexpected match for active turntaxdemo URL")
	}
}

func TestDemoLiveLinks_StaticAuditRejectsUnexpectedHostedLink(t *testing.T) {
	bad := strings.Replace(demoLiveLinksGoodHTML, "http://136.111.250.205:8153/", "http://136.111.250.205:9999/", 1)
	defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
	hasMissing := false
	hasUnexpected := false
	for _, d := range defects {
		if strings.Contains(d, "expected hosted demo link missing: http://136.111.250.205:8153/") {
			hasMissing = true
		}
		if strings.Contains(d, "unexpected hosted demo link: http://136.111.250.205:9999/") {
			hasUnexpected = true
		}
	}
	if !hasMissing || !hasUnexpected {
		t.Fatalf("expected missing and unexpected link defects, got: %v", defects)
	}
}

func TestDemoLiveLinks_StaticAuditRejectsWrongHostedLinkRole(t *testing.T) {
	bad := strings.Replace(
		demoLiveLinksGoodHTML,
		`<a href="http://136.111.250.205/">hub</a>`,
		`<a class="card" href="http://136.111.250.205/">hub</a>`,
		1,
	)
	defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
	hasRoleDefect := false
	for _, d := range defects {
		if strings.Contains(d, "hosted demo link role changed: http://136.111.250.205/ should be a non-card link") {
			hasRoleDefect = true
		}
	}
	if !hasRoleDefect {
		t.Fatalf("expected role changed defect, got: %v", defects)
	}
}

func TestDemoLiveLinks_StaticAuditRejectsStalePathPrefixes(t *testing.T) {
	for _, stale := range []string{"turntax", "ctxdemo", "unsee"} {
		bad := strings.Replace(demoLiveLinksGoodHTML, "http://136.111.250.205:8150/", "http://136.111.250.205/"+stale+"/", 1)
		defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
		hasStale := false
		for _, d := range defects {
			if strings.Contains(d, "stale hosted demo link") {
				hasStale = true
				break
			}
		}
		if !hasStale {
			t.Errorf("expected stale defect for %s, got: %v", stale, defects)
		}
	}
}

func TestDemoLiveLinks_StaticAuditRequiresPlainHTTPDisclosure(t *testing.T) {
	bad := strings.Replace(demoLiveLinksGoodHTML, "plain HTTP", "public demo host", 1)
	defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
	hasDisclosure := false
	for _, d := range defects {
		if strings.Contains(d, "does not disclose plain HTTP") {
			hasDisclosure = true
			break
		}
	}
	if !hasDisclosure {
		t.Fatalf("expected plain HTTP disclosure defect, got: %v", defects)
	}
}

func TestDemoLiveLinks_StaticAuditRejectsHTTPSIPHostLink(t *testing.T) {
	bad := strings.Replace(demoLiveLinksGoodHTML, "http://136.111.250.205:8150/", "https://136.111.250.205:8150/", 1)
	defects := staticDemoLinksAudit(bad, demoLiveLinksHost)
	hasHTTPS := false
	for _, d := range defects {
		if strings.Contains(d, "uses https:// for the IP host") {
			hasHTTPS = true
			break
		}
	}
	if !hasHTTPS {
		t.Fatalf("expected https IP host defect, got: %v", defects)
	}
}

func TestDemoLiveLinks_MetadataAudit(t *testing.T) {
	root := t.TempDir()
	writeTestSocialPreview(t, root, validTestPNGData())

	// Clean metadata
	defects := demoPageMetadataAudit(root, demoLiveLinksGoodHTML)
	if len(defects) != 0 {
		t.Fatalf("expected clean metadata, got: %v", defects)
	}

	// Missing canonical
	bad := strings.Replace(demoLiveLinksGoodHTML, `<link rel="canonical"`, `<!-- <link rel="canonical" -->`, 1)
	defects = demoPageMetadataAudit(root, bad)
	if !stringsContainsAny(defects, "missing canonical") {
		t.Fatalf("expected missing canonical defect, got: %v", defects)
	}

	// Bad social preview asset - not PNG
	writeTestSocialPreview(t, root, []byte("NOTAPNG"+strings.Repeat("x", 2048)))
	defects = demoPageMetadataAudit(root, demoLiveLinksGoodHTML)
	if !stringsContainsAny(defects, "asset is not a PNG") {
		t.Fatalf("expected not-a-png defect, got: %v", defects)
	}

	// Bad social preview asset - too small
	var small bytes.Buffer
	small.WriteString(demoLiveLinksPNGMagic)
	small.Write(bytes.Repeat([]byte("x"), 100))
	writeTestSocialPreview(t, root, small.Bytes())
	defects = demoPageMetadataAudit(root, demoLiveLinksGoodHTML)
	if !stringsContainsAny(defects, "asset is unexpectedly small") {
		t.Fatalf("expected too-small defect, got: %v", defects)
	}

	// Missing social preview asset
	_ = os.Remove(filepath.Join(root, "visuals", "social-preview.png"))
	defects = demoPageMetadataAudit(root, demoLiveLinksGoodHTML)
	if !stringsContainsAny(defects, "asset missing") {
		t.Fatalf("expected asset missing defect, got: %v", defects)
	}
}

func stringsContainsAny(xs []string, substr string) bool {
	for _, x := range xs {
		if strings.Contains(x, substr) {
			return true
		}
	}
	return false
}

func TestDemoLiveLinks_DifferentialAgainstPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	root := repoRoot(t)
	script := filepath.Join(root, "tools", "demo_live_links.py")
	if _, err := os.Stat(script); err != nil {
		t.Skip("tools/demo_live_links.py not found")
	}

	cmd := exec.Command("python3", script)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(root, "tools"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("python tools/demo_live_links.py failed: %v\n%s", err, string(out))
	}

	defects := demoLiveLinksDefects(root)
	if len(defects) != 0 {
		t.Fatalf("Go demoLiveLinksDefects found defects on clean repo: %v", defects)
	}
}
