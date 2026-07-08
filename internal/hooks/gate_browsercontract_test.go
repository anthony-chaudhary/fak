package hooks

import (
	"os/exec"
	"sort"
	"strings"
	"testing"
)

// gate_browsercontract_test.go — parity for the BROWSER_CONTRACT gate. The Python checker
// (tools/demo_browser_contract.py) is workspace-scoped (no --audit-staged / --root), so the staged
// runParity harness does not cover it. Instead we (1) replay the golden fixtures from
// tools/demo_browser_contract_test.py — the oracle's own accept/reject samples — against the ported
// demo_contract_defects / source_default_addr logic, (2) pin the shared registry (demoRegistry,
// demoLifecycle) against the live Python demo_registry, (3) prove the live tracked tree is clean (the
// twin of `make hygiene` running the Python green), and (4) run a verdict-level differential against
// the live Python over the real tree.

// browserMain builds the same main.go fixture as demo_browser_contract_test.browser_main(): a clean
// demo that uses every shared demoui helper.
func browserMain(port, basePath string) string {
	return "\npackage main\n\nconst defaultAddr = \"127.0.0.1:" + port + "\"\n\nfunc main() {\n" +
		"    basePath := demoui.BasePathFlag(flag.CommandLine, \"" + basePath + "\")\n" +
		"    app := http.NewServeMux()\n" +
		"    mux := http.NewServeMux()\n" +
		"    base := demoui.MountWithBasePath(mux, *basePath, app)\n" +
		"    bind := demoui.ListenAddr(*addr, defaultAddr)\n" +
		"    _ = demoui.LocalURL(bind, base)\n}\n"
}

// TestBrowserContract_SourceDefaultAddrExtractsPort is the twin of test_source_default_addr_extracts_port.
func TestBrowserContract_SourceDefaultAddrExtractsPort(t *testing.T) {
	root := t.TempDir()
	demoWrite(t, root, "cmd/guarddemo/main.go", "package main\nconst defaultAddr = \"127.0.0.1:8151\"\n")
	demo := demoReg{name: "guarddemo", basePath: "/guarddemo", apiPath: "api/scenarios", pageMarker: "safety", defaultPort: 8151}
	addr, port := sourceDefaultAddr(root, demo)
	if addr != "127.0.0.1:8151" || port != 8151 {
		t.Fatalf("sourceDefaultAddr = (%q, %d), want (127.0.0.1:8151, 8151)", addr, port)
	}
}

// TestBrowserContract_AcceptsMatchingDocAndReadme is the twin of test_demo_contract_accepts_matching_doc_and_readme.
func TestBrowserContract_AcceptsMatchingDocAndReadme(t *testing.T) {
	root := t.TempDir()
	demo := demoReg{name: "guarddemo", basePath: "/guarddemo", apiPath: "api/scenarios", pageMarker: "safety", defaultPort: 8151}
	demoWrite(t, root, "cmd/guarddemo/main.go", browserMain("8151", "/guarddemo"))
	demoWrite(t, root, "cmd/guarddemo/README.md", "FAK_DEMO_BASE_PATH=/guarddemo go run ./cmd/guarddemo\n")
	runDoc := "\ngo run ./cmd/guarddemo # -> http://127.0.0.1:8151\n" +
		"PORT=8151 ./guarddemo\n" +
		"FAK_DEMO_BASE_PATH=/guarddemo PORT=8151 ./guarddemo\n" +
		"location /guarddemo/ {\n  proxy_pass http://127.0.0.1:8151;\n}\n"
	publicDoc := "go run ./cmd/guarddemo # -> http://127.0.0.1:8151\n"
	if defects := demoContractDefects(root, demo, runDoc, publicDoc); len(defects) != 0 {
		t.Fatalf("expected clean, got defects: %+v", defects)
	}
}

// TestBrowserContract_FlagsMismatchedPortAndMissingReadme is the twin of
// test_demo_contract_flags_mismatched_port_and_missing_readme_command.
func TestBrowserContract_FlagsMismatchedPortAndMissingReadme(t *testing.T) {
	root := t.TempDir()
	demo := demoReg{name: "guarddemo", basePath: "/guarddemo", apiPath: "api/scenarios", pageMarker: "safety", defaultPort: 8151}
	demoWrite(t, root, "cmd/guarddemo/main.go", "const defaultAddr = \"127.0.0.1:9000\"\n")
	demoWrite(t, root, "cmd/guarddemo/README.md", "go run ./cmd/guarddemo\n")
	defects := demoContractDefects(root, demo, "", "")
	if !containsStr(defects, "guarddemo: main.go defaultAddr 127.0.0.1:9000, want 127.0.0.1:8151") {
		t.Fatalf("want defaultAddr-mismatch defect, got %+v", defects)
	}
	if !demoAnyContains(defects, "README missing base-path env command") {
		t.Fatalf("want README-missing defect, got %+v", defects)
	}
	if !demoAnyContains(defects, "missing local loopback URL") {
		t.Fatalf("want loopback-URL defect, got %+v", defects)
	}
}

// TestBrowserContract_FlagsAdHocBasePathHelpers is the twin of test_demo_contract_flags_ad_hoc_base_path_helpers.
func TestBrowserContract_FlagsAdHocBasePathHelpers(t *testing.T) {
	root := t.TempDir()
	demo := demoReg{name: "guarddemo", basePath: "/guarddemo", apiPath: "api/scenarios", pageMarker: "safety", defaultPort: 8151}
	demoWrite(t, root, "cmd/guarddemo/main.go", "\nconst defaultAddr = \"127.0.0.1:8151\"\n"+
		"func main() {\n    _ = flag.String(\"base-path\", os.Getenv(demoui.DemoBasePathEnv), \"path\")\n}\n"+
		"func listenAddr(addr string) string { return addr }\n")
	defects := demoContractDefects(root, demo, "", "")
	for _, want := range []string{
		"missing shared base-path flag helper",
		"defines base-path flag directly",
		"defines local listenAddr",
		"reads guarddemo base-path env directly",
	} {
		if !demoAnyContains(defects, want) {
			t.Fatalf("want a defect containing %q, got %+v", want, defects)
		}
	}
}

// TestBrowserContract_FlagsPublicPageDrift is the twin of test_demo_contract_flags_public_page_drift.
func TestBrowserContract_FlagsPublicPageDrift(t *testing.T) {
	root := t.TempDir()
	demo := demoReg{name: "demorace", basePath: "/demorace", apiPath: "api/run", pageMarker: "race", defaultPort: 8147}
	demoWrite(t, root, "cmd/demorace/main.go", browserMain("8147", "/demorace"))
	demoWrite(t, root, "cmd/demorace/README.md", "FAK_DEMO_BASE_PATH=/demorace go run ./cmd/demorace\n")
	runDoc := "\ngo run ./cmd/demorace # -> http://127.0.0.1:8147\n" +
		"PORT=8147 ./demorace\n" +
		"FAK_DEMO_BASE_PATH=/demorace PORT=8147 ./demorace\n" +
		"location /demorace/ {\n  proxy_pass http://127.0.0.1:8147;\n}\n"
	publicDoc := "<code>./cmd/demorace</code>\n"
	defects := demoContractDefects(root, demo, runDoc, publicDoc)
	for _, want := range []string{
		"docs/demos.html missing public local go-run command",
		"docs/demos.html missing public local loopback URL",
		"docs/demos.html has bare command path",
	} {
		if !demoAnyContains(defects, want) {
			t.Fatalf("want a defect containing %q, got %+v", want, defects)
		}
	}
}

// TestBrowserContract_MissingDefaultPort covers the demo_contract_defects early-return branch: a demo
// with default_port<=0 yields exactly the registry-missing-default_port defect.
func TestBrowserContract_MissingDefaultPort(t *testing.T) {
	root := t.TempDir()
	demo := demoReg{name: "noport", basePath: "/noport", apiPath: "api/x", pageMarker: "x", defaultPort: 0}
	defects := demoContractDefects(root, demo, "", "")
	if len(defects) != 1 || defects[0] != "noport: DEMOS registry missing default_port" {
		t.Fatalf("want single missing-default_port defect, got %+v", defects)
	}
}

// TestBrowserContract_LifecycleDefectsClean asserts the shipped registry's lifecycle decisions are
// internally consistent (the twin of demo_registry.lifecycle_defects() returning []).
func TestBrowserContract_LifecycleDefectsClean(t *testing.T) {
	if d := demoLifecycleDefects(); len(d) != 0 {
		t.Fatalf("shipped registry has lifecycle defects: %+v", d)
	}
}

// TestBrowserContract_RegistryMatchesPython pins demoRegistry + demoLifecycle against the live
// tools/demo_registry.py. A registry add/remove/edit that skips this file reds `go test` here, so the
// Go table can never silently drift from the oracle. Skipped under -short or when python/git is absent.
func TestBrowserContract_RegistryMatchesPython(t *testing.T) {
	if testing.Short() {
		t.Skip("python registry pin skipped under -short")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	clone := repoRoot(t)

	// Dump each demo as "name|base_path|default_port|hosted_path|hosted_port" and each lifecycle
	// decision as "name|state|issue", in registry order, for a deterministic string compare.
	script := "import sys; sys.path.insert(0, 'tools'); import demo_registry as dr\n" +
		"print('DEMOS')\n" +
		"[print('|'.join([d.name, d.base_path, str(d.default_port), d.hosted_path, str(d.hosted_port)])) for d in dr.DEMOS]\n" +
		"print('LIFECYCLE')\n" +
		"[print('|'.join([n, dr.DEMO_LIFECYCLE[n].state, str(dr.DEMO_LIFECYCLE[n].issue)])) for n in sorted(dr.DEMO_LIFECYCLE)]\n"
	args := append(append([]string{}, pyArgs...), "-c", script)
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("python demo_registry dump failed: %v", err)
	}

	var pyDemos, pyLifecycle []string
	section := ""
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		s := strings.TrimSpace(line)
		if s == "DEMOS" || s == "LIFECYCLE" {
			section = s
			continue
		}
		if s == "" {
			continue
		}
		if section == "DEMOS" {
			pyDemos = append(pyDemos, s)
		} else {
			pyLifecycle = append(pyLifecycle, s)
		}
	}

	var goDemos []string
	for _, d := range demoRegistry {
		goDemos = append(goDemos, strings.Join([]string{
			d.name, d.basePath, itoa(int64(d.defaultPort)), d.hostedPath, itoa(int64(d.hostedPort)),
		}, "|"))
	}
	if strings.Join(goDemos, "\n") != strings.Join(pyDemos, "\n") {
		t.Fatalf("demoRegistry drifted from demo_registry.DEMOS:\n  go:\n%s\n  python:\n%s",
			strings.Join(goDemos, "\n"), strings.Join(pyDemos, "\n"))
	}

	var goLifecycle []string
	var names []string
	for n := range demoLifecycle {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		d := demoLifecycle[n]
		goLifecycle = append(goLifecycle, strings.Join([]string{n, d.state, itoa(int64(d.issue))}, "|"))
	}
	if strings.Join(goLifecycle, "\n") != strings.Join(pyLifecycle, "\n") {
		t.Fatalf("demoLifecycle drifted from demo_registry.DEMO_LIFECYCLE:\n  go:\n%s\n  python:\n%s",
			strings.Join(goLifecycle, "\n"), strings.Join(pyLifecycle, "\n"))
	}
}

// TestBrowserContract_LiveTreeClean asserts the real tracked tree carries no browser-demo contract
// drift — the twin of `make hygiene` running tools/demo_browser_contract.py green. Skipped outside a
// git checkout.
func TestBrowserContract_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateBrowserContractTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("browser-demo contract drift on the tracked tree: %+v", findings)
	}
}

// TestBrowserContract_PythonParity is the verdict-level differential: the ported gate and the live
// Python checker must agree (clean vs. defect) over the SAME real workspace. Both default to the repo
// root. Skipped under -short or when python/git is absent.
func TestBrowserContract_PythonParity(t *testing.T) {
	if testing.Short() {
		t.Skip("python parity skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	clone := repoRoot(t)

	tree, err := ReadTrackedTree(clone)
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateBrowserContractTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	goBad := len(findings) > 0

	args := append(append([]string{}, pyArgs...), "tools/demo_browser_contract.py")
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, _ := cmd.CombinedOutput()
	pyBad := cmd.ProcessState.ExitCode() == 1

	if goBad != pyBad {
		t.Fatalf("VERDICT MISMATCH: go bad=%v (%d findings) vs python bad=%v\npython said: %s\ngo findings: %+v",
			goBad, len(findings), pyBad, out, findings)
	}
}
