package hooks

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// gate_browsercontract.go — the BROWSER_CONTRACT gate, a byte-faithful port of
// tools/demo_browser_contract.py. Where DEMO_COMMAND catches docs that tell a reader to run a demo
// command that no longer resolves, this catches metadata DRIFT around each browser demo's documented
// defaults: the Go `const defaultAddr`, the shared demoui base-path/listen/URL helpers in main.go,
// the local run-doc examples (loopback URL, PORT= launch, base-path env, nginx location/proxy_pass),
// the public docs/demos.html go-run command + loopback URL (and the bare-path anti-pattern), and the
// per-demo README base-path env command. It also folds in the registry's lifecycle-decision defects.
// Static and network-free — it never binds a port or runs a demo.
//
// This is a TREE-mode-ONLY gate (like DEMO_COMMAND): demo_browser_contract.py has no --audit-staged
// branch — `make hygiene` invokes `python3 tools/demo_browser_contract.py` over the whole workspace —
// so there is no staged Gate twin, only the HygieneGate wired in tree.go. The Python checker stays on
// disk as the make-hygiene exit-2 fallback and as the parity oracle (gate_browsercontract_test.go).
//
// Parity anchor: demo_browser_contract.py collect() / demo_contract_defects() / source_helper_defects()
// / source_default_addr(). The shared registry it keys on lives in demo_registry.go.

const (
	browserRunDoc    = "docs/run-the-demos.md"
	browserPublicDoc = "docs/demos.html"
)

// demoDefaultAddrRE mirrors demo_browser_contract.DEFAULT_ADDR_RE. Group 1 is the addr, group 2 the
// port. RE2 has no named-group ergonomics here, so the caller reads by index.
var demoDefaultAddrRE = regexp.MustCompile(`const\s+defaultAddr\s*=\s*"(127\.0\.0\.1:(\d+))"`)

// readDemoFile ports _read: return the file text, or "" on any read error (the Python swallows OSError).
func readDemoFile(root string, rel ...string) string {
	parts := append([]string{root}, rel...)
	body, err := os.ReadFile(filepath.Join(parts...))
	if err != nil {
		return ""
	}
	return string(body)
}

// sourceDefaultAddr ports source_default_addr: extract (addr, port) from cmd/<name>/main.go's
// `const defaultAddr = "127.0.0.1:<port>"`, or ("", 0) when it is absent.
func sourceDefaultAddr(root string, demo demoReg) (string, int) {
	text := readDemoFile(root, "cmd", demo.name, "main.go")
	m := demoDefaultAddrRE.FindStringSubmatch(text)
	if m == nil {
		return "", 0
	}
	port, _ := strconv.Atoi(m[2])
	return m[1], port
}

// needleLabel is one ordered (needle, label) containment check — the Go twin of the Python dicts,
// whose insertion order the containment loops depend on.
type needleLabel struct{ needle, label string }

// sourceHelperDefects ports source_helper_defects: main.go must use the shared demoui helpers and
// must NOT hand-roll the base-path flag, listenAddr, or base-path env read.
func sourceHelperDefects(sourceText string, demo demoReg) []string {
	var defects []string
	checks := []needleLabel{
		{`demoui.BasePathFlag(flag.CommandLine, "` + demo.basePath + `")`, "shared base-path flag helper"},
		{"demoui.MountWithBasePath(", "shared base-path mount helper"},
		{"demoui.ListenAddr(", "shared PORT/listen helper"},
		{"demoui.LocalURL(", "shared startup URL helper"},
	}
	for _, c := range checks {
		if !strings.Contains(sourceText, c.needle) {
			defects = append(defects, demo.name+": main.go missing "+c.label+": "+c.needle)
		}
	}
	if strings.Contains(sourceText, `flag.String("base-path"`) {
		defects = append(defects, demo.name+": main.go defines base-path flag directly; use demoui.BasePathFlag")
	}
	if strings.Contains(sourceText, "func listenAddr(") {
		defects = append(defects, demo.name+": main.go defines local listenAddr; use demoui.ListenAddr")
	}
	if strings.Contains(sourceText, "os.Getenv(demoui.DemoBasePathEnv)") {
		defects = append(defects, demo.name+": main.go reads "+demo.name+" base-path env directly; use demoui.BasePathFlag")
	}
	return defects
}

// demoContractDefects ports demo_contract_defects: check one demo's defaultAddr, helper usage, local
// run-doc examples, public-doc go-run/loopback (and bare-path anti-pattern), and README env command.
func demoContractDefects(root string, demo demoReg, runDocText, publicDocText string) []string {
	var defects []string
	if demo.defaultPort <= 0 {
		return []string{demo.name + ": DEMOS registry missing default_port"}
	}

	sourceText := readDemoFile(root, "cmd", demo.name, "main.go")
	addr, port := sourceDefaultAddr(root, demo)
	wantAddr := "127.0.0.1:" + strconv.Itoa(demo.defaultPort)
	if addr != wantAddr || port != demo.defaultPort {
		shown := addr
		if shown == "" {
			shown = "<missing>"
		}
		defects = append(defects, demo.name+": main.go defaultAddr "+shown+", want "+wantAddr)
	}
	defects = append(defects, sourceHelperDefects(sourceText, demo)...)

	portStr := strconv.Itoa(demo.defaultPort)
	runChecks := []needleLabel{
		{"http://127.0.0.1:" + portStr, "local loopback URL"},
		{"PORT=" + portStr + " ./" + demo.name, "PORT launch example"},
		{"FAK_DEMO_BASE_PATH=" + demo.basePath, "base-path env example"},
		{"location " + demo.basePath + "/", "nginx location"},
		{"proxy_pass http://127.0.0.1:" + portStr + ";", "nginx proxy_pass"},
	}
	for _, c := range runChecks {
		if !strings.Contains(runDocText, c.needle) {
			defects = append(defects, demo.name+": "+browserRunDoc+" missing "+c.label+": "+c.needle)
		}
	}

	publicChecks := []needleLabel{
		{"go run ./cmd/" + demo.name, "public local go-run command"},
		{"http://127.0.0.1:" + portStr, "public local loopback URL"},
	}
	for _, c := range publicChecks {
		if !strings.Contains(publicDocText, c.needle) {
			defects = append(defects, demo.name+": "+browserPublicDoc+" missing "+c.label+": "+c.needle)
		}
	}

	barePublicCmd := "<code>./cmd/" + demo.name + "</code>"
	if strings.Contains(publicDocText, barePublicCmd) {
		defects = append(defects, demo.name+": "+browserPublicDoc+" has bare command path, use go run form: "+barePublicCmd)
	}

	readme := readDemoFile(root, "cmd", demo.name, "README.md")
	envCmd := "FAK_DEMO_BASE_PATH=" + demo.basePath + " go run ./cmd/" + demo.name
	if !strings.Contains(readme, envCmd) {
		defects = append(defects, demo.name+": README missing base-path env command: "+envCmd)
	}
	return defects
}

// browserContractDefects ports collect()'s defect derivation over the workspace root: the doc-missing
// probes, then per-demo contract + row-scoped lifecycle defects, then the deduped tail of any
// remaining lifecycle defects. Returns the defect list (empty == clean), in the same order as the
// Python payload["defects"] so a Finding-per-defect gate surfaces identical content.
func browserContractDefects(root string) []string {
	runDocText := readDemoFile(root, filepath.FromSlash(browserRunDoc))
	publicDocText := readDemoFile(root, filepath.FromSlash(browserPublicDoc))

	var defects []string
	if runDocText == "" {
		defects = append(defects, "read "+browserRunDoc+": missing or empty")
	}
	if publicDocText == "" {
		defects = append(defects, "read "+browserPublicDoc+": missing or empty")
	}

	lifecycleDefects := demoLifecycleDefects()

	for _, demo := range demoRegistry {
		rowDefects := demoContractDefects(root, demo, runDocText, publicDocText)
		// row_lifecycle_defects: lifecycle defects scoped to this demo (by "<name>:" prefix or a
		// "cmd/<name>" mention), appended after the contract defects, in lifecycle order.
		for _, d := range lifecycleDefects {
			if strings.HasPrefix(d, demo.name+":") || strings.Contains(d, "cmd/"+demo.name) {
				rowDefects = append(rowDefects, d)
			}
		}
		defects = append(defects, rowDefects...)
	}
	// Append any lifecycle defect not already surfaced above (the Python dedup tail).
	for _, d := range lifecycleDefects {
		if !containsStr(defects, d) {
			defects = append(defects, d)
		}
	}
	return defects
}

// containsStr reports whether xs contains s.
func containsStr(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// gateBrowserContractTree is the BROWSER_CONTRACT --audit-tree gate: run the browser-demo contract
// audit over the tree root and surface each defect as a Finding. Each defect string is self-describing
// (it names the demo and the missing/mismatched contract), so it is placed whole in Detail.
func gateBrowserContractTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, d := range browserContractDefects(t.Root) {
		findings = append(findings, Finding{Gate: "BROWSER_CONTRACT", Detail: d})
	}
	return findings, nil
}
