// Package main implements metal-check: the fail-loud cgo device/build gate for the
// Metal prefill backend (internal/metalgemm). It verifies every prerequisite for a
// cgo Metal build BEFORE attempting one, then triggers command `go run
// ./cmd/metalprobe/linkcheck` — the only place the cgo Metal backend is linked on a
// darwin/arm64 host — and folds the leaf's compiled/available/device report into a
// single verdict line:
//
//	metal: compiled=<bool> available=<bool> device=<tier>
//
// Contract:
//   - any missing prerequisite fails with exit 1 and a message naming exactly which
//     prerequisite is missing (cgo disabled, Xcode CLT, clang, go toolchain, or a
//     failed/short linkcheck report);
//   - cgo Metal code compiled fine but no Metal device could be initialised is a
//     PASS — the runtime falls back to the hand-tuned CPU kernels — and the verdict
//     is explicitly labeled "cpu-fallback";
//   - non-darwin/arm64 hosts (windows/amd64 included) are not applicable: a clear
//     message plus exit 0, so the portable Makefile target can run unchanged there.
//
// The decision core is pure (metalcheck.go) and table-driven-tested without ever
// spawning darwin binaries; the host-specific pieces (goarch, env, toolchain lookups)
// are injected function values collected by probeHostPrereqs + linkcheckLeaf.
package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

// ToolCheck is the raw result of one external tool lookup (e.g. `xcode-select -p`).
type ToolCheck struct {
	OK    bool
	Extra string // extra descriptive text folded into the failure message
}

// Requirements bundles every gate input that varies across hosts and cgo settings.
// It is the injected-input struct the pure Adjudicate function decides over.
type Requirements struct {
	Goos       string // runtime.GOOS of the running gate binary ("darwin", "windows", ...)
	Goarch     string
	CgoEnabled string // CGO_ENABLED from the environment; "" means enabled (Go default)
	GoTool     ToolCheck
	XcodeCLT   ToolCheck
	Clang      ToolCheck
}

// canonicalLower folds a GOARCH value into the pure logic's canonical form: on
// win32/MSYS hosts a GNU make's uname triple can smuggle an msys arch into the
// environment, so the windows runner normalizes to the exact GOARCH the go tool
// would use.
func canonicalLower(goarch string) string {
	switch strings.ToLower(strings.TrimSpace(goarch)) {
	case "amd64", "x86_64", "x86-64", "amd64-w64", "x86_64-w64":
		return "amd64"
	case "arm64":
		return "arm64"
	default:
		return goarch
	}
}

// withCanonicalArch returns a copy of req whose Goarch has been folded through
// canonicalLower. Applied once at entry so pure logic sees a clean GOARCH.
func (req Requirements) withCanonicalArch() Requirements {
	req.Goarch = canonicalLower(req.Goarch)
	return req
}

// cgoDisabled reports whether CGO_ENABLED is explicitly set to 0. Anything else —
// unset, "1", "2" — means cgo is enabled (the Go default).
func cgoDisabled(cgoEnabled string) bool {
	return strings.TrimSpace(cgoEnabled) == "0"
}

// Adjudicate decides the gate's outcome purely from the injected Requirements. The
// returned int is the process exit code; stdout is PopulatedVerdict(req) and stderr
// is FailureReason. It never touches a toolchain, an executable, or the network.
func Adjudicate(req Requirements) int {
	req = req.withCanonicalArch()
	if req.Goos != "darwin" || req.Goarch != "arm64" {
		return 0 // NotApplicable: non-darwin/arm64 hosts pass portably
	}
	if cgoDisabled(req.CgoEnabled) {
		return 1 // CGODisabled: fail with the named prerequisite
	}
	if !req.GoTool.OK {
		return 1 // NoGoToolchain
	}
	if !req.XcodeCLT.OK {
		return 1 // NoXcodeCLT
	}
	if !req.Clang.OK {
		return 1 // NoClang
	}
	return 2 // InvokeLinkcheck: prereqs satisfied; run the cgo leaf
}

// PopulatedVerdict builds the exact one-line verdict the gate prints for a given
// Requirements. While prereqs are failing the verdict is still well-defined — the
// device tier becomes the named prerequisite caveat — so that every exit path
// prints grammar-exact output, deterministic for equal inputs.
func PopulatedVerdict(req Requirements) string {
	req = req.withCanonicalArch()
	if req.Goos != "darwin" || req.Goarch != "arm64" {
		return "metal: compiled=no available=no device=not-applicable"
	}
	if cgoDisabled(req.CgoEnabled) {
		return "metal: compiled=no available=no device=cgo-disabled"
	}
	if !req.GoTool.OK {
		return "metal: compiled=no available=no device=go-toolchain-missing"
	}
	if !req.XcodeCLT.OK {
		return "metal: compiled=no available=no device=xcode-clt-missing"
	}
	if !req.Clang.OK {
		return "metal: compiled=no available=no device=clang-missing"
	}
	// Prereqs pass only after the leaf has compiled and run, so compiled/available
	// both read true here; device naming is the leaf's job via the at-exit variable.
	return "metal: compiled=yes available=yes device=metal"
}

// PopulateFailure builds the exit-1 stderr message naming the exact missing
// prerequisite.
func PopulateFailure(req Requirements) string {
	if cgoDisabled(req.CgoEnabled) {
		return "metal-check failed: cgo is disabled (CGO_ENABLED=0); enable it (unset CGO_ENABLED or set CGO_ENABLED=1) metal prefill requires cgo"
	}
	if !req.GoTool.OK {
		return "metal-check failed: go toolchain missing (" + req.GoTool.Extra + ")"
	}
	if !req.XcodeCLT.OK {
		return "metal-check failed: Xcode Command Line Tools missing (" + req.XcodeCLT.Extra + "; install with: xcode-select --install)"
	}
	if !req.Clang.OK {
		return "metal-check failed: clang missing (" + req.Clang.Extra + "; ships with Xcode Command Line Tools)"
	}
	return "" // no prerequisite failure; linkcheck result decides
}

// PopulateNotApplicable is the stdout for non-darwin/arm64 hosts.
func PopulateNotApplicable(req Requirements) string {
	return fmt.Sprintf("metal-check: not applicable on %s/%s (metal-backend gate requires darwin/arm64); skipping", req.Goos, req.Goarch)
}

// PopulateCPUFallback is the verdict+pass line pair for the device-unavailable-but-
// compiled outcome: allowed pass, explicitly labeled CPU-fallback, both on stdout.
func PopulateCPUFallback(device string) (string, string) {
	if strings.TrimSpace(device) == "" {
		device = "unknown"
	}
	return "metal: compiled=yes available=no device=cpu-fallback(" + device + ")",
		"metal-check: PASS (compiled=yes available=no device=cpu-fallback(" + device + ") -> CPU-fallback build; serving will not use Metal device)"
}

// PopulateDevicePass is the verdict+pass line pair for the device-available outcome.
func PopulateDevicePass(device string) (string, string) {
	if strings.TrimSpace(device) == "" {
		device = "unknown"
	}
	return "metal: compiled=yes available=yes device=" + device,
		"metal-check: PASS (compiled=yes available=yes device=" + device + ")"
}

// PopulateLinkcheckFailure is the stderr message for a linkcheck run that exited
// non-zero, produced unparsable output, or lacked a probe: line.
func PopulateLinkcheckFailure(out string, exitCode int) string {
	return fmt.Sprintf("metal-check failed: linkcheck leaf (go run ./cmd/metalprobe/linkcheck) exited %d with output %q; the cgo Metal backend did not compile or run", exitCode, strings.TrimSpace(out))
}

// runtimeArch is the arch of the running binary: the injected Goarch value for host
// execution, and what `go run` from any make compiles against.
func runtimeArch() string { return runtime.GOARCH }

// linkcheckResult is the payload parsed from linkcheck's single probe line.
type linkcheckResult struct {
	compiled  bool
	available bool
	device    string
}

// parseProbeLine extracts the leaf's single "probe: compiled=... available=...
// device=..." line, tolerating extra stdout noise around it. Device values may
// contain spaces, so the device field consumes the remainder of the line.
func parseProbeLine(out string) (linkcheckResult, bool) {
	var res linkcheckResult
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "probe:") {
			continue
		}
		idx := strings.Index(line, "device=")
		if idx >= 0 {
			res.device = line[idx+len("device="):]
		}
		for _, f := range strings.Fields(line) {
			switch f {
			case "compiled=true":
				res.compiled = true
			case "available=true":
				res.available = true
			}
		}
		return res, true
	}
	return res, false
}

// ArePrereqsSatisfied reports whether all darwin/arm64 build prereqs pass (the
// Adjudicate != 1 complement used by emitVerdict and the failure path).
func ArePrereqsSatisfied(req Requirements) bool {
	return Adjudicate(req) != 1
}

// Run executes the gate for the given goarch, returning the process exit code. All
// host-specific lookups (env, toolchain) are performed here so the pure decision body
// in Adjudicate stays table-testable.
func Run(goarch string) int {
	req := Requirements{
		Goos:       runtime.GOOS,
		Goarch:     goarch,
		CgoEnabled: os.Getenv("CGO_ENABLED"),
		GoTool:     checkGoTool(),
	}
	if code := Adjudicate(req); code != 2 {
		fmt.Println(PopulatedVerdict(req))
		if code == 1 {
			fmt.Fprintln(os.Stderr, PopulateFailure(req))
		} else {
			fmt.Println(PopulateNotApplicable(req))
		}
		return code
	}
	// darwin/arm64 with prereqs satisfied: run the cgo leaf.
	out, exit := runLinkcheck()
	if exit != 0 {
		fmt.Fprintln(os.Stderr, PopulateLinkcheckFailure(out, exit))
		return 1
	}
	line, ok := parseProbeLine(out)
	if !ok {
		fmt.Fprintln(os.Stderr, PopulateLinkcheckFailure(out, exit))
		return 1
	}
	if line.available {
		msg, pass := PopulateDevicePass(line.device)
		fmt.Println(msg)
		fmt.Println(pass)
		return 0
	}
	msg, pass := PopulateCPUFallback(line.device)
	fmt.Println(msg)
	fmt.Println(pass)
	return 0
}
