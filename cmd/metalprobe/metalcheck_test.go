package main

import (
	"runtime"
	"strings"
	"testing"
)

// ok/notOK shorthand tool results keep the case tables readable.
var (
	okTool   = ToolCheck{OK: true, Extra: "present"}
	missTool = ToolCheck{OK: false, Extra: "missing for test"}
)

// TestAdjudicateExitCodes drives every branch of the darwin/arm64 gate logic — plus
// the not-applicable path — over injected inputs only. No darwin binary is executed;
// windows and darwin hits share this single pure table.
func TestAdjudicateExitCodes(t *testing.T) {
	cases := []struct {
		name string
		req  Requirements
		want int
	}{
		{
			name: "darwinarm64 all prereqs pass => run linkcheck",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: okTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: 2,
		},
		{
			name: "darwinarm64 cgo env unset means enabled => run linkcheck",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "",
				GoTool: okTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: 2,
		},
		{
			name: "darwinarm64 CGO_ENABLED=0 => fail",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "0",
				GoTool: okTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: 1,
		},
		{
			name: "darwinarm64 CGO_ENABLED='0 ' (padded) => fail",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: " 0 ",
				GoTool: okTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: 1,
		},
		{
			name: "darwinarm64 go toolchain missing => fail",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: 1,
		},
		{
			name: "darwinarm64 xcode CLT missing => fail",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: okTool, XcodeCLT: missTool, Clang: okTool,
			},
			want: 1,
		},
		{
			name: "darwinarm64 clang missing => fail",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: okTool, XcodeCLT: okTool, Clang: missTool,
			},
			want: 1,
		},
		{
			name: "darwin amd64 => not applicable, exit 0",
			req: Requirements{
				Goos: "darwin", Goarch: "amd64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "linux arm64 => not applicable, exit 0",
			req: Requirements{
				Goos: "linux", Goarch: "arm64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "windows amd64 => not applicable, exit 0",
			req: Requirements{
				Goos: "windows", Goarch: "amd64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "windows arm64 => not applicable, exit 0",
			req: Requirements{
				Goos: "windows", Goarch: "arm64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "freebsd 386 => not applicable, exit 0",
			req: Requirements{
				Goos: "freebsd", Goarch: "386", CgoEnabled: "0",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "msys amd64-w64 triple folds to goamd64 => not applicable, exit 0",
			req: Requirements{
				Goos: "windows", Goarch: "amd64-w64", CgoEnabled: "1",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
		{
			name: "not-applicable beats missing prereqs even with CGO_ENABLED=0",
			req: Requirements{
				Goos: "linux", Goarch: "amd64", CgoEnabled: "0",
				GoTool: missTool, XcodeCLT: missTool, Clang: missTool,
			},
			want: 0,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := Adjudicate(tc.req); got != tc.want {
				t.Fatalf("Adjudicate(%+v) = %d, want %d", tc.req, got, tc.want)
			}
		})
	}
}

// TestFailureMessagesNamePrerequisite checks that every exit-1 darwin path emits a
// stderr message naming the exact missing prerequisite.
func TestFailureMessagesNamePrerequisite(t *testing.T) {
	cases := []struct {
		name string
		req  Requirements
		want string
	}{
		{
			name: "cgo disabled",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "0",
				GoTool: okTool, XcodeCLT: okTool, Clang: okTool,
			},
			want: "cgo is disabled (CGO_ENABLED=0)",
		},
		{
			name: "go toolchain missing",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: ToolCheck{OK: false, Extra: "'go' not found in PATH"}, XcodeCLT: okTool, Clang: okTool,
			},
			want: "go toolchain missing",
		},
		{
			name: "xcode CLT missing",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: okTool, XcodeCLT: ToolCheck{OK: false, Extra: "'xcode-select -p' failed"}, Clang: okTool,
			},
			want: "Xcode Command Line Tools missing",
		},
		{
			name: "clang missing",
			req: Requirements{
				Goos: "darwin", Goarch: "arm64", CgoEnabled: "1",
				GoTool: okTool, XcodeCLT: okTool, Clang: ToolCheck{OK: false, Extra: "'clang' not found in PATH"},
			},
			want: "clang missing",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := PopulateFailure(tc.req)
			if !strings.Contains(got, tc.want) {
				t.Fatalf("PopulateFailure(%+v) = %q, want substring %q", tc.req, got, tc.want)
			}
			if got == "" {
				t.Fatalf("PopulateFailure(%+v) = empty", tc.req)
			}
		})
	}
}

// TestVerdictGrammar locks the exact verdict-line grammar for every outcome class,
// including the CPU-fallback labeling and the not-applicable line.
func TestVerdictGrammar(t *testing.T) {
	cases := []struct {
		name  string
		verdy string
	}{
		{name: "not applicable", verdy: PopulatedVerdict(Requirements{Goos: "windows", Goarch: "amd64", CgoEnabled: "1"})},
		{name: "cgo disabled verdict", verdy: PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "0", GoTool: okTool, XcodeCLT: okTool, Clang: okTool})},
		{name: "go missing verdict", verdy: PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: missTool, XcodeCLT: okTool, Clang: okTool})},
		{name: "xcode missing verdict", verdy: PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: missTool, Clang: okTool})},
		{name: "clang missing verdict", verdy: PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: missTool})},
		{name: "success verdict", verdy: PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: okTool})},
	}
	for _, tc := range cases {
		if !strings.HasPrefix(tc.verdy, "metal: compiled=") {
			t.Fatalf("%s: verdict %q not in metal grammar", tc.name, tc.verdy)
		}
	}
	// Exact grammar spot-checks for the shaped outcomes. The issue's witness
	// grammar is yes/no ("compiled=yes available=<bool> device=<tier>").
	if got := PopulatedVerdict(Requirements{Goos: "windows", Goarch: "amd64"}); got != "metal: compiled=no available=no device=not-applicable" {
		t.Fatalf("not-applicable verdict = %q", got)
	}
	if got := PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "0", GoTool: okTool, XcodeCLT: okTool, Clang: okTool}); got != "metal: compiled=no available=no device=cgo-disabled" {
		t.Fatalf("cgo-disabled verdict = %q", got)
	}
	if got := PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: ToolCheck{OK: false}, XcodeCLT: okTool, Clang: okTool}); got != "metal: compiled=no available=no device=go-toolchain-missing" {
		t.Fatalf("go-missing verdict = %q", got)
	}
	if got := PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: ToolCheck{OK: false}, Clang: okTool}); got != "metal: compiled=no available=no device=xcode-clt-missing" {
		t.Fatalf("xcode-missing verdict = %q", got)
	}
	if got := PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: ToolCheck{OK: false}}); got != "metal: compiled=no available=no device=clang-missing" {
		t.Fatalf("clang-missing verdict = %q", got)
	}
	if got := PopulatedVerdict(Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: okTool}); got != "metal: compiled=yes available=yes device=metal" {
		t.Fatalf("success verdict = %q", got)
	}
	// CPU-fallback label must be explicit on the allowed-pass path (verdict line
	// plus the stdout PASS line naming the fallback).
	fbMsg, fbPass := PopulateCPUFallback("Apple M3 Pro")
	if fbMsg != "metal: compiled=yes available=no device=cpu-fallback(Apple M3 Pro)" {
		t.Fatalf("cpu-fallback verdict = %q", fbMsg)
	}
	if !strings.Contains(fbPass, "CPU-fallback") || !strings.Contains(fbPass, "metal-check: PASS") {
		t.Fatalf("cpu-fallback pass line %q must label the fallback", fbPass)
	}
	// Blank device name still yields a well-formed cpu-fallback line.
	blankMsg, blankPass := PopulateCPUFallback("")
	if blankMsg != "metal: compiled=yes available=no device=cpu-fallback(unknown)" {
		t.Fatalf("blank cpu-fallback verdict = %q", blankMsg)
	}
	if blankPass != "metal-check: PASS (compiled=yes available=no device=cpu-fallback(unknown) -> CPU-fallback build; serving will not use Metal device)" {
		t.Fatalf("blank cpu-fallback pass line = %q", blankPass)
	}
	// Device-available pass line pins the two-line stdout contract.
	dpMsg, dpPass := PopulateDevicePass("Apple M3 Pro")
	if dpMsg != "metal: compiled=yes available=yes device=Apple M3 Pro" {
		t.Fatalf("device-pass verdict = %q", dpMsg)
	}
	if dpPass != "metal-check: PASS (compiled=yes available=yes device=Apple M3 Pro)" {
		t.Fatalf("device-pass pass line = %q", dpPass)
	}
	// LAUNCH failure message pins the leaf path and exit-code context.
	lf := PopulateLinkcheckFailure("boom", 1)
	if !strings.Contains(lf, "./cmd/metalprobe/linkcheck") || !strings.Contains(lf, "exited 1") {
		t.Fatalf("linkcheck failure = %q", lf)
	}
}

// TestVerdictDeterministic asserts the verdict formatter is deterministic: two equal
// inputs must produce byte-identical verdict strings.
func TestVerdictDeterministic(t *testing.T) {
	inputs := []Requirements{
		{Goos: "windows", Goarch: "amd64", CgoEnabled: "1", GoTool: missTool, XcodeCLT: missTool, Clang: missTool},
		{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: okTool},
		{Goos: "darwin", Goarch: "arm64", CgoEnabled: "0", GoTool: missTool, XcodeCLT: missTool, Clang: missTool},
		{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: missTool, Clang: okTool},
	}
	for i, req := range inputs {
		first := PopulatedVerdict(req)
		for j := 0; j < 5; j++ {
			again := PopulatedVerdict(req)
			if again != first {
				t.Fatalf("case %d: verdict not deterministic: %q vs %q", i, first, again)
			}
		}
		if na := PopulateNotApplicable(req); na != PopulateNotApplicable(req) {
			t.Fatalf("case %d: not-applicable message not deterministic", i)
		}
	}
}

// TestArePrereqsSatisfied keeps the used helper aligned with Adjudicate's codes.
func TestArePrereqsSatisfied(t *testing.T) {
	pass := Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "1", GoTool: okTool, XcodeCLT: okTool, Clang: okTool}
	if ArePrereqsSatisfied(pass) != true {
		t.Fatal("prereqs should be satisfied for a fully-passing darwin/arm64 input")
	}
	fail := Requirements{Goos: "darwin", Goarch: "arm64", CgoEnabled: "0", GoTool: okTool, XcodeCLT: okTool, Clang: okTool}
	if ArePrereqsSatisfied(fail) != false {
		t.Fatal("cgo-disabled input must fail the prereq gate")
	}
	if ArePrereqsSatisfied(Requirements{Goos: "windows", Goarch: "amd64"}) != true {
		t.Fatal("not-applicable hosts are not prereq failures")
	}
}

// TestParseProbeLine locks the leaf-output parser: exact form, toleration of
// surrounding log noise, and rejection of absent probe lines.
func TestParseProbeLine(t *testing.T) {
	got, ok := parseProbeLine("probe: compiled=true available=true device=Apple M2 Max\n")
	if !ok || !got.compiled || !got.available || got.device != "Apple M2 Max" {
		t.Fatalf("parseProbeLine(good) = %+v, ok=%v", got, ok)
	}
	got, ok = parseProbeLine("noise\nprobe: compiled=true available=false device=\nmore noise\n")
	if !ok || !got.compiled || got.available || got.device != "" {
		t.Fatalf("parseProbeLine(fallback) = %+v, ok=%v", got, ok)
	}
	if _, ok := parseProbeLine("no probe line here\n"); ok {
		t.Fatal("parseProbeLine should reject input with no probe: prefix")
	}
	if _, ok := parseProbeLine(""); ok {
		t.Fatal("parseProbeLine should reject empty input")
	}
}

// TestCanonicalLower keeps the msys-triple folding well-defined for windows hosts.
func TestCanonicalLower(t *testing.T) {
	cases := map[string]string{
		"amd64":      "amd64",
		"AMD64":      "amd64",
		"arm64":      "arm64",
		"ARM64":      "arm64",
		"x86_64":     "amd64",
		"x86_64-w64": "amd64",
		"amd64-w64":  "amd64",
		"386":        "386",
	}
	for in, want := range cases {
		if got := canonicalLower(in); got != want {
			t.Fatalf("canonicalLower(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestHostRequirementsPopulatesAllPrereqs is the regression witness for the
// metal-check false "Xcode Command Line Tools missing" failure: Run used to build
// Requirements with only GoTool populated, so Adjudicate always saw XcodeCLT.OK and
// Clang.OK as false and failed a correctly-configured host. hostRequirements is the
// extracted wiring under test; it must populate every prerequisite field it gates on
// (each ToolCheck is a status, and OK=false is the honest "not applicable on this
// host" status the shims deliberately emit on non-darwin hosts — Adjudicate returns
// the not-applicable exit 0 before consulting them there).
//
// The gate is only *applicable* on darwin/arm64, so that is the sole host where every
// lookup must actually succeed. On non-darwin hosts (Linux CI, Windows) the probes'
// OK value is host-dependent — `xcode-select` is absent on Linux, and every lookup is
// statically false on Windows — so asserting OK a priori was the bug; the beat under
// test is that hostRequirements wires each field through with a status, not that a
// Linux or Windows box has Xcode. An empty Extra with OK=false would also signal an
// unwired field, so that is rejected.
func TestHostRequirementsPopulatesAllPrereqs(t *testing.T) {
	req := hostRequirements(runtime.GOARCH)
	if req.Goos != runtime.GOOS {
		t.Fatalf("hostRequirements Goos = %q, want %q", req.Goos, runtime.GOOS)
	}
	if req.Goarch != runtime.GOARCH {
		t.Fatalf("hostRequirements Goarch = %q, want %q", req.Goarch, runtime.GOARCH)
	}
	applicable := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
	populated := []struct {
		name string
		got  ToolCheck
	}{
		{"GoTool", req.GoTool},
		{"XcodeCLT", req.XcodeCLT},
		{"Clang", req.Clang},
	}
	for _, tc := range populated {
		if applicable {
			// Sole applicable host: every prerequisite must be genuinely present.
			if !tc.got.OK {
				t.Errorf("hostRequirements left %s unpopulated on applicable darwin/arm64 host: %+v", tc.name, tc.got)
			}
			continue
		}
		// Non-applicable host: the field must still carry a status, not be left as
		// the zero value. A failing lookup names itself in Extra; the windows shims
		// do the same with their "not applicable" text.
		if !tc.got.OK && strings.TrimSpace(tc.got.Extra) == "" {
			t.Errorf("hostRequirements left %s without a status (OK=false and empty Extra): %+v", tc.name, tc.got)
		}
	}
	if applicable {
		if got := Adjudicate(req); got != 2 {
			t.Fatalf("Adjudicate(hostRequirements) = %d, want 2 (prereqs satisfied) on a configured darwin/arm64 host; verdict=%q failure=%q",
				got, PopulatedVerdict(req), PopulateFailure(req))
		}
	} else if got := Adjudicate(req); got != 0 {
		t.Fatalf("Adjudicate(hostRequirements) = %d, want 0 (not applicable) on %s/%s",
			got, runtime.GOOS, runtime.GOARCH)
	}
}
