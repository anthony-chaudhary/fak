package trunkbuildprobe

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

// A real cmd/fak build-failure block (Windows `\` paths, every "no definition" shape).
const realStderr = "# github.com/anthony-chaudhary/fak/cmd/fak\n" +
	"cmd\\fak\\guard.go:167:41: undefined: guardStopHookSameStopFromEnv\n" +
	"cmd\\fak\\guard.go:181:51: undefined: gateway.DefaultVCacheAnchor\n" +
	"cmd\\fak\\guard.go:966:3: unknown field VCacheAnchor in struct literal of type gateway.Config\n" +
	"cmd\\fak\\guard.go:1093:8: srv.SetLogvaultMetricsProvider undefined " +
	"(type *gateway.Server has no field or method SetLogvaultMetricsProvider)\n" +
	"cmd\\fak\\main.go:52:16: undefined: parseVerbArgv\n"

const realStderrMinimal = "# github.com/anthony-chaudhary/fak/cmd/fak\n" +
	"cmd\\fak\\main.go:52:16: undefined: parseVerbArgv\n"

func symSet(b BuildErrors) map[string]bool {
	s := map[string]bool{}
	for _, m := range b.MissingSymbols {
		s[m.Symbol] = true
	}
	return s
}

func TestParseRealBlock(t *testing.T) {
	p := ParseBuildErrors(realStderr)
	if !reflect.DeepEqual(p.FailingPackages, []string{"github.com/anthony-chaudhary/fak/cmd/fak"}) {
		t.Fatalf("failing_packages = %v", p.FailingPackages)
	}
	syms := symSet(p)
	for _, want := range []string{"guardStopHookSameStopFromEnv", "DefaultVCacheAnchor", "VCacheAnchor", "SetLogvaultMetricsProvider", "parseVerbArgv"} {
		if !syms[want] {
			t.Errorf("missing symbol %q not found in %v", want, syms)
		}
	}
	// pkg-qualified gateway.DefaultVCacheAnchor must reduce to the bare identifier.
	if syms["gateway.DefaultVCacheAnchor"] {
		t.Errorf("pkg-qualified symbol should have been reduced to bare identifier")
	}
}

func TestParseCapturesFileAndLine(t *testing.T) {
	p := ParseBuildErrors(realStderr)
	at := map[string]string{}
	for _, m := range p.MissingSymbols {
		at[m.Symbol] = m.At
	}
	if at["guardStopHookSameStopFromEnv"] != "cmd\\fak\\guard.go:167" {
		t.Errorf("at guardStopHookSameStopFromEnv = %q", at["guardStopHookSameStopFromEnv"])
	}
	if at["parseVerbArgv"] != "cmd\\fak\\main.go:52" {
		t.Errorf("at parseVerbArgv = %q", at["parseVerbArgv"])
	}
}

func TestParseAssociatesSymbolWithPackage(t *testing.T) {
	p := ParseBuildErrors(realStderr)
	for _, m := range p.MissingSymbols {
		if m.ReferencedIn != "github.com/anthony-chaudhary/fak/cmd/fak" {
			t.Errorf("referenced_in = %q for %q", m.ReferencedIn, m.Symbol)
		}
	}
}

func TestParseEmpty(t *testing.T) {
	p := ParseBuildErrors("")
	if len(p.FailingPackages) != 0 || len(p.MissingSymbols) != 0 {
		t.Errorf("empty parse not empty: %+v", p)
	}
}

func TestParseDedupesSymbolInSamePackage(t *testing.T) {
	stderr := "# example.com/p\n" +
		"p\\a.go:1:1: undefined: Foo\n" +
		"p\\b.go:2:2: undefined: Foo\n"
	p := ParseBuildErrors(stderr)
	n := 0
	for _, m := range p.MissingSymbols {
		if m.Symbol == "Foo" {
			n++
		}
	}
	if n != 1 {
		t.Errorf("Foo count = %d, want 1", n)
	}
}

func TestParseSameSymbolDistinctAcrossPackages(t *testing.T) {
	stderr := "# example.com/p\n" +
		"p\\a.go:1:1: undefined: Foo\n" +
		"# example.com/q\n" +
		"q\\a.go:1:1: undefined: Foo\n"
	p := ParseBuildErrors(stderr)
	n := 0
	for _, m := range p.MissingSymbols {
		if m.Symbol == "Foo" {
			n++
		}
	}
	if n != 2 {
		t.Errorf("Foo count = %d, want 2", n)
	}
	if !reflect.DeepEqual(p.FailingPackages, []string{"example.com/p", "example.com/q"}) {
		t.Errorf("failing_packages = %v", p.FailingPackages)
	}
}

func TestDefinesSymbolTrueShapes(t *testing.T) {
	cases := []struct{ content, sym string }{
		{"func parseVerbArgv(argv []string) verb {", "parseVerbArgv"},
		{"func (s *Server) SetLogvaultMetricsProvider(p P) {", "SetLogvaultMetricsProvider"},
		{"type VCacheAnchor struct {", "VCacheAnchor"},
		{"\tVCacheAnchor string", "VCacheAnchor"},
		{"const logvaultMetricsVerifySample = 256", "logvaultMetricsVerifySample"},
		{"var DefaultVCacheAnchor = VCacheAnchor{}", "DefaultVCacheAnchor"},
		{"\tresolveLogvaultDir = \"x\"", "resolveLogvaultDir"},
		{"func Foo[T any](x T) T {", "Foo"},
	}
	for _, c := range cases {
		if !DefinesSymbol(c.content, c.sym) {
			t.Errorf("should define %q: %q", c.sym, c.content)
		}
	}
}

func TestDefinesSymbolFalseUses(t *testing.T) {
	cases := []struct{ content, sym string }{
		{"\tfoo := parseVerbArgv(argv)", "parseVerbArgv"},
		{"\tVCacheAnchor: cfg.Anchor,", "VCacheAnchor"},
		{"\tsrv.VCacheAnchor = x", "VCacheAnchor"},
		{"\treturn gateway.DefaultVCacheAnchor", "DefaultVCacheAnchor"},
	}
	for _, c := range cases {
		if DefinesSymbol(c.content, c.sym) {
			t.Errorf("should NOT define %q: %q", c.sym, c.content)
		}
	}
}

func TestDefinesSymbolEmptyIsFalse(t *testing.T) {
	if DefinesSymbol("anything at all", "") {
		t.Errorf("empty symbol should be false")
	}
}

func TestIsGoBuildablePathBuildable(t *testing.T) {
	for _, p := range []string{"cmd/fak/main.go", "internal/gateway/gateway.go", "cmd\\fak\\main.go"} {
		if !IsGoBuildablePath(p) {
			t.Errorf("should be buildable: %q", p)
		}
	}
}

func TestIsGoBuildablePathExcluded(t *testing.T) {
	for _, p := range []string{
		".head_build_check/cmd/fak/guard.go",
		"internal/_scratch/x.go",
		"pkg/testdata/y.go",
		"vendor/z/a.go",
		"README.md",
		".head_build_check\\cmd\\fak\\guard.go",
	} {
		if IsGoBuildablePath(p) {
			t.Errorf("should be excluded: %q", p)
		}
	}
}

func TestFindForgottenOnlyRealGoDefiner(t *testing.T) {
	missing := []MissingSymbol{{Symbol: "parseVerbArgv", ReferencedIn: "cmd/fak"}}
	uncommitted := map[string]string{
		"cmd/fak/main_preamble.go": "func parseVerbArgv(a []string) verb {}\n",
		"docs/notes.md":            "parseVerbArgv is mentioned here\n",
	}
	got := FindForgottenFiles(missing, uncommitted)
	paths := map[string]bool{}
	for _, f := range got {
		paths[f.Path] = true
	}
	if !paths["cmd/fak/main_preamble.go"] {
		t.Errorf("real definer not returned: %v", got)
	}
	if paths["docs/notes.md"] {
		t.Errorf("non-.go definer leaked: %v", got)
	}
}

func TestFindForgottenTestFilesNeverDefiners(t *testing.T) {
	missing := []MissingSymbol{{Symbol: "VCacheAnchor", ReferencedIn: "cmd/fak"}}
	uncommitted := map[string]string{"cmd/fak/vcache_wiring_test.go": "type VCacheAnchor struct{}\n"}
	if got := FindForgottenFiles(missing, uncommitted); len(got) != 0 {
		t.Errorf("test file returned as definer: %v", got)
	}
}

func TestFindForgottenSymbolWithNoDefinerAbsent(t *testing.T) {
	missing := []MissingSymbol{{Symbol: "NeverDefined", ReferencedIn: "cmd/fak"}}
	uncommitted := map[string]string{"cmd/fak/a.go": "func Something() {}\n"}
	if got := FindForgottenFiles(missing, uncommitted); len(got) != 0 {
		t.Errorf("phantom definer: %v", got)
	}
}

func TestFindForgottenGroupsMultipleSymbolsPerFile(t *testing.T) {
	missing := []MissingSymbol{
		{Symbol: "parseVerbArgv", ReferencedIn: "cmd/fak"},
		{Symbol: "recoverUsage", ReferencedIn: "cmd/fak"},
	}
	uncommitted := map[string]string{
		"cmd/fak/main_preamble.go": "func parseVerbArgv() {}\nfunc recoverUsage() {}\n",
	}
	got := FindForgottenFiles(missing, uncommitted)
	if len(got) != 1 {
		t.Fatalf("want 1 file, got %d: %v", len(got), got)
	}
	if got[0].Path != "cmd/fak/main_preamble.go" {
		t.Errorf("path = %q", got[0].Path)
	}
	defs := append([]string{}, got[0].Defines...)
	sort.Strings(defs)
	if !reflect.DeepEqual(defs, []string{"parseVerbArgv", "recoverUsage"}) {
		t.Errorf("defines = %v", got[0].Defines)
	}
}

func TestDiagnoseBuildOK(t *testing.T) {
	d := Diagnose(true, "", map[string]string{}, "abc123")
	if d.Verdict != "BUILD_OK" || !d.Builds {
		t.Errorf("verdict/builds = %q/%v", d.Verdict, d.Builds)
	}
	if len(d.ForgottenFiles) != 0 || len(d.MissingSymbols) != 0 {
		t.Errorf("non-empty on BUILD_OK: %+v", d)
	}
	if !strings.Contains(d.Summary, "not a build break") {
		t.Errorf("summary = %q", d.Summary)
	}
}

func TestDiagnoseBuildBrokenCoherence(t *testing.T) {
	uncommitted := map[string]string{"cmd/fak/main_preamble.go": "func parseVerbArgv() {}\n"}
	d := Diagnose(false, realStderrMinimal, uncommitted, "")
	if d.Verdict != "BUILD_BROKEN_COHERENCE" {
		t.Fatalf("verdict = %q", d.Verdict)
	}
	found := false
	for _, f := range d.ForgottenFiles {
		if f.Path == "cmd/fak/main_preamble.go" {
			found = true
		}
	}
	if !found {
		t.Errorf("forgotten file not surfaced: %v", d.ForgottenFiles)
	}
	if !strings.Contains(d.Summary, "main_preamble.go") || !strings.Contains(d.Summary, "git add") {
		t.Errorf("summary = %q", d.Summary)
	}
}

func TestDiagnoseBuildBrokenOther(t *testing.T) {
	d := Diagnose(false, realStderrMinimal, map[string]string{}, "")
	if d.Verdict != "BUILD_BROKEN_OTHER" {
		t.Fatalf("verdict = %q", d.Verdict)
	}
	if len(d.ForgottenFiles) != 0 {
		t.Errorf("forgotten non-empty: %v", d.ForgottenFiles)
	}
	if !strings.Contains(d.Summary, "genuine") {
		t.Errorf("summary = %q", d.Summary)
	}
}

func TestClassifyDirect(t *testing.T) {
	if Classify(true, nil, nil) != "BUILD_OK" {
		t.Error("builds -> BUILD_OK")
	}
	if Classify(false, []ForgottenFile{{Path: "x"}}, []MissingSymbol{{Symbol: "S"}}) != "BUILD_BROKEN_COHERENCE" {
		t.Error("forgotten -> COHERENCE")
	}
	if Classify(false, nil, []MissingSymbol{{Symbol: "S"}}) != "BUILD_BROKEN_OTHER" {
		t.Error("no forgotten -> OTHER")
	}
}
