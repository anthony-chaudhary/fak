package microagent_test

import (
	"context"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// This file is the executable acceptance witness for issue #2003 — the ToolExec
// SEAM definition (goroutine + subprocess backends behind one interface). The
// seam itself landed under the downstream children #2014 (subprocess backend)
// and #2018 (kernel-floor wiring); those suites (toolexec_test.go,
// toolexec_floor_conformance_test.go) prove the FLOOR property. This file pins
// #2003's OWN two acceptance bullets as guards so the parent seam issue has a
// direct witness rather than only incidental downstream coverage:
//
//	1. "ToolExec has >=2 backends behind the SAME interface (goroutine +
//	   subprocess)."  -> TestSeamHasTwoBackendsBehindOneInterface drives the SAME
//	   Run(ctx, ToolAction) (ToolResult, error) seam method against BOTH backends
//	   and asserts each executes and captures output — one seam type fronts every
//	   isolation level.
//	2. "Every microagent action is routed through the seam (no direct exec in the
//	   loop)."  -> TestSeamIsSingleExecRoutingPoint parses the package's own
//	   non-test sources and asserts os/exec is imported ONLY by the seam file
//	   (toolexec.go). A future direct exec.Command added to the host/loop trips
//	   this guard.
//
// Generation frame (gen/second-next) closing evidence — see toolexec.go for the
// full memo; this test is the compatibility/acceptance artifact that binds the
// generation option's interface boundary to an executable check.

// TestSeamHasTwoBackendsBehindOneInterface is the #2003 acceptance-1 witness:
// the goroutine and subprocess backends are two implementations behind ONE
// interface, both driven through the identical ToolExec.Run seam signature.
func TestSeamHasTwoBackendsBehindOneInterface(t *testing.T) {
	// >=2 backends registered, and goroutine + subprocess are both present.
	names := microagent.RegisteredBackends()
	for _, want := range []string{microagent.BackendGoroutine, microagent.BackendSubprocess} {
		if !contains(names, want) {
			t.Fatalf("registered backends %v missing %q — the seam must ship both the goroutine and subprocess tiers", names, want)
		}
	}
	if len(names) < 2 {
		t.Fatalf("registered backends = %v, want >=2 behind the same interface", names)
	}

	// Both concrete backends satisfy the SAME Backend interface (compile-time).
	var _ microagent.Backend = microagent.NewGoroutineBackend()

	// Drive one allowed action shape through EACH backend via the SAME seam
	// method Run(ctx, ToolAction) (ToolResult, error). Different isolation
	// mechanics, one interface, one call site.

	// Goroutine tier: an in-process func runs and its output is captured.
	gb := microagent.NewGoroutineBackend()
	if err := gb.Register("run_shell", func(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
		return microagent.ToolResult{Stdout: []byte("goroutine-out"), ExitCode: 0}, nil
	}); err != nil {
		t.Fatalf("goroutine Register: %v", err)
	}
	teGo, err := microagent.NewToolExecBackend(allowKernel("run_shell"), gb)
	if err != nil {
		t.Fatalf("NewToolExecBackend(goroutine): %v", err)
	}
	resGo, err := teGo.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
	if err != nil {
		t.Fatalf("goroutine Run: %v", err)
	}
	if !resGo.Ran || string(resGo.Stdout) != "goroutine-out" {
		t.Fatalf("goroutine backend: Ran=%v stdout=%q, want Ran=true stdout=%q", resGo.Ran, resGo.Stdout, "goroutine-out")
	}

	// Subprocess tier, reached BY NAME through the registry: the same seam,
	// executing an os/exec child, captured through the identical ToolResult.
	teProc, err := microagent.NewToolExecFor(microagent.BackendSubprocess, allowKernel("run_shell"))
	if err != nil {
		t.Fatalf("NewToolExecFor(subprocess): %v", err)
	}
	t.Setenv(helperModeEnv, "echo")
	resProc, err := teProc.Run(context.Background(), microagent.ToolAction{
		Tool:    "run_shell",
		Path:    os.Args[0],
		Argv:    []string{"-test.run=^TestHelperProcess$"},
		Timeout: 30 * time.Second,
	})
	if err != nil {
		t.Fatalf("subprocess Run: %v", err)
	}
	if !resProc.Ran || string(resProc.Stdout) != "out-marker" {
		t.Fatalf("subprocess backend: Ran=%v stdout=%q, want Ran=true stdout=%q", resProc.Ran, resProc.Stdout, "out-marker")
	}
}

// TestSeamIsSingleExecRoutingPoint is the #2003 acceptance-2 witness: os/exec is
// confined behind the seam. It parses THIS package's own non-test source files
// and asserts the exec import appears only in the seam file (toolexec.go) — so
// the host/loop code (microagent.go, slotsched.go, hibernate.go, journalsink.go,
// …) has no direct exec. Concrete Microagent.Step bodies live out-of-package
// (the #2001 RunArm extraction is still open); this guard pins the property for
// the package that OWNS the seam, which is where a direct-exec regression would
// most easily creep in.
func TestSeamIsSingleExecRoutingPoint(t *testing.T) {
	const execImport = "os/exec"
	const seamFile = "toolexec.go"

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("ReadDir(package dir): %v", err)
	}
	fset := token.NewFileSet()
	var importers []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, imp := range f.Imports {
			if strings.Trim(imp.Path.Value, `"`) == execImport {
				importers = append(importers, name)
				break
			}
		}
	}
	sort.Strings(importers)

	if len(importers) != 1 || importers[0] != seamFile {
		t.Fatalf("os/exec is imported by %v, want ONLY %q — every microagent action must route through the ToolExec seam; a direct exec elsewhere breaks #2003 acceptance 2", importers, seamFile)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
