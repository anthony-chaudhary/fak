package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/executionroute"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func cmdExecutionRoute(argv []string) { os.Exit(runExecutionRoute(os.Stdout, os.Stderr, argv)) }

func runExecutionRoute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("execution-route", flag.ContinueOnError)
	fs.SetOutput(stderr)
	candidates := fs.String("harnesses", "", "ordered comma-separated harness profile ids or executable names")
	wire := fs.String("wire", "", "required harness wire: anthropic|openai-chat|openai-responses|gemini")
	repoint := fs.String("repoint", "", "required repoint mechanism: env|settings-file|cli-config|wrapper")
	rotatable := fs.Bool("rotatable", false, "require a harness with declared account rotation identity")
	aspect := fs.String("aspect", string(modelroute.AspectRequest), "model-routing aspect")
	tool := fs.String("tool", "", "tool name when aspect=tool_call")
	complexity := fs.String("complexity", string(modelroute.ComplexityMedium), "model-routing complexity")
	sessionID := fs.String("session", "", "prior session id")
	continuity := fs.Bool("continuity", false, "preserve prior-session continuity")
	portable := fs.Bool("portable", false, "prior session state can fork across harness/model boundaries")
	context := fs.Float64("context-utilization", 0, "prior session context utilization from 0 to 1")
	compactAt := fs.Float64("compact-at", .80, "compact+resume threshold from 0 to 1")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "execution-route: unexpected positional arguments")
		return 2
	}
	if *context < 0 || *context > 1 || *compactAt <= 0 || *compactAt > 1 {
		fmt.Fprintln(stderr, "execution-route: context-utilization must be 0..1 and compact-at must be >0..1")
		return 2
	}
	var ordered []string
	for _, item := range strings.Split(*candidates, ",") {
		if item = strings.TrimSpace(item); item != "" {
			ordered = append(ordered, item)
		}
	}
	decision, err := executionroute.Route(executionroute.Request{
		HarnessCandidates: ordered,
		Harness:           executionroute.HarnessRequirements{Wire: harnessprofile.Wire(*wire), Repoint: harnessprofile.RepointMechanism(*repoint), Rotatable: *rotatable},
		Model:             modelroute.Subject{Aspect: modelroute.Aspect(*aspect), Tool: *tool, Complexity: modelroute.Complexity(*complexity)},
		Session:           executionroute.SessionSubject{ID: *sessionID, PreserveContinuity: *continuity, Portable: *portable, ContextUtilization: *context, CompactAt: *compactAt},
	}, harnessprofile.Profiles(), modelroute.DefaultManifest())
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(decision); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}
