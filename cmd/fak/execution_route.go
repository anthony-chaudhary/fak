package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/executionroute"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
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
	sourceDesc := fs.String("source-descriptor", "", "source session descriptor: path to a JSON file, or inline JSON (with --target-descriptor, computes eligibility instead of the boolean fallback)")
	targetDesc := fs.String("target-descriptor", "", "target envelope descriptor: path to a JSON file, or inline JSON (with --source-descriptor, computes eligibility instead of the boolean fallback)")
	fleetStatus := fs.String("fleet-status", "", "live harness health: path to (or inline JSON of) a `fak fleet-accounts status --json` report; excludes unavailable/draining/cooldown candidates and records a reason for each")
	healthMaxAge := fs.Int64("health-max-age", 0, "freshness bound in seconds for --fleet-status readings (0 = no bound); a reading older than this is treated as stale and its candidate excluded")
	requireHealth := fs.Bool("require-health", false, "with --fleet-status, exclude a candidate that has no live health reading (fail closed)")
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
	if (*sourceDesc == "") != (*targetDesc == "") {
		fmt.Fprintln(stderr, "execution-route: --source-descriptor and --target-descriptor must be supplied together (either alone leaves the boolean fallback in charge, which would silently ignore it)")
		return 2
	}
	source, err := loadSessionDescriptor(*sourceDesc)
	if err != nil {
		fmt.Fprintf(stderr, "execution-route: source descriptor: %v\n", err)
		return 2
	}
	target, err := loadSessionDescriptor(*targetDesc)
	if err != nil {
		fmt.Fprintf(stderr, "execution-route: target descriptor: %v\n", err)
		return 2
	}
	health, err := loadFleetHealth(*fleetStatus, *healthMaxAge, *requireHealth)
	if err != nil {
		fmt.Fprintf(stderr, "execution-route: fleet status: %v\n", err)
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
		Health:            health,
		Model:             modelroute.Subject{Aspect: modelroute.Aspect(*aspect), Tool: *tool, Complexity: modelroute.Complexity(*complexity)},
		Session:           executionroute.SessionSubject{ID: *sessionID, PreserveContinuity: *continuity, Portable: *portable, ContextUtilization: *context, CompactAt: *compactAt, Source: source, Target: target},
	}, harnessprofile.Profiles(), modelroute.DefaultManifest())
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	// Surface every candidate excluded ahead of the winner on stderr so the
	// operator sees WHY a harness was skipped (health/requirement) without parsing
	// the JSON on stdout, which stays the machine-readable decision alone.
	for _, r := range decision.Harness.Rejected {
		fmt.Fprintf(stderr, "execution-route: skipped harness %q: %s\n", r.Candidate, r.Reason)
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(decision); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

// loadFleetHealth turns a --fleet-status flag value (a path to, or inline JSON of,
// a `fak fleet-accounts status --json` report) into the routing health input via
// the fleetaccounts adapter. An empty spec yields an inert report, leaving harness
// selection on the static requirements alone. maxAgeSeconds is the freshness bound
// and requireEvidence makes an unmeasured candidate fail closed.
func loadFleetHealth(spec string, maxAgeSeconds int64, requireEvidence bool) (executionroute.HealthReport, error) {
	if strings.TrimSpace(spec) == "" {
		return executionroute.HealthReport{}, nil
	}
	raw := []byte(spec)
	if !strings.HasPrefix(strings.TrimSpace(spec), "{") {
		b, err := os.ReadFile(spec)
		if err != nil {
			return executionroute.HealthReport{}, err
		}
		raw = b
	}
	var report fleetaccounts.StatusReport
	if err := json.Unmarshal(raw, &report); err != nil {
		return executionroute.HealthReport{}, err
	}
	health := executionroute.HealthFromFleetStatus(report.Accounts, maxAgeSeconds)
	health.RequireEvidence = requireEvidence
	return health, nil
}

// loadSessionDescriptor reads a SessionDescriptor from a flag value that is
// either a path to a JSON file or inline JSON (a value starting with '{').
// An empty spec means the flag was not supplied and yields nil, leaving the
// boolean fallback in charge. Interpretability (version, known state kinds)
// is judged by the descriptor's own Validate inside RouteCompat, not here.
func loadSessionDescriptor(spec string) (*executionroute.SessionDescriptor, error) {
	if spec == "" {
		return nil, nil
	}
	raw := []byte(spec)
	if !strings.HasPrefix(strings.TrimSpace(spec), "{") {
		b, err := os.ReadFile(spec)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	var d executionroute.SessionDescriptor
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, err
	}
	return &d, nil
}
