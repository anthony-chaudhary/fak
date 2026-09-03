package main

// signal.go - `fak signal`, JOB CONTROL for a running served session: the OS-process-model
// vocabulary over the live session-control plane. It is the answer to Claude Code #21419
// ("send input to a running agent without stopping it" - SIGCONT + stdin), which the field
// closed as a duplicate. fak can answer it:
//
//	fak signal <id> pause    # hold at the next turn boundary   (SIGSTOP)
//	fak signal <id> resume   # un-pause, a live state flip      (SIGCONT)
//	fak signal <id> stop     # clean stop, drained at the boundary (SIGTERM)
//	fak signal <id> steer "..."   # send INPUT to the running session, taken at its next boundary
//
// pause/resume/stop are the OS job-control names over the ALREADY-shipped session control
// verbs (the same /v1/fak/session/{id}/run write `fak session pause` uses) - signal is the
// process-model framing, not a second control plane. steer is the genuinely new verb: it
// POSTs to /v1/fak/session/{id}/steer, which the serve process adjudicates and enqueues onto
// its a2achan bus (Session locale) for the running loop to consume at its next turn boundary
// - never mid-decode. A tainted/over-scoped steer is refused by the kernel's default-deny
// floor and surfaces as a 422.
//
// Read-back of the resulting run-state is via `fak session status <id>` (or `fak ps`).
// Connection flags mirror `fak session`: --addr ($FAK_ADDR or http://127.0.0.1:8080),
// --key ($FAK_KEY). This reuses the sessionClient defined in session_cmd.go - it does NOT
// duplicate the HTTP client.

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

// cmdSignal is the `fak signal` entry point; it maps the testable core's exit code to the
// process exit code, mirroring cmdSession.
func cmdSignal(argv []string) { os.Exit(runSignal(os.Stdout, os.Stderr, argv)) }

// runSignal is the testable core: it returns the process exit code (0 ok, 1 a transport/HTTP
// error, 2 a usage error) and takes its streams explicitly so a test can drive it against an
// httptest gateway.
func runSignal(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		signalUsage(stderr)
		return 2
	}
	verb := argv[0]
	args := argv[1:]

	// Every verb takes the session id first; steer additionally takes its text from --text
	// (or stdin). The id comes before any flags so `fak signal sess-1 pause --json` parses.
	known := map[string]bool{"pause": true, "resume": true, "stop": true, "steer": true}
	if !known[verb] {
		if len(argv) >= 2 && known[argv[1]] {
			verb = argv[1]
			args = append([]string{argv[0]}, argv[2:]...)
		} else {
			fmt.Fprintf(stderr, "fak signal: unknown verb %q (want pause|resume|stop|steer)\n", verb)
			signalUsage(stderr)
			return 2
		}
	}
	if len(args) < 1 {
		fmt.Fprintf(stderr, "fak signal %s: missing session id\n", verb)
		signalUsage(stderr)
		return 2
	}
	id := args[0]
	flagArgs := args[1:]

	fs := flag.NewFlagSet("signal "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw JSON instead of the human line")
	ifRev := fs.Uint64("if-rev", 0, "optimistic-concurrency guard: apply only if the session's current rev matches (0 = no guard)")
	reason := fs.String("reason", "", "reason token recorded on stop")
	text := fs.String("text", "", "steer: the input to deliver to the running session (or read from stdin if empty)")
	stdin := fs.Bool("stdin", false, "steer: read the input text from stdin")
	regDir := fs.String("reg-dir", "", "registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	if err := fs.Parse(flagArgs); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak signal %s: unexpected argument %q (the id comes first, then flags)\n", verb, fs.Arg(0))
		return 2
	}

	targetID := id
	targetAddr := strings.TrimRight(*addr, "/")
	if !sessionAddrExplicit(fs) {
		resolvedID, resolvedAddr, code, handled := resolveSignalTarget(stderr, verb, id, resolveSweepRegDir(*regDir))
		if handled {
			if code != 0 {
				return code
			}
			targetID = resolvedID
			targetAddr = resolvedAddr
		}
	}

	c := &sessionClient{base: targetAddr, key: *key, hc: &http.Client{Timeout: 15 * time.Second}}

	switch verb {
	case "pause":
		return c.runVerb(stdout, stderr, *asJSON, targetID, "paused", *reason, *ifRev)
	case "resume":
		return c.runVerb(stdout, stderr, *asJSON, targetID, "running", *reason, *ifRev)
	case "stop":
		return c.runVerb(stdout, stderr, *asJSON, targetID, "stopped", *reason, *ifRev)
	case "steer":
		body, code := resolveSteerText(stdin, text, stderr)
		if code != 0 {
			return code
		}
		return c.renderSteer(stdout, stderr, *asJSON, targetID, body)
	}
	return 2 // unreachable: the known-verb gate already rejected unknown verbs
}

// resolveSteerText reads the steer payload from --text, or from stdin when --stdin is set or
// --text is empty. An empty payload (no text, no stdin) is a usage error.
func resolveSteerText(stdin *bool, text *string, stderr io.Writer) (string, int) {
	if *stdin || strings.TrimSpace(*text) == "" {
		raw, err := io.ReadAll(io.LimitReader(os.Stdin, maxSessionRespBytes))
		if err != nil {
			fmt.Fprintf(stderr, "fak signal steer: read stdin: %v\n", err)
			return "", 1
		}
		piped := strings.TrimSpace(string(raw))
		if piped != "" {
			return piped, 0
		}
	}
	if strings.TrimSpace(*text) == "" {
		fmt.Fprintln(stderr, "fak signal steer: provide the input via --text \"...\" or on stdin")
		return "", 2
	}
	return *text, 0
}

// steer POSTs the operator input to /v1/fak/session/{id}/steer. The server adjudicates and
// enqueues it; a refused (tainted/over-scoped) steer comes back as a 422, surfaced by
// httpStatusError. The 202 response body is decoded into a small ack.
func (c *sessionClient) steer(id, text string) (steerAck, error) {
	return c.steerAs(id, text, "")
}

// steerAs is steer with an explicit source attribution: principal names a non-operator
// steer source (e.g. "doomloop-guard" for a machine re-anchor nudge, #3529). Empty principal
// ⇒ the human "operator" default. The principal is attribution only and never elevates trust
// (the a2achan floor gates on caps+taint+scope, not the from-string), so it is safe for the
// CLI to name a machine source without buying the input more authority than an operator steer.
func (c *sessionClient) steerAs(id, text, principal string) (steerAck, error) {
	var ack steerAck
	err := c.req(http.MethodPost, "/v1/fak/session/"+url.PathEscape(id)+"/steer",
		gateway.SteerRequest{Text: text, Principal: principal}, &ack)
	return ack, err
}

// steerAck is the gateway's accept response for a steer.
type steerAck struct {
	TraceID string `json:"trace_id"`
	Steered bool   `json:"steered"`
}

// renderSteer sends a steer and prints the ack (JSON or a one-line human form), mapping any
// transport/HTTP error to exit 1.
func (c *sessionClient) renderSteer(stdout, stderr io.Writer, asJSON bool, id, text string) int {
	ack, err := c.steer(id, text)
	if err != nil {
		fmt.Fprintf(stderr, "fak signal steer: %v\n", err)
		return 1
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, ack)
	}
	fmt.Fprintf(stdout, "steered %s (%d bytes) - delivered at the session's next turn boundary\n", ack.TraceID, len(text))
	return 0
}

func resolveSignalTarget(stderr io.Writer, verb, query, regDir string) (string, string, int, bool) {
	rows := guardsessions.Load(regDir)
	res := guardsessions.Resolve(rows, query)
	switch {
	case res.Matched == 1:
		gw := strings.TrimRight(strings.TrimSpace(res.Row.GatewayURL), "/")
		if gw == "" {
			fmt.Fprintf(stderr, "fak signal %s: guard session %s published no gateway_url — cannot control cross-process without --addr\n", verb, res.Row.Handle)
			return "", "", 1, true
		}
		actualID := strings.TrimSpace(res.Row.TraceID)
		if actualID == "" {
			actualID = res.Row.Handle
		}
		return actualID, gw, 0, true
	case res.Matched > 1:
		fmt.Fprintf(stderr, "fak signal %s: %q matches %d guard sessions — narrow the prefix:\n", verb, query, res.Matched)
		for _, c := range res.Candidates {
			fmt.Fprintf(stderr, "  %s  %s\n", c.Handle, orDash(c.TraceID))
		}
		return "", "", 3, true
	}
	return query, "", 0, false
}

func signalUsage(w io.Writer) {
	fmt.Fprint(w, `fak signal - job control for a running served session (the SIGCONT+stdin gap)

  fak signal <id> pause              hold at the next turn boundary       (SIGSTOP)
  fak signal <id> resume             un-pause, a live state flip          (SIGCONT)
  fak signal <id> stop [--reason R]  clean stop, drained at the boundary  (SIGTERM)
  fak signal <id> steer --text "..." send INPUT to the running session, taken at its
                                     next boundary (or pipe the text on stdin)

  pause/resume/stop are the job-control names over the live session-control verbs;
  steer rides the adjudicated a2achan bus (a tainted/over-scoped steer is refused).
  Read back the run-state with: fak session status <id>  (or fak ps)

flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)
       --if-rev N (optimistic-concurrency guard)  --json
`)
}
