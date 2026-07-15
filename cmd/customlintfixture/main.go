// Command customlintfixture is the hostile-behavior fixture for the custom-linter ABI.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

const schema = "fak-custom-lint/1"

type request struct {
	Schema  string          `json:"schema"`
	Hook    string          `json:"hook"`
	Subject json.RawMessage `json:"subject"`
}

type finding struct {
	ID       string `json:"id"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
	Evidence string `json:"evidence,omitempty"`
}

type response struct {
	Schema      string    `json:"schema"`
	Disposition string    `json:"disposition"`
	Findings    []finding `json:"findings,omitempty"`
}

func run(out, errw io.Writer, in io.Reader, args []string) int {
	fs := flag.NewFlagSet("customlintfixture", flag.ContinueOnError)
	fs.SetOutput(errw)
	mode := fs.String("mode", "echo", "echo|malformed|overflow|sleep|crash")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	switch *mode {
	case "malformed":
		fmt.Fprint(out, "{not json")
		return 0
	case "overflow":
		fmt.Fprint(out, strings.Repeat("x", 2<<20))
		return 0
	case "sleep":
		time.Sleep(10 * time.Second)
		return 0
	case "crash":
		fmt.Fprintln(errw, "fixture crash")
		return 7
	case "echo":
	default:
		fmt.Fprintf(errw, "unknown mode %q\n", *mode)
		return 2
	}
	var req request
	dec := json.NewDecoder(in)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}
	if req.Schema != schema || req.Hook == "" || !json.Valid(req.Subject) {
		fmt.Fprintln(errw, "bad request")
		return 2
	}
	var subject struct {
		Text string `json:"text"`
	}
	_ = json.Unmarshal(req.Subject, &subject)
	resp := response{Schema: schema, Disposition: "allow"}
	if name, ok := strings.CutPrefix(subject.Text, "env:"); ok && os.Getenv(name) != "" {
		resp.Disposition = "deny"
		resp.Findings = []finding{{ID: "fixture.env-visible", Severity: "error", Message: "environment variable is visible", Evidence: name}}
	}
	if strings.Contains(strings.ToLower(subject.Text), "deny-me") {
		resp.Disposition = "deny"
		resp.Findings = []finding{{ID: "fixture.deny", Severity: "error", Message: "deny marker found", Evidence: "deny-me"}}
	}
	if err := json.NewEncoder(out).Encode(resp); err != nil {
		fmt.Fprintln(errw, err)
		return 1
	}
	return 0
}

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Stdin, os.Args[1:])) }
