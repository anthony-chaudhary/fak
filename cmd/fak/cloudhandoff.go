package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/cloudhandoff"
)

func cmdCloudHandoff(argv []string) {
	os.Exit(runCloudHandoff(os.Stdout, os.Stderr, os.Stdin, argv))
}

type cloudHandoffInput struct {
	Policy       cloudhandoff.Policy  `json:"policy"`
	Request      cloudhandoff.Request `json:"request"`
	LocalAttempt cloudhandoff.Attempt `json:"local_attempt"`
	PreApproved  bool                 `json:"pre_approved"`
	SendError    string               `json:"send_error,omitempty"`
}

type cloudHandoffOutput struct {
	Schema  string               `json:"schema"`
	Receipt cloudhandoff.Receipt `json:"receipt"`
	Events  []cloudhandoff.Event `json:"events"`
	Error   string               `json:"error,omitempty"`
}

func runCloudHandoff(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("cloud-handoff", flag.ContinueOnError)
	fs.SetOutput(stderr)

	selfTest := fs.Bool("self-test", false, "run self-test validation on representative fixtures")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	if *selfTest {
		return runCloudHandoffSelfTest(stdout, stderr, *asJSON)
	}

	args := fs.Args()
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak cloud-handoff [flags] <input.json|->")
		fmt.Fprintln(stderr, "       fak cloud-handoff --self-test [--json]")
		return 2
	}

	var data []byte
	var err error
	if args[0] == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(args[0])
	}
	if err != nil {
		fmt.Fprintf(stderr, "cloud-handoff: read input: %v\n", err)
		return 1
	}

	var in cloudHandoffInput
	if err := json.Unmarshal(data, &in); err != nil {
		fmt.Fprintf(stderr, "cloud-handoff: parse input: %v\n", err)
		return 1
	}

	broker := cloudhandoff.New()
	var approve cloudhandoff.Approval
	if in.Policy.Consent == cloudhandoff.ConsentAsk {
		approve = func(ev cloudhandoff.Event) bool {
			return in.PreApproved
		}
	}

	var send cloudhandoff.Transport = func(pkg cloudhandoff.Package) error {
		if in.SendError != "" {
			return fmt.Errorf("%s", in.SendError)
		}
		return nil
	}

	receipt, handoffErr := broker.Handoff(in.Policy, in.Request, approve, send, in.LocalAttempt)
	errStr := ""
	if handoffErr != nil {
		errStr = handoffErr.Error()
	}

	if *asJSON {
		out := cloudHandoffOutput{
			Schema:  cloudhandoff.Schema,
			Receipt: receipt,
			Events:  broker.Events(),
			Error:   errStr,
		}
		_ = writeIndentedJSONNoEscape(stdout, out)
	} else {
		if handoffErr != nil {
			fmt.Fprintf(stderr, "cloud-handoff: terminal=%s reason=%v\n", receipt.Terminal, handoffErr)
		} else {
			fmt.Fprintf(stdout, "cloud-handoff: terminal=%s remote_completed=%t local_completed=%t\n",
				receipt.Terminal, receipt.RemoteCompleted, receipt.LocalCompleted)
		}
	}

	if handoffErr != nil {
		return 1
	}
	return 0
}

func runCloudHandoffSelfTest(stdout, stderr io.Writer, asJSON bool) int {
	policy := cloudhandoff.Policy{
		Eligible:        true,
		Consent:         cloudhandoff.ConsentPreapproved,
		Destinations:    []string{"vendor-cloud"},
		AllowedTriggers: []cloudhandoff.Trigger{cloudhandoff.TriggerFault, cloudhandoff.TriggerDeadline},
	}
	req := cloudhandoff.Request{
		OperationID:      "op-selftest-1",
		Trigger:          cloudhandoff.TriggerFault,
		Data:             []cloudhandoff.DataClass{{Name: "tokens", LocalRequired: false}},
		DestinationClass: "vendor-cloud",
		Consequence:      "cost",
		Alternatives:     []string{"abort", "retry-local"},
		Payload:          []byte("hello payload"),
	}
	local := cloudhandoff.Attempt{
		Engine:   "fak-native",
		Location: "local",
		Outcome:  "failed",
	}

	broker := cloudhandoff.New()
	receipt, err := broker.Handoff(policy, req, nil, func(pkg cloudhandoff.Package) error {
		return nil
	}, local)

	if err != nil {
		fmt.Fprintf(stderr, "cloud-handoff self-test failed: %v\n", err)
		return 1
	}

	if asJSON {
		out := cloudHandoffOutput{
			Schema:  cloudhandoff.Schema,
			Receipt: receipt,
			Events:  broker.Events(),
		}
		_ = writeIndentedJSONNoEscape(stdout, out)
	} else {
		fmt.Fprintln(stdout, "cloud-handoff: self-test passed (valid transition verified)")
	}
	return 0
}
