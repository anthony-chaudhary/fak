package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func cmdOrchestration(args []string) {
	os.Exit(runOrchestration(os.Stdout, os.Stderr, args))
}

func runOrchestration(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 || args[0] != "plan" {
		fmt.Fprintln(stderr, "usage: fak orchestration plan --profile off|auto|ultracode (--task FIXTURE | --task-text TEXT) [--json] [--strict] [--selfcheck]")
		return 2
	}
	fs := flag.NewFlagSet("orchestration plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	profile := fs.String("profile", "auto", "orchestration profile: off, auto, or ultracode")
	taskPath := fs.String("task", "", "versioned task fixture JSON")
	taskText := fs.String("task-text", "", "current task text (converted to a typed task without persisting prompt text)")
	strict := fs.Bool("strict", false, "reject any capability degradation")
	jsonOut := fs.Bool("json", false, "emit the stable resolution JSON")
	selfcheck := fs.Bool("selfcheck", false, "verify stable JSON round-trip without launching work")
	capset := fs.String("capabilities", "native", "harness fixture: native or unsupported")
	maxWorkers := orchestrationOptionalInt{}
	maxTokens := orchestrationOptionalInt64{}
	attended := orchestrationOptionalBool{}
	fs.Var(&maxWorkers, "max-workers", "operator worker cap")
	fs.Var(&maxTokens, "max-tokens", "operator token cap")
	fs.Var(&attended, "attended", "operator interaction policy (true or false)")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if (*taskPath == "") == (strings.TrimSpace(*taskText) == "") || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak orchestration plan: exactly one of --task or --task-text is required and positional arguments are not accepted")
		return 2
	}
	var task orchestration.TaskSpec
	var err error
	if *taskPath != "" {
		var data []byte
		data, err = os.ReadFile(*taskPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak orchestration plan: read task: %v\n", err)
			return 1
		}
		task, err = orchestration.ParseTask(data)
	} else {
		task, err = orchestration.TaskFromText(*taskText)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak orchestration plan: %v\n", err)
		return 2
	}
	caps, err := orchestrationCapabilities(*capset)
	if err != nil {
		fmt.Fprintf(stderr, "fak orchestration plan: %v\n", err)
		return 2
	}
	req := orchestration.OrchestrationProfile{Name: orchestration.Profile(*profile), Strict: *strict}
	if maxWorkers.set {
		req.MaxWorkers = &maxWorkers.value
	}
	if maxTokens.set {
		req.MaxTokens = &maxTokens.value
	}
	if attended.set {
		req.Attended = &attended.value
	}
	resolved, err := orchestration.Resolve(req, task, caps)
	if err != nil {
		if errors.Is(err, orchestration.ErrStrictDegradation) {
			fmt.Fprintf(stderr, "fak orchestration plan: %v\n", err)
			return 3
		}
		fmt.Fprintf(stderr, "fak orchestration plan: %v\n", err)
		return 1
	}
	stable, err := orchestration.StableJSON(resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak orchestration plan: encode: %v\n", err)
		return 1
	}
	if *selfcheck {
		roundTrip, err := orchestration.ParseResolution(stable)
		if err != nil {
			fmt.Fprintf(stderr, "fak orchestration plan: selfcheck decode: %v\n", err)
			return 1
		}
		again, err := orchestration.StableJSON(roundTrip)
		if err != nil || string(stable) != string(again) {
			fmt.Fprintln(stderr, "fak orchestration plan: selfcheck unstable JSON")
			return 1
		}
		fmt.Fprintf(stderr, "SELFCHECK PASS schema=%s offline=true launched=0\n", resolved.Schema)
	}
	if *jsonOut {
		_, _ = stdout.Write(append(stable, '\n'))
		return 0
	}
	fmt.Fprintln(stdout, strings.Join(resolved.Resolved.Explanation, " "))
	for _, d := range resolved.Degradations {
		fmt.Fprintf(stdout, "DEGRADED %s: required=%s available=%s reason=%s\n", d.Capability, d.Required, d.Available, d.Reason)
	}
	return 0
}

func orchestrationCapabilities(name string) (orchestration.HarnessCapabilities, error) {
	switch strings.ToLower(name) {
	case "native":
		return orchestration.HarnessCapabilities{
			Concurrency:        orchestration.SupportNative,
			TaskMessaging:      orchestration.SupportNative,
			Cancellation:       orchestration.SupportNative,
			Leases:             orchestration.SupportNative,
			IndependentWitness: orchestration.SupportNative,
		}, nil
	case "unsupported":
		return orchestration.HarnessCapabilities{}, nil
	default:
		return orchestration.HarnessCapabilities{}, fmt.Errorf("unknown capabilities fixture %q", name)
	}
}

type orchestrationOptionalInt struct {
	value int
	set   bool
}

func (v *orchestrationOptionalInt) String() string { return strconv.Itoa(v.value) }
func (v *orchestrationOptionalInt) Set(s string) error {
	n, err := strconv.Atoi(s)
	if err != nil {
		return err
	}
	v.value, v.set = n, true
	return nil
}

type orchestrationOptionalInt64 struct {
	value int64
	set   bool
}

func (v *orchestrationOptionalInt64) String() string { return strconv.FormatInt(v.value, 10) }
func (v *orchestrationOptionalInt64) Set(s string) error {
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return err
	}
	v.value, v.set = n, true
	return nil
}

type orchestrationOptionalBool struct {
	value bool
	set   bool
}

func (v *orchestrationOptionalBool) String() string   { return strconv.FormatBool(v.value) }
func (v *orchestrationOptionalBool) IsBoolFlag() bool { return true }
func (v *orchestrationOptionalBool) Set(s string) error {
	n, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}
	v.value, v.set = n, true
	return nil
}

var _ json.Marshaler
