package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func cmdOrchestration(args []string) {
	os.Exit(runOrchestration(os.Stdout, os.Stderr, args))
}

func runOrchestration(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 && args[0] == "status" {
		return runOrchestrationStatus(stdout, stderr, args[1:])
	}
	if len(args) == 0 || args[0] != "plan" {
		fmt.Fprintln(stderr, "usage: fak orchestration plan --profile off|auto|ultracode (--task FIXTURE | --task-text TEXT) [--json] [--strict] [--launch] [--selfcheck]")
		return 2
	}
	fs := flag.NewFlagSet("orchestration plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	profile := fs.String("profile", "auto", "orchestration profile: off, auto, or ultracode")
	taskPath := fs.String("task", "", "versioned task fixture JSON")
	taskText := fs.String("task-text", "", "current task text (converted to a typed task without persisting prompt text)")
	strict := fs.Bool("strict", false, "reject any capability degradation")
	jsonOut := fs.Bool("json", false, "emit the stable resolution JSON")
	launch := fs.Bool("launch", false, "launch resolved ultracode workers or emit a typed direct-decline receipt")
	selfcheck := fs.Bool("selfcheck", false, "verify stable JSON round-trip without launching work")
	codexHome := fs.String("codex-home", "", "Codex home used for a session-linked invocation receipt")
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
	if (*taskPath == "") == (strings.TrimSpace(*taskText) == "") || fs.NArg() != 0 || (*launch && *selfcheck) {
		fmt.Fprintln(stderr, "fak orchestration plan: exactly one of --task or --task-text is required, positional arguments are not accepted, and --launch conflicts with --selfcheck")
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
	if err == nil && taskText != nil && *taskText != "" {
		orchestration.RouteResolution(&resolved, *taskText, guardCodexDefaultModelID)
	}
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
		fmt.Fprintf(stdout, "SELFCHECK PASS schema=%s offline=true launched=0\n", resolved.Schema)
	}
	sessionID := strings.TrimSpace(os.Getenv("CODEX_THREAD_ID"))
	if *launch {
		if err := validateCodexOrchestrationArtifactHome(*codexHome); err != nil {
			fmt.Fprintf(stderr, "fak orchestration plan: %v\n", err)
			return 1
		}
	}
	if sessionID != "" && !*selfcheck {
		if err := writeCodexOrchestrationInvocationReceipt(*codexHome, codexOrchestrationInvocationReceipt{
			Schema: "fak.codex_orchestration_invocation.v1", SessionID: sessionID,
			InvokedAt: time.Now().UTC().Format(time.RFC3339Nano), TaskID: task.ID,
			Requested: resolved.Requested.Name, Resolved: resolved.Resolved.Profile,
			WorkClass: resolved.Resolved.WorkClass, MaxWorkers: resolved.Resolved.Budget.MaxWorkers,
		}); err != nil {
			fmt.Fprintf(stderr, "fak orchestration plan: persist Codex invocation receipt: %v\n", err)
			return 1
		}
	}
	var launchReceipt *codexOrchestrationLaunchReceipt
	if *launch {
		if sessionID == "" {
			fmt.Fprintln(stderr, "fak orchestration plan: --launch requires CODEX_THREAD_ID so the launch can join to the guarded first-turn witness")
			return 2
		}
		capabilityProfile := "native"
		if *capset == "unsupported" {
			capabilityProfile = "unsupported"
		}
		launched, launchErr := launchCodexOrchestrationWorkers(*codexHome, sessionID, *profile, capabilityProfile, *taskText, resolved)
		if launchErr != nil {
			fmt.Fprintf(stderr, "fak orchestration plan: %v\n", launchErr)
			return 1
		}
		launchReceipt = &launched
		if !*jsonOut {
			fmt.Fprintf(stdout, "launch=%s run_id=%s workers=%d decline=%s\n", launched.Status, launched.RunID, len(launched.Workers), launched.DeclineReason)
		}
	}
	if *jsonOut {
		if launchReceipt != nil {
			return encodeJSONOrFail(stdout, stderr, struct {
				Schema string                          `json:"schema"`
				Plan   orchestration.Resolution        `json:"plan"`
				Launch codexOrchestrationLaunchReceipt `json:"launch"`
			}{Schema: "fak.codex_orchestration_launch_result.v1", Plan: resolved, Launch: *launchReceipt}, "fak orchestration plan")
		}
		_, _ = stdout.Write(append(stable, '\n'))
		return 0
	}
	fmt.Fprintln(stdout, strings.Join(resolved.Resolved.Explanation, " "))
	for _, d := range resolved.Degradations {
		fmt.Fprintf(stdout, "DEGRADED %s: required=%s available=%s reason=%s\n", d.Capability, d.Required, d.Available, d.Reason)
	}
	return 0
}

type codexOrchestrationInvocationReceipt struct {
	Schema     string                  `json:"schema"`
	SessionID  string                  `json:"session_id"`
	InvokedAt  string                  `json:"invoked_at"`
	TaskID     string                  `json:"task_id"`
	Requested  orchestration.Profile   `json:"requested_profile"`
	Resolved   orchestration.Profile   `json:"resolved_profile"`
	WorkClass  orchestration.WorkClass `json:"work_class"`
	MaxWorkers int                     `json:"max_workers"`
}

func codexOrchestrationInvocationReceiptPath(codexHome, sessionID string) (string, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return "", err
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" || filepath.Base(sessionID) != sessionID {
		return "", errors.New("invalid Codex session id for orchestration receipt")
	}
	return filepath.Join(home, "fak-orchestration-invocations", sessionID+".json"), nil
}

func writeCodexOrchestrationInvocationReceipt(codexHome string, receipt codexOrchestrationInvocationReceipt) error {
	path, err := codexOrchestrationInvocationReceiptPath(codexHome, receipt.SessionID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(body, '\n'), 0o600)
}

func readCodexOrchestrationInvocationReceipt(codexHome, sessionID string) (codexOrchestrationInvocationReceipt, bool) {
	path, err := codexOrchestrationInvocationReceiptPath(codexHome, sessionID)
	if err != nil {
		return codexOrchestrationInvocationReceipt{}, false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return codexOrchestrationInvocationReceipt{}, false
	}
	var receipt codexOrchestrationInvocationReceipt
	ok := json.Unmarshal(raw, &receipt) == nil && receipt.Schema == "fak.codex_orchestration_invocation.v1" && receipt.SessionID == sessionID
	return receipt, ok
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
