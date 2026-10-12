package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// Execution policies for `fak serve --execution-policy`. production keeps the
// GFX1151 device-only guards; bringup lifts them on a node the operator dedicates
// to bringing up the served model, so a host/offload parity path is never refused
// before the device path exists.
const (
	serveExecutionPolicyFlag       = "execution-policy"
	serveExecutionPolicyProduction = "production"
	serveExecutionPolicyBringup    = "bringup"

	// ServeBringupDedicatedEnv names the model ids (comma-separated) this node is
	// dedicated to bringing up. The fleet side writes it into the serve environment
	// from the node's bringup_dedicated record; it must name the served --model.
	ServeBringupDedicatedEnv = "FAK_BRINGUP_DEDICATED"

	ExecutionPolicyUnknown      = "EXECUTION_POLICY_UNKNOWN"
	ExecutionPolicyNotDedicated = "EXECUTION_POLICY_NOT_DEDICATED"

	// executionPolicyBypassDeviceOnlySession is the session-level device-only
	// execution policy a halo serve otherwise installs on every native session.
	executionPolicyBypassDeviceOnlySession = "device-only-session"
)

// executionPolicyError is the typed refusal for an unusable --execution-policy.
// Code is closed vocabulary.
type executionPolicyError struct {
	Code   string
	Detail string
}

func (e *executionPolicyError) Error() string {
	return fmt.Sprintf("fak serve: --%s refused (%s): %s", serveExecutionPolicyFlag, e.Code, e.Detail)
}

// serveExecutionPolicy is the resolved policy. A nil or production policy
// bypasses nothing; an admitted bringup policy records every guard it lifted.
type serveExecutionPolicy struct {
	Name     string
	Model    string
	Admitted bool
	Bypassed []string
	log      io.Writer
}

// resolveServeExecutionPolicy validates the flag against the served model and
// the dedication the serve environment carries. bringup is admitted only when
// the dedication names the served model; it never silently degrades to
// production, because the operator asked for host/offload execution.
func resolveServeExecutionPolicy(name, model, dedicated string, log io.Writer) (*serveExecutionPolicy, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		name = serveExecutionPolicyProduction
	}
	switch name {
	case serveExecutionPolicyProduction:
		return &serveExecutionPolicy{Name: name}, nil
	case serveExecutionPolicyBringup:
	default:
		return nil, &executionPolicyError{Code: ExecutionPolicyUnknown, Detail: fmt.Sprintf("%q is not one of %s, %s", name, serveExecutionPolicyProduction, serveExecutionPolicyBringup)}
	}
	model = strings.TrimSpace(model)
	if model == "" || !slices.Contains(splitBringupDedication(dedicated), model) {
		return nil, &executionPolicyError{Code: ExecutionPolicyNotDedicated, Detail: fmt.Sprintf("%s=%q does not dedicate this node to --model %q", ServeBringupDedicatedEnv, strings.TrimSpace(dedicated), model)}
	}
	return &serveExecutionPolicy{Name: name, Model: model, Admitted: true, log: log}, nil
}

func splitBringupDedication(v string) []string {
	var out []string
	for _, f := range strings.Split(v, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// admit returns err unless it is a GFX1151 device-only refusal an admitted
// bringup policy lifts; a lifted refusal is recorded and logged by its token.
func (p *serveExecutionPolicy) admit(err error) error {
	var halo *haloGPURequiredError
	if err == nil || p == nil || !p.Admitted || !errors.As(err, &halo) {
		return err
	}
	p.bypass(halo.Reason, halo.Detail)
	return nil
}

func (p *serveExecutionPolicy) bypass(token, detail string) {
	if slices.Contains(p.Bypassed, token) {
		return
	}
	p.Bypassed = append(p.Bypassed, token)
	if p.log != nil {
		fmt.Fprintf(p.log, "fak serve: --%s=%s bypassed GFX1151 device-only refusal %s for model %q (%s)\n", serveExecutionPolicyFlag, p.Name, token, p.Model, detail)
	}
}

// requireDeviceExecution is the session device-only decision for a halo host.
// An admitted bringup policy admits host execution and records the lift.
func (p *serveExecutionPolicy) requireDeviceExecution(halo bool) bool {
	if !halo {
		return false
	}
	if p == nil || !p.Admitted {
		return true
	}
	p.bypass(executionPolicyBypassDeviceOnlySession, "native sessions may execute on the host")
	return false
}

// report is the /healthz execution_policy block; production reports nothing so
// a default serve's /healthz is unchanged.
func (p *serveExecutionPolicy) report() *gateway.ExecutionPolicyReport {
	if p == nil || !p.Admitted {
		return nil
	}
	return &gateway.ExecutionPolicyReport{Policy: p.Name, Model: p.Model, Dedicated: true, Bypassed: slices.Clone(p.Bypassed)}
}

const (
	serveDryAdmitFlag   = "dry-admit"
	serveDryAdmitSchema = "fak.serve.dry_admit.v1"
	serveDryAdmitAdmit  = "ADMIT"
)

// serveDryAdmitVerdict is the --dry-admit stdout document. A refusal never
// reaches it: the admission stage that refuses exits non-zero with its typed
// stderr line, exactly as a real launch would.
type serveDryAdmitVerdict struct {
	Schema                 string                         `json:"schema"`
	Verdict                string                         `json:"verdict"`
	Model                  string                         `json:"model"`
	GGUF                   string                         `json:"gguf,omitempty"`
	Backend                string                         `json:"backend"`
	ExecutionPolicy        string                         `json:"execution_policy"`
	Bypassed               []string                       `json:"bypassed"`
	RequireDeviceExecution bool                           `json:"require_device_execution"`
	NativeContextTokens    int                            `json:"native_context_tokens,omitempty"`
	Policy                 *gateway.ExecutionPolicyReport `json:"execution_policy_report,omitempty"`
}

func newServeDryAdmitVerdict(rt *serveRuntime, sf *serveFlags) serveDryAdmitVerdict {
	v := serveDryAdmitVerdict{
		Schema: serveDryAdmitSchema, Verdict: serveDryAdmitAdmit,
		Model: derefString(sf.model), GGUF: derefString(sf.ggufPath),
		Backend: "cpu", ExecutionPolicy: serveExecutionPolicyProduction, Bypassed: []string{},
		RequireDeviceExecution: rt.requireDeviceExecution, NativeContextTokens: rt.nativeContext.ResolvedTokens,
	}
	if rt.useMetal {
		v.Backend = "metal"
	}
	if rt.chatBackend != nil {
		v.Backend = rt.chatBackend.Name()
	}
	if p := rt.executionPolicy; p != nil {
		v.ExecutionPolicy = p.Name
		v.Bypassed = append(v.Bypassed, p.Bypassed...)
		v.Policy = p.report()
	}
	return v
}

// runServeDryAdmit runs the header-only admission gates loadModel runs before
// any tensor read, then prints the verdict. Nothing loads and no listener binds.
func (rt *serveRuntime) runServeDryAdmit(sf *serveFlags, w io.Writer) error {
	if path := strings.TrimSpace(derefString(sf.ggufPath)); path != "" {
		if err := applyServeNCPUMoE(*sf.nCPUMoE); err != nil {
			return err
		}
		if err := applyServeExpertRingBytes(*sf.expertRingBytes); err != nil {
			return err
		}
		ranks := 1
		if rt.ep.sharded {
			ranks = rt.ep.ranks
		}
		if err := rt.resolveNativeContext(sf, ranks); err != nil {
			return err
		}
		if _, err := preflightServeBackendForward(path, rt.chatBackend); err != nil {
			return err
		}
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(newServeDryAdmitVerdict(rt, sf))
}
