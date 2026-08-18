package scmbridge

// scmdefinition.go projects the portable fak.service.v1 desired-state contract
// (#4749) into the concrete Windows SCM service DEFINITION (#4756, parent
// #4748) — the Windows twin of internal/systemservice's systemd unit rendering
// (#4750). Everything here is pure and deterministic: no SCM, registry, or
// sc.exe calls — the installer and the reconciler both read what this file
// renders, so the two planes cannot drift apart.
//
// The projection rules this file encodes (and refuses violations of):
//
//   - The failure-action ladder is DERIVED from the ONE portable
//     servicespec.RestartPolicy by replaying RestartPolicy.Decide: each rung is
//     the bounded exponential backoff delay, and the restart-window circuit
//     opening projects as a terminal "none" action — SCM repeats the last
//     action for further failures, so an open circuit stays open natively.
//   - desired-stopped is preserved, not deleted: the service stays installed
//     with start type "disabled" so a reboot or a reconciler cannot resurrect
//     an intentional stop. There is no uninstall in this projection.
//   - Least privilege by construction: every machine service definition runs
//     as NT AUTHORITY\LocalService (MachinePrincipal); there is no input that
//     can name a stronger account.
//   - depends_on projects onto native SCM service dependencies (the BindsTo
//     analog): the SCM starts dependencies first and cascades stops, which is
//     how servicespec.ExitDependencyLoss surfaces natively.
//   - A watchdog (liveness) deadline requires notify readiness — SCM has no
//     native watchdog, so without a heartbeat channel there is no liveness
//     signal to miss, and the request is refused rather than silently dropped.
//     Accepted watchdog wiring is carried AS DATA for the S4U watchdog task
//     (RoleWatchdog) to enforce; a missed deadline classifies as
//     servicespec.ExitWatchdog.
//   - Secrets are referenced, never serialized: an EnvRef.SecretRef requires a
//     protected env-file reference; rendering a secret value into a definition
//     is impossible by construction.
//   - Rendering is refused, never escaped, when a value would make the sc.exe
//     text ambiguous (embedded double quotes, control characters), so the
//     rendered form is deterministic byte-for-byte.

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/linefmt"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// SCMDefinitionSchemaV1 names the versioned SCM service-definition document.
const SCMDefinitionSchemaV1 = "fak.service.scm.v1"

// StartType is the closed SCM start-type vocabulary the definition uses.
// Demand exists for read-back comparison; the v1 spec itself only projects
// auto, delayed-auto, and disabled.
type StartType string

const (
	StartTypeAuto        StartType = "auto"
	StartTypeDelayedAuto StartType = "delayed-auto"
	StartTypeDemand      StartType = "demand"
	StartTypeDisabled    StartType = "disabled"
)

// FailureActionKind is what SCM does on one failure. Reboot and run-command
// are deliberately not in the vocabulary: a fak workload never earns the right
// to reboot the host or run an arbitrary command from a recovery slot.
type FailureActionKind string

const (
	FailureRestart FailureActionKind = "restart"
	FailureNone    FailureActionKind = "none"
)

// FailureAction is one SCM recovery rung: restart after DelayMS, or nothing.
type FailureAction struct {
	Kind    FailureActionKind `json:"kind"`
	DelayMS int64             `json:"delay_ms,omitempty"`
}

// FailureActions is the full SERVICE_FAILURE_ACTIONS projection.
type FailureActions struct {
	// ResetPeriodSec is SCM's failure-count reset period, derived from the
	// spec's stable-run reset (the same mapping Projection.RecoveryResetSec
	// uses): a run that stays up this long starts the ladder over.
	ResetPeriodSec int64 `json:"reset_period_sec"`
	// Actions is the ladder replayed from RestartPolicy.Decide. SCM applies
	// Actions[n] to the n-th failure and repeats the LAST action beyond the
	// ladder — which is why a circuit-open policy ends in FailureNone.
	Actions []FailureAction `json:"actions"`
	// OnNonCrashFailures is SCM's FAILURE_ACTIONS_ON_NONCRASH_FAILURES flag
	// (sc.exe failureflag). True for a KindService: any exit — clean included —
	// leaves desired=running unmet, so every stop without an operator intent
	// counts as a failure.
	OnNonCrashFailures bool `json:"on_non_crash_failures"`
}

// Definition is the desired Windows SCM form of one machine-owned workload —
// the deterministic document an installer applies (CreateService /
// ChangeServiceConfig2) and a reconciler diffs read-back against.
type Definition struct {
	Schema      string               `json:"schema"`
	Identity    servicespec.Identity `json:"identity"`
	ServiceName string               `json:"service_name"`
	DisplayName string               `json:"display_name"`
	Description string               `json:"description"`
	// BinPath is the rendered service command line (quoted image path + args).
	BinPath   string    `json:"bin_path"`
	StartType StartType `json:"start_type"`
	// Account is always MachinePrincipal — least privilege by construction.
	Account string `json:"account"`
	// Dependencies are native SCM service dependencies from depends_on.
	Dependencies []string `json:"dependencies,omitempty"`
	// Environment carries literal NAME=value bindings (sorted by the spec);
	// the installer writes them to the service's registry Environment value.
	// SecretRef entries are never here — they resolve through EnvFile.
	Environment []string `json:"environment,omitempty"`
	// EnvFile is the protected env-file REFERENCE covering every SecretRef.
	EnvFile string         `json:"env_file,omitempty"`
	Failure FailureActions `json:"failure"`
	// Watchdog wiring as data, enforced by the S4U watchdog task (SCM itself
	// has no watchdog): the liveness deadline and the recommended keepalive
	// cadence (half the deadline). Zero means no watchdog.
	WatchdogMS  int64 `json:"watchdog_ms,omitempty"`
	KeepaliveMS int64 `json:"keepalive_ms,omitempty"`
	// Readiness carries the opaque probe contract through from the spec.
	ReadinessKind   string `json:"readiness_kind,omitempty"`
	ReadinessTarget string `json:"readiness_target,omitempty"`
	StartTimeoutMS  int64  `json:"start_timeout_ms,omitempty"`
	// DesiredStopped projects desired=stopped: installed but start-disabled,
	// preserved across reconcile and reboot, never auto-resurrected.
	DesiredStopped bool `json:"desired_stopped"`
	// ConfigText is the deterministic sc.exe command sequence expressing the
	// SCM-expressible part of this definition (environment and env-file are
	// registry-applied by the installer; sc.exe cannot express them).
	ConfigText string `json:"config_text"`
}

// DefinitionInput carries the host-specific facts the portable spec does not
// encode.
type DefinitionInput struct {
	// DelayedStart projects auto start as delayed-auto (start after the boot
	// storm). Ignored when the spec is desired-stopped.
	DelayedStart bool
	// WatchdogMS is the liveness deadline; requires notify readiness.
	WatchdogMS int64
	// EnvFile is the protected env-file reference every EnvRef.SecretRef must
	// be covered by; secret values are never rendered.
	EnvFile string
}

// Definition refusals — the closed reasons ProjectDefinition fails.
var (
	ErrBadServiceName      = errors.New("scmbridge: identity/dependency is not a valid SCM service name")
	ErrQuoteInArgument     = errors.New("scmbridge: value contains a double quote; sc.exe rendering would be ambiguous")
	ErrSecretNeedsEnvFile  = errors.New("scmbridge: env secret_ref requires a protected env-file reference; a secret value is never rendered into a service definition")
	ErrWatchdogNeedsNotify = errors.New("scmbridge: a watchdog deadline requires notify readiness (kind \"notify\"); without a heartbeat channel there is no liveness signal")
)

// ReadinessNotify is the servicespec.Readiness.Kind this projection interprets
// as a native heartbeat channel (the Windows twin of sd_notify readiness).
const ReadinessNotify = "notify"

// scmNameToken is the strict charset for service and dependency names. SCM
// formally forbids only "/" and "\"; the projection keeps the same strict
// token set as the systemd twin so one identity is valid on every platform.
var scmNameToken = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

// maxFailureLadder bounds the replayed ladder: a window cap beyond this many
// rungs is truncated to a final max-backoff restart (which SCM repeats), an
// honest approximation of an effectively uncapped policy.
const maxFailureLadder = 16

// ProjectDefinition derives the deterministic SCM service definition of the
// spec's machine role. Placement is enforced here exactly as in Project: a
// recurring job never becomes an SCM service.
func ProjectDefinition(spec *servicespec.Spec, in DefinitionInput) (*Definition, error) {
	if spec == nil {
		return nil, errors.New("scmbridge: nil spec")
	}
	s := *spec
	s.Normalize()
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if s.Kind == servicespec.KindJob {
		return nil, ErrJobOnSCM
	}
	if !scmNameToken.MatchString(s.Identity.Workload) {
		return nil, fmt.Errorf("%w: workload %q", ErrBadServiceName, s.Identity.Workload)
	}
	for _, d := range s.DependsOn {
		if !scmNameToken.MatchString(d) {
			return nil, fmt.Errorf("%w: dependency %q", ErrBadServiceName, d)
		}
	}
	named := map[string]string{
		"identity.node": s.Identity.Node, "identity.service": s.Identity.Service,
		"dir": s.Dir, "checkpoint_dir": s.CheckpointDir, "env_file": in.EnvFile,
	}
	for name, v := range named {
		if err := renderable(name, v); err != nil {
			return nil, err
		}
	}
	for _, a := range s.Command {
		if err := renderable("command argument", a); err != nil {
			return nil, err
		}
	}
	for _, e := range s.Env {
		if err := renderable("env "+e.Name, e.Name+e.Value); err != nil {
			return nil, err
		}
		if e.SecretRef != "" && in.EnvFile == "" {
			return nil, fmt.Errorf("%w (env %q)", ErrSecretNeedsEnvFile, e.Name)
		}
	}
	var readinessKind, readinessTarget string
	var startTimeout int64
	if s.Readiness != nil {
		readinessKind, readinessTarget, startTimeout = s.Readiness.Kind, s.Readiness.Target, s.Readiness.TimeoutMS
	}
	if in.WatchdogMS > 0 && readinessKind != ReadinessNotify {
		return nil, ErrWatchdogNeedsNotify
	}

	d := &Definition{
		Schema:          SCMDefinitionSchemaV1,
		Identity:        s.Identity,
		ServiceName:     s.Identity.Workload,
		DisplayName:     "fak workload " + s.Identity.Workload,
		Description:     fmt.Sprintf("fak workload %s (service %s) on node %s, projected from %s", s.Identity.Workload, s.Identity.Service, s.Identity.Node, servicespec.SchemaV1),
		BinPath:         scmCommandLine(s.Command),
		Account:         MachinePrincipal,
		Dependencies:    append([]string(nil), s.DependsOn...),
		EnvFile:         in.EnvFile,
		WatchdogMS:      in.WatchdogMS,
		KeepaliveMS:     in.WatchdogMS / 2,
		ReadinessKind:   readinessKind,
		ReadinessTarget: readinessTarget,
		StartTimeoutMS:  startTimeout,
		DesiredStopped:  s.Desired == servicespec.DesiredStopped,
	}
	switch {
	case d.DesiredStopped:
		d.StartType = StartTypeDisabled
	case in.DelayedStart:
		d.StartType = StartTypeDelayedAuto
	default:
		d.StartType = StartTypeAuto
	}
	for _, e := range s.Env {
		if e.SecretRef != "" {
			continue // referenced via EnvFile, never serialized
		}
		d.Environment = append(d.Environment, e.Name+"="+e.Value)
	}
	d.Failure = FailureActions{
		ResetPeriodSec:     s.Restart.StableRunResetMS / 1000,
		Actions:            failureLadder(s.Restart),
		OnNonCrashFailures: s.Kind == servicespec.KindService,
	}
	d.ConfigText = scmConfigText(d)
	return d, nil
}

// failureLadder replays RestartPolicy.Decide crash-by-crash: rung n is the
// decision for the n-th consecutive failure inside the restart window. The
// first non-restart decision (the circuit opening) becomes a terminal
// FailureNone — SCM repeats the last action, so the circuit stays open until
// the reset period restarts the count.
func failureLadder(rp servicespec.RestartPolicy) []FailureAction {
	out := make([]FailureAction, 0, 8)
	for attempt := 0; attempt < maxFailureLadder; attempt++ {
		dec := rp.Decide(servicespec.RestartInput{
			Kind:        servicespec.KindService,
			Desired:     servicespec.DesiredRunning,
			Class:       servicespec.ExitCrash,
			Attempt:     attempt,
			WindowCount: attempt,
		})
		if !dec.Restart {
			return append(out, FailureAction{Kind: FailureNone})
		}
		out = append(out, FailureAction{Kind: FailureRestart, DelayMS: dec.DelayMS})
	}
	return out
}

// renderable refuses values that would make the rendered definition ambiguous:
// control characters and embedded double quotes are never escaped, always
// refused, so rendering is total and deterministic on everything it accepts.
func renderable(name, v string) error {
	if strings.ContainsAny(v, "\x00\r\n") {
		return fmt.Errorf("scmbridge: %s contains a control character", name)
	}
	if strings.ContainsRune(v, '"') {
		return fmt.Errorf("%w (%s)", ErrQuoteInArgument, name)
	}
	return nil
}

// scmCommandLine joins the command with Windows quoting: an argument with
// spaces or tabs is wrapped in double quotes (embedded quotes were refused).
func scmCommandLine(args []string) string {
	quoted := make([]string, 0, len(args))
	for _, a := range args {
		quoted = append(quoted, winQuote(a))
	}
	return strings.Join(quoted, " ")
}

func winQuote(a string) string {
	if a == "" || strings.ContainsAny(a, " \t") {
		return `"` + a + `"`
	}
	return a
}

// scmConfigText is the deterministic sc.exe command sequence for the
// SCM-expressible part of the definition. Line and field order are fixed. The
// sequence configures only — it never starts or deletes: starting is the
// reconciler's decision and desired-stopped keeps the service installed.
func scmConfigText(d *Definition) string {
	var b strings.Builder
	w := linefmt.Writer(&b)
	// binPath= is a quoted argument whose value embeds our own quoting; only
	// quotes introduced by winQuote can appear (input quotes were refused), so
	// escaping them as \" is unambiguous.
	w(`sc.exe create %s binPath= "%s" start= %s obj= "%s" DisplayName= "%s"`,
		d.ServiceName, strings.ReplaceAll(d.BinPath, `"`, `\"`), d.StartType, d.Account, d.DisplayName)
	if len(d.Dependencies) > 0 {
		w("sc.exe config %s depend= %s", d.ServiceName, strings.Join(d.Dependencies, "/"))
	}
	w(`sc.exe description %s "%s"`, d.ServiceName, d.Description)
	rungs := make([]string, 0, len(d.Failure.Actions))
	for _, a := range d.Failure.Actions {
		if a.Kind == FailureRestart {
			rungs = append(rungs, fmt.Sprintf("restart/%d", a.DelayMS))
		} else {
			rungs = append(rungs, `""/0`)
		}
	}
	w("sc.exe failure %s reset= %d actions= %s", d.ServiceName, d.Failure.ResetPeriodSec, strings.Join(rungs, "/"))
	if d.Failure.OnNonCrashFailures {
		w("sc.exe failureflag %s 1", d.ServiceName)
	}
	return b.String()
}
