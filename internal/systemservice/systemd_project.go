package systemservice

// systemd_project.go projects the portable fak.service.v1 desired-state
// contract (#4749) into its Linux systemd form (#4750, parent #4748): a
// deterministic unit rendering plus the sd_notify/watchdog wiring AS DATA.
// This is the Linux twin of internal/scmbridge's fak.service.windows.v1
// projection: pure and deterministic, no systemctl or D-Bus calls — cmd/fak
// shells and operators install/read-back what this leaf renders.
//
// The projection rules the leaf encodes (and refuses violations of):
//
//   - Readiness kind "notify" projects Type=notify: systemd itself owns the
//     starting -> ready edge (READY=1) and, when a watchdog interval is set,
//     liveness (WATCHDOG=1 keepalives; a miss is servicespec.ExitWatchdog).
//   - Any other readiness kind projects the fallback: Type=exec, where
//     "running" stands in for "ready" at the manager and the supervisor probes
//     readiness itself from the NotifyWiring kind/target. A watchdog interval
//     without sd_notify support is refused — there is no heartbeat channel.
//   - Secrets are referenced, never serialized: an EnvRef.SecretRef requires a
//     root-owned EnvironmentFile= reference; rendering a secret value inline
//     is impossible by construction.
//   - desired-stopped is preserved, not deleted: the unit stays installed with
//     Enabled=false so a reboot cannot resurrect an intentional stop.
//   - Least privilege by default: no named user means DynamicUser=yes.

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/linefmt"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// SystemdProjectionSchemaV1 names the versioned systemd projection document.
const SystemdProjectionSchemaV1 = "fak.service.systemd.v1"

// SystemdScope is which systemd manager owns the unit.
type SystemdScope string

const (
	// ScopeSystem is a PID-1-owned system unit (WantedBy=multi-user.target).
	ScopeSystem SystemdScope = "system"
	// ScopeUser is a per-user manager unit (WantedBy=default.target).
	ScopeUser SystemdScope = "user"
)

// ReadinessNotify is the servicespec.Readiness.Kind this supervisor interprets
// as native sd_notify support (servicespec keeps the kind opaque; the platform
// projection interprets it).
const ReadinessNotify = "notify"

// NotifyMode is how the starting -> ready edge is decided.
type NotifyMode string

const (
	// NotifySdNotify: Type=notify — the workload sends READY=1/WATCHDOG=1/
	// STOPPING=1 over $NOTIFY_SOCKET and systemd owns readiness + liveness.
	NotifySdNotify NotifyMode = "sd-notify"
	// NotifyFallback: Type=exec — the manager equates running with ready; the
	// fak supervisor probes real readiness from the wiring's kind/target.
	NotifyFallback NotifyMode = "fallback-probe"
)

// Notify message vocabulary (sd_notify(3)).
const (
	MsgReady    = "READY=1"
	MsgStopping = "STOPPING=1"
	MsgWatchdog = "WATCHDOG=1"
)

// NotifyWiring is the readiness/watchdog wiring as data: what the workload
// must send (or the supervisor must probe) and what the manager enforces.
type NotifyWiring struct {
	Mode NotifyMode `json:"mode"`
	// sd-notify mode: the exact messages and cadence.
	ReadyMessage    string `json:"ready_message,omitempty"`
	StoppingMessage string `json:"stopping_message,omitempty"`
	WatchdogMessage string `json:"watchdog_message,omitempty"`
	// WatchdogMS is the WatchdogSec= deadline (0 = watchdog off). A missed
	// deadline is classified servicespec.ExitWatchdog.
	WatchdogMS int64 `json:"watchdog_ms,omitempty"`
	// KeepaliveMS is the recommended WATCHDOG=1 send interval: half the
	// watchdog deadline, per sd_watchdog_enabled(3).
	KeepaliveMS int64 `json:"keepalive_ms,omitempty"`
	// Fallback mode: the readiness probe the fak supervisor owns (opaque
	// kind/target carried through from the spec; empty kind means "running is
	// ready").
	ReadinessKind   string `json:"readiness_kind,omitempty"`
	ReadinessTarget string `json:"readiness_target,omitempty"`
	// StartTimeoutMS bounds starting -> ready in both modes (TimeoutStartSec=).
	StartTimeoutMS int64 `json:"start_timeout_ms,omitempty"`
}

// SystemdInput carries the host-specific facts the portable spec does not
// encode.
type SystemdInput struct {
	Scope SystemdScope
	// User/Group is the least-privilege identity for system-scope units; empty
	// User means DynamicUser=yes (systemd allocates a transient identity).
	User  string
	Group string
	// EnvironmentFiles are root-owned env-file REFERENCES (EnvironmentFile=).
	// Every EnvRef.SecretRef must be covered by one; secret values are never
	// rendered.
	EnvironmentFiles []string
	// WatchdogMS enables the systemd watchdog (sd-notify readiness only).
	WatchdogMS int64
}

// SystemdProjection is the desired systemd form of one workload — the document
// both the installer and the reconciler read, derived from the ONE portable
// spec so the two planes cannot drift apart.
type SystemdProjection struct {
	Schema   string               `json:"schema"`
	Identity servicespec.Identity `json:"identity"`
	Kind     servicespec.Kind     `json:"kind"`
	Scope    SystemdScope         `json:"scope"`
	UnitName string               `json:"unit_name"`
	// UnitText is the deterministic rendered unit file.
	UnitText string       `json:"unit_text"`
	Notify   NotifyWiring `json:"notify"`
	// Enabled projects intent onto boot persistence: enable --now when true.
	Enabled bool `json:"enabled"`
	// DesiredStopped: unit installed but disabled and stopped — preserved
	// across reconcile and reboot, never auto-resurrected as a crash.
	DesiredStopped bool `json:"desired_stopped"`
}

// Projection refusals — the closed reasons ProjectSystemd fails.
var (
	ErrNilSpec          = errors.New("systemservice: nil spec")
	ErrUnknownScope     = errors.New("systemservice: unknown systemd scope")
	ErrBadUnitToken     = errors.New("systemservice: identity/dependency is not a valid systemd unit token")
	ErrSecretNeedsFile  = errors.New("systemservice: env secret_ref requires an EnvironmentFile reference; a secret value is never rendered into a unit")
	ErrWatchdogNoNotify = errors.New("systemservice: watchdog keepalives require sd_notify readiness (kind \"notify\"); fallback readiness has no heartbeat channel")
)

// unitToken is the strict charset for workload and dependency names embedded
// in unit names ("@" is excluded so a name can never become a template).
var unitToken = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

// SystemdUnitNameFor derives the deterministic unit name for a workload.
func SystemdUnitNameFor(workload string) string { return "fak-" + workload + ".service" }

// ProjectSystemd derives the desired systemd form of the spec. The notify/
// watchdog placement rule is enforced here, once: sd_notify readiness gets
// Type=notify (+ optional watchdog), everything else gets the fallback, and a
// watchdog without a notify channel is refused.
func ProjectSystemd(spec *servicespec.Spec, in SystemdInput) (*SystemdProjection, error) {
	if spec == nil {
		return nil, ErrNilSpec
	}
	s := *spec
	s.Normalize()
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if in.Scope != ScopeSystem && in.Scope != ScopeUser {
		return nil, fmt.Errorf("%w: %q", ErrUnknownScope, in.Scope)
	}
	if !unitToken.MatchString(s.Identity.Workload) {
		return nil, fmt.Errorf("%w: workload %q", ErrBadUnitToken, s.Identity.Workload)
	}
	for _, d := range s.DependsOn {
		if !unitToken.MatchString(d) {
			return nil, fmt.Errorf("%w: dependency %q", ErrBadUnitToken, d)
		}
	}
	for name, v := range map[string]string{
		"identity.node": s.Identity.Node, "identity.service": s.Identity.Service,
		"dir": s.Dir, "checkpoint_dir": s.CheckpointDir, "user": in.User, "group": in.Group,
	} {
		if strings.ContainsAny(v, "\x00\r\n") {
			return nil, fmt.Errorf("systemservice: %s contains a control character", name)
		}
	}
	for _, a := range s.Command {
		if strings.ContainsAny(a, "\x00\r\n") {
			return nil, errors.New("systemservice: command argument contains a control character")
		}
	}
	for _, f := range in.EnvironmentFiles {
		if strings.TrimSpace(f) == "" || strings.ContainsAny(f, "\x00\r\n") {
			return nil, errors.New("systemservice: environment file reference is empty or contains a control character")
		}
	}
	for _, e := range s.Env {
		if strings.ContainsAny(e.Name+e.Value, "\x00\r\n") {
			return nil, fmt.Errorf("systemservice: env %q contains a control character", e.Name)
		}
		if e.SecretRef != "" && len(in.EnvironmentFiles) == 0 {
			return nil, fmt.Errorf("%w (env %q)", ErrSecretNeedsFile, e.Name)
		}
	}

	notify := notifyWiring(&s, in)
	if notify.Mode != NotifySdNotify && in.WatchdogMS > 0 {
		return nil, ErrWatchdogNoNotify
	}

	p := &SystemdProjection{
		Schema:         SystemdProjectionSchemaV1,
		Identity:       s.Identity,
		Kind:           s.Kind,
		Scope:          in.Scope,
		UnitName:       SystemdUnitNameFor(s.Identity.Workload),
		Notify:         notify,
		DesiredStopped: s.Desired == servicespec.DesiredStopped,
	}
	p.Enabled = !p.DesiredStopped
	p.UnitText = unitFileText(&s, in, notify)
	return p, nil
}

// notifyWiring interprets the opaque readiness contract for systemd.
func notifyWiring(s *servicespec.Spec, in SystemdInput) NotifyWiring {
	var kind, target string
	var timeout int64
	if s.Readiness != nil {
		kind, target, timeout = s.Readiness.Kind, s.Readiness.Target, s.Readiness.TimeoutMS
	}
	if kind == ReadinessNotify {
		return NotifyWiring{
			Mode:            NotifySdNotify,
			ReadyMessage:    MsgReady,
			StoppingMessage: MsgStopping,
			WatchdogMessage: MsgWatchdog,
			WatchdogMS:      in.WatchdogMS,
			KeepaliveMS:     in.WatchdogMS / 2,
			StartTimeoutMS:  timeout,
		}
	}
	return NotifyWiring{
		Mode:            NotifyFallback,
		ReadinessKind:   kind,
		ReadinessTarget: target,
		StartTimeoutMS:  timeout,
	}
}

// unitFileText renders the deterministic unit text. Field order is fixed; every
// value either passed the control-character refusals above or is quoted.
func unitFileText(s *servicespec.Spec, in SystemdInput, n NotifyWiring) string {
	var b strings.Builder
	w := linefmt.Writer(&b)

	w("[Unit]")
	w("Description=fak workload %s (service %s) on node %s", s.Identity.Workload, s.Identity.Service, s.Identity.Node)
	w("Documentation=https://github.com/anthony-chaudhary/fak")
	after := "network-online.target"
	var binds []string
	for _, d := range s.DependsOn {
		u := SystemdUnitNameFor(d)
		after += " " + u
		binds = append(binds, u)
	}
	w("After=%s", after)
	w("Wants=network-online.target")
	if len(binds) > 0 {
		// BindsTo projects servicespec.ExitDependencyLoss: losing a dependency
		// stops this unit instead of leaving it silently degraded.
		w("BindsTo=%s", strings.Join(binds, " "))
	}
	// Start-rate limiting is the restart-window circuit: exhausting
	// window_max_restarts inside window_ms parks the unit failed (= fenced).
	w("StartLimitIntervalSec=%s", msSpan(s.Restart.WindowMS))
	w("StartLimitBurst=%d", s.Restart.WindowMaxRestarts)
	w("")

	w("[Service]")
	if n.Mode == NotifySdNotify {
		w("Type=notify")
		w("NotifyAccess=main")
		if n.WatchdogMS > 0 {
			w("WatchdogSec=%s", msSpan(n.WatchdogMS))
		}
	} else {
		w("Type=exec")
	}
	if n.StartTimeoutMS > 0 {
		w("TimeoutStartSec=%s", msSpan(n.StartTimeoutMS))
	}
	args := make([]string, 0, len(s.Command))
	for _, a := range s.Command {
		args = append(args, systemdQuote(a))
	}
	w("ExecStart=%s", strings.Join(args, " "))
	if s.Dir != "" {
		w("WorkingDirectory=%s", systemdQuote(s.Dir))
	}
	for _, f := range in.EnvironmentFiles {
		w("EnvironmentFile=%s", systemdQuote(f))
	}
	for _, e := range s.Env {
		if e.SecretRef != "" {
			continue // referenced via EnvironmentFile, never serialized
		}
		w("Environment=%s", systemdQuote(e.Name+"="+e.Value))
	}
	w("Environment=FAK_SERVICE_MANAGER=systemd-%s", in.Scope)
	// The durable workload identity survives process and invocation identity
	// changes — the acceptance witness's "same workload, new process" axis.
	w("Environment=%s", systemdQuote("FAK_SERVICE_NODE="+s.Identity.Node))
	w("Environment=%s", systemdQuote("FAK_SERVICE_NAME="+s.Identity.Service))
	w("Environment=%s", systemdQuote("FAK_SERVICE_WORKLOAD="+s.Identity.Workload))
	if s.Kind == servicespec.KindService {
		// Always-on: any exit — clean included — leaves desire unmet.
		w("Restart=always")
	} else {
		// Recurring job: a clean exit is completion; only failures retry.
		w("Restart=on-failure")
	}
	w("RestartSec=%s", msSpan(s.Restart.InitialBackoffMS))
	if steps := backoffSteps(s.Restart); steps > 0 {
		// systemd >= 254 walks RestartSec up to RestartMaxDelaySec in
		// RestartSteps increments — the same bounded exponential contract as
		// servicespec.RestartPolicy. Older systemd ignores the pair and keeps
		// the flat RestartSec floor.
		w("RestartSteps=%d", steps)
		w("RestartMaxDelaySec=%s", msSpan(s.Restart.MaxBackoffMS))
	}
	w("KillMode=control-group")
	w("UMask=0077")
	w("NoNewPrivileges=yes")
	if in.Scope == ScopeSystem {
		if in.User == "" {
			w("DynamicUser=yes")
		} else {
			w("User=%s", in.User)
			if in.Group != "" {
				w("Group=%s", in.Group)
			}
		}
		w("PrivateTmp=yes")
		w("ProtectSystem=strict")
		w("ProtectHome=read-only")
		var rw []string
		if s.Dir != "" {
			rw = append(rw, systemdQuote(s.Dir))
		}
		if s.CheckpointDir != "" && s.CheckpointDir != s.Dir {
			rw = append(rw, systemdQuote(s.CheckpointDir))
		}
		if len(rw) > 0 {
			w("ReadWritePaths=%s", strings.Join(rw, " "))
		}
	}
	w("")

	w("[Install]")
	if in.Scope == ScopeSystem {
		w("WantedBy=multi-user.target")
	} else {
		w("WantedBy=default.target")
	}
	return b.String()
}

// msSpan renders integer milliseconds as a systemd time span, whole seconds
// when exact so the common cases read naturally.
func msSpan(ms int64) string {
	if ms%1000 == 0 {
		return fmt.Sprintf("%ds", ms/1000)
	}
	return fmt.Sprintf("%dms", ms)
}

// backoffSteps counts the factor-multiplications from the initial to the max
// backoff — systemd's RestartSteps for the same bounded exponential ladder.
// Zero (flat policy) omits the pair.
func backoffSteps(p servicespec.RestartPolicy) int {
	if p.BackoffFactor <= 1 || p.InitialBackoffMS >= p.MaxBackoffMS || p.InitialBackoffMS <= 0 {
		return 0
	}
	steps, d := 0, p.InitialBackoffMS
	for d < p.MaxBackoffMS {
		if d > p.MaxBackoffMS/p.BackoffFactor {
			return steps + 1
		}
		d *= p.BackoffFactor
		steps++
	}
	return steps
}
