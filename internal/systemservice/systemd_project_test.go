package systemservice

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// notifySpec is a representative always-on service: sd_notify readiness, one
// dependency, a literal env, a referenced secret, and a checkpoint dir.
func notifySpec() *servicespec.Spec {
	return &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "node-a", Service: "gateway"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"/opt/fak/bin/fak", "serve", "--addr", ":8080"},
		Dir:      "/var/lib/fak",
		Env: []servicespec.EnvRef{
			{Name: "FAK_TOKEN", SecretRef: "secret://fak/token"},
			{Name: "FAK_MODE", Value: "fleet"},
		},
		Readiness:     &servicespec.Readiness{Kind: ReadinessNotify, TimeoutMS: 15000},
		CheckpointDir: "/var/lib/fak/ckpt",
		DependsOn:     []string{"registry"},
	}
}

const goldenNotifySystemUnit = `[Unit]
Description=fak workload gateway (service gateway) on node node-a
Documentation=https://github.com/anthony-chaudhary/fak
After=network-online.target fak-registry.service
Wants=network-online.target
BindsTo=fak-registry.service
StartLimitIntervalSec=600s
StartLimitBurst=5

[Service]
Type=notify
NotifyAccess=main
WatchdogSec=30s
TimeoutStartSec=15s
ExecStart="/opt/fak/bin/fak" "serve" "--addr" ":8080"
WorkingDirectory="/var/lib/fak"
EnvironmentFile="/etc/fak/gateway.env"
Environment="FAK_MODE=fleet"
Environment=FAK_SERVICE_MANAGER=systemd-system
Environment="FAK_SERVICE_NODE=node-a"
Environment="FAK_SERVICE_NAME=gateway"
Environment="FAK_SERVICE_WORKLOAD=gateway"
Restart=always
RestartSec=1s
RestartSteps=6
RestartMaxDelaySec=60s
KillMode=control-group
UMask=0077
NoNewPrivileges=yes
DynamicUser=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths="/var/lib/fak" "/var/lib/fak/ckpt"

[Install]
WantedBy=multi-user.target
`

func TestProjectSystemdGoldenNotifySystemUnit(t *testing.T) {
	in := SystemdInput{
		Scope:            ScopeSystem,
		EnvironmentFiles: []string{"/etc/fak/gateway.env"},
		WatchdogMS:       30000,
	}
	p, err := ProjectSystemd(notifySpec(), in)
	if err != nil {
		t.Fatal(err)
	}
	if p.Schema != SystemdProjectionSchemaV1 {
		t.Fatalf("schema = %q", p.Schema)
	}
	if p.UnitName != "fak-gateway.service" {
		t.Fatalf("unit name = %q", p.UnitName)
	}
	if !p.Enabled || p.DesiredStopped {
		t.Fatalf("desired-running must project enabled: %+v", p)
	}
	if p.UnitText != goldenNotifySystemUnit {
		t.Fatalf("unit text drifted from golden.\n--- got ---\n%s\n--- want ---\n%s", p.UnitText, goldenNotifySystemUnit)
	}
	n := p.Notify
	if n.Mode != NotifySdNotify {
		t.Fatalf("notify mode = %q", n.Mode)
	}
	if n.ReadyMessage != "READY=1" || n.StoppingMessage != "STOPPING=1" || n.WatchdogMessage != "WATCHDOG=1" {
		t.Fatalf("sd_notify vocabulary drifted: %+v", n)
	}
	if n.WatchdogMS != 30000 || n.KeepaliveMS != 15000 {
		t.Fatalf("watchdog keepalive must be half the deadline: %+v", n)
	}
	if n.StartTimeoutMS != 15000 {
		t.Fatalf("start timeout = %d", n.StartTimeoutMS)
	}
	if strings.Contains(p.UnitText, "secret://fak/token") {
		t.Fatal("secret reference leaked into the unit text")
	}
	// Determinism: projecting the same spec twice is byte-identical.
	q, err := ProjectSystemd(notifySpec(), in)
	if err != nil {
		t.Fatal(err)
	}
	if q.UnitText != p.UnitText {
		t.Fatal("projection is not deterministic")
	}
}

const goldenFallbackUserUnit = `[Unit]
Description=fak workload sweeper (service sweeper) on node node-b
Documentation=https://github.com/anthony-chaudhary/fak
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60s
StartLimitBurst=3

[Service]
Type=exec
TimeoutStartSec=5s
ExecStart="/usr/local/bin/fak" "sweep"
Environment=FAK_SERVICE_MANAGER=systemd-user
Environment="FAK_SERVICE_NODE=node-b"
Environment="FAK_SERVICE_NAME=sweeper"
Environment="FAK_SERVICE_WORKLOAD=sweeper"
Restart=on-failure
RestartSec=2s
KillMode=control-group
UMask=0077
NoNewPrivileges=yes

[Install]
WantedBy=default.target
`

func TestProjectSystemdGoldenFallbackUserJob(t *testing.T) {
	spec := &servicespec.Spec{
		Schema:    servicespec.SchemaV1,
		Identity:  servicespec.Identity{Node: "node-b", Service: "sweeper"},
		Kind:      servicespec.KindJob,
		Desired:   servicespec.DesiredStopped,
		Command:   []string{"/usr/local/bin/fak", "sweep"},
		Readiness: &servicespec.Readiness{Kind: "http", Target: "http://127.0.0.1:9090/healthz", TimeoutMS: 5000},
		Restart: servicespec.RestartPolicy{
			InitialBackoffMS: 2000, MaxBackoffMS: 2000, BackoffFactor: 2,
			WindowMS: 60000, WindowMaxRestarts: 3,
		},
	}
	p, err := ProjectSystemd(spec, SystemdInput{Scope: ScopeUser})
	if err != nil {
		t.Fatal(err)
	}
	if p.UnitText != goldenFallbackUserUnit {
		t.Fatalf("unit text drifted from golden.\n--- got ---\n%s\n--- want ---\n%s", p.UnitText, goldenFallbackUserUnit)
	}
	// desired-stopped is preserved, not deleted: unit present, disabled.
	if p.Enabled || !p.DesiredStopped {
		t.Fatalf("desired-stopped must project installed-but-disabled: %+v", p)
	}
	n := p.Notify
	if n.Mode != NotifyFallback {
		t.Fatalf("fallback readiness must not project sd-notify: %+v", n)
	}
	if n.ReadinessKind != "http" || n.ReadinessTarget != "http://127.0.0.1:9090/healthz" {
		t.Fatalf("fallback probe lost the readiness contract: %+v", n)
	}
	if n.WatchdogMS != 0 || n.KeepaliveMS != 0 {
		t.Fatalf("fallback mode must not claim a watchdog: %+v", n)
	}
}

func TestProjectSystemdDesiredStopIsUnitInvariant(t *testing.T) {
	in := SystemdInput{Scope: ScopeSystem, EnvironmentFiles: []string{"/etc/fak/gateway.env"}, WatchdogMS: 30000}
	running, err := ProjectSystemd(notifySpec(), in)
	if err != nil {
		t.Fatal(err)
	}
	stopped := notifySpec()
	stopped.Desired = servicespec.DesiredStopped
	sp, err := ProjectSystemd(stopped, in)
	if err != nil {
		t.Fatal(err)
	}
	// Same unit either way: intent moves ONLY the enablement bit, so a
	// reconcile or reboot cannot resurrect an intentional stop.
	if sp.UnitText != running.UnitText {
		t.Fatal("desired-stopped changed the rendered unit; it must only flip enablement")
	}
	if sp.Enabled || !sp.DesiredStopped {
		t.Fatalf("desired-stopped enablement wrong: %+v", sp)
	}
}

func TestProjectSystemdNamedLeastPrivilegeIdentity(t *testing.T) {
	p, err := ProjectSystemd(notifySpec(), SystemdInput{
		Scope: ScopeSystem, User: "fak", Group: "fak",
		EnvironmentFiles: []string{"/etc/fak/gateway.env"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\nUser=fak\n", "\nGroup=fak\n"} {
		if !strings.Contains(p.UnitText, want) {
			t.Fatalf("unit missing %q:\n%s", want, p.UnitText)
		}
	}
	if strings.Contains(p.UnitText, "DynamicUser=") {
		t.Fatal("named identity must not also allocate a dynamic user")
	}
}

func TestProjectSystemdRefusals(t *testing.T) {
	base := func() *servicespec.Spec { return notifySpec() }
	in := SystemdInput{Scope: ScopeSystem, EnvironmentFiles: []string{"/etc/fak/gateway.env"}}

	if _, err := ProjectSystemd(nil, in); !errors.Is(err, ErrNilSpec) {
		t.Fatalf("nil spec: %v", err)
	}
	if _, err := ProjectSystemd(base(), SystemdInput{Scope: "session", EnvironmentFiles: in.EnvironmentFiles}); !errors.Is(err, ErrUnknownScope) {
		t.Fatalf("unknown scope: %v", err)
	}
	// A referenced secret with no environment-file reference must refuse —
	// rendering the value inline is impossible by construction.
	if _, err := ProjectSystemd(base(), SystemdInput{Scope: ScopeSystem}); !errors.Is(err, ErrSecretNeedsFile) {
		t.Fatalf("secret without env file: %v", err)
	}
	// A watchdog needs the sd_notify channel; fallback readiness has none.
	s := base()
	s.Readiness = &servicespec.Readiness{Kind: "http", Target: "http://127.0.0.1:1/healthz"}
	if _, err := ProjectSystemd(s, SystemdInput{Scope: ScopeSystem, EnvironmentFiles: in.EnvironmentFiles, WatchdogMS: 30000}); !errors.Is(err, ErrWatchdogNoNotify) {
		t.Fatalf("watchdog without notify: %v", err)
	}
	// Unit-token strictness: a workload that could template or traverse.
	s = base()
	s.Identity.Workload = "evil@template"
	if _, err := ProjectSystemd(s, in); !errors.Is(err, ErrBadUnitToken) {
		t.Fatalf("bad workload token: %v", err)
	}
	s = base()
	s.DependsOn = []string{"reg/istry"}
	if _, err := ProjectSystemd(s, in); !errors.Is(err, ErrBadUnitToken) {
		t.Fatalf("bad dependency token: %v", err)
	}
	// Control-character injection anywhere in rendered values must refuse.
	s = base()
	s.Command = []string{"/opt/fak/bin/fak\nExecStart=evil"}
	if _, err := ProjectSystemd(s, in); err == nil {
		t.Fatal("command injection accepted")
	}
	s = base()
	s.Dir = "/var/lib/fak\nReadWritePaths=/"
	if _, err := ProjectSystemd(s, in); err == nil {
		t.Fatal("dir injection accepted")
	}
	s = base()
	s.Env = append(s.Env, servicespec.EnvRef{Name: "X", Value: "a\nEnvironment=evil"})
	if _, err := ProjectSystemd(s, in); err == nil {
		t.Fatal("env injection accepted")
	}
}
