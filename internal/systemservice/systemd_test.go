package systemservice

import (
	"strings"
	"testing"
)

func TestRenderSystemdSystemUnitIsPID1OwnedAndHardened(t *testing.T) {
	u, err := RenderSystemdSystemUnit(SystemdConfig{Executable: "/opt/fak/bin/fak", StateDir: "/var/lib/fak"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"DynamicUser=yes", "ExecStart=\"/opt/fak/bin/fak\" service run --interval 15s --notify systemd",
		"Restart=always", "KillMode=control-group", "NoNewPrivileges=yes", "ProtectSystem=strict",
		"ReadWritePaths=\"/var/lib/fak\"", "MemoryMax=1G", "TasksMax=256", "CPUQuota=50%",
		"WantedBy=multi-user.target", "FAK_SERVICE_MANAGER=systemd-system",
	} {
		if !strings.Contains(u, want) {
			t.Fatalf("unit missing %q:\n%s", want, u)
		}
	}
	for _, forbidden := range []string{"WantedBy=default.target", "DISPLAY=", "graphical-session.target", "--user", "StateDirectory=fak", "FAK_SERVICE_NOTIFY"} {
		if strings.Contains(u, forbidden) {
			t.Fatalf("PID-1 control-plane unit contains user/UI ownership %q: %s", forbidden, u)
		}
	}
}

func TestRenderSystemdRejectsInjectionAndMissingPrincipal(t *testing.T) {
	for _, c := range []SystemdConfig{
		{Executable: "/tmp/fak\nExecStart=evil", StateDir: "/var/lib/fak"},
		{Executable: "/opt/fak", StateDir: ""},
		{Executable: "/opt/fak", StateDir: "/var/lib/fak\nReadWritePaths=/"},
	} {
		if _, e := RenderSystemdSystemUnit(c); e == nil {
			t.Fatalf("unsafe config accepted: %+v", c)
		}
	}
}
