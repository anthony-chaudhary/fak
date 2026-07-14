package systemservice

import (
	"strings"
	"testing"
)

func TestRenderSystemdUserUnitOwnsAndHardensControlPlane(t *testing.T) {
	u, err := RenderSystemdUserUnit(SystemdConfig{Executable: "/home/a b/fak", StateDir: "/home/a/.local/state/fak"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"ExecStart=\"/home/a b/fak\" service run --interval 15s", "Restart=always", "KillMode=control-group", "NoNewPrivileges=yes", "MemoryMax=1G", "TasksMax=256", "CPUQuota=50%", "WantedBy=default.target"} {
		if !strings.Contains(u, want) {
			t.Fatalf("unit missing %q:\n%s", want, u)
		}
	}
	if strings.Contains(u, "Terminal") || strings.Contains(u, "DISPLAY=") {
		t.Fatalf("control-plane unit depends on UI: %s", u)
	}
}
func TestRenderSystemdRejectsNewlines(t *testing.T) {
	if _, e := RenderSystemdUserUnit(SystemdConfig{Executable: "/tmp/fak\nExecStart=evil", StateDir: "/tmp"}); e == nil {
		t.Fatal("injection accepted")
	}
}
