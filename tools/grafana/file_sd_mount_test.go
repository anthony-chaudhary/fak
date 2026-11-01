package grafana

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestFleetNodeFileSDMountContract keeps the supervisor-generated fleet-node
// target inventory inside the running Prometheus container. The target file is
// runtime state, so the stable contract is the provisioning-directory mount
// plus the matching in-container file-SD path—not tracking generated node
// identities.
func TestFleetNodeFileSDMountContract(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	read := func(name string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return string(raw)
	}

	compose := read("docker-compose.yml")
	prometheus := read("prometheus.yml")
	const (
		mount = "- ./provisioning:/etc/prometheus/provisioning:ro"
		path  = "- /etc/prometheus/provisioning/fleet-node-targets.json"
		old   = "- /etc/prometheus/fleet-node-targets.json"
	)
	if !strings.Contains(compose, mount) {
		t.Errorf("Prometheus must mount the generated provisioning directory read-only; missing %q", mount)
	}
	if !strings.Contains(prometheus, path) {
		t.Errorf("fleet_node file-SD must read the mounted provisioning path; missing %q", path)
	}
	if strings.Contains(prometheus, old) {
		t.Errorf("fleet_node file-SD still points at the unmounted root path %q", old)
	}
}
