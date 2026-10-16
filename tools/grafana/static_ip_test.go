package grafana

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestComposeServicesPinStaticIPs keeps every published service on a fixed
// address. Docker Desktop's port forwarder caches the container IP behind a
// published port, so containers that swap IPs on restart leave the host port
// answering with an empty reply while the containers report healthy.
func TestComposeServicesPinStaticIPs(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(file), "docker-compose.yml"))
	if err != nil {
		t.Fatalf("read docker-compose.yml: %v", err)
	}
	compose := string(raw)
	const prefix = "${FAK_OBS_SUBNET_PREFIX:-172.30.90}"
	if !strings.Contains(compose, "- subnet: "+prefix+".0/24") {
		t.Errorf("default network must declare a fixed subnet %s.0/24", prefix)
	}
	for _, svc := range []string{"prometheus", "alertmanager", "grafana"} {
		start := strings.Index(compose, "\n  "+svc+":\n")
		if start < 0 {
			t.Fatalf("service %s not found", svc)
		}
		body := compose[start+1:]
		next := len(body)
		for _, other := range []string{"\n  prometheus:\n", "\n  alertmanager:\n", "\n  grafana:\n", "\nnetworks:\n", "\nvolumes:\n"} {
			if i := strings.Index(body[1:], other); i >= 0 && i+1 < next {
				next = i + 1
			}
		}
		if !strings.Contains(body[:next], "ipv4_address: "+prefix+".") {
			t.Errorf("service %s must pin ipv4_address on %s.0/24", svc, prefix)
		}
	}
}
