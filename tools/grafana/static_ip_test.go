package grafana

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testenv/repotest"
)

// TestComposeServicesPinStaticIPs keeps every published service on a fixed
// address. Docker Desktop's port forwarder caches the container IP behind a
// published port, so containers that swap IPs on restart leave the host port
// answering with an empty reply while the containers report healthy.
func TestComposeServicesPinStaticIPs(t *testing.T) {
	compose := readCompose(t)
	const prefix = "${FAK_OBS_SUBNET_PREFIX:-172.30.90}"
	if !strings.Contains(compose, "- subnet: "+prefix+".0/24") {
		t.Errorf("default network must declare a fixed subnet %s.0/24", prefix)
	}
	for _, svc := range []string{"prometheus", "alertmanager", "grafana"} {
		if !strings.Contains(composeServiceBody(t, compose, svc), "ipv4_address: "+prefix+".") {
			t.Errorf("service %s must pin ipv4_address on %s.0/24", svc, prefix)
		}
	}
}

// TestComposeUnauthenticatedAPIsBindLoopback keeps the Prometheus and
// Alertmanager HTTP APIs, which have no auth, off the LAN. Grafana reaches both
// over the compose network, so only the host-published port is narrowed.
func TestComposeUnauthenticatedAPIsBindLoopback(t *testing.T) {
	compose := readCompose(t)
	for svc, port := range map[string]string{"prometheus": "9091", "alertmanager": "9093"} {
		body := composeServiceBody(t, compose, svc)
		mapping := regexp.MustCompile(`^((\[[^\]]*\]|[0-9.]+):)?[0-9]+:` + port + `(/tcp)?$`)
		var mappings []string
		for _, line := range strings.Split(body, "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "- ") {
				continue
			}
			if item := strings.Trim(strings.TrimSpace(line[2:]), `"'`); mapping.MatchString(item) {
				mappings = append(mappings, item)
			}
		}
		if len(mappings) == 0 {
			t.Errorf("service %s publishes no :%s mapping", svc, port)
		}
		for _, m := range mappings {
			if m != "127.0.0.1:"+port+":"+port {
				t.Errorf("service %s publishes %q; want 127.0.0.1:%s:%s", svc, m, port, port)
			}
		}
	}
	if strings.Contains(composeServiceBody(t, compose, "prometheus"), "--web.enable-lifecycle") {
		t.Error("prometheus must not enable the unauthenticated lifecycle API (/-/quit, /-/reload)")
	}
}

func readCompose(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repotest.PackageDir(t), "docker-compose.yml"))
	if err != nil {
		t.Fatalf("read docker-compose.yml: %v", err)
	}
	return string(raw)
}

func composeServiceBody(t *testing.T, compose, svc string) string {
	t.Helper()
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
	return body[:next]
}
