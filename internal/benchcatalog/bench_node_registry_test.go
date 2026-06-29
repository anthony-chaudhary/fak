package benchcatalog

import (
	"encoding/json"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestBenchNodeExampleRegistryIsTrackedAndRealRegistryIsNot(t *testing.T) {
	root := repoRootForBenchNodeRegistryTest(t)
	out, err := exec.Command("git", "-C", root, "ls-files").Output()
	if err != nil {
		t.Fatalf("git ls-files: %v", err)
	}
	tracked := "\n" + string(out) + "\n"
	if !strings.Contains(tracked, "\ntools/bench_nodes.example.json\n") {
		t.Fatal("tools/bench_nodes.example.json must be tracked")
	}
	if strings.Contains(tracked, "\ntools/bench_nodes.json\n") {
		t.Fatal("tools/bench_nodes.json carries live identity and must stay untracked")
	}
}

func TestBenchNodeExampleRegistryPlaceholders(t *testing.T) {
	d := loadBenchNodeExample(t)
	if d.Schema != "fleet-bench-nodes/1" {
		t.Fatalf("schema = %q", d.Schema)
	}
	if len(d.Nodes) == 0 || len(d.Onboarding.Roster) == 0 {
		t.Fatal("nodes and onboarding roster must be populated")
	}
	nodes := map[string]benchNode{}
	for _, n := range d.Nodes {
		nodes[n.Name] = n
		if n.TailnetIP == "" {
			if n.ResolveCmd == "" {
				t.Fatalf("%s needs tailnet_ip or resolve_cmd", n.Name)
			}
		} else {
			ip := net.ParseIP(n.TailnetIP)
			if ip == nil || !isDocIP(ip) {
				t.Fatalf("%s tailnet_ip %q must be an RFC-5737 documentation address", n.Name, n.TailnetIP)
			}
			if strings.HasSuffix(n.TailnetIP, ".") || strings.HasPrefix(n.TailnetIP, "100.") {
				t.Fatalf("%s tailnet_ip %q looks like live identity", n.Name, n.TailnetIP)
			}
		}
		if !strings.HasSuffix(n.MagicDNS, ".example.ts.net") {
			t.Fatalf("%s magicdns %q must use .example.ts.net", n.Name, n.MagicDNS)
		}
		if n.SSHUser != "user" {
			t.Fatalf("%s ssh_user %q must be placeholder user", n.Name, n.SSHUser)
		}
		if n.HostKey != "" {
			t.Fatalf("%s host_key must be empty in the public template", n.Name)
		}
	}
	for _, r := range d.Onboarding.Roster {
		n, ok := nodes[r.Node]
		if !ok {
			t.Fatalf("roster node %q missing from nodes[]", r.Node)
		}
		if n.SanitizedName != r.SanitizedName {
			t.Fatalf("%s sanitized_name mismatch: node %q roster %q", r.Node, n.SanitizedName, r.SanitizedName)
		}
		if r.Status != "onboarded" && r.Status != "pending" {
			t.Fatalf("%s bad status %q", r.Node, r.Status)
		}
		if r.Status == "onboarded" && r.Verified == "" {
			t.Fatalf("%s onboarded but has no verified date", r.Node)
		}
		if r.Status == "pending" && r.Blocker == "" {
			t.Fatalf("%s pending but has no blocker", r.Node)
		}
		delete(nodes, r.Node)
	}
	if len(nodes) != 0 {
		t.Fatalf("nodes missing roster entries: %v", nodes)
	}
	gcp := d.nodeByName("gcp-gpu")
	if gcp == nil {
		t.Fatal("gcp-gpu placeholder missing")
	}
	if gcp.TailnetIP != "" {
		t.Fatalf("gcp-gpu must not commit a static IP; got %q", gcp.TailnetIP)
	}
	if !strings.Contains(gcp.ResolveCmd, "tools/gcp_bench.py --resolve-ip") {
		t.Fatalf("gcp-gpu resolve_cmd must call the GCP bench provisioner resolver: %q", gcp.ResolveCmd)
	}
	if !strings.Contains(gcp.ResolveCmd, "BENCH_NODE_GCP_INSTANCE") ||
		!strings.Contains(gcp.ResolveCmd, "BENCH_NODE_GCP_ZONE") {
		t.Fatalf("gcp-gpu resolve_cmd must keep real instance/zone in the gitignored registry/env: %q", gcp.ResolveCmd)
	}
}

type benchNodeExample struct {
	Schema     string      `json:"schema"`
	Nodes      []benchNode `json:"nodes"`
	Onboarding struct {
		Roster []benchNodeRoster `json:"roster"`
	} `json:"_onboarding"`
}

type benchNode struct {
	Name          string `json:"name"`
	SanitizedName string `json:"sanitized_name"`
	TailnetIP     string `json:"tailnet_ip"`
	ResolveCmd    string `json:"resolve_cmd"`
	MagicDNS      string `json:"magicdns"`
	SSHUser       string `json:"ssh_user"`
	HostKey       string `json:"host_key"`
}

type benchNodeRoster struct {
	Node          string `json:"node"`
	SanitizedName string `json:"sanitized_name"`
	Status        string `json:"status"`
	Verified      string `json:"verified"`
	Blocker       string `json:"blocker"`
}

func loadBenchNodeExample(t *testing.T) benchNodeExample {
	t.Helper()
	root := repoRootForBenchNodeRegistryTest(t)
	b, err := os.ReadFile(filepath.Join(root, "tools", "bench_nodes.example.json"))
	if err != nil {
		t.Fatalf("read bench_nodes.example.json: %v", err)
	}
	var d benchNodeExample
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("decode bench_nodes.example.json: %v", err)
	}
	return d
}

func (d benchNodeExample) nodeByName(name string) *benchNode {
	for i := range d.Nodes {
		if d.Nodes[i].Name == name {
			return &d.Nodes[i]
		}
	}
	return nil
}

func repoRootForBenchNodeRegistryTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found")
		}
		dir = parent
	}
}

func isDocIP(ip net.IP) bool {
	for _, cidr := range []string{"192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"} {
		_, n, _ := net.ParseCIDR(cidr)
		if n.Contains(ip) {
			return true
		}
	}
	return false
}
