package egressfloor_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// fak-test:runtime fast est=1s
func TestSSHDestinationOmitsWindowsExecutable(t *testing.T) {
	command := "C:/Windows/System32/OpenSSH/ssh.exe -F none -T -o StrictHostKeyChecking=yes operator@node.example /bin/hostname"
	hosts := egressfloor.Destinations("Bash", map[string]any{"command": command})
	if !slices.Equal(hosts, []string{"node.example"}) {
		t.Fatalf("destinations = %v, want only node.example", hosts)
	}
}

// fak-test:runtime fast est=1s
func TestSSHDestinationHardFloor(t *testing.T) {
	for _, target := range []string{"operator@169.254.169.254", "operator@metadata.google.internal", "operator@metadata", "operator@[fd00:ec2::254]"} {
		t.Run(target, func(t *testing.T) {
			args := map[string]any{"command": "/usr/bin/ssh -F none -T -a -x -o BatchMode=yes " + target + " /bin/hostname"}
			if host, _ := egressfloor.Classify("Bash", args); host == "" {
				t.Fatal("SSH destination escaped metadata floor")
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSSHDestinationConfiguredPolicies(t *testing.T) {
	args := map[string]any{"command": "C:/Windows/System32/OpenSSH/ssh.exe -F none -T -o StrictHostKeyChecking=yes operator@blocked.example /bin/hostname"}
	if host, _ := egressfloor.Classify("Bash", args, "blocked.example"); host != "blocked.example" {
		t.Fatalf("extra deny host = %q", host)
	}
	hosts := egressfloor.Destinations("Bash", args)
	if !slices.Contains(hosts, "blocked.example") || slices.Contains(hosts, "operator@blocked.example") {
		t.Fatalf("SSH hosts = %v", hosts)
	}
	for _, kind := range []egresslist.Kind{egresslist.Block, egresslist.Allow} {
		list := egresslist.NewBuilder().AddRules("operator", []string{"blocked.example"}, kind).Build()
		if decision := list.Decide(hosts[slices.Index(hosts, "blocked.example")]); decision.Kind != kind {
			t.Fatalf("projected host policy = %v, want %v", decision.Kind, kind)
		}
	}
}

// fak-test:runtime fast est=1s
func TestSSHDestinationPreservesOtherExtraction(t *testing.T) {
	for _, command := range []string{
		"/usr/bin/ssh -o ProxyCommand=curl https://proxy.example operator@node.example /bin/hostname",
		"/usr/bin/ssh operator@node.example curl https://remote.example",
		"curl https://user:password@url.example/path",
	} {
		hosts := egressfloor.Destinations("Bash", map[string]any{"command": command})
		for _, host := range []string{"proxy.example", "remote.example", "url.example"} {
			if strings.Contains(command, host) && !slices.Contains(hosts, host) {
				t.Fatalf("lost %s from %v", host, hosts)
			}
		}
	}
	if host, _ := egressfloor.Classify("write", map[string]any{"content": "ssh user@169.254.169.254"}); host != "" {
		t.Fatal("document content became egress")
	}
}
