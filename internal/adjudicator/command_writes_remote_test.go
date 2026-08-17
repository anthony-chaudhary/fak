package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestSSHRemotePayloadDoesNotNameLocalWriteTargets(t *testing.T) {
	guarded := DefaultPolicy().SelfModifyGlobs
	credential := guardCredentialGlobs
	cases := []struct {
		name  string
		cmd   string
		globs []string
	}{
		{"remote test", "ssh node 'test -e internal/abi/x.go'", guarded},
		{"remote grep", "ssh node 'grep pattern internal/abi/x.go'", guarded},
		{"remote copy", "ssh node 'cp build/out internal/abi/x.go'", guarded},
		{"remote move", "ssh node 'mv tmp dos.toml'", guarded},
		{"remote redirect", "ssh node 'echo x > internal/abi/x.go'", guarded},
		{"option before destination", "ssh -o ConnectTimeout=2 node 'test -e .git/config'", credential},
		{"env wrapper", "env ssh node 'cp x internal/abi/x.go'", guarded},
		{"absolute executable", "/usr/bin/ssh node 'mv x dos.toml'", guarded},
		{"windows executable", "ssh.exe node 'echo x > .env'", credential},
		{"tailscale hostname", "tailscale ssh anthony@100.68.66.42 'hostname; test -f ~/.profile'", credential},
		{"tailscale utc date", "tailscale.exe ssh anthony@100.68.66.42 'hostname; date -u; test -e /etc/hosts'", credential},
		{"absolute tailscale executable", "/usr/bin/tailscale ssh node 'test -e .git/config'", credential},
		{"powershell probe", "Test-NetConnection node -Port 22", guarded},
		{"socket probe", "nc -z node 22", guarded},
		{"http head mention", "curl --head https://example.invalid/.git/", credential},
		{"literal mention", "printf '%s' 'internal/abi/'", guarded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandSelfModify(map[string]any{"command": tc.cmd}, tc.globs); got != "" {
				t.Fatalf("read-only/local-safe command matched %q: %s", got, tc.cmd)
			}
		})
	}
}

func TestSSHLocalExecutionSurfacesRemainGuarded(t *testing.T) {
	guarded := DefaultPolicy().SelfModifyGlobs
	cases := []struct {
		name string
		cmd  string
		want string
	}{
		{"outer redirect quoted payload", "ssh node 'remote read' > internal/abi/probe.log", "internal/abi/"},
		{"outer redirect operands", "ssh node cat > dos.toml", "dos.toml"},
		{"proxy command", "ssh -o 'ProxyCommand=sh -c \"cp x internal/abi/x.go\"' node", "internal/abi/"},
		{"local command", "ssh -o 'LocalCommand=sh -c \"echo x > dos.toml\"' node", "dos.toml"},
		{"local nested shell", "sh -c 'cp x internal/abi/x.go'", "internal/abi/"},
		{"plain local copy", "cp x internal/abi/x.go", "internal/abi/"},
		{"preceding local mutation", "sed -i s/a/b/ internal/abi/x.go && ssh host echo done", "internal/abi/"},
		{"tailscale outer redirect", "tailscale ssh node 'hostname' > dos.toml", "dos.toml"},
		{"tailscale preceding local mutation", "cp x internal/abi/x.go && tailscale ssh node hostname", "internal/abi/"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandSelfModify(map[string]any{"command": tc.cmd}, guarded); got != tc.want {
				t.Fatalf("command match = %q, want %q: %s", got, tc.want, tc.cmd)
			}
		})
	}
}

func TestSSHRemotePayloadBoundaryParsing(t *testing.T) {
	allowed := []string{
		"ssh -- node 'cp x internal/abi/x.go'",
		"ssh -oProxyCommand=none node 'cp x internal/abi/x.go'",
		"ssh -p22 node 'mv x dos.toml'",
		"ssh -vv node 'echo x > internal/abi/x.go'",
		"tailscale ssh node 'test -e .git/config'",
		"tailscale.exe ssh node 'test -e /etc/hosts'",
	}
	for _, cmd := range allowed {
		if got := commandSelfModify(map[string]any{"command": cmd}, DefaultPolicy().SelfModifyGlobs); got != "" {
			t.Errorf("understood ssh spelling matched %q: %s", got, cmd)
		}
	}
}

func TestSSHLocalCommandDecisionKeepsBoundedSelfModifyWitness(t *testing.T) {
	cmd := `ssh -o 'LocalCommand=sh -c "echo x > dos.toml"' node`
	v := New(DefaultPolicy()).Adjudicate(context.Background(), inlineCall("Bash", `{"command":`+strconvQuote(cmd)+`}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("decision = (%v, %v), want (DENY, SELF_MODIFY)", v.Kind, v.Reason)
	}
	w, ok := v.Payload.(abi.WitnessPayload)
	if !ok || w.Claim != "dos.toml" {
		t.Fatalf("witness = %#v, want bounded matched glob %q", v.Payload, "dos.toml")
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
