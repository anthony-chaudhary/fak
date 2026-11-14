package sandbox

import (
	"context"
	"runtime"
	"strings"
	"testing"
)

// TestL1InstanceChildEnvStripsGatewayKeys is the #14450 witness for the L1 engine:
// with provider credentials and a key-principal variable set in the PARENT process,
// a real child executed through the instance sees none of them; an explicit
// non-secret spec addition still passes, and a spec trying to add a protected name
// is refused.
func TestL1InstanceChildEnvStripsGatewayKeys(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-parent-secret-openai")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-parent-secret")
	t.Setenv("FAK_TENANT_PRINCIPAL", "tenant-a")

	ws := t.TempDir()
	spec := Spec{
		Tier:         TierL1NativeOS,
		WorkspaceDir: ws,
		EgressPolicy: EgressBlocked,
		Env:          []string{"CHILD_ENV_PROBE=visible", "FAK_TENANT_PRINCIPAL=should-not-pass"},
		ProtectedEnv: []string{"FAK_TENANT_PRINCIPAL"},
	}
	prov := NewL1Provider()
	inst, err := prov.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer inst.Close()

	var req ExecutionRequest
	if runtime.GOOS == "windows" {
		req = ExecutionRequest{Command: "cmd.exe", Argv: []string{"/d", "/s", "/c", "set"}}
	} else {
		req = ExecutionRequest{Command: "env"}
	}
	res, err := inst.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute failed: %v (stderr=%s)", err, res.Stderr)
	}
	stdout := string(res.Stdout)
	for _, secret := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FAK_TENANT_PRINCIPAL"} {
		if strings.Contains(stdout, secret) {
			t.Errorf("child env leaked %s:\n%s", secret, stdout)
		}
	}
	if !strings.Contains(stdout, "CHILD_ENV_PROBE=visible") {
		t.Errorf("explicit spec addition did not reach the child:\n%s", stdout)
	}
	if !strings.Contains(stdout, "PATH=") {
		t.Errorf("allowlisted PATH missing from child env:\n%s", stdout)
	}
}
