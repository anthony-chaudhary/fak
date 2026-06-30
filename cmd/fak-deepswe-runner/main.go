// Command fak-deepswe-runner emits deterministic DeepSWE adapter fixtures for
// SWE-bench runner-contract tests.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/swebench"
)

const requestSchema = "fak.swebench.deepswe-request.v1"

type adapterRequest struct {
	Schema   string            `json:"schema"`
	Instance swebench.Instance `json:"instance"`
	Model    string            `json:"model,omitempty"`
	MaxSteps int               `json:"max_steps,omitempty"`
	Repo     string            `json:"repo,omitempty"`
	Runner   string            `json:"runner"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("fak-deepswe-runner", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fixture := fs.Bool("fixture", false, "emit a deterministic adapter-contract fixture prediction")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*fixture {
		fmt.Fprintln(stderr, "fak-deepswe-runner: --fixture is required; this command is a contract fixture, not a real DeepSWE runner")
		return 2
	}

	var req adapterRequest
	if err := json.NewDecoder(stdin).Decode(&req); err != nil {
		fmt.Fprintf(stderr, "decode request: %v\n", err)
		return 2
	}
	if req.Schema != requestSchema {
		fmt.Fprintf(stderr, "bad schema %q, want %q\n", req.Schema, requestSchema)
		return 2
	}
	if req.Runner != string(swebench.RunnerDeepSWE) {
		fmt.Fprintf(stderr, "bad runner %q, want %q\n", req.Runner, swebench.RunnerDeepSWE)
		return 2
	}
	if strings.TrimSpace(req.Instance.InstanceID) == "" {
		fmt.Fprintln(stderr, "missing instance_id")
		return 2
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = "DeepSWE-Preview-fixture"
	}
	pred := swebench.Prediction{
		InstanceID:      req.Instance.InstanceID,
		ModelNameOrPath: model,
		ModelPatch:      fixturePatch(req.Instance),
	}
	if err := json.NewEncoder(stdout).Encode(pred); err != nil {
		fmt.Fprintf(stderr, "encode prediction: %v\n", err)
		return 1
	}
	return 0
}

func fixturePatch(in swebench.Instance) string {
	name := strings.ReplaceAll(in.InstanceID, "__", "_")
	if name == "" {
		name = "deepswe_fixture"
	}
	return fmt.Sprintf(`diff --git a/%[1]s_fak_deepswe_fixture.txt b/%[1]s_fak_deepswe_fixture.txt
new file mode 100644
index 0000000..f1c7000
--- /dev/null
+++ b/%[1]s_fak_deepswe_fixture.txt
@@ -0,0 +1,2 @@
+DeepSWE adapter contract fixture for %[2]s
+This patch is grader-consumable shape evidence, not a benchmark score.
`, sanitizePatchStem(name), in.InstanceID)
}

func sanitizePatchStem(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "deepswe_fixture"
	}
	return out
}
