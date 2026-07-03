package loopmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestEmbeddedDefaultMatchesOperatorTemplate pins the binary-embedded default
// policy (internal/loopmgr/default-policy.json, the one a background loop
// inherits) in sync with the operator-facing template (tools/loop-policy.default.json,
// the one a human copies to .fak/loop-policy.json). The two are the SAME sane
// defaults viewed from two doors; if they drift, an operator reading the template
// would configure a loop differently from what the binary actually ships. The
// template carries `_doc` annotation keys the embedded copy omits, so the compare
// is over the parsed Policies value, not the raw bytes.
func TestEmbeddedDefaultMatchesOperatorTemplate(t *testing.T) {
	embedded := DefaultPolicies()

	root := repoRootForDefaultPolicyTest(t)
	raw, err := os.ReadFile(filepath.Join(root, "tools", "loop-policy.default.json"))
	if err != nil {
		t.Fatalf("read operator template: %v", err)
	}
	var template Policies
	if err := json.Unmarshal(raw, &template); err != nil {
		t.Fatalf("decode operator template: %v", err)
	}

	if !reflect.DeepEqual(embedded, template) {
		t.Fatalf("embedded default and operator template have drifted:\n embedded=%+v\n template=%+v\n"+
			"keep internal/loopmgr/default-policy.json and tools/loop-policy.default.json in sync",
			embedded, template)
	}
}

// repoRootForDefaultPolicyTest walks up from the test's working directory to the
// module root (go.mod), so the test can read the operator template under tools/.
func repoRootForDefaultPolicyTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
