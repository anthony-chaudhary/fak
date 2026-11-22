package adjudicator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestDefaultPostureMatchesSecurityDoc pins the SECURITY.md "Capability floor"
// row to the shipped default posture, so the public guarantee cannot drift from
// what DefaultPolicy actually enforces.
func TestDefaultPostureMatchesSecurityDoc(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "SECURITY.md"))
	if err != nil {
		t.Fatalf("read SECURITY.md: %v", err)
	}
	var row string
	for _, line := range strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(line, "| **Capability floor** |") {
			row = line
			break
		}
	}
	if row == "" {
		t.Fatal(`SECURITY.md has no "| **Capability floor** |" table row`)
	}

	def := DefaultPolicy()
	shipped := def.Posture.String()
	if !strings.Contains(row, "shipped `"+shipped+"` posture") {
		t.Errorf("Capability floor row must name the shipped posture `%s` that DefaultPolicy returns; row: %s", shipped, row)
	}
	strict := "--posture " + PostureFailClosed.String()
	if !strings.Contains(row, strict) {
		t.Errorf("Capability floor row must name the strict lever %q; row: %s", strict, row)
	}

	const unknownTool = "security_doc_unlisted_tool"
	if def.Allow[unknownTool] {
		t.Fatalf("%s must stay off the default allow-list for this check", unknownTool)
	}
	ctx := context.Background()

	if def.Posture != PostureDefaultOpen {
		t.Fatalf("doc row describes an unlisted tool as permitted under the shipped posture, but DefaultPolicy posture is %s", shipped)
	}
	if !strings.Contains(row, "unlisted tool is permitted") {
		t.Errorf("Capability floor row must state that an unlisted tool is permitted under %s; row: %s", shipped, row)
	}
	v := New(def).Adjudicate(ctx, inlineCall(unknownTool, `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Errorf("default policy, unlisted tool: got %v (%s), want ALLOW as SECURITY.md states", v.Kind, abi.ReasonName(v.Reason))
	}

	strictPolicy := DefaultPolicy()
	strictPolicy.Posture = PostureFailClosed
	v = New(strictPolicy).Adjudicate(ctx, inlineCall(unknownTool, `{}`))
	if v.Kind != abi.VerdictDeny {
		t.Errorf("fail_closed policy, unlisted tool: got %v (%s), want DENY as SECURITY.md states", v.Kind, abi.ReasonName(v.Reason))
	}
}
