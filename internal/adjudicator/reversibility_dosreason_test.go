package adjudicator

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// reversibilityEmittedReasonToken is the wire spelling of abi.VerdictRequireWitness —
// the token the gateway surfaces in-band as the refusal reason when a verdict carries
// no closed-vocabulary Reason (internal/gateway/wire.go maps VerdictRequireWitness to
// "REQUIRE_WITNESS"; internal/gateway/reversibility_note.go reasonOrKind returns the
// Reason name if set, else this KIND). The adjudicator does not own that string, so it
// is pinned here as a constant tied to that mapping.
const reversibilityEmittedReasonToken = "REQUIRE_WITNESS"

// TestReversibilityGateEmittedReasonIsClassified is the closed-vocabulary conformance
// witness for #2776. The reversibility rung refuses with Kind=VerdictRequireWitness and
// sets NO Reason (reversibilityGateVerdict), so the in-band render falls back to the
// KIND and the agent sees "REQUIRE_WITNESS" as the reason it was refused with. That
// emitted token MUST resolve to a declared, refusable dos.toml [reasons] table —
// otherwise `dos_check_reason REQUIRE_WITNESS` returns known=false / category=UNCLASSIFIED,
// exactly the verdict-KIND-into-the-reason-slot drift the closed vocabulary exists to
// kill (the gate emitting the invalid reason the kernel's own contract rejects).
//
// It drives the REAL classifier and verdict builder rather than a hand-built verdict,
// so it binds the actual production shape: an outward-facing `git push` -> RequireWitness
// with an empty Reason -> the KIND-fallback token -> a declared dos.toml reason. It fails
// loudly on the pre-#2776 state (the [reasons.REQUIRE_WITNESS] table absent or not marked
// refusal = true) and passes while the declaration stands — the regression guard for the
// dos.toml registration (committed 858b06ed0) that resolves the UNCLASSIFIED drift.
//
// The dos.toml-reading helpers below mirror internal/toon and internal/assumecheck's
// dosreasons_test.go (which copy the same helpers because the packages cannot import one
// another's test code).
func TestReversibilityGateEmittedReasonIsClassified(t *testing.T) {
	// Drive the real classifier + verdict builder for an outward-facing call.
	env := ClassifyReversibility("git_push", map[string]any{"command": "git push origin main"})
	if env.Class == ReversibilityReversible {
		t.Fatalf("git push should classify as an escalated (non-reversible) call; got %q", env.Class)
	}
	v := reversibilityGateVerdict(env)

	// The exact condition #2776 names: the gate surfaces its KIND and carries no
	// closed-vocabulary Reason, so the render falls back to the KIND token.
	if v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("reversibility verdict Kind = %v, want VerdictRequireWitness", v.Kind)
	}

	// The agent-visible refusal token is the Reason name when the verdict sets one,
	// else the KIND (internal/gateway reasonOrKind). This gate sets no Reason.
	token := reversibilityEmittedReasonToken
	if v.Reason != abi.ReasonNone {
		token = abi.ReasonName(v.Reason)
	}

	// A core abi reason is inherently classified; any other emitted token (the KIND
	// leak here) must be declared in the workspace dos.toml [reasons] table, or
	// dos_check_reason resolves it UNCLASSIFIED.
	if _, core := abi.ReasonByName(token); core {
		return
	}
	content := readRepoDosToml(t)
	header := "[reasons." + token + "]"
	if !strings.Contains(content, header) {
		t.Fatalf("reversibility gate emits refusal token %q with no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED drift, #2776)", token, header)
	}
	block := dosReasonBlock(content, header)
	if !reasonFieldTrue(block, "refusal") {
		t.Fatalf("reason %q is declared but not marked refusal = true — dos_check_reason would resolve it as non-refusable (#2776)", token)
	}
}

// dosReasonBlock returns the text of the [reasons.<TOKEN>] table named by header:
// from the header line up to (but excluding) the next top-level [section] or EOF, so
// the field assertions scope to a single reason's table rather than a sibling's.
func dosReasonBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// reasonFieldTrue reports whether block contains a `field = true` line, tolerant of the
// aligned whitespace the dos.toml [reasons] tables use (e.g. "refusal  = true").
func reasonFieldTrue(block, field string) bool {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") && strings.TrimSpace(rest[1:]) == "true" {
			return true
		}
	}
	return false
}

// readRepoDosToml reads the repo-root dos.toml located relative to this test's own
// source path (internal/adjudicator), so the lookup is independent of the test's
// working directory.
func readRepoDosToml(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}
