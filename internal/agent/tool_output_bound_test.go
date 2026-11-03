package agent

// tool_output_bound_test.go — acceptance witness for the settle-time tool-output
// bound (ticket oss-port-agent-tool-output-spill): an ADAPT of opencode's
// settle-time result truncation (packages/core/src/tool/registry.ts@4eb29a64,
// MIT) onto fak's existing internal/toolbound seam. The turn loop settles the
// RoleTool message with the bounded head/tail preview plus the spill notice
// while the full bytes live in the managed spill file. Driven end-to-end
// through RunArm with a scripted planner and a routed big-result engine:
//
//   - option off (unarmed): the RoleTool message carries the 1 MiB result
//     byte-for-byte unchanged;
//   - option on (armed): the message content stays within the byte cap, keeps
//     the head/tail preview around the spill notice, and the spill file holds
//     the full original bytes.
//
// Fixture envelope: the fake engine is registered under the engine id "exec",
// a search/exec-class name in ctxmmu's output classification
// (mmu_durability.go isSearchExecTool). Such results stay in context up to
// SearchExecOversizeBytes (exactly 1 MiB, strict >), so a 1 MiB result is the
// largest payload that reaches the loop's settlement point RAW — precisely the
// case the settle-time bound exists for. Threat-shaped or larger results are
// normgate/ctxmmu territory upstream of this seam and are not this ticket's
// subject.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/toolbound"
)

const spillOversizeSize = 1 << 20 // the 1 MiB tool result the contract pins

// oversizeBody builds the deterministic 1 MiB payload. Deliberately
// threat-free AND non-degenerate: ctxmmu's admission screen holds out long
// runs of one repeated 16-byte chunk as pollution (repeats), so the body
// walks a 26-letter phase pattern whose period never aligns with the chunk
// boundary — no two adjacent chunks are equal, and no injection/secret/PII
// shape matches. That is what lets the bytes reach the loop's settlement
// point unmodified.
func oversizeBody() []byte {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	b := make([]byte, spillOversizeSize)
	for i := range b {
		b[i] = alphabet[(i*7)%len(alphabet)]
	}
	return b
}

// bigResultEngine answers every dispatched call with the fixed oversize body,
// so the settlement path receives a 1 MiB result without touching the
// filesystem.
type bigResultEngine struct{ body []byte }

func (e *bigResultEngine) Caps() []abi.Capability { return nil }

func (e *bigResultEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return engineResult(ctx, c, nil, e.body, false, "exec"), nil
}

// spillManifest routes get_user_details to the "exec" id so the kernel
// dispatches to the fake engine without disturbing the shared "localtools"
// registration.
func spillManifest() *modelroute.Manifest {
	return &modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "default", Role: "primary"}}},
		Rules: []modelroute.Rule{
			{
				Name:  "get-user-to-big-engine",
				Match: modelroute.Match{Aspect: modelroute.AspectToolCall, Tool: toolGetUser},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "exec"}}},
			},
		},
	}
}

// runSpillScript drives RunArm over two turns (one get_user_details call, then
// a final answer) and returns the RoleTool message the second model boundary
// saw. user_id is unique per scenario so the vDSO result cache can never serve
// another test's entry.
func runSpillScript(t *testing.T, userID string) Message {
	t.Helper()
	Configure() // registers localtools + the agent policy
	abi.RegisterEngine("exec", &bigResultEngine{body: oversizeBody()})

	var roleTool Message
	found := false
	observer := func(b ModelRequestBoundary) error {
		for _, m := range b.Messages {
			if m.Role == RoleTool {
				roleTool = m
				found = true
			}
		}
		return nil
	}
	_, err := RunArm(context.Background(), &scriptedPlanner{turns: []*Completion{
		toolCallTurn(toolGetUser, `{"user_id":"`+userID+`"}`),
		{Message: Message{Content: "done"}},
	}}, "task", true, 5, nil, WithRouteManifest(spillManifest()), WithModelRequestObserver(observer))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if !found {
		t.Fatal("no RoleTool message reached the model boundary — the tool result was not settled")
	}
	return roleTool
}

func TestAgentToolResultSettlementSpillsOversize(t *testing.T) {
	body := oversizeBody()

	t.Run("option_off_content_unchanged", func(t *testing.T) {
		msg := runSpillScript(t, "u_spill_off")
		if !bytes.Equal([]byte(msg.Content), body) {
			t.Fatalf("unarmed settlement changed the content: got %d bytes, want the raw %d bytes byte-for-byte", len(msg.Content), len(body))
		}
	})

	t.Run("option_on_spills_oversize", func(t *testing.T) {
		spillDir := t.TempDir()
		const maxBytes = 64 << 10
		ArmToolOutputBound(toolbound.Options{MaxBytes: maxBytes, SpillDir: spillDir})
		defer DisarmToolOutputBound()

		msg := runSpillScript(t, "u_spill_on")

		// Within the cap: head + tail previews plus the notice, nowhere near 1 MiB.
		if len(msg.Content) >= len(body) {
			t.Fatalf("armed settlement did not bound the content: %d bytes vs %d raw", len(msg.Content), len(body))
		}
		if len(msg.Content) > maxBytes+512 {
			t.Fatalf("bounded content %d bytes exceeds the %d-byte cap plus notice margin", len(msg.Content), maxBytes)
		}
		// Head/tail preview preserved around the spill notice.
		if !strings.HasPrefix(msg.Content, string(body[:maxBytes/2])) {
			t.Error("bounded content lost the head preview")
		}
		if !strings.HasSuffix(msg.Content, string(body[len(body)-maxBytes/2:])) {
			t.Error("bounded content lost the tail preview")
		}
		if !strings.Contains(msg.Content, "output truncated: spilled") {
			t.Errorf("bounded content lacks the spill notice (len=%d)", len(msg.Content))
		}
		// The spill file holds the full original bytes.
		entries, err := os.ReadDir(spillDir)
		if err != nil {
			t.Fatalf("read spill dir: %v", err)
		}
		var files []string
		for _, e := range entries {
			if !e.IsDir() {
				files = append(files, e.Name())
			}
		}
		if len(files) != 1 {
			t.Fatalf("spill dir holds %d files (%v), want exactly one spill file", len(files), files)
		}
		full, err := os.ReadFile(filepath.Join(spillDir, files[0]))
		if err != nil {
			t.Fatalf("read spill file: %v", err)
		}
		if !bytes.Equal(full, body) {
			t.Fatalf("spill file holds %d bytes, want the full original %d bytes", len(full), len(body))
		}
	})
}
