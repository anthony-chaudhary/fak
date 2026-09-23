package ctxmmu_test

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestOversizeToolClassThresholds(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		tool   string
		engine string
		size   int
		want   abi.VerdictKind
	}{
		{"bash", "", 8 << 10, abi.VerdictAllow},
		{"powershell", "", 8 << 10, abi.VerdictAllow},
		{"rg", "", 8 << 10, abi.VerdictAllow},
		{"grep", "", 8 << 10, abi.VerdictAllow},
		{"shell_command", "", 8 << 10, abi.VerdictAllow},
		{"functions.shell_command", "", 8 << 10, abi.VerdictAllow},
		{"run_terminal_cmd", "", 8 << 10, abi.VerdictAllow},
		{"tool", "codetools.bash", 8 << 10, abi.VerdictAllow},
		{"tool", "codetools.grep", 8 << 10, abi.VerdictAllow},
		{"read", "", 8 << 10, abi.VerdictAllow},
		{"bash", "", (1 << 20) + 1, abi.VerdictTransform},
	} {
		t.Run(tc.tool+"_"+tc.engine+"_"+strconv.Itoa(tc.size), func(t *testing.T) {
			m := ctxmmu.New()
			body := distinctOversize(tc.size)
			c := call(tc.tool)
			c.Engine = tc.engine
			r := result(c, body)
			verdict := m.Admit(ctx, c, r)
			if verdict.Kind != tc.want {
				t.Fatalf("tool=%q bytes=%d verdict=%v, want %v", tc.tool, tc.size, verdict.Kind, tc.want)
			}
			if tc.want != abi.VerdictTransform {
				if got := resolveBody(t, ctx, r.Payload); !bytes.Equal(got, body) {
					t.Fatalf("tool=%q %d-byte output changed", tc.tool, tc.size)
				}
				return
			}
			var pointer struct {
				Paged         bool   `json:"_paged"`
				Ref           string `json:"ref"`
				RetrievalTool string `json:"retrieval_tool"`
			}
			transform, ok := verdict.Payload.(abi.TransformPayload)
			if !ok {
				t.Fatalf("tool=%q Transform payload type = %T", tc.tool, verdict.Payload)
			}
			pointerBody := resolveBody(t, ctx, transform.NewArgs)
			if err := json.Unmarshal(pointerBody, &pointer); err != nil {
				t.Fatalf("decode pointer: %v", err)
			}
			if !pointer.Paged || pointer.Ref == "" || pointer.RetrievalTool != "fak_context_restore" {
				t.Fatalf("not a recoverable pointer: %+v body=%q", pointer, pointerBody[:min(len(pointerBody), 300)])
			}
			if restored, ok := m.ResolvePagedRef(ctx, pointer.Ref); !ok || !bytes.Equal(restored, body) {
				t.Fatalf("restore tool=%q failed: found=%t restored bytes=%d", tc.tool, ok, len(restored))
			}
		})
	}
}
