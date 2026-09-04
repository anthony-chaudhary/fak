package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestReadTransformToFakRead(t *testing.T) {
	a := New(Policy{})
	ctx := context.Background()

	tests := []struct {
		name       string
		tool       string
		args       map[string]any
		wantPath   string
		wantOffset any
		wantLimit  any
	}{
		{
			name:       "camelCase filePath",
			tool:       "Read",
			args:       map[string]any{"filePath": "internal/kernel/kernel.go", "offset": 10, "limit": 20},
			wantPath:   "internal/kernel/kernel.go",
			wantOffset: float64(10),
			wantLimit:  float64(20),
		},
		{
			name:       "short path",
			tool:       "Read",
			args:       map[string]any{"path": "cmd/fak/main.go", "offset": 0, "limit": 100},
			wantPath:   "cmd/fak/main.go",
			wantOffset: float64(0),
			wantLimit:  float64(100),
		},
		{
			name:       "canonical snake_case file_path",
			tool:       "Read",
			args:       map[string]any{"file_path": "README.md"},
			wantPath:   "README.md",
			wantOffset: nil,
			wantLimit:  nil,
		},
		{
			name:       "lowercase read with filePath",
			tool:       "read",
			args:       map[string]any{"filePath": "go.mod", "limit": 50},
			wantPath:   "go.mod",
			wantOffset: nil,
			wantLimit:  float64(50),
		},
		{
			name:       "uppercase READ with path",
			tool:       "READ",
			args:       map[string]any{"path": "LICENSE", "offset": 5},
			wantPath:   "LICENSE",
			wantOffset: float64(5),
			wantLimit:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.args)
			if err != nil {
				t.Fatalf("marshal args: %v", err)
			}
			call := inlineCall(tc.tool, string(b))
			v := a.Adjudicate(ctx, call)

			if v.Kind != abi.VerdictTransform {
				t.Fatalf("got verdict kind %v, want VerdictTransform", v.Kind)
			}
			if v.By != "monitor/read_to_fak_read" {
				t.Fatalf("got By=%q, want %q", v.By, "monitor/read_to_fak_read")
			}
			if v.Meta["reversibility_autorepair"] != "read_to_fak_read" {
				t.Fatalf("got Meta[reversibility_autorepair]=%q, want %q", v.Meta["reversibility_autorepair"], "read_to_fak_read")
			}

			tp, ok := v.Payload.(abi.TransformPayload)
			if !ok {
				t.Fatalf("payload type = %T, want TransformPayload", v.Payload)
			}
			if tp.NewTool != "fak_read" {
				t.Fatalf("NewTool = %q, want %q", tp.NewTool, "fak_read")
			}

			resBytes := refBytes(ctx, tp.NewArgs)
			var gotArgs map[string]any
			if err := json.Unmarshal(resBytes, &gotArgs); err != nil {
				t.Fatalf("unmarshal transformed args: %v", err)
			}

			if gotArgs["file_path"] != tc.wantPath {
				t.Errorf("file_path = %v, want %v", gotArgs["file_path"], tc.wantPath)
			}
			if _, exists := gotArgs["filePath"]; exists {
				t.Errorf("legacy filePath was not stripped from args: %v", gotArgs)
			}
			if _, exists := gotArgs["path"]; exists {
				t.Errorf("legacy path was not stripped from args: %v", gotArgs)
			}
			if tc.wantOffset != nil {
				if gotArgs["offset"] != tc.wantOffset {
					t.Errorf("offset = %v, want %v", gotArgs["offset"], tc.wantOffset)
				}
			}
			if tc.wantLimit != nil {
				if gotArgs["limit"] != tc.wantLimit {
					t.Errorf("limit = %v, want %v", gotArgs["limit"], tc.wantLimit)
				}
			}
		})
	}
}

func TestReadWithoutPathDoesNotTransform(t *testing.T) {
	a := New(Policy{})
	ctx := context.Background()

	// Read without path/filePath/file_path falls through (not transformed).
	call := inlineCall("Read", `{"other":"value"}`)
	v := a.Adjudicate(ctx, call)
	if v.Kind == abi.VerdictTransform {
		t.Fatalf("Read without path should not transform, got VerdictTransform")
	}
}
