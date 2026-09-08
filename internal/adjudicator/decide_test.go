package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestTransparentTransformGrepAndGlob(t *testing.T) {
	a := New(Policy{})
	ctx := context.Background()

	t.Run("grep with regex and filePath transforms to fak_grep", func(t *testing.T) {
		args := map[string]any{"regex": "foo", "filePath": "internal/"}
		b, err := json.Marshal(args)
		if err != nil {
			t.Fatalf("marshal args: %v", err)
		}
		call := inlineCall("grep", string(b))
		v := a.Adjudicate(ctx, call)

		if v.Kind != abi.VerdictTransform {
			t.Fatalf("got verdict kind %v, want VerdictTransform", v.Kind)
		}
		if v.By != "monitor/grep_to_fak_grep" {
			t.Fatalf("got By=%q, want %q", v.By, "monitor/grep_to_fak_grep")
		}
		if v.Meta["reversibility_autorepair"] != "grep_to_fak_grep" {
			t.Fatalf("got Meta[reversibility_autorepair]=%q, want %q", v.Meta["reversibility_autorepair"], "grep_to_fak_grep")
		}

		tp, ok := v.Payload.(abi.TransformPayload)
		if !ok {
			t.Fatalf("payload type = %T, want TransformPayload", v.Payload)
		}
		if tp.NewTool != "fak_grep" {
			t.Fatalf("NewTool = %q, want %q", tp.NewTool, "fak_grep")
		}

		resBytes := refBytes(ctx, tp.NewArgs)
		var gotArgs map[string]any
		if err := json.Unmarshal(resBytes, &gotArgs); err != nil {
			t.Fatalf("unmarshal transformed args: %v", err)
		}
		if gotArgs["pattern"] != "foo" {
			t.Errorf("pattern = %v, want %q", gotArgs["pattern"], "foo")
		}
		if gotArgs["path"] != "internal/" {
			t.Errorf("path = %v, want %q", gotArgs["path"], "internal/")
		}
		if _, exists := gotArgs["regex"]; exists {
			t.Errorf("legacy regex was not stripped: %v", gotArgs)
		}
		if _, exists := gotArgs["filePath"]; exists {
			t.Errorf("legacy filePath was not stripped: %v", gotArgs)
		}
	})

	t.Run("glob with pattern and path transforms to fak_glob", func(t *testing.T) {
		args := map[string]any{"pattern": "*.go", "path": "cmd/"}
		b, err := json.Marshal(args)
		if err != nil {
			t.Fatalf("marshal args: %v", err)
		}
		call := inlineCall("glob", string(b))
		v := a.Adjudicate(ctx, call)

		if v.Kind != abi.VerdictTransform {
			t.Fatalf("got verdict kind %v, want VerdictTransform", v.Kind)
		}
		if v.By != "monitor/glob_to_fak_glob" {
			t.Fatalf("got By=%q, want %q", v.By, "monitor/glob_to_fak_glob")
		}
		if v.Meta["reversibility_autorepair"] != "glob_to_fak_glob" {
			t.Fatalf("got Meta[reversibility_autorepair]=%q, want %q", v.Meta["reversibility_autorepair"], "glob_to_fak_glob")
		}

		tp, ok := v.Payload.(abi.TransformPayload)
		if !ok {
			t.Fatalf("payload type = %T, want TransformPayload", v.Payload)
		}
		if tp.NewTool != "fak_glob" {
			t.Fatalf("NewTool = %q, want %q", tp.NewTool, "fak_glob")
		}

		resBytes := refBytes(ctx, tp.NewArgs)
		var gotArgs map[string]any
		if err := json.Unmarshal(resBytes, &gotArgs); err != nil {
			t.Fatalf("unmarshal transformed args: %v", err)
		}
		if gotArgs["pattern"] != "*.go" {
			t.Errorf("pattern = %v, want %q", gotArgs["pattern"], "*.go")
		}
		if gotArgs["path"] != "cmd/" {
			t.Errorf("path = %v, want %q", gotArgs["path"], "cmd/")
		}
	})

	t.Run("grep aliases and preserves include", func(t *testing.T) {
		args := map[string]any{"query": "bar", "dir": "pkg/", "include": "*.go"}
		b, err := json.Marshal(args)
		if err != nil {
			t.Fatalf("marshal args: %v", err)
		}
		call := inlineCall("rg", string(b))
		v := a.Adjudicate(ctx, call)

		if v.Kind != abi.VerdictTransform {
			t.Fatalf("got verdict kind %v, want VerdictTransform", v.Kind)
		}
		if v.By != "monitor/grep_to_fak_grep" {
			t.Fatalf("got By=%q, want %q", v.By, "monitor/grep_to_fak_grep")
		}

		tp, ok := v.Payload.(abi.TransformPayload)
		if !ok || tp.NewTool != "fak_grep" {
			t.Fatalf("NewTool = %q, want fak_grep", tp.NewTool)
		}

		resBytes := refBytes(ctx, tp.NewArgs)
		var gotArgs map[string]any
		if err := json.Unmarshal(resBytes, &gotArgs); err != nil {
			t.Fatalf("unmarshal args: %v", err)
		}
		if gotArgs["pattern"] != "bar" {
			t.Errorf("pattern = %v, want %q", gotArgs["pattern"], "bar")
		}
		if gotArgs["path"] != "pkg/" {
			t.Errorf("path = %v, want %q", gotArgs["path"], "pkg/")
		}
		if gotArgs["include"] != "*.go" {
			t.Errorf("include = %v, want %q", gotArgs["include"], "*.go")
		}
		if _, exists := gotArgs["query"]; exists {
			t.Errorf("legacy query was not stripped: %v", gotArgs)
		}
		if _, exists := gotArgs["dir"]; exists {
			t.Errorf("legacy dir was not stripped: %v", gotArgs)
		}
	})

	t.Run("find transforms to fak_glob with aliases", func(t *testing.T) {
		args := map[string]any{"glob": "*.md", "directory": "docs/"}
		b, err := json.Marshal(args)
		if err != nil {
			t.Fatalf("marshal args: %v", err)
		}
		call := inlineCall("find", string(b))
		v := a.Adjudicate(ctx, call)

		if v.Kind != abi.VerdictTransform {
			t.Fatalf("got verdict kind %v, want VerdictTransform", v.Kind)
		}
		if v.By != "monitor/glob_to_fak_glob" {
			t.Fatalf("got By=%q, want %q", v.By, "monitor/glob_to_fak_glob")
		}

		tp, ok := v.Payload.(abi.TransformPayload)
		if !ok || tp.NewTool != "fak_glob" {
			t.Fatalf("NewTool = %q, want fak_glob", tp.NewTool)
		}

		resBytes := refBytes(ctx, tp.NewArgs)
		var gotArgs map[string]any
		if err := json.Unmarshal(resBytes, &gotArgs); err != nil {
			t.Fatalf("unmarshal args: %v", err)
		}
		if gotArgs["pattern"] != "*.md" {
			t.Errorf("pattern = %v, want %q", gotArgs["pattern"], "*.md")
		}
		if gotArgs["path"] != "docs/" {
			t.Errorf("path = %v, want %q", gotArgs["path"], "docs/")
		}
		if _, exists := gotArgs["glob"]; exists {
			t.Errorf("legacy glob was not stripped: %v", gotArgs)
		}
		if _, exists := gotArgs["directory"]; exists {
			t.Errorf("legacy directory was not stripped: %v", gotArgs)
		}
	})

	t.Run("DefaultPolicy allows fak_grep and fak_glob", func(t *testing.T) {
		p := DefaultPolicy()
		if !p.Allow["fak_grep"] {
			t.Errorf("DefaultPolicy Allow missing fak_grep")
		}
		if !p.Allow["fak_glob"] {
			t.Errorf("DefaultPolicy Allow missing fak_glob")
		}
	})
}
