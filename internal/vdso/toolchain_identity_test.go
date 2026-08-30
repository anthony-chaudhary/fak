package vdso

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func toolchainDependentCall(tool, args, binaryID, toolchainID string) *abi.ToolCall {
	c := roCall(tool, args)
	c.Meta[MetaToolchainDependent] = "true"
	if binaryID != "" {
		c.Meta[MetaBinaryIdentity] = binaryID
	}
	if toolchainID != "" {
		c.Meta[MetaToolchainIdentity] = toolchainID
	}
	return c
}

func TestToolchainDependentCacheIdentity(t *testing.T) {
	v := New(8)
	const (
		tool = "compile_shader"
		args = `{"source":"kernel"}`
	)

	first := toolchainDependentCall(tool, args, "fak-bin-a", "/opt/compiler-a")
	same := toolchainDependentCall(tool, args, "fak-bin-a", "/opt/compiler-a")
	otherCompiler := toolchainDependentCall(tool, args, "fak-bin-a", "/opt/compiler-b")
	otherBinary := toolchainDependentCall(tool, args, "fak-bin-b", "/opt/compiler-a")

	ordinary := roCall(tool, args)
	wantOrdinary := tool + ":" + rawArgHash([]byte(args)) + ":0"
	if got := v.keyFor(ordinary, []byte(args)); got != wantOrdinary {
		t.Fatalf("ordinary cache key changed: got %q want %q", got, wantOrdinary)
	}

	firstKey := v.keyFor(first, []byte(args))
	if got := v.keyFor(same, []byte(args)); got != firstKey {
		t.Fatalf("same binary/toolchain witness key = %q, want stable %q", got, firstKey)
	}
	if got := v.keyFor(otherCompiler, []byte(args)); got == firstKey {
		t.Fatalf("different compiler witness reused key %q", got)
	}
	if got := v.keyFor(otherBinary, []byte(args)); got == firstKey {
		t.Fatalf("different binary witness reused key %q", got)
	}

	// Length framing must keep arbitrary identity components unambiguous.
	left := toolchainDependentCall(tool, args, "a:b", "c")
	right := toolchainDependentCall(tool, args, "a", "b:c")
	if leftKey, rightKey := v.keyFor(left, []byte(args)), v.keyFor(right, []byte(args)); leftKey == rightKey {
		t.Fatalf("framing collision for distinct identities: %q", leftKey)
	}
}

func TestToolchainDependentCacheFailsClosedWithoutIdentity(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	const (
		tool = "compile_shader"
		args = `{"source":"kernel"}`
	)

	known := toolchainDependentCall(tool, args, "fak-bin-a", "/opt/compiler-a")
	v.Emit(completeEvent(known, "artifact-a"))
	if got, ok := v.Lookup(ctx, known); !ok || string(resolveBytes(t, got.Payload)) != "artifact-a" {
		t.Fatalf("same witnessed call did not reuse its fill: ok=%v result=%v", ok, got)
	}

	changed := toolchainDependentCall(tool, args, "fak-bin-a", "/opt/compiler-b")
	if got, ok := v.Lookup(ctx, changed); ok || got != nil {
		t.Fatalf("changed toolchain shared prior artifact: ok=%v result=%v", ok, got)
	}

	for _, tc := range []struct {
		name        string
		binaryID    string
		toolchainID string
	}{
		{name: "missing binary", toolchainID: "/opt/compiler-a"},
		{name: "missing toolchain", binaryID: "fak-bin-a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			unknown := toolchainDependentCall(tool, args, tc.binaryID, tc.toolchainID)
			if got, ok := v.Lookup(ctx, unknown); ok || got != nil {
				t.Fatalf("unknown identity shared prior artifact: ok=%v result=%v", ok, got)
			}
			_, _, before, _ := v.Stats()
			v.Emit(completeEvent(unknown, "ambiguous-artifact"))
			_, _, after, _ := v.Stats()
			if after != before {
				t.Fatalf("unknown identity was admitted: fills %d -> %d", before, after)
			}
		})
	}
}
