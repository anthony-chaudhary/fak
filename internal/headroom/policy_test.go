package headroom

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestWorthCompressing pins the "when to compress" floor against the documented
// defaults (min 48 bytes; worth it iff saved >= 256 bytes OR >= 15%).
func TestWorthCompressing(t *testing.T) {
	cases := []struct {
		orig, neu int
		want      bool
		why       string
	}{
		{10, 5, false, "below the min-bytes floor"},
		{1000, 1000, false, "no saving"},
		{1000, 1100, false, "expanded (defensive)"},
		{100000, 90000, true, "big absolute win (10000 bytes) even at 10%"},
		{100, 60, true, "small but 40% ratio"},
		{1000, 970, false, "3% / 30 bytes — marginal"},
		{2000, 1745, false, "255 bytes (<256) and 12.75% (<15%)"},
		{2000, 1740, true, "260 bytes clears the absolute floor"},
	}
	for _, c := range cases {
		if got := worthCompressing(c.orig, c.neu); got != c.want {
			t.Errorf("worthCompressing(%d,%d)=%v, want %v (%s)", c.orig, c.neu, got, c.want, c.why)
		}
	}
}

// TestWorthCompressingTunable: the floor is operator-tunable (env-backed vars) —
// lowering the ratio floor admits a saving that was previously not worth taking.
func TestWorthCompressingTunable(t *testing.T) {
	old := minSavedRatio
	t.Cleanup(func() { minSavedRatio = old })
	minSavedRatio = 0.01
	if !worthCompressing(1000, 970) { // 3% now clears a 1% floor
		t.Fatal("lowering the ratio floor should admit a 3% saving")
	}
}

// TestGateLeavesMarginalSavingRaw is the load-bearing "knowing when NOT to" test:
// the native compressor CAN shrink the body (trailing-whitespace trim), but the
// GATE leaves it raw because the saving does not clear the worth-it floor — the
// model gets the verbatim bytes and no preserve-write is spent.
func TestGateLeavesMarginalSavingRaw(t *testing.T) {
	withSelected(t, NativeName)
	var sb strings.Builder
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&sb, "data row %02d with a trailing space   \n", i) // unique lines, 3 trailing spaces
	}
	body := []byte(sb.String())

	// Precondition: the compressor finds a (small) saving, but it is below the floor.
	out, _ := nativeCompressor{}.Compress(context.Background(), Input{Bytes: body})
	if !out.Compressed {
		t.Fatal("precondition: native should find a small trim saving")
	}
	if strings.Contains(out.Codec, "dedup") || strings.Contains(out.Codec, "fold") {
		t.Fatalf("precondition: lines must be unique so only trim fires, got codec %q", out.Codec)
	}
	if worthCompressing(out.OrigLen, out.NewLen) {
		t.Fatalf("precondition: the trim saving should be below the worth-it floor (orig=%d new=%d)", out.OrigLen, out.NewLen)
	}

	// The gate leaves it raw.
	r := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}}
	v := NewGate().Admit(context.Background(), &abi.ToolCall{Tool: "Bash"}, r)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("a not-worth-it saving must be admitted raw (VerdictAllow), got %v", v.Kind)
	}
}

// TestGateTakesWorthItSaving is the contrast: a saving that clears the floor still
// transforms (a colorized log shrunk well past 15%).
func TestGateTakesWorthItSaving(t *testing.T) {
	withSelected(t, NativeName)
	const e = "\x1b"
	var sb strings.Builder
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&sb, e+"[32mok"+e+"[0m   github.com/x/pkg%02d\t0.0%ds\n", i, i%9)
	}
	body := []byte(sb.String())
	r := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}}
	v := NewGate().Admit(context.Background(), &abi.ToolCall{Tool: "Bash"}, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("a worth-it saving must Transform, got %v", v.Kind)
	}
}
