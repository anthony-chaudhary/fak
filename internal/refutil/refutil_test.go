package refutil

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestBytesReturnsInlinePayload(t *testing.T) {
	want := []byte("payload")
	got := Bytes(context.Background(), abi.Ref{Kind: abi.RefInline, Inline: want})
	if string(got) != string(want) {
		t.Fatalf("Bytes(inline) = %q, want %q", got, want)
	}
}

func TestBytesFailsClosedWithoutResolver(t *testing.T) {
	if got := Bytes(context.Background(), abi.Ref{Kind: abi.RefBlob, Handle: 99}); got != nil {
		t.Fatalf("Bytes(unresolved) = %q, want nil", got)
	}
}
