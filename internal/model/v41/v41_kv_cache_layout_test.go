package v41

import (
	"bytes"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/v41kv"
	"testing"
)

func TestV41KVNeutralCompatibility(t *testing.T) {
	if ErrV41KVCacheLayout != v41kv.ErrV41KVCacheLayout {
		t.Fatal("error identity changed")
	}
	for _, layout := range V41KVCacheLayouts() {
		canonical, ok := v41kv.V41KVCacheLayoutFor(layout.Format)
		if !ok || canonical != layout {
			t.Fatal("layout alias changed")
		}
		nope, rope := make([]float32, layout.DNoPE), make([]float32, layout.DRoPE)
		got, err := layout.V41KVQuantizeRow(nope, rope)
		want, wantErr := canonical.V41KVQuantizeRow(nope, rope)
		if err != nil || wantErr != nil || !bytes.Equal(got, want) {
			t.Fatal("public wrapper differs")
		}
	}
	if err := (V41KVCacheLayout{}).Validate(); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatal("invalid layout refusal changed")
	}
}
