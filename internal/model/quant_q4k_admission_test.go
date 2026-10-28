package model

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestLazyQ4KAdmission(t *testing.T) {
	const name = "model.layers.0.mlp.down_proj.weight"
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name     string
		shape    []int
		bytes    int
		reason   string
		expected string
	}{
		// GGUF [128,2] has one complete total block, but neither model row is valid.
		{"unaligned_row", []int{2, 128}, 144, "multiple of 256", "unavailable"},
		{"zero_rows", []int{0, 256}, 0, "positive", "unavailable"},
		{"negative_rows", []int{-1, 256}, -144, "positive", "unavailable"},
		{"zero_columns", []int{1, 0}, 0, "positive", "unavailable"},
		{"negative_columns", []int{1, -256}, -144, "positive", "unavailable"},
		{"short", []int{1, 256}, 143, "mismatch", "144"},
		{"long", []int{1, 256}, 145, "mismatch", "144"},
		{"negative_bytes", []int{1, 256}, -1, "mismatch", "144"},
		// The old multiplication wrapped to -144 and accepted this descriptor.
		{"byte_overflow", []int{maxInt, 256}, -144, "overflows int", "unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewQuantBuilder(Config{HiddenSize: 256}, false)
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("AddLazyQ4K panicked: %v; want typed malformed-payload error", p)
				}
			}()
			err := b.AddLazyQ4K(name, tc.shape, LazyQ4KRange{Bytes: tc.bytes})
			// Keep the same regression compilable on the unfixed parent, where the
			// new exported error type does not exist yet.
			if fmt.Sprintf("%T", err) != "*model.LazyQ4KPayloadError" {
				t.Fatalf("error = %T %v; want *model.LazyQ4KPayloadError", err, err)
			}
			for _, part := range []string{name, "Q4_K", fmt.Sprint(tc.shape), tc.reason,
				"expected " + tc.expected, fmt.Sprintf("actual %d", tc.bytes)} {
				if !strings.Contains(err.Error(), part) {
					t.Errorf("error %q missing %q", err, part)
				}
			}
			if len(b.m.q4kw) != 0 {
				t.Fatal("malformed descriptor was stored")
			}
		})
	}

	t.Run("valid_lazy_and_failed_replacement", func(t *testing.T) {
		b := NewQuantBuilder(Config{HiddenSize: 256}, false)
		payload := bytes.Repeat([]byte{0x5a}, q4kBlockBytes)
		r := &issue9073ReaderAt{data: payload}
		if err := b.AddLazyQ4K(name, []int{1, 256}, LazyQ4KRange{Reader: r, Bytes: len(payload)}); err != nil {
			t.Fatal(err)
		}
		q := b.m.q4kw[name]
		if q == nil || q.lazy == nil || len(q.raw) != 0 || r.reads != 0 {
			t.Fatal("valid descriptor was not retained lazily")
		}
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("malformed replacement panicked: %v", p)
			}
		}()
		if err := b.AddLazyQ4K(name, []int{1, 256}, LazyQ4KRange{Bytes: 143}); err == nil {
			t.Fatal("malformed replacement accepted")
		}
		if b.m.q4kw[name] != q {
			t.Fatal("malformed replacement changed stored descriptor")
		}
		got, err := q.materializeRaw()
		if err != nil || !bytes.Equal(got, payload) || r.reads != 1 {
			t.Fatalf("materialize valid descriptor: error=%v reads=%d equal=%v", err, r.reads, bytes.Equal(got, payload))
		}
	})

	t.Run("existing_admission_order", func(t *testing.T) {
		b := NewQuantBuilder(Config{HiddenSize: 256}, false)
		for _, skip := range []struct {
			name  string
			shape []int
		}{
			{"model.norm.weight", []int{1, 128}},
			{name, []int{256}},
			{"model.layers.0.self_attn.qkv_proj.weight", []int{1, 128}},
		} {
			if err := b.AddLazyQ4K(skip.name, skip.shape, LazyQ4KRange{}); err != nil {
				t.Fatalf("ineligible tensor should still skip: %v", err)
			}
		}
		if len(b.m.q4kw) != 0 {
			t.Fatal("ineligible tensor stored")
		}
		b.built = true
		if err := b.AddLazyQ4K(name, []int{1, 128}, LazyQ4KRange{}); err == nil || !strings.Contains(err.Error(), "already built") {
			t.Fatalf("built guard lost precedence: %v", err)
		}
		v41 := NewQuantBuilder(Config{ModelType: "deepseek_v41"}, false)
		if err := v41.AddLazyQ4K(name, []int{1, 128}, LazyQ4KRange{}); !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("V41 refusal lost precedence: %v", err)
		}
	})
}
