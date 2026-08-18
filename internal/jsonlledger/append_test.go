package jsonlledger

import (
	"bytes"
	"errors"
	"testing"
)

func TestAppendValidated(t *testing.T) {
	var out bytes.Buffer
	if err := AppendValidated(&out, struct {
		N int `json:"n"`
	}{2}, func(v struct {
		N int `json:"n"`
	}) error {
		return nil
	}); err != nil || out.String() != "{\"n\":2}\n" {
		t.Fatalf("append=%q, %v", out.String(), err)
	}
	want := errors.New("bad")
	if err := AppendValidated(&out, 1, func(int) error { return want }); !errors.Is(err, want) {
		t.Fatalf("validation error=%v", err)
	}
}
