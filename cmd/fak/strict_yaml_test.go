package main

import (
	"errors"
	"testing"
)

func TestStrictYAML(t *testing.T) {
	ok := "model: qwen\nserver:\n  port: 8000\n  host: a\nclient:\n  port: 9000\n"
	if err := strictYAMLCheck(ok); err != nil {
		t.Fatalf("same key under different parents is legal: %v", err)
	}
	dup := "model: qwen\nserver:\n  port: 8000\n  port: 8001\n"
	err := strictYAMLCheck(dup)
	var d strictYAMLDuplicate
	if !errors.As(err, &d) {
		t.Fatalf("err = %v, want strictYAMLDuplicate", err)
	}
	if d.Key != "port" || d.Line != 4 || d.Col != 3 || d.FirstLine != 3 {
		t.Fatalf("duplicate = %+v, want key port at line 4 col 3 (first line 3)", d)
	}
	top := "a: 1\nb: 2\na: 3\n"
	if err := strictYAMLCheck(top); err == nil {
		t.Fatal("top-level duplicate must be rejected")
	}
}
