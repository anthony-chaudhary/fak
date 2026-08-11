package main

import (
	"example.com/diamond/internal/b"
	"example.com/diamond/internal/c"
	"testing"
)

func TestDiamond(t *testing.T) {
	if b.Value()+c.Value() != 5 {
		t.Fatal("diamond")
	}
}
