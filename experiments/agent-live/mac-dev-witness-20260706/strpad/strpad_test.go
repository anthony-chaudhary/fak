package strpad

import "testing"

func TestLeftPadNoPadding(t *testing.T) {
	if got := LeftPad("gopher", 3, '.'); got != "gopher" {
		t.Fatalf("LeftPad() = %q, want %q", got, "gopher")
	}
}

func TestLeftPadASCIIPadding(t *testing.T) {
	if got := LeftPad("go", 4, '.'); got != "..go" {
		t.Fatalf("LeftPad() = %q, want %q", got, "..go")
	}
}

func TestLeftPadUnicodeWidth(t *testing.T) {
	if got := LeftPad("猫", 2, '*'); got != "*猫" {
		t.Fatalf("LeftPad() = %q, want %q", got, "*猫")
	}
}
