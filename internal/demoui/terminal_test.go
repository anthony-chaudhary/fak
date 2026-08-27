package demoui

import (
	"os"
	"reflect"
	"testing"
)

func TestTerminalPalettePlainForRedirectedOutput(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	f, err := os.CreateTemp(t.TempDir(), "redirected-output")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if got := TerminalPalette(f); got != (Palette{}) {
		t.Fatalf("TerminalPalette(regular file) = %#v, want plain palette", got)
	}
}

func TestTerminalPalettePlainOnStatError(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	f, err := os.CreateTemp(t.TempDir(), "closed-output")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if got := TerminalPalette(f); got != (Palette{}) {
		t.Fatalf("TerminalPalette(closed file) = %#v, want plain palette", got)
	}
}

func TestTerminalPaletteCharacterDeviceAndANSIFields(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Skipf("open character device: %v", err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeCharDevice == 0 {
		t.Skipf("%s is not reported as a character device on this platform", os.DevNull)
	}

	want := Palette{
		Red:    "\033[31m",
		Amber:  "\033[33m",
		Green:  "\033[32m",
		Blue:   "\033[34m",
		Cyan:   "\033[36m",
		Yellow: "\033[33m",
		Dim:    "\033[2m",
		Bold:   "\033[1m",
		Reset:  "\033[0m",
	}
	if got := TerminalPalette(f); !reflect.DeepEqual(got, want) {
		t.Fatalf("TerminalPalette(character device) = %#v, want %#v", got, want)
	}
}

func TestTerminalPaletteHonorsNOColor(t *testing.T) {
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Skipf("open character device: %v", err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeCharDevice == 0 {
		t.Skipf("%s is not reported as a character device on this platform", os.DevNull)
	}
	t.Setenv("NO_COLOR", "1")

	if got := TerminalPalette(f); got != (Palette{}) {
		t.Fatalf("TerminalPalette with NO_COLOR = %#v, want plain palette", got)
	}
}

func TestPalettePaint(t *testing.T) {
	p := Palette{Reset: "<reset>"}
	if got := p.Paint("", "plain"); got != "plain" {
		t.Fatalf("Paint(empty code) = %q, want %q", got, "plain")
	}
	if got := p.Paint("<red>", "colored"); got != "<red>colored<reset>" {
		t.Fatalf("Paint(code) = %q, want exact reset wrapping", got)
	}
}
