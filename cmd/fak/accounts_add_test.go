package main

import (
	"path/filepath"
	"testing"
)

func TestAccountDir(t *testing.T) {
	home := filepath.FromSlash("/h")
	cases := []struct {
		name, suffix, want string
	}{
		{"day26", "-netra", filepath.Join(home, ".claude-day26-netra")},
		{"day26-netra", "-netra", filepath.Join(home, ".claude-day26-netra")}, // already suffixed, not doubled
		{"plain", "", filepath.Join(home, ".claude-plain")},                   // no suffix
	}
	for _, c := range cases {
		if got := accountDir(home, c.name, c.suffix); got != c.want {
			t.Errorf("accountDir(%q,%q) = %q, want %q", c.name, c.suffix, got, c.want)
		}
	}
}

func TestExtractToken(t *testing.T) {
	cases := map[string]string{
		"sk-ant-oat01-abc":                          "sk-ant-oat01-abc",
		"Paste this token:\nsk-ant-oat01-xyz\nDone": "sk-ant-oat01-xyz",
		"  sk-ant-oat01-trimmed  ":                  "sk-ant-oat01-trimmed",
		"no token here":                             "no token here",
	}
	for in, want := range cases {
		if got := extractToken(in); got != want {
			t.Errorf("extractToken(%q) = %q, want %q", in, got, want)
		}
	}
}
