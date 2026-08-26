package shelltoken

import "testing"

func TestIsAssign(t *testing.T) {
	tests := []struct {
		token string
		want  bool
	}{
		{"NAME=value", true},
		{"_NAME9=", true},
		{"9NAME=value", false},
		{"BAD-NAME=value", false},
		{"=value", false},
		{"NAME", false},
	}
	for _, tt := range tests {
		if got := IsAssign(tt.token); got != tt.want {
			t.Errorf("IsAssign(%q) = %v, want %v", tt.token, got, tt.want)
		}
	}
}

func TestShortCluster(t *testing.T) {
	tests := []struct {
		token   string
		short   bool
		flag    byte
		present bool
	}{
		{"-am", true, 'a', true},
		{"-am", true, 'm', true},
		{"-a=value", true, 'v', false},
		{"--amend", false, 'a', true},
		{"-", false, 'a', false},
	}
	for _, tt := range tests {
		if got := IsShortCluster(tt.token); got != tt.short {
			t.Errorf("IsShortCluster(%q) = %v, want %v", tt.token, got, tt.short)
		}
		if got := ClusterHas(tt.token, tt.flag); got != tt.present {
			t.Errorf("ClusterHas(%q, %q) = %v, want %v", tt.token, tt.flag, got, tt.present)
		}
	}
}

func TestProgramBasename(t *testing.T) {
	tests := map[string]string{
		"git":                          "git",
		"/usr/bin/GIT":                 "git",
		`C:\Program Files\Git\git.exe`: "git",
		"git-secret":                   "git-secret",
		"tool.EXE.bak":                 "tool.exe.bak",
	}
	for token, want := range tests {
		if got := ProgramBasename(token); got != want {
			t.Errorf("ProgramBasename(%q) = %q, want %q", token, got, want)
		}
	}
}
