// Package childenv makes a child process's PWD name the directory it runs in.
//
// exec.Cmd.Dir moves the working directory but not the PWD variable a shell
// ancestor exported; under Git Bash (msys) PWD names the launcher's directory
// and survives into every inheriting descendant, so a tool that trusts PWD
// over its real cwd acts on the wrong directory (docs/AGENT-CONTROL.md).
package childenv

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ForDir returns a copy of env with PWD and OLDPWD dropped (case-insensitive,
// as Windows treats variable names) and PWD naming dir, made absolute; an
// empty dir uses the launcher's working directory. It never modifies env.
// Unlike exec.Cmd.Env, nil does not mean "inherit"; pass os.Environ() or Apply.
func ForDir(env []string, dir string) []string {
	if dir == "" {
		dir, _ = os.Getwd()
	} else if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if DirVar(name) {
			continue
		}
		out = append(out, kv)
	}
	if dir != "" {
		out = append(out, "PWD="+dir)
	}
	return out
}

// DirVar reports whether name is PWD or OLDPWD, compared case-insensitively.
// Shells keep them in step with their own cwd, so a launcher must never
// forward them to a child running elsewhere.
func DirVar(name string) bool {
	return strings.EqualFold(name, "PWD") || strings.EqualFold(name, "OLDPWD")
}

// Apply sets cmd.Env to ForDir(cmd.Env, cmd.Dir); a nil cmd.Env ("inherit")
// becomes a corrected copy of os.Environ(). Call it after cmd.Dir is set; a
// PWD appended afterwards wins over the one Apply set.
func Apply(cmd *exec.Cmd) {
	env := cmd.Env
	if env == nil {
		env = os.Environ()
	}
	cmd.Env = ForDir(env, cmd.Dir)
}
