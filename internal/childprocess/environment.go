package childprocess

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// WorkdirEnv returns one absolute working directory and its matching child
// environment. A nil environment inherits the parent; an explicitly empty one
// does not. Without a directory override, both inputs are returned unchanged.
// The caller must use the returned directory and environment together.
func WorkdirEnv(dir string, env []string) (string, []string, error) {
	if dir == "" {
		return dir, env, nil
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", nil, err
	}
	if env == nil {
		env = os.Environ()
	}
	child := make([]string, 0, len(env)+1)
	for _, entry := range env {
		name, _, _ := strings.Cut(entry, "=")
		isDirectory := name == "PWD" || name == "OLDPWD"
		if runtime.GOOS == "windows" {
			isDirectory = strings.EqualFold(name, "PWD") || strings.EqualFold(name, "OLDPWD")
		}
		if !isDirectory {
			child = append(child, entry)
		}
	}
	return abs, append(child, "PWD="+abs), nil
}
