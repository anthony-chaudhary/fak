//go:build darwin

package taskrun

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func platformSandboxAvailable() error {
	info, err := os.Stat("/usr/bin/sandbox-exec")
	if err != nil || !info.Mode().IsRegular() || info.Mode()&0111 == 0 {
		return &SandboxUnavailableError{Reason: "/usr/bin/sandbox-exec is not executable"}
	}
	return nil
}

func platformSandboxName() string { return "darwin-sandbox-exec" }

func platformSandboxCommand(ctx context.Context, cfg sandboxCommandConfig) (*exec.Cmd, error) {
	profile := sandboxProfileForAccess(cfg.Trial, cfg.Oracle, cfg.Cache, cfg.Temp, cfg.GoRoot, cfg.TrialWritable)
	cmd := exec.CommandContext(ctx, "/usr/bin/sandbox-exec", "-p", profile, filepath.Join(cfg.GoRoot, "bin", "go"), "test", "./...", "-count=1")
	cmd.Dir = cfg.Trial
	cmd.Env = []string{
		"GOTOOLCHAIN=local", "GOENV=off", "GOPROXY=off", "GOSUMDB=off",
		"GOCACHE=" + cfg.Cache, "GOTMPDIR=" + cfg.Temp, "TMPDIR=" + cfg.Temp, "HOME=" + cfg.Temp,
		"PATH=" + filepath.Join(cfg.GoRoot, "bin") + ":/usr/bin:/bin",
	}
	return cmd, nil
}

func sandboxProfile(trial, oracle, cache, temp, goRoot string) string {
	return sandboxProfileForAccess(trial, oracle, cache, temp, goRoot, true)
}

func sandboxProfileForAccess(trial, oracle, cache, temp, goRoot string, trialWritable bool) string {
	literal := func(path string) string {
		path = strings.ReplaceAll(path, `\`, `\\`)
		path = strings.ReplaceAll(path, `"`, `\"`)
		return `(subpath "` + path + `")`
	}
	reads := []string{`(literal "/dev/null")`, literal(trial), literal(cache), literal(temp), literal(goRoot), literal("/usr/lib"), literal("/System/Library")}
	writes := []string{literal(cache), literal(temp)}
	if trialWritable {
		writes = append(writes, literal(trial))
	}
	return `(version 1)
(deny default)
(import "system.sb")
(allow process*)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read* ` + strings.Join(reads, " ") + `)
(allow file-write* ` + strings.Join(writes, " ") + `)
(deny file-read* ` + literal(oracle) + `)
(deny file-write* ` + literal(oracle) + `)
(deny network*)`
}
