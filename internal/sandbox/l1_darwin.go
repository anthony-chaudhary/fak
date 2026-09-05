//go:build darwin

package sandbox

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type darwinConfinement struct {
	spec    Spec
	profile string
}

func newOSConfinement(spec Spec) (osConfinement, error) {
	profile := CompileSeatbeltProfile(spec)
	return &darwinConfinement{
		spec:    spec,
		profile: profile,
	}, nil
}

// CompileSeatbeltProfile generates a dynamic TinyScheme/Seatbelt profile
// that permits reading and execution but denies writes outside granted LaneTree
// and temporary directories.
func CompileSeatbeltProfile(spec Spec) string {
	var b strings.Builder
	b.WriteString("(version 1)\n")
	b.WriteString("(deny default)\n")
	b.WriteString("(allow process-exec)\n")
	b.WriteString("(allow process-fork)\n")
	b.WriteString("(allow sysctl-read)\n")
	b.WriteString("(allow file-read*)\n")

	b.WriteString("(allow file-write*\n")
	b.WriteString("    (subpath \"/tmp\")\n")
	b.WriteString("    (subpath \"/private/tmp\")\n")
	b.WriteString("    (subpath \"/var/tmp\")\n")
	b.WriteString("    (subpath \"/private/var/tmp\")\n")

	for _, wp := range spec.WritablePaths {
		clean := filepath.Clean(wp)
		if clean != "" {
			b.WriteString(fmt.Sprintf("    (subpath %q)\n", clean))
		}
	}

	ws := filepath.Clean(spec.WorkspaceDir)
	if len(spec.LaneTree) > 0 {
		for _, lane := range spec.LaneTree {
			trimmed := strings.TrimSuffix(lane, "/**")
			trimmed = strings.TrimSuffix(trimmed, "/*")
			absLane := filepath.Clean(filepath.Join(ws, trimmed))
			b.WriteString(fmt.Sprintf("    (subpath %q)\n", absLane))
		}
	} else if ws != "" {
		b.WriteString(fmt.Sprintf("    (subpath %q)\n", ws))
	}

	b.WriteString(")\n")
	return b.String()
}

func (d *darwinConfinement) PrepareCommand(cmd *exec.Cmd, req ExecutionRequest) error {
	if _, err := os.Stat("/usr/bin/sandbox-exec"); err == nil {
		origPath := cmd.Path
		origArgs := cmd.Args
		cmd.Path = "/usr/bin/sandbox-exec"
		cmd.Args = append([]string{"sandbox-exec", "-p", d.profile, origPath}, origArgs[1:]...)
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 2 * time.Second
	return nil
}

func (d *darwinConfinement) OnProcessStart(pid int) error {
	return nil
}

func (d *darwinConfinement) PostProcess(res *ExecutionResult) error {
	return nil
}

func (d *darwinConfinement) Close() error {
	return nil
}
