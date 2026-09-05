//go:build linux

package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const (
	sysLandlockCreateRuleset     = 444
	sysLandlockAddRule           = 445
	sysLandlockRestrictSelf      = 446
	landlockCreateRulesetVersion = 1 << 0
	landlockRulePathBeneath      = 1
	oPath                        = 0x200000
)

const (
	fsExecute    = 1 << 0
	fsWriteFile  = 1 << 1
	fsReadFile   = 1 << 2
	fsReadDir    = 1 << 3
	fsRemoveDir  = 1 << 4
	fsRemoveFile = 1 << 5
	fsMakeChar   = 1 << 6
	fsMakeDir    = 1 << 7
	fsMakeReg    = 1 << 8
	fsMakeSock   = 1 << 9
	fsMakeFifo   = 1 << 10
	fsMakeBlock  = 1 << 11
	fsMakeSym    = 1 << 12
	fsRefer      = 1 << 13
	fsTruncate   = 1 << 14
	fsIoctlDev   = 1 << 15
)

const (
	landlockReadClass      = fsReadFile | fsReadDir | fsExecute
	landlockReadWriteClass = landlockReadClass | fsWriteFile | fsRemoveDir | fsRemoveFile |
		fsMakeChar | fsMakeDir | fsMakeReg | fsMakeSock | fsMakeFifo | fsMakeBlock | fsMakeSym | fsTruncate
)

// LandlockRule represents a path-beneath access grant rule.
type LandlockRule struct {
	Path          string
	AllowedAccess uint64
	ReadOnly      bool
}

// CompileLandlockRules maps Spec.LaneTree + scratch space into Landlock read-write rules
// and read-only rules for system binaries (/bin, /usr, /lib, etc.).
func CompileLandlockRules(spec Spec) []LandlockRule {
	var rules []LandlockRule

	// System read-only roots
	systemReadOnly := []string{"/bin", "/usr", "/lib", "/lib64", "/sbin", "/etc", "/dev", "/proc"}
	for _, sysPath := range systemReadOnly {
		if fi, err := os.Stat(sysPath); err == nil && fi.IsDir() {
			rules = append(rules, LandlockRule{
				Path:          sysPath,
				AllowedAccess: landlockReadClass,
				ReadOnly:      true,
			})
		}
	}

	// Writable temp paths
	tempDirs := []string{"/tmp", "/var/tmp"}
	for _, td := range tempDirs {
		if fi, err := os.Stat(td); err == nil && fi.IsDir() {
			rules = append(rules, LandlockRule{
				Path:          td,
				AllowedAccess: landlockReadWriteClass,
				ReadOnly:      false,
			})
		}
	}
	for _, wp := range spec.WritablePaths {
		if fi, err := os.Stat(wp); err == nil && fi.IsDir() {
			rules = append(rules, LandlockRule{
				Path:          wp,
				AllowedAccess: landlockReadWriteClass,
				ReadOnly:      false,
			})
		}
	}

	ws := filepath.Clean(spec.WorkspaceDir)
	if len(spec.LaneTree) > 0 {
		if fi, err := os.Stat(ws); err == nil && fi.IsDir() {
			rules = append(rules, LandlockRule{
				Path:          ws,
				AllowedAccess: landlockReadClass,
				ReadOnly:      true,
			})
		}
		for _, lane := range spec.LaneTree {
			trimmed := strings.TrimSuffix(lane, "/**")
			trimmed = strings.TrimSuffix(trimmed, "/*")
			absLane := filepath.Clean(filepath.Join(ws, trimmed))
			if fi, err := os.Stat(absLane); err == nil && fi.IsDir() {
				rules = append(rules, LandlockRule{
					Path:          absLane,
					AllowedAccess: landlockReadWriteClass,
					ReadOnly:      false,
				})
			}
		}
	} else if ws != "" {
		if fi, err := os.Stat(ws); err == nil && fi.IsDir() {
			rules = append(rules, LandlockRule{
				Path:          ws,
				AllowedAccess: landlockReadWriteClass,
				ReadOnly:      false,
			})
		}
	}

	return rules
}

type linuxConfinement struct {
	spec  Spec
	rules []LandlockRule
}

func newOSConfinement(spec Spec) (osConfinement, error) {
	rules := CompileLandlockRules(spec)
	return &linuxConfinement{
		spec:  spec,
		rules: rules,
	}, nil
}

func (l *linuxConfinement) PrepareCommand(cmd *exec.Cmd, req ExecutionRequest) error {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	cmd.SysProcAttr.Pdeathsig = syscall.SIGKILL
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 2 * time.Second
	return nil
}

func (l *linuxConfinement) OnProcessStart(pid int) error {
	return nil
}

func (l *linuxConfinement) PostProcess(res *ExecutionResult) error {
	return nil
}

func (l *linuxConfinement) Close() error {
	return nil
}
