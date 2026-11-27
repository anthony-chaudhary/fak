package main

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// armGuardToolprocReuse arms content-digest tool-call reuse on a loopback
// guard gateway. Off-loopback listeners and empty workdirs fail-safe unarmed.
func armGuardToolprocReuse(srv *gateway.Server, mode string, loopbackOnly bool, workDir string) (armed bool, reason string) {
	m := strings.ToLower(strings.TrimSpace(mode))
	if m == "reads" {
		switch env := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_TOOLPROC_REUSE"))); env {
		case "off", "reads", "all":
			m = env
		}
	}
	if m == "off" {
		return false, "disabled"
	}
	if !loopbackOnly || strings.TrimSpace(workDir) == "" {
		return false, "non_loopback_or_no_workdir"
	}
	switch m {
	case "", "auto", "reads":
		srv.SetToolprocReuse(toolproc.ArmedConfig{CoalesceQueries: false}, toolproc.FileDigest(workDir))
		return true, "reads"
	case "all":
		srv.SetToolprocReuse(toolproc.ArmedConfig{CoalesceQueries: true}, toolproc.FileDigest(workDir))
		return true, "all"
	default:
		return false, "unknown_mode"
	}
}
