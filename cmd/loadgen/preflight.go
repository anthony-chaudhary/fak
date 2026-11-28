// Adapted from ai-dynamo/aiperf, src/aiperf/cli_runner/_preflight.py:31 at
// ab610add4e8876fca223197596c728e17365f381. Apache-2.0; re-expressed as a Go
// preflight gate with injectable probes so the checks stay hermetic in tests.

package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"
)

// PreflightStatus is the closed verdict set of one preflight check.
type PreflightStatus string

const (
	PreflightPass PreflightStatus = "pass"
	PreflightWarn PreflightStatus = "warn"
	PreflightFail PreflightStatus = "fail"
)

// PreflightCheck is one named check outcome with an operator hint.
type PreflightCheck struct {
	Name   string
	Status PreflightStatus
	Hint   string
}

// PreflightConfig declares what the run needs before bootstrap. NoFile and
// Dial are probes; nil NoFile skips the descriptor check, nil Dial uses net.
type PreflightConfig struct {
	ArtifactDir string
	Endpoint    string // host:port, empty skips reachability
	WantNoFile  uint64 // descriptors the run needs, 0 skips
	NoFile      func() (soft, hard uint64, err error)
	Dial        func(addr string, timeout time.Duration) error
	DialTimeout time.Duration
}

// RunPreflight runs every check, containing each failure so one bad probe
// cannot hide the others, and reports whether the run may proceed.
func RunPreflight(cfg PreflightConfig) ([]PreflightCheck, bool) {
	checks := []PreflightCheck{checkArtifactDir(cfg.ArtifactDir)}
	if cfg.WantNoFile > 0 && cfg.NoFile != nil {
		checks = append(checks, checkNoFile(cfg.WantNoFile, cfg.NoFile))
	}
	if cfg.Endpoint != "" {
		checks = append(checks, checkEndpoint(cfg))
	}
	ok := true
	for _, c := range checks {
		if c.Status == PreflightFail {
			ok = false
		}
	}
	return checks, ok
}

func checkArtifactDir(dir string) PreflightCheck {
	c := PreflightCheck{Name: "artifact_dir"}
	if dir == "" {
		c.Status, c.Hint = PreflightFail, "artifact dir is empty"
		return c
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		c.Status, c.Hint = PreflightFail, fmt.Sprintf("cannot create %s: %v", dir, err)
		return c
	}
	probe, err := os.CreateTemp(dir, ".preflight-*")
	if err != nil {
		c.Status, c.Hint = PreflightFail, fmt.Sprintf("%s is not writable: %v", dir, err)
		return c
	}
	name := probe.Name()
	probe.Close()
	os.Remove(name)
	c.Status, c.Hint = PreflightPass, filepath.Clean(dir)+" is writable"
	return c
}

func checkNoFile(want uint64, probe func() (uint64, uint64, error)) PreflightCheck {
	c := PreflightCheck{Name: "nofile_limit"}
	soft, hard, err := probe()
	switch {
	case err != nil:
		c.Status, c.Hint = PreflightWarn, fmt.Sprintf("descriptor limit unreadable: %v", err)
	case soft >= want:
		c.Status, c.Hint = PreflightPass, fmt.Sprintf("soft limit %d covers %d", soft, want)
	case hard >= want:
		c.Status, c.Hint = PreflightWarn, fmt.Sprintf("soft limit %d below %d; raise to %d (hard limit allows it)", soft, want, want)
	default:
		c.Status, c.Hint = PreflightFail, fmt.Sprintf("hard limit %d below needed %d", hard, want)
	}
	return c
}

func checkEndpoint(cfg PreflightConfig) PreflightCheck {
	c := PreflightCheck{Name: "endpoint"}
	timeout := cfg.DialTimeout
	if timeout <= 0 {
		timeout = 3 * time.Second
	}
	dial := cfg.Dial
	if dial == nil {
		dial = func(addr string, d time.Duration) error {
			conn, err := net.DialTimeout("tcp", addr, d)
			if err == nil {
				conn.Close()
			}
			return err
		}
	}
	if err := dial(cfg.Endpoint, timeout); err != nil {
		c.Status, c.Hint = PreflightFail, fmt.Sprintf("%s unreachable: %v", cfg.Endpoint, err)
		return c
	}
	c.Status, c.Hint = PreflightPass, cfg.Endpoint+" reachable"
	return c
}
