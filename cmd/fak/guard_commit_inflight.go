package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

type guardCommitProbeResult struct {
	InFlight bool
	Source   string
	Details  string
}

var isGuardCommitInFlight = probeGuardCommitInFlight
var guardProcCollector = procguard.CollectRelations

func probeGuardCommitInFlight(childPID int, repoRoot string) (bool, string) {
	res := probeGuardCommitInFlightDetailed(childPID, repoRoot)
	return res.InFlight, res.Details
}

func probeGuardCommitInFlightDetailed(childPID int, repoRoot string) guardCommitProbeResult {
	if gitDir := resolveGitDir(repoRoot); gitDir != "" {
		lockPath := filepath.Join(gitDir, "index.lock")
		if fi, err := os.Stat(lockPath); err == nil && !fi.IsDir() {
			age := time.Since(fi.ModTime())
			if age < 5*time.Minute {
				return guardCommitProbeResult{
					InFlight: true,
					Source:   "index_lock",
					Details:  fmt.Sprintf(".git/index.lock exists (age=%s)", age.Round(time.Second)),
				}
			}
		}
	}
	if p, found := findActiveCommitProcessInTree(childPID); found {
		return guardCommitProbeResult{
			InFlight: true,
			Source:   "process_tree",
			Details:  fmt.Sprintf("active commit process %s (PID %d)", p.Name, p.PID),
		}
	}
	return guardCommitProbeResult{
		InFlight: false,
	}
}

func resolveGitDir(root string) string {
	if strings.TrimSpace(root) == "" {
		return ""
	}
	gitPath := filepath.Join(root, ".git")
	fi, err := os.Stat(gitPath)
	if err != nil {
		return ""
	}
	if fi.IsDir() {
		return gitPath
	}
	data, err := os.ReadFile(gitPath)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "gitdir:") {
			target := strings.TrimSpace(strings.TrimPrefix(line, "gitdir:"))
			if !filepath.IsAbs(target) {
				target = filepath.Join(root, target)
			}
			return filepath.Clean(target)
		}
	}
	return ""
}

func findActiveCommitProcessInTree(rootPID int) (procguard.Proc, bool) {
	if rootPID <= 0 {
		return procguard.Proc{}, false
	}
	procs, errStr := guardProcCollector()
	if errStr != "" || len(procs) == 0 {
		return procguard.Proc{}, false
	}
	byPID := make(map[int]procguard.Proc, len(procs))
	children := make(map[int][]int)
	for _, p := range procs {
		byPID[p.PID] = p
		if p.PPID != nil {
			children[*p.PPID] = append(children[*p.PPID], p.PID)
		}
	}

	if rootProc, ok := byPID[rootPID]; ok {
		if isCommitProcess(rootProc.Name, rootProc.Cmdline) {
			return rootProc, true
		}
	}

	visited := make(map[int]bool)
	visited[rootPID] = true
	queue := append([]int(nil), children[rootPID]...)
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if visited[curr] {
			continue
		}
		visited[curr] = true
		if p, ok := byPID[curr]; ok {
			if isCommitProcess(p.Name, p.Cmdline) {
				return p, true
			}
		}
		for _, childPID := range children[curr] {
			if !visited[childPID] {
				queue = append(queue, childPID)
			}
		}
	}
	return procguard.Proc{}, false
}

func isCommitProcess(name, cmd string) bool {
	cleanName := strings.ReplaceAll(name, "/", string(filepath.Separator))
	baseName := strings.ToLower(filepath.Base(cleanName))
	baseName = strings.TrimSuffix(baseName, ".exe")

	args := strings.Fields(strings.ToLower(cmd))
	if baseName != "git" && baseName != "fak" && len(args) > 0 {
		firstToken := strings.ToLower(filepath.Base(strings.ReplaceAll(args[0], "/", string(filepath.Separator))))
		firstToken = strings.TrimSuffix(firstToken, ".exe")
		if firstToken == "git" || firstToken == "fak" {
			baseName = firstToken
		}
	}

	switch baseName {
	case "git":
		for _, arg := range args {
			switch arg {
			case "commit", "add", "merge", "rebase", "update-ref":
				return true
			}
		}
	case "fak":
		for _, arg := range args {
			switch arg {
			case "commit", "sweep", "sync":
				return true
			}
		}
	}
	return false
}

type guardCommitGraceOutcome int

const (
	guardCommitGraceCompleted guardCommitGraceOutcome = iota
	guardCommitGraceExpired
	guardCommitGraceChildExited
)

func pollGuardCommitGrace(
	childPID int,
	repoRoot string,
	gracePeriod time.Duration,
	pollInterval time.Duration,
	wait <-chan error,
) (guardCommitGraceOutcome, error) {
	if pollInterval <= 0 {
		pollInterval = 250 * time.Millisecond
	}
	graceDeadline := time.Now().Add(gracePeriod)
	pollTicker := time.NewTicker(pollInterval)
	defer pollTicker.Stop()

	for {
		select {
		case err := <-wait:
			return guardCommitGraceChildExited, err
		case now := <-pollTicker.C:
			inFlight, _ := isGuardCommitInFlight(childPID, repoRoot)
			if !inFlight {
				return guardCommitGraceCompleted, nil
			}
			if !now.Before(graceDeadline) {
				return guardCommitGraceExpired, nil
			}
		}
	}
}
