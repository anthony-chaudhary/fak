//go:build linux

package treedoctor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
)

func listGoTmpProcesses() ([]GoTmpProcess, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("read /proc process snapshot: %w", err)
	}
	uid := uint32(os.Geteuid())
	processes := make([]GoTmpProcess, 0, len(entries))
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || !entry.IsDir() {
			continue
		}
		procDir := filepath.Join("/proc", entry.Name())
		info, err := os.Stat(procDir)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("stat process %d: %w", pid, err)
		}
		stat, ok := info.Sys().(*syscall.Stat_t)
		if !ok || stat.Uid != uid {
			continue
		}
		cmdline, cmdErr := os.ReadFile(filepath.Join(procDir, "cmdline"))
		executable, exeErr := os.Readlink(filepath.Join(procDir, "exe"))
		// A zombie has neither field and a process may vanish between the directory scan
		// and these reads. One readable field is still an exact reference witness; both
		// unreadable while the process remains is ambiguous and fails the whole snapshot.
		if cmdErr != nil && exeErr != nil {
			if _, statErr := os.Stat(procDir); statErr == nil &&
				!errors.Is(cmdErr, os.ErrNotExist) && !errors.Is(exeErr, os.ErrNotExist) {
				return nil, fmt.Errorf("read process %d command/executable references: %v; %v", pid, cmdErr, exeErr)
			}
			continue
		}
		processes = append(processes, GoTmpProcess{
			PID:            pid,
			CommandLine:    strings.TrimSpace(strings.ReplaceAll(string(cmdline), "\x00", " ")),
			ExecutablePath: executable,
		})
	}
	sort.Slice(processes, func(i, j int) bool { return processes[i].PID < processes[j].PID })
	return processes, nil
}

func goTmpIsReparse(_ string, info os.FileInfo) (bool, error) {
	return info.Mode()&os.ModeSymlink != 0, nil
}
