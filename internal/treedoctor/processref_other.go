//go:build !windows && !linux

package treedoctor

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

func listGoTmpProcesses() ([]GoTmpProcess, error) {
	output, err := exec.Command("ps", "-axo", "pid=,command=").Output()
	if err != nil {
		return nil, fmt.Errorf("enumerate Unix process command references: %w", err)
	}
	processes := make([]GoTmpProcess, 0)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		pid, err := strconv.Atoi(fields[0])
		if err != nil {
			continue
		}
		command := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), fields[0]))
		processes = append(processes, GoTmpProcess{PID: pid, CommandLine: command, ExecutablePath: fields[1]})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan Unix process command references: %w", err)
	}
	sort.Slice(processes, func(i, j int) bool { return processes[i].PID < processes[j].PID })
	return processes, nil
}

func goTmpIsReparse(_ string, info os.FileInfo) (bool, error) {
	return info.Mode()&os.ModeSymlink != 0, nil
}
