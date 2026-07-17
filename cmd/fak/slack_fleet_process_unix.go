//go:build !windows

package main

import (
	"bufio"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
)

func collectBackgroundProcesses() ([]loopfleet.Process, error) {
	out, err := exec.Command("ps", "-eo", "pid=,ppid=,etimes=,args=").Output()
	if err != nil {
		return nil, fmt.Errorf("process inventory: %w", err)
	}
	var got []loopfleet.Process
	scan := bufio.NewScanner(strings.NewReader(string(out)))
	for scan.Scan() {
		fields := strings.Fields(scan.Text())
		if len(fields) < 4 {
			continue
		}
		pid, e1 := strconv.Atoi(fields[0])
		ppid, e2 := strconv.Atoi(fields[1])
		if e1 != nil || e2 != nil {
			continue
		}
		got = append(got, loopfleet.Process{PID: pid, ParentPID: ppid, Command: strings.Join(fields[3:], " ")})
	}
	return got, scan.Err()
}
