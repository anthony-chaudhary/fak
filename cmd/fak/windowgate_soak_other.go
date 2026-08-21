//go:build !windows

package main

import "time"

func snapshotDesktopConsoleSoakProcesses() (desktopConsoleSoakProcessCounts, error) {
	return desktopConsoleSoakProcessCounts{Families: map[string]int{}}, nil
}

func waitDesktopConsoleSoakProcessBaseline(before desktopConsoleSoakProcessCounts, _ time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error) {
	return before, nil, nil
}

func waitDesktopConsoleSoakSurvivors([]int, time.Duration) []int { return nil }
