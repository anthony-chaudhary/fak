package main

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

type loopDriveRuntime struct {
	clock    func() time.Time
	sleep    func(time.Duration)
	floor    time.Duration
	goalPath string
	cleanup  func()
}

func defaultGoalSpecPath() string {
	if envGoal := strings.TrimSpace(os.Getenv("FAK_GOAL_SPEC")); envGoal != "" {
		return envGoal
	}
	scratchGoal := filepath.Join("_scratch", "goals", "GOAL.md")
	if _, err := os.Stat(scratchGoal); err == nil {
		return scratchGoal
	}
	return "GOAL.md"
}

func prepareLoopDriveRuntime(opt *loopDriveOptions) (loopDriveRuntime, error) {
	runtime := loopDriveRuntime{
		clock:    opt.Clock,
		sleep:    opt.Sleep,
		floor:    opt.MinIterationFloor,
		goalPath: strings.TrimSpace(opt.GoalPath),
	}
	if runtime.clock == nil {
		runtime.clock = time.Now
	}
	if runtime.sleep == nil {
		runtime.sleep = time.Sleep
	}
	if runtime.floor <= 0 {
		runtime.floor = time.Second
	}
	if runtime.goalPath == "" {
		runtime.goalPath = defaultGoalSpecPath()
	}
	opt.GoalPath = runtime.goalPath
	handoffFile, cleanup, err := loopDriveHandoffFile(opt.HandoffFile)
	if err != nil {
		return loopDriveRuntime{}, err
	}
	opt.HandoffFile = handoffFile
	runtime.cleanup = cleanup
	return runtime, nil
}
