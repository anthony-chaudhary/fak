package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
)

type validateWorkspaceRequest struct {
	stdout, stderr io.Writer
	ctx            context.Context
	result         *validateResult
	recorder       *validateRecorder
	root, tip      string
	mine           pathList
	testRun        string
	wslTests       bool
	asJSON         bool
}

type preparedValidateWorkspace struct {
	paths        []string
	testRun      string
	dir          string
	wslWorkspace bool
	cleanup      func()
}

func prepareValidateWorkspace(req validateWorkspaceRequest) (preparedValidateWorkspace, int, bool) {
	phase := req.recorder.start("normalize_mine")
	paths, err := normalizeMinePathsWithin(req.ctx, req.root, req.mine)
	if code, failed := finishValidateRequiredPhase(req.stdout, req.stderr, req.result, req.recorder, phase, "normalize_mine", err, req.asJSON,
		fmt.Sprintf("fak validate: %v", err)); failed {
		return preparedValidateWorkspace{}, code, true
	}
	req.result.Mine = paths
	req.result.Overlays.Skipped = append([]string(nil), paths...)
	effectiveTestRun := strings.TrimSpace(req.testRun)
	if effectiveTestRun != "" {
		req.result.TestRun = effectiveTestRun
		req.result.TestScope = "explicit"
	} else if inferred, inferErr := ownedTestRunExpression(req.root, paths); inferErr == nil && inferred != "" {
		effectiveTestRun = inferred
		req.result.TestRun = inferred
		req.result.TestScope = "owned-test-files"
	}
	wslWorkspace := runtime.GOOS == "windows" && req.wslTests
	phase = req.recorder.start("extract_tip")
	var dir string
	if wslWorkspace {
		dir, err = extractCommittedTipWSLWithin(req.ctx, req.root, req.tip)
	} else {
		dir, err = extractCommittedTipWithin(req.ctx, req.root, req.tip)
	}
	if req.ctx.Err() != nil || errors.Is(err, context.DeadlineExceeded) {
		detail := context.DeadlineExceeded.Error()
		if err != nil {
			detail = err.Error()
		}
		phase.finishAs("timeout", detail)
		return preparedValidateWorkspace{}, finishValidateTimeout(req.stdout, req.result, req.recorder, "extract_tip", req.asJSON), true
	}
	phase.finish(err)
	if err != nil {
		fmt.Fprintf(req.stderr, "fak validate: cannot materialize tip %s: %v\n", short(req.tip), err)
		return preparedValidateWorkspace{}, 2, true
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	if wslWorkspace {
		cleanup = func() { cleanupValidateWSLDir(dir) }
	}
	return preparedValidateWorkspace{paths: paths, testRun: effectiveTestRun, dir: dir, wslWorkspace: wslWorkspace, cleanup: cleanup}, 0, false
}
