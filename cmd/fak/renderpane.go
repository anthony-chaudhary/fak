package main

// Render-pane fault boundary (#2170 acceptance row 2, split to #3139): a TUI
// render surface is a FAULT BOUNDARY. When a pane's paint loop panics or exits
// unexpectedly, the crash must fold to a STRUCTURED child failure — the typed
// toolprocgate.ConsoleRendererExit class, recorded through
// Supervisor.ExitConsoleFault — and the supervisor plus every sibling session
// must stay alive. Before this seam the render surfaces in this package were
// pure functions with no owner: a renderer crash rode the goroutine straight
// up to the process and took the supervisor (and every unrelated agent pane)
// with it, and nothing durable named the crash.
//
// runRenderPane is the embedder contract in executable form: the future
// TUI-as-attachable-client refactor (#2170 discussion) wraps each pane's run
// loop in it; today the row-2 regression witness (renderpane_test.go) proves
// the isolation, and the fault rows it journals are the same JSONL
// `fak toolproc console-faults` folds for the operator (#2170 row 4).

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// runRenderPane executes one render surface's run loop under the console-fault
// boundary. callID must already be Spawned on sup (the pane is a supervised
// child like any tool call).
//
//   - render returns nil: the pane detached cleanly; the supervisor records a
//     normal exit(ok) and no fault is journaled.
//   - render panics or returns an error: the crash is CONTAINED — never
//     re-panicked — and recorded via Supervisor.ExitConsoleFault as the typed
//     ConsoleRendererExit class on the renderer surface, so the supervisor's
//     table stays consistent, enforcement stays scoped to the dead pane, and
//     sibling sessions are untouched. The typed row is appended to faultSink
//     (nil = no durable sink) in the fail-closed console-fault JSONL schema.
//
// faulted reports whether a renderer fault was recorded; err reports a seam
// failure (unknown call, sink write refusal) — never the renderer's own crash,
// which is the payload of the returned event, not an error of the boundary.
func runRenderPane(sup *toolprocgate.Supervisor, callID string, nowMS func() int64, faultSink io.Writer, render func() error) (ev toolprocgate.ConsoleFaultEvent, faulted bool, err error) {
	detail, crashed := renderPaneCrash(render)
	if !crashed {
		if exitErr := sup.Exit(callID, nowMS(), "ok"); exitErr != nil {
			return toolprocgate.ConsoleFaultEvent{}, false, exitErr
		}
		return toolprocgate.ConsoleFaultEvent{}, false, nil
	}
	ev, err = sup.ExitConsoleFault(callID, nowMS(), toolprocgate.ConsoleRendererExit, toolprocgate.ConsoleSurfaceRenderer, detail)
	if err != nil {
		return toolprocgate.ConsoleFaultEvent{}, true, err
	}
	if faultSink != nil {
		if appendErr := toolprocgate.AppendConsoleFaultEvent(faultSink, ev); appendErr != nil {
			return ev, true, appendErr
		}
	}
	return ev, true, nil
}

// renderPaneCrash runs one renderer to completion, converting a panic or an
// error return into bounded crash detail. crashed is false only for a clean
// nil return (an intentional detach). The recover here IS the fault boundary:
// a paint panic stops at this frame instead of unwinding into the supervisor.
func renderPaneCrash(render func() error) (detail string, crashed bool) {
	defer func() {
		if r := recover(); r != nil {
			detail = fmt.Sprintf("renderer panic: %v", r)
			crashed = true
		}
	}()
	if err := render(); err != nil {
		return "renderer exit: " + err.Error(), true
	}
	return "", false
}
