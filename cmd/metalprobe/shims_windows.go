//go:build windows

// Windows host lookups for the metal-check gate. A windows/amd64 host can never
// reach the darwin/arm64 critique path — Run dispatches to the not-applicable exit
// before any of these would be consulted — so each lookup returns a statically false
// ToolCheck describing itself, keeping the package self-contained and honest if a
// future darwin-only caller ever re-uses it.
package main

// checkGoTool on windows: never consulted (not-applicable exit precedes it).
func checkGoTool() ToolCheck {
	return ToolCheck{OK: false, Extra: "unavailable on windows host (gate is not applicable here)"}
}

// checkXcodeCLT on windows: never consulted (not-applicable exit precedes it).
func checkXcodeCLT() ToolCheck {
	return ToolCheck{OK: false, Extra: "unavailable on windows host (gate is not applicable here)"}
}

// checkClang on windows: never consulted (not-applicable exit precedes it).
func checkClang() ToolCheck {
	return ToolCheck{OK: false, Extra: "unavailable on windows host (gate is not applicable here)"}
}

// runLinkcheck on windows: never invoked (the leaf is darwin-constrained).
func runLinkcheck() (string, int) {
	return "", 1
}
