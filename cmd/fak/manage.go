package main

import "os"

// cmdManage is the primary agent-management surface. It intentionally delegates
// to the mature guard implementation while that implementation is renamed in
// place, so manage and its short alias m inherit every launch flag and operator
// subcommand. The legacy `fak guard` entry point remains a compatibility alias.
func cmdManage(argv []string) {
	if len(argv) > 0 && argv[0] == "hook" {
		cmdManageNativeHook(argv[1:], os.Stdin, os.Stdout)
		return
	}
	if len(argv) > 0 && argv[0] == "parity" {
		cmdLaunchParityCheck(argv[1:])
		return
	}
	dispatchManageLaunch(argv, cmdCodex, func(args []string) { cmdManageCommand("manage", args) })
}

// dispatchManageLaunch keeps the convenient bare managed-Codex spelling on the
// dedicated launcher, where freshness admission and Codex-specific setup are
// already proven. Any flags, delimiter, or child arguments retain the generic
// manage contract rather than being reinterpreted here.
func dispatchManageLaunch(argv []string, codex func([]string), generic func([]string)) {
	dispatchManageLaunchWithOpencode(argv, codex, cmdOpencode, generic)
}

func dispatchManageLaunchWithOpencode(argv []string, codex, opencode func([]string), generic func([]string)) {
	if len(argv) == 1 && argv[0] == "codex" {
		codex(nil)
		return
	}
	if len(argv) == 1 && argv[0] == "opencode" {
		opencode(nil)
		return
	}
	generic(argv)
}
