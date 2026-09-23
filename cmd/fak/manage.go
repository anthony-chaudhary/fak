package main

import (
	"fmt"
	"os"
)

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
	dispatchManageCommand(
		argv,
		func(args []string) { cmdManageCommand("manage", args) },
		cmdCodex,
		cmdOpencode,
		cmdManageDirect,
	)
}

func dispatchManageCommand(argv []string, managed, codex, opencode, direct func([]string)) {
	if manageOperatorCommand(argv) {
		managed(argv)
		return
	}
	if len(argv) > 0 && argv[0] == "--guard" {
		managed(argv[1:])
		return
	}
	dispatchManageLaunchWithOpencode(argv, codex, opencode, direct)
}

func manageOperatorCommand(argv []string) bool {
	if len(argv) == 0 {
		return true
	}
	switch argv[0] {
	case "allow", "deny", "disable", "policy", "compile", "restart-audit", "sessions", "resume", "--resume", "--help", "-h":
		return true
	default:
		return false
	}
}

func cmdManageDirect(argv []string) {
	if len(argv) > 0 && argv[0] == "--" {
		argv = argv[1:]
	}
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "fak manage: missing agent command (pass --guard to enable kernel adjudication)")
		return
	}
	os.Exit(execOpencodeLaunchChild(os.Stdout, os.Stderr, argv, os.Environ()))
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
