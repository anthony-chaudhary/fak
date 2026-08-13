package main

// cmdManage is the primary agent-management surface. It intentionally delegates
// to the mature guard implementation while that implementation is renamed in
// place, so manage and its short alias m inherit every launch flag and operator
// subcommand. The legacy `fak guard` entry point remains a compatibility alias.
func cmdManage(argv []string) {
	if len(argv) > 0 && argv[0] == "parity" {
		cmdLaunchParityCheck(argv[1:])
		return
	}
	cmdManageCommand("manage", argv)
}
