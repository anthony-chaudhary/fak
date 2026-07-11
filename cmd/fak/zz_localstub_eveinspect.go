package main

// LOCAL SCRATCH — NOT FOR COMMIT.
//
// The committed eve.go dispatch (case "inspect") calls runEveInspect, whose real
// definition lives in the *untracked* eve_inspect.go and depends on evebridge.InspectFS
// (#2601), which was never committed — so cmd/fak does not compile at HEAD. This stub
// satisfies eve.go's reference (with the untracked eve_inspect.go / eve_inspect_test.go
// temporarily disabled) purely so the package builds and this session's real work can be
// compile- and test-verified. It is deleted, and the .disabled files restored, at the end
// of the session. It invents no evebridge API.
import "io"

func runEveInspect(stdout, stderr io.Writer, argv []string) int {
	_ = stdout
	_ = stderr
	_ = argv
	return 1
}
