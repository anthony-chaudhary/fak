package agentbench

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

// runChildCLI keeps process-isolation entry points out of the public flag surface.
// The parent task runner invokes these with an immutable config path.
func runChildCLI(ctx context.Context, stdout, stderr io.Writer, args []string) (bool, int) {
	if len(args) == 0 || (args[0] != "task-child" && args[0] != "test-child") {
		return false, 0
	}
	kind := args[0]
	flags := flag.NewFlagSet(kind, flag.ContinueOnError)
	flags.SetOutput(stderr)
	config := flags.String("config", "", "child configuration path")
	if err := flags.Parse(args[1:]); err != nil {
		return true, 2
	}
	if *config == "" || flags.NArg() != 0 {
		fmt.Fprintf(stderr, "agentbench: %s requires exactly --config PATH\n", kind)
		return true, 2
	}
	var err error
	if kind == "task-child" {
		err = taskrun.RunChild(ctx, *config, stdout)
	} else {
		err = taskrun.RunTestChild(ctx, *config, stdout)
	}
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: %s: %v\n", kind, err)
		return true, 1
	}
	return true, 0
}
