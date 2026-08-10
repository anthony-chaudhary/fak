package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// parseFlagsRejectArgs parses argv into fs and refuses any positional argument —
// these commands are all flags-only front doors, so a stray positional is a typo
// worth failing on, never a silently ignored word.
//
// It returns (exitCode, done) in CALLER order: done=true means "stop now and
// return exitCode" — a parse error, a `-h` usage request, or an unexpected
// positional. done=false means parsing succeeded and the command should proceed.
// The bool was previously the opposite (ok-shaped) while both call sites read it
// as done-shaped, which inverted every front door that used it: valid flags
// returned 0 immediately having done nothing, and a positional argument fell
// through into the command body instead of being rejected. Nothing pinned it, so
// the contract is stated here and pinned by TestParseFlagsRejectArgs.
func parseFlagsRejectArgs(fs *flag.FlagSet, argv []string, stderr io.Writer) (int, bool) {
	if err := fs.Parse(argv); err != nil {
		return 2, true
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected positional arguments: %s\n", fs.Name(), strings.Join(fs.Args(), " "))
		return 2, true
	}
	return 0, false
}

func writeIndentedJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func resolveRoot(explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	if root, err := filepath.Abs("."); err == nil {
		return root
	}
	return "."
}

func intList(nums []int) string {
	if len(nums) == 0 {
		return ""
	}
	cp := append([]int(nil), nums...)
	sort.Ints(cp)
	parts := make([]string, len(cp))
	for i, n := range cp {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ",")
}

var _ = os.ErrNotExist

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return ""
}

func stripJSONFence(text string) string {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "```") {
		return text
	}
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		text = text[i+1:]
	}
	text = strings.TrimSpace(text)
	text = strings.TrimSuffix(text, "```")
	return strings.TrimSpace(text)
}

const issueGHTimeout = 30 * time.Second

func runTaskHandoffGH(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), issueGHTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		fmt.Fprintf(&errb, "\ngh timed out after %s", issueGHTimeout)
		return out.String(), errb.String(), false
	}
	return out.String(), errb.String(), err == nil
}

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	windowgate.ConfigureDetachedCommand(cmd)
}

func runBoundedGHCmd(ctx context.Context, cmd *exec.Cmd, timeout time.Duration) (string, string, bool) {
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		fmt.Fprintf(&errb, "\ngh timed out after %s", timeout)
		return out.String(), errb.String(), false
	}
	return out.String(), errb.String(), err == nil
}

func writeJSON(w io.Writer, v any) int {
	if err := writeIndentedJSON(w, v); err != nil {
		return 2
	}
	return 0
}
