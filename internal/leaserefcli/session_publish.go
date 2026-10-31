// Package leaserefcli contains the shared, inference-independent session descriptor CLI.
package leaserefcli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessionref"
)

// RunSessionPublish implements both fak leaseref session-publish and the standalone
// fak-leaseref session-publish spelling. The flags, diagnostics and JSON are shared.
// It updates a local session descriptor only: it neither acquires a lease nor
// publishes to a remote, and does not replace ownership or fencing checks.
func RunSessionPublish(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref session-publish", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	session := fs.String("session", "", "session id to publish under refs/fak/locks/session-<id>")
	host := fs.String("host", "", "host/node label (default: os hostname)")
	state := fs.String("state", "RUNNING", "session PCB state")
	ttl := fs.Int64("ttl", 0, "descriptor lifetime in seconds (0 = no expiry)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *session == "" {
		fmt.Fprintln(stderr, "fak leaseref session-publish: --session is required")
		return 2
	}
	if *host == "" {
		if h, err := os.Hostname(); err == nil {
			*host = h
		}
	}
	now := time.Now()
	desc := sessionref.SessionDescriptor{
		ID:        *session,
		Host:      *host,
		PCBState:  strings.TrimSpace(*state),
		UpdatedAt: now.Unix(),
		TTLSecs:   *ttl,
	}
	if desc.PCBState == "" {
		desc.PCBState = "RUNNING"
	}
	store := sessionref.NewInDir(*dir)
	if _, err := store.PublishSession(context.Background(), desc); err != nil {
		fmt.Fprintf(stderr, "fak leaseref session-publish: %v\n", err)
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, desc, "session-publish")
}

func parseFlagsRejectArgs(fs *flag.FlagSet, argv []string, stderr io.Writer) (int, bool) {
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0, true
		}
		return 2, true
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), fs.Arg(0))
		return 2, true
	}
	return 0, false
}

func writeIndentedJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func encodeJSONOrFail(stdout, stderr io.Writer, v any, label string) int {
	return encodeJSONOrFailPrefixed(stdout, stderr, v, label+": encode json")
}

func encodeJSONOrFailPrefixed(stdout, stderr io.Writer, v any, errPrefix string) int {
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", errPrefix, err)
		return 1
	}
	return 0
}

func emitLeaserefJSON(stdout, stderr io.Writer, v any, sub string) int {
	return encodeJSONOrFail(stdout, stderr, v, "fak leaseref "+sub)
}
