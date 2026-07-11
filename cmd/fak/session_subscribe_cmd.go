package main

// session_subscribe_cmd.go — `fak session subscribe`, the operator re-attach verb
// (#2767, epic #2753): drain ONE running session's drive-state event stream by its
// stable trace handle, resumable from a cursor across a controller disconnect.
//
//	fak session subscribe <id>              # the whole retained tail + the next cursor
//	fak session subscribe <id> --since N    # resume after the last event seen (re-attach)
//
// The cursor printed on every drain is the feed's global seq; a controller that
// disconnects re-attaches losslessly by passing it back as --since. complete=false
// on the reply means the bounded feed trimmed events the cursor never saw — re-read
// `fak session status <id>` to re-sync. This is the LIVE re-attach, distinct from
// cold resume: the session keeps running throughout; after re-attaching, every
// control verb (`fak session pause|priority|…`) works on the same handle.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// runSessionSubscribe is the testable core of `fak session subscribe`, following
// runSession's exit-code convention (0 ok, 1 transport/HTTP error, 2 usage).
func runSessionSubscribe(stdout, stderr io.Writer, args []string) int {
	if len(args) < 1 || strings.HasPrefix(args[0], "-") {
		fmt.Fprintln(stderr, "usage: fak session subscribe <id> [--since N] [--json] [--addr URL] [--key K]")
		return 2
	}
	id := args[0]

	fs := flag.NewFlagSet("session subscribe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw JSON instead of the human table")
	since := fs.Uint64("since", 0, "resume cursor: the seq of the last event seen (0 = the whole retained tail)")
	if rc, ok := parseFlagsOrHelp(fs, args[1:]); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session subscribe: unexpected argument %q (the id comes first, then flags)\n", fs.Arg(0))
		return 2
	}

	c := &sessionClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}
	sub, err := c.subscribe(id, *since)
	if err != nil {
		fmt.Fprintf(stderr, "fak session subscribe: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(sub); err != nil {
			fmt.Fprintf(stderr, "fak session subscribe: encode: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "%s  events=%d  cursor=%d  complete=%v\n", sub.TraceID, len(sub.Events), sub.Cursor, sub.Complete)
	if !sub.Complete {
		fmt.Fprintln(stdout, "  (gap: events since --since were trimmed from the feed; re-read `fak session status` and resume from the cursor above)")
	}
	for _, ev := range sub.Events {
		line := fmt.Sprintf("  seq=%-6d rev=%-6d run=%s", ev.Seq, ev.Rev, ev.Run)
		if ev.Reason != "" {
			line += "  reason=" + ev.Reason
		}
		fmt.Fprintln(stdout, line)
	}
	fmt.Fprintf(stdout, "re-attach later with: fak session subscribe %s --since %d\n", sub.TraceID, sub.Cursor)
	return 0
}

// subscribe drains one session's revision events after the cursor
// (GET /v1/fak/session/{id}/subscribe?since=N — the #2767 re-attach op).
func (c *sessionClient) subscribe(id string, since uint64) (gateway.SessionSubscribeResponse, error) {
	var sub gateway.SessionSubscribeResponse
	path := "/v1/fak/session/" + url.PathEscape(id) + "/subscribe"
	if since > 0 {
		path += "?since=" + strconv.FormatUint(since, 10)
	}
	err := c.req(http.MethodGet, path, nil, &sub)
	return sub, err
}
