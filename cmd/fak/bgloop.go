package main

// bgloop.go — `fak bgloop`, the surface for the IN-KERNEL background-loop runtime
// (internal/bgloop): the loops `fak serve` keeps progressing while it is up.
//
//	fak bgloop status        one aligned row per supervised loop on a live fak serve
//	fak bgloop status --json the raw GET /v1/fak/loops body, machine-readable
//	fak bgloop status --watch refresh every --interval until interrupted
//	fak bgloop demo          OFFLINE witness — run a heartbeat + a panicking loop and
//	                         show progress AND panic-containment (no server, no GPU)
//
// `status` is the read-only twin of `fak ps`: it folds GET /v1/fak/loops (the loop
// runtime snapshot the gateway serves) into a table, issuing no control verb. `demo`
// needs no server at all — it spins a supervisor in-process so the runtime's two
// load-bearing properties (a loop keeps ticking; a panicking loop is contained and
// the process stays up) are witnessable in ~1s with no dependencies.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func cmdBgloop(argv []string) { os.Exit(runBgloop(os.Stdout, os.Stderr, argv)) }

// runBgloop is the testable core: it dispatches the subcommand and returns the
// process exit code (0 ok, 1 a transport/runtime error, 2 a usage error).
func runBgloop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runBgloopStatus(stdout, stderr, nil)
	}
	switch argv[0] {
	case "status":
		return runBgloopStatus(stdout, stderr, argv[1:])
	case "demo":
		return runBgloopDemo(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		bgloopUsage(stdout)
		return 0
	default:
		// `fak bgloop --addr ...` (a leading flag) means status with flags.
		if strings.HasPrefix(argv[0], "-") {
			return runBgloopStatus(stdout, stderr, argv)
		}
		fmt.Fprintf(stderr, "fak bgloop: unknown subcommand %q\n", argv[0])
		bgloopUsage(stderr)
		return 2
	}
}

// runBgloopStatus fetches GET /v1/fak/loops from a live gateway and renders it.
func runBgloopStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bgloop status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { bgloopUsage(stderr) }
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw GET /v1/fak/loops JSON instead of the human table")
	watch := fs.Bool("watch", false, "refresh continuously until interrupted")
	interval := fs.Duration("interval", 2*time.Second, "watch refresh cadence")
	frames := fs.Int("frames", 0, "watch: stop after N frames (0 = until interrupted)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak bgloop status: unexpected argument %q (status takes only flags)\n", fs.Arg(0))
		return 2
	}

	c := &bgloopClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}
	if !*watch {
		resp, err := c.list()
		if err != nil {
			fmt.Fprintf(stderr, "fak bgloop: %v\n", err)
			return 1
		}
		if *asJSON {
			return emitBgloopJSON(stdout, stderr, resp)
		}
		renderBgloopTable(stdout, resp)
		return 0
	}

	iv := *interval
	if iv <= 0 {
		iv = 2 * time.Second
	}
	for i := 0; *frames <= 0 || i < *frames; i++ {
		fmt.Fprint(stdout, "\033[H\033[2J")
		fmt.Fprintf(stdout, "fak bgloop — in-kernel loops @ %s (every %s; Ctrl-C to stop)\n\n", c.base, iv)
		resp, err := c.list()
		switch {
		case err != nil:
			fmt.Fprintf(stderr, "fak bgloop: %v\n", err)
		case *asJSON:
			emitBgloopJSON(stdout, stderr, resp)
		default:
			renderBgloopTable(stdout, resp)
		}
		if *frames > 0 && i == *frames-1 {
			break
		}
		time.Sleep(iv)
	}
	return 0
}

// runBgloopDemo runs the in-kernel supervisor in-process — no server, no GPU — so the
// runtime's two load-bearing properties are witnessable in ~1s: a heartbeat loop keeps
// ticking, and a loop that panics every tick is contained (counted + restarted) while
// the process stays up and shuts down cleanly.
func runBgloopDemo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bgloop demo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { bgloopUsage(stderr) }
	dur := fs.Duration("duration", 600*time.Millisecond, "how long to run the demo loops")
	tickIv := fs.Duration("interval", 50*time.Millisecond, "loop tick interval")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	var beats atomic.Int64
	sup := bgloop.New(bgloop.WithBackoff(10*time.Millisecond, 80*time.Millisecond))
	_ = sup.Register(bgloop.Loop{Name: "heartbeat", Interval: *tickIv, Tick: func(context.Context) error {
		beats.Add(1)
		return nil
	}})
	_ = sup.Register(bgloop.Loop{Name: "flaky", Interval: *tickIv, Tick: func(context.Context) error {
		panic("simulated loop crash")
	}})

	fmt.Fprintf(stdout, "fak bgloop demo — 2 in-kernel loops for %s (no server, no GPU, no key)\n\n", *dur)
	ctx, cancel := context.WithCancel(context.Background())
	sup.Start(ctx)
	time.Sleep(*dur)
	renderBgloopTable(stdout, gateway.BgloopsResponse{Loops: sup.Snapshot()})

	cancel()
	shctx, shcancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer shcancel()
	shErr := sup.Shutdown(shctx)

	hb, _ := sup.Get("heartbeat")
	fl, _ := sup.Get("flaky")
	fmt.Fprintln(stdout)
	if hb.Ticks > 0 && fl.Panics > 0 && shErr == nil {
		fmt.Fprintf(stdout, "WITNESS: heartbeat kept progressing (%d ticks) while the panicking loop was contained "+
			"(%d panics, %d restarts) — the kernel stayed up and shut down cleanly.\n", hb.Ticks, fl.Panics, fl.Restarts)
		return 0
	}
	fmt.Fprintf(stderr, "demo did not witness expected behavior: heartbeat ticks=%d flaky panics=%d shutdownErr=%v\n",
		hb.Ticks, fl.Panics, shErr)
	return 1
}

// bgloopClient is the read-only HTTP client for GET /v1/fak/loops.
type bgloopClient struct {
	base string
	key  string
	hc   *http.Client
}

func (c *bgloopClient) list() (gateway.BgloopsResponse, error) {
	var out gateway.BgloopsResponse
	req, err := http.NewRequest(http.MethodGet, c.base+"/v1/fak/loops", nil)
	if err != nil {
		return out, err
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("GET /v1/fak/loops: %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return out, fmt.Errorf("decode /v1/fak/loops: %w", err)
	}
	return out, nil
}

func renderBgloopTable(w io.Writer, resp gateway.BgloopsResponse) {
	if len(resp.Loops) == 0 {
		fmt.Fprintln(w, "no background loops registered")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tSTATE\tINTERVAL\tTICKS\tERRORS\tPANICS\tRESTARTS\tLAST_TICK\tLAST_ERR")
	for _, st := range resp.Loops {
		last := "-"
		if !st.LastTickAt.IsZero() {
			last = st.LastTickAt.Format("15:04:05")
		}
		lastErr := st.LastErr
		if lastErr == "" {
			lastErr = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\n",
			st.Name, st.State, st.Interval, st.Ticks, st.Errors, st.Panics, st.Restarts, last, lastErr)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "%d loop(s)\n", len(resp.Loops))
}

func emitBgloopJSON(stdout, stderr io.Writer, resp gateway.BgloopsResponse) int {
	b, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak bgloop: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}

func bgloopUsage(w io.Writer) {
	fmt.Fprint(w, `fak bgloop — the in-kernel background-loop runtime (the loops fak serve runs while it is up)

  fak bgloop status            one row per supervised loop on a live fak serve (GET /v1/fak/loops)
  fak bgloop status --json     the raw /v1/fak/loops body, machine-readable
  fak bgloop status --watch    refresh every --interval until interrupted
  fak bgloop demo              OFFLINE witness: a heartbeat + a panicking loop, showing progress + containment

status flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)  --watch  --interval D  --frames N  --json
demo flags:   --duration D (default 600ms)  --interval D (default 50ms)

read-only: fak bgloop status issues no control verb; it only renders the runtime snapshot.
`)
}
