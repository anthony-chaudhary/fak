package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

func cmdArms(argv []string) {
	os.Exit(runArms(os.Stdout, os.Stderr, argv))
}

func armsUsage(w io.Writer) {
	fmt.Fprintln(w, `Usage: fak arms <subcommand> [flags]

Tooling for research project arms running on the same machine: origin attribution,
request inspection, concurrency limits, and lease-based admission control.

Subcommands:
  status, list   Display all recognized research project arms, limits, and leases (default)
  who, traffic   Inspect current in-flight requests and caller processes on the machine
  lease          Manage server leases (acquire, release, list)
  limit          Set concurrency limits for a specific research arm

Flags (common):
  --addr string       Gateway server base URL (default "http://127.0.0.1:8080" or $FAK_ADDR)
  --key string        Bearer credential (default $FAK_KEY)
  --json              Emit machine-readable JSON output
  --watch             Continuously refresh output (top mode)
  --interval duration Refresh interval in watch mode (default 2s)

Examples:
  fak arms status                             # view active arms, limits, and request volume
  fak arms who                                # see who is currently hitting the server (PIDs, arms)
  fak arms lease acquire --arm eval-sweep --concurrency 2 --ttl 10m
  fak arms lease acquire --arm perf-test --exclusive
  fak arms lease release --arm eval-sweep
  fak arms lease list
  fak arms limit --arm codex-agent --max-concurrency 4`)
}

func runArms(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runArmsStatus(stdout, stderr, argv)
	}

	sub := argv[0]
	if strings.HasPrefix(sub, "-") {
		return runArmsStatus(stdout, stderr, argv)
	}

	switch sub {
	case "status", "list":
		return runArmsStatus(stdout, stderr, argv[1:])
	case "who", "traffic":
		return runArmsTraffic(stdout, stderr, argv[1:])
	case "lease":
		return runArmsLease(stdout, stderr, argv[1:])
	case "limit":
		return runArmsLimit(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		armsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak arms: unknown subcommand %q (run 'fak arms help')\n", sub)
		return 2
	}
}

type armsClient struct {
	base string
	key  string
	hc   *http.Client
}

func newArmsClient(base, key string) *armsClient {
	if base == "" {
		base = defaultSessionAddr()
	}
	base = strings.TrimRight(base, "/")
	return &armsClient{
		base: base,
		key:  key,
		hc:   &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *armsClient) get(path string, out any) error {
	req, err := http.NewRequest("GET", c.base+path, nil)
	if err != nil {
		return err
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server error (%d): %s", resp.StatusCode, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *armsClient) post(path string, body any, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequest("POST", c.base+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server error (%d): %s", resp.StatusCode, string(b))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *armsClient) delete(path string, out any) error {
	req, err := http.NewRequest("DELETE", c.base+path, nil)
	if err != nil {
		return err
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server error (%d): %s", resp.StatusCode, string(b))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func runArmsStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	asJSON := fs.Bool("json", false, "emit JSON output")
	watch := fs.Bool("watch", false, "refresh continuously")
	interval := fs.Duration("interval", 2*time.Second, "refresh interval")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	client := newArmsClient(*addr, *key)
	for {
		var snap researcharm.Snapshot
		err := client.get("/v1/fak/arms", &snap)
		if err != nil && strings.Contains(err.Error(), "404") {
			// Fallback: server is running older build; synthesize arms from /v1/fak/sessions
			var fallback *researcharm.Snapshot
			fallback, err = fallbackArmsStatus(client)
			if fallback != nil {
				snap = *fallback
			}
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak arms status: error contacting gateway at %s: %v\n", client.base, err)
			return 1
		}

		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(snap)
		} else {
			renderArmsTable(stdout, snap)
		}

		if !*watch {
			break
		}
		time.Sleep(*interval)
		fmt.Fprint(stdout, psClearScreen)
	}
	return 0
}

func renderArmsTable(w io.Writer, snap researcharm.Snapshot) {
	fmt.Fprintf(w, "RESEARCH PROJECT ARMS (%d registered, %d active requests, %d active leases)\n\n",
		snap.TotalArms, snap.TotalInflight, len(snap.ActiveLeases))

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ARM ID\tGROUP\tACTIVE\tLIMIT\tTOTAL REQS\tTOKENS\tERRORS\tPIDS\tLEASE\tLAST SEEN")

	now := time.Now()
	for _, a := range snap.Arms {
		limitStr := "unlimited"
		if a.MaxConcurrency > 0 {
			limitStr = strconv.Itoa(a.MaxConcurrency)
		}
		pids := "-"
		if len(a.RecentPIDs) > 0 {
			var pidStrs []string
			for _, p := range a.RecentPIDs {
				pidStrs = append(pidStrs, strconv.Itoa(p))
			}
			pids = strings.Join(pidStrs, ",")
		}
		leaseStr := "none"
		if a.ActiveLease != nil {
			rem := a.ActiveLease.ExpiresAt.Sub(now).Round(time.Second)
			if rem > 0 {
				leaseStr = fmt.Sprintf("%s (%s left)", a.ActiveLease.Mode, rem)
			}
		}
		lastSeenStr := "-"
		if !a.LastSeen.IsZero() {
			ago := now.Sub(a.LastSeen).Round(time.Second)
			lastSeenStr = fmt.Sprintf("%s ago", ago)
		}

		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n",
			a.ID, a.Group, a.ActiveRequests, limitStr, a.TotalRequests, a.TotalTokens, a.ErrorCount, pids, leaseStr, lastSeenStr)
	}
	_ = tw.Flush()
}

func runArmsTraffic(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms who", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	asJSON := fs.Bool("json", false, "emit JSON output")
	watch := fs.Bool("watch", false, "refresh continuously")
	interval := fs.Duration("interval", 1*time.Second, "refresh interval")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	client := newArmsClient(*addr, *key)
	for {
		var inflight []researcharm.InflightRequest
		err := client.get("/v1/fak/arms/traffic", &inflight)
		if err != nil && strings.Contains(err.Error(), "404") {
			// Fallback: discover live socket connections to server on host
			inflight, err = fallbackArmsTraffic(client)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak arms who: error contacting gateway at %s: %v\n", client.base, err)
			return 1
		}

		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(inflight)
		} else {
			renderTrafficTable(stdout, inflight)
		}

		if !*watch {
			break
		}
		time.Sleep(*interval)
		fmt.Fprint(stdout, psClearScreen)
	}
	return 0
}

func renderTrafficTable(w io.Writer, reqs []researcharm.InflightRequest) {
	if len(reqs) == 0 {
		fmt.Fprintln(w, "No active in-flight requests on the server right now.")
		fmt.Fprintln(w, "(Run 'fak arms status' to view arm request history, limits, and leases)")
		return
	}

	fmt.Fprintf(w, "CURRENT IN-FLIGHT REQUESTS (%d active)\n\n", len(reqs))
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PID\tPROCESS\tARM ID\tGROUP\tENDPOINT\tIN-FLIGHT\tTRACE ID\tREMOTE ADDR")

	now := time.Now()
	for _, req := range reqs {
		pidStr := "-"
		if req.CallerPID > 0 {
			pidStr = strconv.Itoa(req.CallerPID)
		}
		proc := req.CallerProcess
		if proc == "" {
			proc = "-"
		}
		dur := now.Sub(req.StartedAt).Round(100 * time.Millisecond)
		trace := req.TraceID
		if trace == "" {
			trace = "-"
		}

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			pidStr, proc, req.ArmID, req.ArmGroup, req.Endpoint, dur, trace, req.RemoteAddr)
	}
	_ = tw.Flush()
}

func runArmsLease(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runArmsLeaseList(stdout, stderr, argv)
	}

	sub := argv[0]
	switch sub {
	case "acquire":
		return runArmsLeaseAcquire(stdout, stderr, argv[1:])
	case "release":
		return runArmsLeaseRelease(stdout, stderr, argv[1:])
	case "list":
		return runArmsLeaseList(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak arms lease: unknown action %q (try: acquire, release, list)\n", sub)
		return 2
	}
}

func runArmsLeaseAcquire(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms lease acquire", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	arm := fs.String("arm", "", "research project arm ID (required)")
	exclusive := fs.Bool("exclusive", false, "acquire exclusive lease (locks out all other arms)")
	concurrency := fs.Int("concurrency", 0, "maximum concurrent requests for this arm (0 = default)")
	ttl := fs.Duration("ttl", 5*time.Minute, "lease duration")
	pid := fs.Int("pid", os.Getpid(), "holder process PID")
	asJSON := fs.Bool("json", false, "emit JSON output")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if strings.TrimSpace(*arm) == "" {
		fmt.Fprintln(stderr, "fak arms lease acquire: --arm is required")
		return 2
	}

	mode := researcharm.LeaseModeShared
	if *exclusive {
		mode = researcharm.LeaseModeExclusive
	}

	reqBody := researcharm.LeaseRequest{
		ArmID:       *arm,
		HolderPID:   *pid,
		Mode:        mode,
		Concurrency: *concurrency,
		TTL:         *ttl,
	}

	client := newArmsClient(*addr, *key)
	var lease researcharm.LeaseInfo
	err := client.post("/v1/fak/arms/lease", reqBody, &lease)
	if err != nil {
		fmt.Fprintf(stderr, "fak arms lease acquire failed: %v\n", err)
		return 1
	}

	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(lease)
	} else {
		fmt.Fprintf(stdout, "Lease acquired successfully!\n")
		fmt.Fprintf(stdout, "  ID:          %s\n", lease.ID)
		fmt.Fprintf(stdout, "  Arm:         %s\n", lease.ArmID)
		fmt.Fprintf(stdout, "  Mode:        %s\n", lease.Mode)
		if lease.Concurrency > 0 {
			fmt.Fprintf(stdout, "  Concurrency: %d\n", lease.Concurrency)
		}
		fmt.Fprintf(stdout, "  Holder PID:  %d\n", lease.HolderPID)
		fmt.Fprintf(stdout, "  Expires:     %s (%s)\n", lease.ExpiresAt.Format(time.RFC3339), lease.ExpiresAt.Sub(time.Now()).Round(time.Second))
		fmt.Fprintf(stdout, "  Token:       %s (save this to release the lease)\n", lease.Token)
	}
	return 0
}

func runArmsLeaseRelease(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms lease release", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	arm := fs.String("arm", "", "research project arm ID")
	id := fs.String("lease-id", "", "lease ID")
	token := fs.String("token", "", "lease token")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	target := *id
	if target == "" {
		target = *arm
	}
	if target == "" {
		fmt.Fprintln(stderr, "fak arms lease release: --arm or --lease-id is required")
		return 2
	}

	client := newArmsClient(*addr, *key)
	path := "/v1/fak/arms/lease?id=" + target
	if *token != "" {
		path += "&token=" + *token
	}

	var res map[string]any
	err := client.delete(path, &res)
	if err != nil {
		fmt.Fprintf(stderr, "fak arms lease release failed: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "Lease %s released successfully.\n", target)
	return 0
}

func runArmsLeaseList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms lease list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	asJSON := fs.Bool("json", false, "emit JSON output")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	client := newArmsClient(*addr, *key)
	var leases []researcharm.LeaseInfo
	err := client.get("/v1/fak/arms/lease", &leases)
	if err != nil {
		fmt.Fprintf(stderr, "fak arms lease list failed: %v\n", err)
		return 1
	}

	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(leases)
		return 0
	}

	if len(leases) == 0 {
		fmt.Fprintln(stdout, "No active leases held.")
		return 0
	}

	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "LEASE ID\tARM ID\tMODE\tCONCURRENCY\tHOLDER PID\tEXPIRES IN")
	now := time.Now()
	for _, l := range leases {
		concStr := "unlimited"
		if l.Concurrency > 0 {
			concStr = strconv.Itoa(l.Concurrency)
		}
		rem := l.ExpiresAt.Sub(now).Round(time.Second)
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\n",
			l.ID, l.ArmID, l.Mode, concStr, l.HolderPID, rem)
	}
	_ = tw.Flush()
	return 0
}

func runArmsLimit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("arms limit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential")
	arm := fs.String("arm", "", "research project arm ID (required)")
	maxConcurrency := fs.Int("max-concurrency", -1, "max concurrent requests (0 = unlimited)")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if strings.TrimSpace(*arm) == "" {
		fmt.Fprintln(stderr, "fak arms limit: --arm is required")
		return 2
	}
	if *maxConcurrency < 0 {
		fmt.Fprintln(stderr, "fak arms limit: --max-concurrency is required (0 for unlimited, or positive integer)")
		return 2
	}

	client := newArmsClient(*addr, *key)
	reqBody := researcharm.LimitRequest{
		ArmID:          *arm,
		MaxConcurrency: *maxConcurrency,
	}

	var res map[string]any
	err := client.post("/v1/fak/arms/limits", reqBody, &res)
	if err != nil {
		fmt.Fprintf(stderr, "fak arms limit failed: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "Updated concurrency limit for arm %q to %d\n", *arm, *maxConcurrency)
	return 0
}

func fallbackArmsStatus(client *armsClient) (*researcharm.Snapshot, error) {
	var resp struct {
		Sessions []struct {
			TraceID    string `json:"trace_id"`
			Run        string `json:"run"`
			TokensUsed int64  `json:"tokens_used"`
			TokenUsage int64  `json:"token_usage"`
			Rev        int64  `json:"rev"`
		} `json:"sessions"`
		Count int `json:"count"`
	}
	if err := client.get("/v1/fak/sessions", &resp); err != nil {
		return nil, err
	}

	armMap := make(map[string]*researcharm.ArmInfo)
	totalInflight := 0
	for _, sess := range resp.Sessions {
		origin := researcharm.ExtractOrigin(nil, sess.TraceID)
		armID := origin.ArmID
		a, ok := armMap[armID]
		if !ok {
			a = &researcharm.ArmInfo{
				ID:          armID,
				Group:       origin.ArmGroup,
				DisplayName: armID,
				LastSeen:    time.Now(),
			}
			armMap[armID] = a
		}
		a.TotalRequests++
		tokens := sess.TokensUsed
		if tokens == 0 {
			tokens = sess.TokenUsage
		}
		a.TotalTokens += tokens
		if sess.Run == "running" {
			a.ActiveRequests++
			totalInflight++
		}
	}

	var arms []researcharm.ArmInfo
	for _, a := range armMap {
		arms = append(arms, *a)
	}
	sort.Slice(arms, func(i, j int) bool {
		if arms[i].ActiveRequests != arms[j].ActiveRequests {
			return arms[i].ActiveRequests > arms[j].ActiveRequests
		}
		return arms[i].TotalRequests > arms[j].TotalRequests
	})

	return &researcharm.Snapshot{
		Timestamp:     time.Now(),
		TotalArms:     len(arms),
		TotalInflight: totalInflight,
		Arms:          arms,
	}, nil
}

func fallbackArmsTraffic(client *armsClient) ([]researcharm.InflightRequest, error) {
	u, err := url.Parse(client.base)
	if err != nil {
		return nil, err
	}
	portStr := u.Port()
	if portStr == "" {
		portStr = "8080"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	var reqs []researcharm.InflightRequest
	now := time.Now()

	if runtime.GOOS == "darwin" {
		cmd := exec.CommandContext(ctx, "lsof", "-iTCP:"+portStr, "-sTCP:ESTABLISHED", "-P", "-n")
		out, err := cmd.Output()
		if err == nil {
			lines := strings.Split(string(out), "\n")
			for _, line := range lines {
				fields := strings.Fields(line)
				if len(fields) >= 9 {
					comm := fields[0]
					pid, _ := strconv.Atoi(fields[1])
					node := fields[8]
					if strings.HasSuffix(node, ":"+portStr) {
						procArgs := comm
						psOut, psErr := exec.CommandContext(ctx, "ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
						if psErr == nil {
							trimmed := strings.TrimSpace(string(psOut))
							if trimmed != "" {
								procArgs = trimmed
							}
						}

						armID, group := inferArmFromProcess(comm, procArgs)

						reqs = append(reqs, researcharm.InflightRequest{
							RequestID:     "conn-" + strconv.Itoa(pid),
							ArmID:         armID,
							ArmGroup:      group,
							CallerPID:     pid,
							CallerProcess: procArgs,
							Endpoint:      "(connected socket)",
							RemoteAddr:    node,
							StartedAt:     now,
						})
					}
				}
			}
		}
	}
	return reqs, nil
}

func inferArmFromProcess(comm, fullCmd string) (string, string) {
	lower := strings.ToLower(fullCmd)
	switch {
	case strings.Contains(lower, "chat"):
		return "interactive-chat", "interactive"
	case strings.Contains(lower, "curl"):
		return "curl-client", "direct"
	case strings.Contains(lower, "prometheu"):
		return "prometheus", "monitoring"
	case strings.Contains(lower, "codex"):
		return "codex", "agent"
	case strings.Contains(lower, "orch"):
		return "orch", "orch"
	case strings.Contains(lower, "guard"):
		return "guard", "guard"
	default:
		return comm, "local"
	}
}
