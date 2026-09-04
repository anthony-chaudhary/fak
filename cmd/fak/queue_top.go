package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// QueueTopSchema identifies the structured JSON schema emitted by fak queue top --json.
const QueueTopSchema = "fak-queue-top/1"

const (
	defaultQueueTopInterval = 1 * time.Second
	defaultQueueTopSlots    = 16
	queueTopClearScreen     = "\033[H\033[2J"
)

// QueueTopContract is the serialized form of one active contract in queue top output.
type QueueTopContract struct {
	TicketID    string `json:"ticket_id"`
	Holder      string `json:"holder"`
	State       string `json:"state"`
	Age         string `json:"age"`
	AgeSeconds  int64  `json:"age_seconds"`
	TokenBudget int64  `json:"token_budget"`
	TokensUsed  int64  `json:"tokens_used"`
	Tokens      string `json:"tokens"`
	PaceTier    string `json:"pace_tier,omitempty"`
	Generation  int64  `json:"generation,omitempty"`
	WorktreeDir string `json:"worktree_dir,omitempty"`
	VerifyCmd   string `json:"verify_cmd,omitempty"`
}

// QueueTopPacing holds aggregated pacing telemetry and headroom numbers.
type QueueTopPacing struct {
	FrontierCount    int     `json:"frontier_count"`
	CommodityCount   int     `json:"commodity_count"`
	EvalOnlyCount    int     `json:"eval_only_count"`
	OtherCount       int     `json:"other_count"`
	TotalTokensUsed  int64   `json:"total_tokens_used"`
	TotalTokenBudget int64   `json:"total_token_budget"`
	TokensHeadroom   int64   `json:"tokens_headroom"`
	HeadroomPct      float64 `json:"headroom_pct"`
}

// QueueTopBacklog holds queue backlog counts partitioned by state.
type QueueTopBacklog struct {
	TotalLive int `json:"total_live"`
	Active    int `json:"active"`
	Executing int `json:"executing"`
	YieldedIO int `json:"yielded_io"`
	Verifying int `json:"verifying"`
	Pending   int `json:"pending"`
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
	Expired   int `json:"expired"`
}

// QueueTopSlots holds concurrency slot utilization metrics.
type QueueTopSlots struct {
	TotalSlots     int `json:"total_slots"`
	UsedSlots      int `json:"used_slots"`
	AvailableSlots int `json:"available_slots"`
}

// QueueTopReport is the top-level payload conforming to fak-queue-top/1.
type QueueTopReport struct {
	Schema          string             `json:"schema"`
	GeneratedAt     string             `json:"generated_at"`
	Backlog         QueueTopBacklog    `json:"backlog"`
	Pacing          QueueTopPacing     `json:"pacing"`
	Slots           QueueTopSlots      `json:"slots"`
	ActiveContracts []QueueTopContract `json:"active_contracts"`
}

func formatContractAge(now time.Time, acquiredAt int64) (string, int64) {
	if acquiredAt <= 0 {
		return "0s", 0
	}
	t := time.Unix(acquiredAt, 0)
	d := now.Sub(t)
	if d < 0 {
		d = 0
	}
	sec := int64(d.Seconds())
	if sec < 60 {
		return fmt.Sprintf("%ds", sec), sec
	}
	min := sec / 60
	remSec := sec % 60
	if min < 60 {
		return fmt.Sprintf("%dm%02ds", min, remSec), sec
	}
	hr := min / 60
	remMin := min % 60
	return fmt.Sprintf("%dh%02dm", hr, remMin), sec
}

func formatContractTokens(used, budget int64) string {
	if budget > 0 {
		return fmt.Sprintf("%d/%d", used, budget)
	}
	return fmt.Sprintf("%d", used)
}

func isActiveContractState(st leaseref.ContractState) bool {
	switch st {
	case leaseref.ContractStateExecuting, leaseref.ContractStateYieldedIO, leaseref.ContractStateVerifying:
		return true
	default:
		return false
	}
}

func buildQueueTopReport(live []leaseref.ContractRecord, expired []string, totalSlots int, now time.Time) QueueTopReport {
	if totalSlots <= 0 {
		totalSlots = defaultQueueTopSlots
	}
	rep := QueueTopReport{
		Schema:          QueueTopSchema,
		GeneratedAt:     now.UTC().Format(time.RFC3339),
		ActiveContracts: make([]QueueTopContract, 0),
		Slots: QueueTopSlots{
			TotalSlots: totalSlots,
		},
	}

	rep.Backlog.TotalLive = len(live)
	rep.Backlog.Expired = len(expired)

	for _, rec := range live {
		switch rec.State {
		case leaseref.ContractStateExecuting:
			rep.Backlog.Executing++
		case leaseref.ContractStateYieldedIO:
			rep.Backlog.YieldedIO++
		case leaseref.ContractStateVerifying:
			rep.Backlog.Verifying++
		case leaseref.ContractStatePending:
			rep.Backlog.Pending++
		case leaseref.ContractStateSucceeded:
			rep.Backlog.Succeeded++
		case leaseref.ContractStateFailed:
			rep.Backlog.Failed++
		}

		if isActiveContractState(rec.State) {
			ageStr, ageSec := formatContractAge(now, rec.AcquiredAt)
			tokensStr := formatContractTokens(rec.TokensUsed, rec.TokenBudget)

			rep.ActiveContracts = append(rep.ActiveContracts, QueueTopContract{
				TicketID:    rec.TicketID,
				Holder:      rec.Holder,
				State:       string(rec.State),
				Age:         ageStr,
				AgeSeconds:  ageSec,
				TokenBudget: rec.TokenBudget,
				TokensUsed:  rec.TokensUsed,
				Tokens:      tokensStr,
				PaceTier:    rec.PaceTier,
				Generation:  rec.Generation,
				WorktreeDir: rec.WorktreeDir,
				VerifyCmd:   rec.VerifyCmd,
			})

			switch strings.ToLower(rec.PaceTier) {
			case leaseref.PaceTierFrontier:
				rep.Pacing.FrontierCount++
			case leaseref.PaceTierCommodity:
				rep.Pacing.CommodityCount++
			case leaseref.PaceTierEvalOnly:
				rep.Pacing.EvalOnlyCount++
			default:
				if rec.PaceTier != "" {
					rep.Pacing.OtherCount++
				}
			}

			rep.Pacing.TotalTokensUsed += rec.TokensUsed
			rep.Pacing.TotalTokenBudget += rec.TokenBudget
		}
	}

	rep.Backlog.Active = len(rep.ActiveContracts)

	// Sort active contracts by TicketID for stable and deterministic output.
	sort.Slice(rep.ActiveContracts, func(i, j int) bool {
		return rep.ActiveContracts[i].TicketID < rep.ActiveContracts[j].TicketID
	})

	if rep.Pacing.TotalTokenBudget > 0 {
		headroom := rep.Pacing.TotalTokenBudget - rep.Pacing.TotalTokensUsed
		if headroom < 0 {
			headroom = 0
		}
		rep.Pacing.TokensHeadroom = headroom
		rep.Pacing.HeadroomPct = (float64(headroom) / float64(rep.Pacing.TotalTokenBudget)) * 100.0
	} else {
		rep.Pacing.TokensHeadroom = 0
		rep.Pacing.HeadroomPct = 100.0
	}

	rep.Slots.UsedSlots = rep.Backlog.Active
	avail := rep.Slots.TotalSlots - rep.Slots.UsedSlots
	if avail < 0 {
		avail = 0
	}
	rep.Slots.AvailableSlots = avail

	return rep
}

func renderQueueTopTable(w io.Writer, report QueueTopReport) {
	fmt.Fprintln(w, "fak queue top — contract states, pacing headroom & slots")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "QUEUE BACKLOG SUMMARY:")
	fmt.Fprintf(w, "  Total Live: %d | Active: %d (Executing: %d, Yielded IO: %d, Verifying: %d) | Pending: %d | Expired: %d\n",
		report.Backlog.TotalLive,
		report.Backlog.Active,
		report.Backlog.Executing,
		report.Backlog.YieldedIO,
		report.Backlog.Verifying,
		report.Backlog.Pending,
		report.Backlog.Expired,
	)
	fmt.Fprintf(w, "  Slots: %d/%d used (%d available) | Pacing Headroom: %d / %d tokens (%.1f%%)\n",
		report.Slots.UsedSlots,
		report.Slots.TotalSlots,
		report.Slots.AvailableSlots,
		report.Pacing.TokensHeadroom,
		report.Pacing.TotalTokenBudget,
		report.Pacing.HeadroomPct,
	)
	fmt.Fprintf(w, "  Tiers: frontier=%d, commodity=%d, eval_only=%d\n",
		report.Pacing.FrontierCount,
		report.Pacing.CommodityCount,
		report.Pacing.EvalOnlyCount,
	)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "ACTIVE CONTRACTS:")
	if len(report.ActiveContracts) == 0 {
		fmt.Fprintln(w, "  (no active contracts)")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "TICKET ID\tHOLDER\tSTATE\tAGE\tTOKENS\tTIER\tGEN")
	for _, c := range report.ActiveContracts {
		holder := c.Holder
		if holder == "" {
			holder = "-"
		}
		tier := c.PaceTier
		if tier == "" {
			tier = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%d\n",
			c.TicketID,
			holder,
			c.State,
			c.Age,
			c.Tokens,
			tier,
			c.Generation,
		)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "\n%d active contract(s)\n", len(report.ActiveContracts))
}

const queueTopUsage = `fak queue top - real-time TUI for contract states, pacing headroom, and slots (#11163)

Usage:
  fak queue top [flags]

Flags:
  --json          emit JSON schema fak-queue-top/1
  --snapshot      render snapshot and exit (non-interactive)
  --watch         refresh continuously
  --interval      refresh interval in watch mode (default: 1s)
  --frames        stop after N frames in watch mode (0 = run until interrupted)
  --slots         concurrency slot capacity (default: 16)
  --dir           repo dir (default: git discovery from cwd)
`

func cmdQueueTop(argv []string) {
	if len(argv) > 0 && argv[0] == "top" {
		argv = argv[1:]
	}
	os.Exit(runQueueTop(os.Stdout, os.Stderr, argv))
}

func runQueueTop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "top" {
		argv = argv[1:]
	}
	if len(argv) > 0 && (argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help") {
		fmt.Fprint(stdout, queueTopUsage)
		return 0
	}

	fs := flag.NewFlagSet("fak queue top", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, queueTopUsage) }

	asJSON := fs.Bool("json", false, "emit JSON schema fak-queue-top/1")
	snapshot := fs.Bool("snapshot", false, "render snapshot and exit")
	watch := fs.Bool("watch", false, "refresh continuously")
	interval := fs.Duration("interval", defaultQueueTopInterval, "watch refresh interval")
	frames := fs.Int("frames", 0, "stop after N frames in watch mode (0 = run until interrupted)")
	slots := fs.Int("slots", defaultQueueTopSlots, "concurrency slot capacity")
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")

	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	var watchSet bool
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "watch" {
			watchSet = true
		}
	})

	isTerminal := writerIsTerminal(stdout)
	snapshotMode := *snapshot || *asJSON || !isTerminal
	if watchSet && *watch {
		snapshotMode = false
	}

	*dir = pathutil.ExpandTilde(*dir)
	var store *leaseref.Store
	if strings.TrimSpace(*dir) != "" {
		store = leaseref.NewInDir(*dir)
	} else {
		store = leaseref.New()
	}

	if snapshotMode {
		now := time.Now()
		live, expired, err := store.LiveContracts(context.Background(), now)
		if err != nil {
			fmt.Fprintf(stderr, "fak queue top: %v\n", err)
			return 1
		}
		report := buildQueueTopReport(live, expired, *slots, now)
		return emitJSONOrRender(stdout, stderr, "fak queue top", *asJSON, report, func(w io.Writer) {
			renderQueueTopTable(w, report)
		})
	}

	return runQueueTopWatch(stdout, stderr, store, *slots, *asJSON, *interval, *frames)
}

func runQueueTopWatch(stdout, stderr io.Writer, store *leaseref.Store, slots int, asJSON bool, interval time.Duration, frames int) int {
	if interval <= 0 {
		interval = defaultQueueTopInterval
	}
	for i := 0; frames <= 0 || i < frames; i++ {
		now := time.Now()
		live, expired, err := store.LiveContracts(context.Background(), now)
		if err != nil {
			fmt.Fprintf(stderr, "fak queue top: %v\n", err)
			if frames > 0 && i == frames-1 {
				return 1
			}
			time.Sleep(interval)
			continue
		}
		report := buildQueueTopReport(live, expired, slots, now)
		if !asJSON {
			fmt.Fprint(stdout, queueTopClearScreen)
		}
		code := emitJSONOrRender(stdout, stderr, "fak queue top", asJSON, report, func(w io.Writer) {
			renderQueueTopTable(w, report)
		})
		if code != 0 {
			return code
		}
		if frames > 0 && i == frames-1 {
			break
		}
		time.Sleep(interval)
	}
	return 0
}
