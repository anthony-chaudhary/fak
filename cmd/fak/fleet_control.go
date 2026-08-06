package main

// `fak fleet control` — the operator's end of the fleet control bus (#5600, epic
// #5599). One point that commands every live instance, and — the half that makes it a
// control plane rather than a megaphone — reads back what each one actually did.
//
//	fak fleet control send --op OP [--payload TEXT] --all|--instance|--machine|--role
//	                       [--lane L] [--wave W] [--label X] [--ttl D] [--reason R]
//	                       [--wait D] [--bus DIR] [--json]
//	fak fleet control status --directive ID [--bus DIR] [--json]
//	fak fleet control instances [--ttl D] [--bus DIR] [--json]
//
// Two rules shape the surface:
//
// Publishing into an empty fleet is REFUSED (FLEETBUS_NO_TARGET), never accepted and
// left to time out. An accepted directive nobody will ever apply is the same phantom
// the gateway refuses a 202 for.
//
// Exit 0 means WITNESSED APPLIED — every addressed instance acked applied within the
// wait. A published-but-unwitnessed directive exits 1, including under `--wait 0`,
// where the operator explicitly asked not to look. The enqueue is not the witness.
//
// There is deliberately no `drain` verb here. A CLI process cannot apply a directive to
// a serve process's session table; a CLI drainer would only steal the claim from the
// instance that can and ack it refused. The one drainer is `fak serve`.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
)

func runFleetControl(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak fleet control: missing subcommand (send | status | instances)")
		return 2
	}
	switch argv[0] {
	case "send":
		return runFleetControlSend(stdout, stderr, argv[1:])
	case "status":
		return runFleetControlStatus(stdout, stderr, argv[1:])
	case "instances":
		return runFleetControlInstances(stdout, stderr, argv[1:])
	}
	fmt.Fprintf(stderr, "fak fleet control: unknown subcommand %q (want: send | status | instances)\n", argv[0])
	return 2
}

// fleetControlNow is the clock seam, matching fleetNow's role in the sibling verbs.
var fleetControlNow = time.Now
var fleetControlSleep = time.Sleep

// defaultFleetBusDir puts the bus beside the registry the fleet tools already agree on,
// so a fleet that shares FLEET_STATE_DIR shares a bus without being told twice.
func defaultFleetBusDir() string {
	if v := strings.TrimSpace(os.Getenv("FAK_FLEET_BUS")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("FLEET_STATE_DIR")); v != "" {
		return filepath.Join(v, "bus")
	}
	return filepath.Join(filepath.Dir(defaultFleetRegistryDir()), "bus")
}

// --- send ------------------------------------------------------------------ //

func runFleetControlSend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet control send", flag.ContinueOnError)
	fs.SetOutput(stderr)
	op := fs.String("op", "", "control op to fan out (the applier owns its meaning; e.g. steer, pause, resume)")
	payload := fs.String("payload", "", "argument for the op (e.g. the steer text)")
	// --text is the spelling #5600's done-condition uses for the steer case. It is the
	// SAME field, not a second one: an op takes at most one argument, and two flag names
	// that could disagree is a way to send a directive whose payload nobody can predict.
	text := fs.String("text", "", "alias for --payload, in the steer spelling")
	all := fs.Bool("all", false, "address every live instance — must be STATED, never implied by an absent filter")
	instance := fs.String("instance", "", "comma-separated instance ids to address")
	machine := fs.String("machine", "", "comma-separated machines to address")
	role := fs.String("role", "", "comma-separated roles to address (e.g. serve)")
	lane := fs.String("lane", "", "narrow to this lane WITHIN each addressed instance")
	wave := fs.String("wave", "", "narrow to this wave WITHIN each addressed instance")
	label := fs.String("label", "", "narrow to this label WITHIN each addressed instance")
	ttl := fs.Duration("ttl", 5*time.Minute, "how long the directive stays applyable (0 = never expires)")
	reason := fs.String("reason", "", "why — carried through to each instance's local record")
	issuer := fs.String("issuer", "", "control point name for attribution (default: hostname)")
	wait := fs.Duration("wait", 10*time.Second, "how long to wait for acks; 0 publishes without witnessing (and exits 1)")
	busDir := fs.String("bus", "", "bus directory (default: FAK_FLEET_BUS, else <FLEET_STATE_DIR>/bus)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	rosterTTL := fs.Duration("roster-ttl", fleetbus.DefaultInstanceTTL, "fresh-instance roster window used for this fold")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak fleet control send: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*op) == "" {
		fmt.Fprintln(stderr, "fak fleet control send: --op is required")
		return 2
	}
	arg, ok := resolveControlPayload(*payload, *text)
	if !ok {
		fmt.Fprintln(stderr, "fak fleet control send: --payload and --text are the same field and disagree; state one")
		return 2
	}

	sel := fleetbus.Selector{
		All:      *all,
		Instance: splitTokenList(*instance),
		Machine:  splitTokenList(*machine),
		Role:     splitTokenList(*role),
		Lane:     strings.TrimSpace(*lane),
		Wave:     strings.TrimSpace(*wave),
		Label:    strings.TrimSpace(*label),
	}
	now := fleetControlNow()
	d, refusal := fleetbus.NewDirective(controlIssuer(*issuer), fleetbus.Op(strings.TrimSpace(*op)),
		arg, sel, *ttl, strings.TrimSpace(*reason), now)
	if refusal != nil {
		fmt.Fprintf(stderr, "fak fleet control send: %s\n", refusal)
		return 2
	}

	bus, code := openControlBus(stderr, "send", *busDir)
	if bus == nil {
		return code
	}
	if *rosterTTL <= 0 {
		fmt.Fprintln(stderr, "fak fleet control send: --roster-ttl must be positive")
		return 2
	}
	roster, err := bus.Instances(now, *rosterTTL)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet control send: read roster: %v\n", err)
		return 2
	}
	// Refuse at the edge rather than accept and let it time out: a directive addressed
	// to nobody is the accepted-but-never-applied phantom in its purest form.
	targets := fleetbus.PublishTargets(sel, roster)
	if len(targets) == 0 {
		fmt.Fprintf(stderr, "fak fleet control send: %s: selector %q addresses none of the %d live instance(s) in %s\n",
			fleetbus.NoTarget, sel.String(), len(roster), bus.Root)
		return 2
	}
	// Stamp WHO was addressed onto the directive before publishing it. Without this the
	// fold re-derives its denominator from whatever roster is live at fold time, and an
	// instance that took the directive to its grave would quietly leave the count —
	// turning "one instance never answered" into a clean exit 0.
	d = d.WithTargets(targets)
	if err := bus.Publish(d); err != nil {
		fmt.Fprintf(stderr, "fak fleet control send: publish: %v\n", err)
		return 2
	}

	rep, waited, err := awaitFleetControl(bus, d, *wait, *rosterTTL)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet control send: %v\n", err)
		return 1
	}
	return emitFleetControl(stdout, stderr, rep, waited, *wait, *asJSON, "send")
}

// awaitFleetControl polls the return path until every addressed instance has answered
// or the budget runs out. It re-reads the ROSTER each pass, not just the acks: an
// instance that announces after the publish is a real new target, and a denominator
// frozen at publish time would report it as never having existed.
func awaitFleetControl(bus *fleetbus.DirBus, d fleetbus.Directive, wait, rosterTTL time.Duration) (fleetbus.Report, time.Duration, error) {
	const poll = 250 * time.Millisecond
	start := fleetControlNow()
	for {
		rep, err := foldFleetControl(bus, d, rosterTTL)
		waited := fleetControlNow().Sub(start)
		if err != nil {
			return fleetbus.Report{}, waited, err
		}
		if wait <= 0 || rep.Complete || waited >= wait {
			return rep, waited, nil
		}
		remaining := wait - waited
		if remaining > poll {
			remaining = poll
		}
		fleetControlSleep(remaining)
	}
}

func foldFleetControl(bus *fleetbus.DirBus, d fleetbus.Directive, rosterTTL time.Duration) (fleetbus.Report, error) {
	now := fleetControlNow()
	roster, err := bus.Instances(now, rosterTTL)
	if err != nil {
		return fleetbus.Report{}, fmt.Errorf("read roster: %w", err)
	}
	acks, err := bus.Acks(d.ID)
	if err != nil {
		return fleetbus.Report{}, fmt.Errorf("read acknowledgements: %w", err)
	}
	return fleetbus.Fold(d, roster, acks, now), nil
}

// --- status ---------------------------------------------------------------- //

func runFleetControlStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet control status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	directive := fs.String("directive", "", "directive id to fold (as printed by `send`)")
	busDir := fs.String("bus", "", "bus directory (default: FAK_FLEET_BUS, else <FLEET_STATE_DIR>/bus)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	rosterTTL := fs.Duration("roster-ttl", fleetbus.DefaultInstanceTTL, "fresh-instance roster window used for this fold")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak fleet control status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*directive) == "" {
		fmt.Fprintln(stderr, "fak fleet control status: --directive is required")
		return 2
	}

	bus, code := openControlBus(stderr, "status", *busDir)
	if bus == nil {
		return code
	}
	directives, err := bus.Directives()
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet control status: read directives: %v\n", err)
		return 2
	}
	want := strings.TrimSpace(*directive)
	for _, d := range directives {
		if d.ID == want {
			if *rosterTTL <= 0 {
				fmt.Fprintln(stderr, "fak fleet control status: --roster-ttl must be positive")
				return 2
			}
			rep, err := foldFleetControl(bus, d, *rosterTTL)
			if err != nil {
				fmt.Fprintf(stderr, "fak fleet control status: %v\n", err)
				return 1
			}
			return emitFleetControl(stdout, stderr, rep, 0, 0, *asJSON, "status")
		}
	}
	// Say which of the two this is: an id that was never issued and one whose ledger
	// generation has rotated away are different problems with different fixes.
	fmt.Fprintf(stderr, "fak fleet control status: directive %q is not in %s (%d directive(s) retained; a rotated generation drops history, never a live directive)\n",
		want, bus.Root, len(directives))
	return 2
}

// --- instances ------------------------------------------------------------- //

func runFleetControlInstances(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleet control instances", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ttl := fs.Duration("ttl", fleetbus.DefaultInstanceTTL, "how recent an announcement must be to count as live")
	busDir := fs.String("bus", "", "bus directory (default: FAK_FLEET_BUS, else <FLEET_STATE_DIR>/bus)")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak fleet control instances: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	bus, code := openControlBus(stderr, "instances", *busDir)
	if bus == nil {
		return code
	}
	roster, err := bus.Instances(fleetControlNow(), *ttl)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet control instances: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"bus":       bus.Root,
			"ttl_sec":   int(ttl.Seconds()),
			"instances": roster,
		}, "fak fleet control instances")
	}
	fmt.Fprintf(stdout, "fleet control instances [%s]: %d live (ttl %s)\n", bus.Root, len(roster), *ttl)
	if len(roster) == 0 {
		// Not an error: an empty fleet is a fact, and it is the fact that makes the
		// next `send` refuse with FLEETBUS_NO_TARGET instead of hanging.
		fmt.Fprintln(stdout, "  (none — a send would refuse with "+string(fleetbus.NoTarget)+")")
		return 0
	}
	for _, inst := range roster {
		ops := "-"
		if len(inst.Ops) > 0 {
			parts := make([]string, 0, len(inst.Ops))
			for _, op := range inst.Ops {
				parts = append(parts, string(op))
			}
			ops = strings.Join(parts, ",")
		}
		fmt.Fprintf(stdout, "  %-24s %-16s %-10s pid=%-8d seen=%s ops=%s\n",
			inst.ID, dashIfEmpty(inst.Machine), dashIfEmpty(inst.Role), inst.PID, inst.SeenUTC, ops)
	}
	return 0
}

// --- shared ---------------------------------------------------------------- //

func openControlBus(stderr io.Writer, verb, dir string) (*fleetbus.DirBus, int) {
	if strings.TrimSpace(dir) == "" {
		dir = defaultFleetBusDir()
	}
	bus, err := fleetbus.OpenDir(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleet control %s: %v\n", verb, err)
		return nil, 2
	}
	return bus, 0
}

// emitFleetControl renders the fold and maps it to the exit code. Exit 0 is reserved
// for WITNESSED APPLIED — anything short of that (a refusal, a lapse, an instance still
// silent, or an operator who passed --wait 0 and chose not to look) is 1.
func emitFleetControl(stdout, stderr io.Writer, rep fleetbus.Report, waited, budget time.Duration, asJSON bool, verb string) int {
	if asJSON {
		if code := encodeJSONOrFail(stdout, stderr, fleetControlJSON{
			Report:    rep,
			WaitedSec: waited.Seconds(),
			Witnessed: rep.Complete && rep.Applied == rep.Targeted,
		}, "fak fleet control "+verb); code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, renderFleetControl(rep, waited, budget, verb))
	}
	if rep.Complete && rep.Applied == rep.Targeted {
		return 0
	}
	return 1
}

type fleetControlJSON struct {
	fleetbus.Report
	// WaitedSec is how long the control point actually watched the return path.
	WaitedSec float64 `json:"waited_sec"`
	// Witnessed is the exit-0 condition spelled out: every addressed instance acked
	// APPLIED. It is deliberately not the same field as Complete, which only means
	// everybody answered.
	Witnessed bool `json:"witnessed"`
}

func renderFleetControl(rep fleetbus.Report, waited, budget time.Duration, verb string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fleet control %s: op=%s selector=%s issuer=%s issued=%s\n",
		rep.Directive, rep.Op, rep.Selector, rep.Issuer, rep.IssuedUTC)
	for _, row := range rep.Rows {
		detail := ""
		switch row.Status {
		case fleetbus.RowApplied:
			detail = fmt.Sprintf("affected=%d %s", row.Affected, dashIfEmpty(row.Witness))
		case fleetbus.RowOutstanding:
			detail = "no ack yet"
		default:
			detail = strings.TrimSpace(string(row.Reason) + " " + row.Detail)
		}
		roster := ""
		if !row.InRoster {
			roster = " (left the roster)"
		}
		fmt.Fprintf(&b, "  %-24s %-16s %-12s %s%s\n",
			row.Instance, dashIfEmpty(row.Machine), row.Status, detail, roster)
	}
	fmt.Fprintf(&b, "  targeted=%d applied=%d refused=%d expired=%d outstanding=%d affected=%d\n",
		rep.Targeted, rep.Applied, rep.Refused, rep.Expired, rep.Outstanding, rep.AffectedTotal)
	fmt.Fprintf(&b, "  verdict: %s\n", fleetControlVerdict(rep, waited, budget, verb))
	return b.String()
}

// fleetControlVerdict says which of the three states this is in the operator's own
// terms. "not yet" is a first-class outcome, distinct from a refusal: the instances may
// still answer, and the report says whether the TTL leaves them any room to.
// A verb of "status" is a READ of a directive already published, not a wait: it has no
// --wait flag and measured no latency, so it must not explain an outstanding directive
// as a wait-budget choice the operator never made, nor print "after 0s" as if it had
// timed something.
func fleetControlVerdict(rep fleetbus.Report, waited, budget time.Duration, verb string) string {
	read := verb == "status"
	switch {
	case rep.Targeted == 0:
		return "no target — nobody was addressed, so nothing can have applied"
	case rep.Complete && rep.Applied == rep.Targeted && read:
		return fmt.Sprintf("applied — all %d addressed instance(s) witnessed it", rep.Targeted)
	case rep.Complete && rep.Applied == rep.Targeted:
		return fmt.Sprintf("applied — all %d addressed instance(s) witnessed it, after %s", rep.Targeted, waited.Round(time.Millisecond))
	case rep.Complete:
		return fmt.Sprintf("answered, not applied — %d of %d applied; %d refused, %d expired",
			rep.Applied, rep.Targeted, rep.Refused, rep.Expired)
	case rep.DirectiveExpired:
		return fmt.Sprintf("not yet, and never — the ttl lapsed with %d instance(s) still silent; they will not answer now", rep.Outstanding)
	case read:
		return fmt.Sprintf("not yet — %d of %d applied, %d still silent; re-read with this same command once they have had time to drain",
			rep.Applied, rep.Targeted, rep.Outstanding)
	case budget <= 0:
		return fmt.Sprintf("not yet — published, but --wait 0 means no ack was witnessed; run `fak fleet control status --directive %s`", rep.Directive)
	default:
		return fmt.Sprintf("not yet — %d of %d applied, %d still silent after %s; run `fak fleet control status --directive %s`",
			rep.Applied, rep.Targeted, rep.Outstanding, waited.Round(time.Millisecond), rep.Directive)
	}
}

// resolveControlPayload folds the two spellings of the op's one argument into one value,
// reporting false when they were both stated and disagree. Silently preferring one would
// mean the fleet receives a payload the operator can see they did not send — and the only
// place that becomes visible is N instances later.
func resolveControlPayload(payload, text string) (string, bool) {
	switch {
	case payload == "":
		return text, true
	case text == "", text == payload:
		return payload, true
	default:
		return "", false
	}
}

// splitTokenList parses a comma-separated axis, dropping empties so a trailing comma is
// not a phantom member.
func splitTokenList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func controlIssuer(flagVal string) string {
	if v := strings.TrimSpace(flagVal); v != "" {
		return v
	}
	if host, err := os.Hostname(); err == nil && strings.TrimSpace(host) != "" {
		return sanitizeBusToken(host)
	}
	return "control"
}

// sanitizeBusToken maps an arbitrary host string onto fleetbus's token alphabet. A
// hostname is not the operator's choice, so refusing one for a stray character would
// block the send over something nobody can fix at the call site.
func sanitizeBusToken(s string) string {
	var b strings.Builder
	for i := 0; i < len(s) && b.Len() < 128; i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_', c == '.':
			b.WriteByte(c)
		default:
			b.WriteByte('-')
		}
	}
	out := b.String()
	if out == "" || out == "." || out == ".." {
		return "control"
	}
	return out
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}
