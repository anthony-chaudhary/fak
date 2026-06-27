package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// cmdProduct posts product-direction content to the #product Slack channel.
//
//	fak product post --status                         # fold the product scorecard into a card
//	fak product post --persona                        # fold the persona-readiness scorecard
//	fak product post --from card.json --debt-key product_debt
//	fak product post --title "Direction" --notes "..."        # free-form product prose
//	fak product post --title "Direction" --notes-file note.md # prose from a file (- for stdin)
//
// It targets the SAME workspace as `fak scoreboard` (team FAK_SCOREBOARD_TEAM) but a
// DIFFERENT channel: #scoreboard carries scores/numbers, #product carries product
// direction, persona findings, and product-status snapshots. The bot token is shared
// (FAK_SCOREBOARD_TOKEN); only the channel differs (FAK_PRODUCT_CHANNEL). A product post
// never falls back to #scoreboard — that would put direction prose in the number feed.
func cmdProduct(argv []string) {
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "fak product: missing subcommand (post)")
		os.Exit(2)
	}
	switch argv[0] {
	case "post":
		os.Exit(runProductPost(os.Stdout, os.Stderr, argv[1:]))
	default:
		fmt.Fprintf(os.Stderr, "fak product: unknown subcommand %q (want: post)\n", argv[0])
		os.Exit(2)
	}
}

func runProductPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak product post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	status := fs.Bool("status", false, "fold tools/product_scorecard.py --json into the card (product-status snapshot)")
	persona := fs.Bool("persona", false, "fold tools/persona_readiness_scorecard.py --json into the card (persona findings)")
	from := fs.String("from", "", "read a pkg/scorecard control-pane JSON payload from this file (- for stdin)")
	debtKey := fs.String("debt-key", "", "with --from/--status/--persona: which corpus integer is the headline debt (default: product_debt / persona_debt)")
	title := fs.String("title", "", "post title (default: derived from the scorecard schema, or required for a free-form --notes post)")
	notes := fs.String("notes", "", "free-form product-direction body (mutually exclusive with --notes-file)")
	notesFile := fs.String("notes-file", "", "read the free-form body from this file (- for stdin)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_PRODUCT_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	up, err := buildProductUpdate(*status, *persona, *from, *debtKey, *title, *notes, *notesFile, *source)
	if err != nil {
		fmt.Fprintf(stderr, "fak product post: %v\n", err)
		return 2
	}

	if *dryRun {
		fmt.Fprintln(stdout, up.Text())
		return 0
	}

	ch := *channel
	if ch == "" {
		ch = scoreboard.ResolveProductChannel()
	}
	if ch == "" {
		fmt.Fprintln(stderr, "fak product post: no channel: pass --channel, set FAK_PRODUCT_CHANNEL, or add it to .env.slack.local")
		return 2
	}
	client, err := scoreboard.NewClient(*token)
	if err != nil {
		fmt.Fprintf(stderr, "fak product post: %v\n", err)
		return 2
	}
	ts, err := client.Post(ctx(), ch, up.Text(), up.Blocks())
	if err != nil {
		fmt.Fprintf(stderr, "fak product post: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}

// buildProductUpdate assembles the Update from exactly one content source: a scorecard
// payload (--status / --persona / --from) or a free-form note (--title + --notes/--notes-file).
func buildProductUpdate(status, persona bool, from, debtKey, title, notes, notesFile, source string) (scoreboard.Update, error) {
	src := source
	if src == "" {
		src = defaultSource()
	}

	// Free-form product note: a title plus a body, no scorecard involved.
	if notes != "" || notesFile != "" {
		if status || persona || from != "" {
			return scoreboard.Update{}, fmt.Errorf("--notes/--notes-file is a free-form post; do not combine it with --status/--persona/--from")
		}
		if notes != "" && notesFile != "" {
			return scoreboard.Update{}, fmt.Errorf("pass --notes or --notes-file, not both")
		}
		body := notes
		if notesFile != "" {
			raw, err := readFromFile(notesFile)
			if err != nil {
				return scoreboard.Update{}, err
			}
			body = string(raw)
		}
		t := title
		if t == "" {
			return scoreboard.Update{}, fmt.Errorf("a free-form --notes post needs a --title")
		}
		return scoreboard.Update{Title: t, Notes: body, Source: src}, nil
	}

	// Scorecard-derived card. --status / --persona run the tool and capture its JSON;
	// --from reads a payload already on disk. Exactly one must be selected.
	var raw []byte
	var defKey string
	switch {
	case status:
		if persona || from != "" {
			return scoreboard.Update{}, fmt.Errorf("pass exactly one of --status / --persona / --from")
		}
		b, err := runScorecardJSON("tools/product_scorecard.py")
		if err != nil {
			return scoreboard.Update{}, err
		}
		raw, defKey = b, "product_debt"
	case persona:
		if from != "" {
			return scoreboard.Update{}, fmt.Errorf("pass exactly one of --status / --persona / --from")
		}
		b, err := runScorecardJSON("tools/persona_readiness_scorecard.py")
		if err != nil {
			return scoreboard.Update{}, err
		}
		raw, defKey = b, "persona_debt"
	case from != "":
		b, err := readFromFile(from)
		if err != nil {
			return scoreboard.Update{}, err
		}
		raw = b
	default:
		return scoreboard.Update{}, fmt.Errorf("nothing to post: pass --status, --persona, --from <payload.json>, or --title + --notes")
	}

	var p scorecard.Payload
	if err := json.Unmarshal(raw, &p); err != nil {
		return scoreboard.Update{}, fmt.Errorf("parse scorecard payload: %w", err)
	}
	key := debtKey
	if key == "" {
		key = defKey
	}
	t := title
	if t == "" {
		t = p.Schema
	}
	if t == "" {
		t = "product"
	}
	up := scoreboard.FromPayload(t, p, key)
	up.Source = src
	return up, nil
}

// runScorecardJSON runs a project python scorecard with --json and returns its stdout.
// It tries `python` then `python3` so the same call works on Windows and POSIX.
//
// A scorecard's EXIT CODE is its verdict, not a run failure: it exits non-zero precisely
// when the verdict is ACTION (there is debt) while still printing the full, valid JSON
// payload to stdout — and that payload is exactly what we want to post. So a non-zero exit
// with JSON-shaped stdout is a SUCCESS here; only a missing interpreter or empty/non-JSON
// stdout is a real error (we never post a half card).
func runScorecardJSON(toolPath string) ([]byte, error) {
	var lastErr error
	for _, bin := range []string{"python", "python3"} {
		cmd := exec.Command(bin, toolPath, "--json")
		var out, errb bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &errb
		runErr := cmd.Run()
		if out.Len() > 0 && bytes.HasPrefix(bytes.TrimSpace(out.Bytes()), []byte("{")) {
			// Valid-looking JSON on stdout: the verdict (exit code) is the payload, not a failure.
			return out.Bytes(), nil
		}
		if runErr != nil {
			lastErr = fmt.Errorf("%s %s --json: %w (%s)", bin, toolPath, runErr, errb.String())
			continue
		}
		return nil, fmt.Errorf("%s %s --json produced no JSON output", bin, toolPath)
	}
	return nil, lastErr
}
