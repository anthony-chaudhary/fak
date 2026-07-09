package main

// Tests for the `fak slack outbox calls` operator verb — the per-source Slack API-call gauge
// behind "the session cards are wasting our Slack limits". It binds the CLI to the durable
// CallStats fold end to end: enqueue a realistic mix, drain it through a fake Slack, then read
// the reduction off the command's output.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestRunSlackOutboxCallsJSONAttributesPerSource(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	ob, err := openOutbox()
	if err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	// Two one-shot posts from one surface and one from another — three chat.postMessage calls
	// across two sources, so the fold must attribute two to "runcard" and one to "slack-send".
	for _, r := range []slackoutbox.Row{
		{Channel: "C1", Text: "shipped a", Source: "runcard"},
		{Channel: "C1", Text: "shipped b", Source: "runcard"},
		{Channel: "C1", Text: "hi", Source: "slack-send"},
	} {
		if _, err := ob.Enqueue(r); err != nil {
			t.Fatalf("enqueue: %v", err)
		}
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}

	out.Reset()
	errb.Reset()
	if rc := runSlackOutbox(&out, &errb, []string{"calls", "--json"}); rc != 0 {
		t.Fatalf("calls --json rc=%d stderr=%s", rc, errb.String())
	}
	var cs slackoutbox.CallStats
	if err := json.Unmarshal(out.Bytes(), &cs); err != nil {
		t.Fatalf("decode call stats: %v\n%s", err, out.String())
	}
	if cs.TotalSent != 3 {
		t.Fatalf("total_sent = %d, want 3\n%s", cs.TotalSent, out.String())
	}
	bySrc := map[string]slackoutbox.SourceCalls{}
	for _, sc := range cs.Sources {
		bySrc[sc.Source] = sc
	}
	if g := bySrc["runcard"]; g.Posts != 2 || g.Sent() != 2 {
		t.Fatalf("runcard footprint wrong: %+v", g)
	}
	if g := bySrc["slack-send"]; g.Posts != 1 || g.Sent() != 1 {
		t.Fatalf("slack-send footprint wrong: %+v", g)
	}
	// Loudest-first: runcard (2 sent) before slack-send (1 sent).
	if len(cs.Sources) < 2 || cs.Sources[0].Source != "runcard" {
		t.Fatalf("sources not sorted loudest-first: %+v", cs.Sources)
	}
}

func TestRunSlackOutboxCallsHumanLine(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	if _, err := openOutbox(); err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"calls"}); rc != 0 {
		t.Fatalf("calls rc=%d stderr=%s", rc, errb.String())
	}
	// An empty spool still renders the headline gauge (zero calls spent).
	if !strings.Contains(out.String(), "sent 0 call") {
		t.Fatalf("calls human line missing the zero-spend headline:\n%s", out.String())
	}
}
