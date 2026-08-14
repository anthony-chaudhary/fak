package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// runSessionOpen is the reference terminal client for the same session-client protocol used
// by the browser page. It deliberately does not read local session files or reconstruct state.
func runSessionOpen(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "--selfcheck" {
		return runSessionOpenSelfcheck(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("session open", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("url", "http://127.0.0.1:8080", "fak gateway base URL")
	bearer := fs.String("bearer", "", "gateway bearer token")
	localToken := fs.String("local-token", strings.TrimSpace(os.Getenv("FAK_SESSION_CLIENT_TOKEN")), "per-user local session capability token (default FAK_SESSION_CLIENT_TOKEN)")
	workspace := fs.String("workspace", "default", "workspace identity bound to the logical session")
	since := fs.Uint64("since", 0, "last applied logical event address")
	input := fs.String("input", "", "submit one text input while the attachment holds the lease")
	decision := fs.String("decision", "", "keyboard decision: approve or deny")
	closeSession := fs.Bool("close", false, "close the logical session after attaching")
	jsonOut := fs.Bool("json", false, "emit the shared wire response as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak session open [--url URL] [--since N] [--input TEXT] SESSION_ID")
		return 2
	}
	sessionID := strings.TrimSpace(fs.Arg(0))
	endpoint := strings.TrimRight(*base, "/") + "/v1/fak/session/" + url.PathEscape(sessionID)
	client := &sessionOpenClient{http: http.DefaultClient, bearer: strings.TrimSpace(*bearer), localToken: strings.TrimSpace(*localToken)}
	var desc gateway.SessionClientDescriptor
	if err := client.call(http.MethodGet, endpoint+"/client", nil, &desc); err != nil {
		fmt.Fprintf(stderr, "fak session open: describe: %v\n", err)
		return 1
	}
	attachReq := gateway.SessionClientAttachRequest{ClientKind: "terminal", WorkspaceID: strings.TrimSpace(*workspace), Since: *since}
	var attached gateway.SessionClientAttachResponse
	if err := client.call(http.MethodPost, endpoint+"/attach", attachReq, &attached); err != nil {
		fmt.Fprintf(stderr, "fak session open: attach: %v\n", err)
		return 1
	}
	result := attached
	if strings.TrimSpace(*input) != "" {
		req := gateway.SessionClientActionRequest{AttachmentID: attached.AttachmentID, ExecutionEpoch: attached.Descriptor.ExecutionEpoch, Text: *input, Principal: "terminal"}
		if err := client.call(http.MethodPost, endpoint+"/input", req, &result); err != nil {
			fmt.Fprintf(stderr, "fak session open: input: %v\n", err)
			return 1
		}
	}
	if strings.TrimSpace(*decision) != "" {
		var response any
		if err := client.call(http.MethodPost, endpoint+"/decision", gateway.SessionClientDecisionRequest{AttachmentID: attached.AttachmentID, Decision: *decision}, &response); err != nil {
			fmt.Fprintf(stderr, "fak session open: decision: %v\n", err)
			return 1
		}
	}
	if *closeSession {
		if err := client.call(http.MethodPost, endpoint+"/close", gateway.SessionClientDetachRequest{AttachmentID: attached.AttachmentID}, nil); err != nil {
			fmt.Fprintf(stderr, "fak session open: close: %v\n", err)
			return 1
		}
		return 0
	}
	defer func() {
		_ = client.call(http.MethodPost, endpoint+"/detach", gateway.SessionClientDetachRequest{AttachmentID: attached.AttachmentID}, nil)
	}()
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(result)
		return 0
	}
	renderSessionOpen(stdout, result)
	return 0
}

type sessionOpenClient struct {
	http       *http.Client
	bearer     string
	localToken string
}

func (c *sessionOpenClient) call(method, endpoint string, body, dst any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, endpoint, rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.bearer != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearer)
	}
	if c.localToken != "" {
		req.Header.Set(gateway.SessionClientTokenHeader, c.localToken)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var envelope struct {
			Error struct {
				Code    any    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&envelope)
		return fmt.Errorf("%s: %s", fmt.Sprint(envelope.Error.Code), envelope.Error.Message)
	}
	if dst != nil {
		return json.NewDecoder(resp.Body).Decode(dst)
	}
	return nil
}

func renderSessionOpen(w io.Writer, result gateway.SessionClientAttachResponse) {
	d := result.Descriptor
	fmt.Fprintf(w, "fak session %s\n", d.SessionID)
	fmt.Fprintf(w, "  execution_epoch=%s event_head=%d attachment=%s input_lease=%t\n", d.ExecutionEpoch, d.EventHead, result.AttachmentID, result.InputLease)
	fmt.Fprintf(w, "  capability_digest=%s\n", d.CapabilityDigest)
	fmt.Fprintf(w, "  capabilities=%s\n", strings.Join(d.Capabilities, ","))
	for _, action := range d.Actions {
		if action.Available {
			fmt.Fprintf(w, "  action=%s available route=%s %s\n", action.ID, action.Method, action.Route)
		} else {
			fmt.Fprintf(w, "  action=%s unavailable=%s reason=%s handoff=%s\n", action.ID, action.UnavailableCode, action.UnavailableReason, action.Handoff)
		}
	}
	fmt.Fprintf(w, "  state=%s rev=%d pending=%s\n", d.State.Run, d.State.Rev, blankSessionOpen(d.Pending))
	if d.RecoveryDependency != "" {
		fmt.Fprintf(w, "  recovery_dependency=%s\n", d.RecoveryDependency)
	}
	fmt.Fprintf(w, "  terminal_bytes=%d terminal_digest=%s\n", d.Terminal.ByteLength, d.Terminal.Digest)
	if d.Terminal.Transcript != "" {
		fmt.Fprintln(w, "  terminal transcript:")
		fmt.Fprint(w, d.Terminal.Transcript)
		if !strings.HasSuffix(d.Terminal.Transcript, "\n") {
			fmt.Fprintln(w)
		}
	}
	for _, effect := range d.Effects {
		fmt.Fprintf(w, "  effect=%s verdict=%s check=%s\n", effect.ID, effect.Verdict, blankSessionOpen(effect.Check))
	}
	fmt.Fprintf(w, "  replay events=%d cursor=%d\n", len(result.Events), result.Cursor)
	for _, ev := range result.Events {
		fmt.Fprintf(w, "    #%d state=%s rev=%d\n", ev.Seq, ev.Run, ev.Rev)
	}
	fmt.Fprintf(w, "  browser=%s/open\n", d.Endpoint)
}

func blankSessionOpen(v string) string {
	if strings.TrimSpace(v) == "" {
		return "-"
	}
	return v
}
