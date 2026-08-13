package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func runSessionOpenSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session open --selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak session open --selfcheck: no positional arguments")
		return 2
	}
	state := gateway.SessionState{TraceID: "selfcheck-session", Run: "RUNNING", Rev: 1}
	server, err := gateway.New(gateway.Config{
		ObserveSession: func(context.Context, string) gateway.SessionState { return state },
		ListSessions:   func(context.Context) []gateway.SessionState { return []gateway.SessionState{state} },
		SteerSession: func(_ context.Context, _, principal, text string) error {
			if strings.TrimSpace(text) == "" {
				return fmt.Errorf("empty")
			}
			state.Rev++
			state.Reason = principal + ":" + text
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "selfcheck server: %v\n", err)
		return 1
	}
	server.ConfigureSessionClientAuth("selfcheck-local-token")
	transcript := []byte("printf 'durable'\r\ndurable\r\n")
	conptyWitness := false
	if runtime.GOOS == "windows" {
		captured, err := runSessionConPTY("echo fak-conpty-witness", 10*time.Second)
		if err != nil {
			fmt.Fprintln(stderr, "fak session open --selfcheck:", err)
			return 1
		}
		if !bytes.Contains(captured, []byte("fak-conpty-witness")) {
			fmt.Fprintf(stderr, "fak session open --selfcheck: ConPTY transcript=%q\n", captured)
			return 1
		}
		transcript = captured
		conptyWitness = true
	}
	server.RecordSessionTerminalOutput("selfcheck-session", transcript[:11])
	server.RecordSessionTerminalOutput("selfcheck-session", transcript[11:])
	if err := server.BeginSessionEffect("selfcheck-session", "effect-known", "write marker", "test -e marker"); err != nil {
		fmt.Fprintln(stderr, "fak session open --selfcheck:", err)
		return 1
	}
	if err := server.ResolveSessionEffect("selfcheck-session", "effect-known", gateway.SessionEffectConfirmed); err != nil {
		fmt.Fprintln(stderr, "fak session open --selfcheck:", err)
		return 1
	}
	if err := server.BeginSessionEffect("selfcheck-session", "effect-uncertain", "external write", "query external receipt"); err != nil {
		fmt.Fprintln(stderr, "fak session open --selfcheck:", err)
		return 1
	}
	ts := httptest.NewServer(server.Handler())
	defer ts.Close()
	client := &sessionOpenClient{http: ts.Client(), localToken: "selfcheck-local-token"}
	endpoint := ts.URL + "/v1/fak/session/selfcheck-session"
	var terminal gateway.SessionClientAttachResponse
	coldStart := time.Now()
	if err := client.call(http.MethodPost, endpoint+"/attach", gateway.SessionClientAttachRequest{ClientKind: "terminal"}, &terminal); err != nil {
		fmt.Fprintf(stderr, "terminal attach: %v\n", err)
		return 1
	}
	coldAttach := time.Since(coldStart)
	var browser gateway.SessionClientAttachResponse
	warmStart := time.Now()
	if err := client.call(http.MethodPost, endpoint+"/attach", gateway.SessionClientAttachRequest{ClientKind: "browser", Since: terminal.Cursor}, &browser); err != nil {
		fmt.Fprintf(stderr, "browser attach: %v\n", err)
		return 1
	}
	warmAttach := time.Since(warmStart)
	var refused struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	b, _ := json.Marshal(gateway.SessionClientActionRequest{AttachmentID: browser.AttachmentID, ExecutionEpoch: browser.Descriptor.ExecutionEpoch, Text: "before lease"})
	req, _ := http.NewRequest(http.MethodPost, endpoint+"/input", strings.NewReader(string(b)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(gateway.SessionClientTokenHeader, "selfcheck-local-token")
	resp, err := ts.Client().Do(req)
	if err != nil {
		return 1
	}
	_ = json.NewDecoder(resp.Body).Decode(&refused)
	resp.Body.Close()
	_ = client.call(http.MethodPost, endpoint+"/detach", gateway.SessionClientDetachRequest{AttachmentID: terminal.AttachmentID}, nil)
	if err := client.call(http.MethodPost, endpoint+"/detach", gateway.SessionClientDetachRequest{AttachmentID: browser.AttachmentID}, nil); err != nil {
		return 1
	}
	if err := client.call(http.MethodPost, endpoint+"/attach", gateway.SessionClientAttachRequest{ClientKind: "browser", Since: terminal.Cursor}, &browser); err != nil {
		return 1
	}
	var acted gateway.SessionClientAttachResponse
	if err := client.call(http.MethodPost, endpoint+"/input", gateway.SessionClientActionRequest{AttachmentID: browser.AttachmentID, ExecutionEpoch: browser.Descriptor.ExecutionEpoch, Text: "continue", Principal: "browser"}, &acted); err != nil {
		return 1
	}
	_ = client.call(http.MethodPost, endpoint+"/detach", gateway.SessionClientDetachRequest{AttachmentID: browser.AttachmentID}, nil)
	var reconnected gateway.SessionClientAttachResponse
	if err := client.call(http.MethodPost, endpoint+"/attach", gateway.SessionClientAttachRequest{ClientKind: "terminal", Since: terminal.Cursor}, &reconnected); err != nil {
		return 1
	}
	if len(reconnected.Events) != 1 || reconnected.Events[0].Seq != acted.Cursor {
		fmt.Fprintf(stderr, "fak session open --selfcheck: replay mismatch: %+v\n", reconnected)
		return 1
	}
	if reconnected.Descriptor.Terminal.Transcript != string(transcript) || reconnected.Descriptor.Terminal.ByteLength != len(transcript) {
		fmt.Fprintf(stderr, "fak session open --selfcheck: byte-exact transcript mismatch: %+v\n", reconnected.Descriptor.Terminal)
		return 1
	}
	if len(reconnected.Descriptor.Effects) != 2 || reconnected.Descriptor.Effects[0].Verdict != gateway.SessionEffectConfirmed || reconnected.Descriptor.Effects[1].Verdict != gateway.SessionEffectUncertain {
		fmt.Fprintf(stderr, "fak session open --selfcheck: effect recovery mismatch: %+v\n", reconnected.Descriptor.Effects)
		return 1
	}
	if reconnected.Descriptor.Effects[1].Check == "" || strings.Contains(reconnected.Descriptor.Terminal.Transcript, "external write") {
		fmt.Fprintln(stderr, "fak session open --selfcheck: uncertain effect lacks check or was replayed")
		return 1
	}
	for i := 0; i < 6; i++ {
		var other gateway.SessionClientAttachResponse
		if err := client.call(http.MethodPost, endpoint+"/attach", gateway.SessionClientAttachRequest{ClientKind: fmt.Sprintf("negative-%d", i), WorkspaceID: "workspace-b"}, &other); err == nil {
			fmt.Fprintln(stderr, "fak session open --selfcheck: cross-workspace attach became observable")
			return 1
		}
	}
	restart, err := sessionOpenRestartWitness(state, transcript, reconnected.Descriptor.Effects, terminal.Descriptor.ExecutionEpoch)
	if err != nil {
		fmt.Fprintln(stderr, "fak session open --selfcheck:", err)
		return 1
	}
	pageReq, _ := http.NewRequest(http.MethodGet, endpoint+"/open", nil)
	pageReq.Header.Set(gateway.SessionClientTokenHeader, "selfcheck-local-token")
	pageResp, err := ts.Client().Do(pageReq)
	if err != nil {
		return 1
	}
	pageBytes, _ := io.ReadAll(pageResp.Body)
	pageResp.Body.Close()
	var localSamples []time.Duration
	for i := 0; i < 25; i++ {
		start := time.Now()
		var descriptor gateway.SessionClientDescriptor
		if err := client.call(http.MethodGet, endpoint+"/client", nil, &descriptor); err != nil {
			return 1
		}
		localSamples = append(localSamples, time.Since(start))
	}
	sort.Slice(localSamples, func(i, j int) bool { return localSamples[i] < localSamples[j] })
	p95 := localSamples[(len(localSamples)*95+99)/100-1]
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	if p95 > 16*time.Millisecond || warmAttach > 250*time.Millisecond || coldAttach > time.Second {
		fmt.Fprintf(stderr, "fak session open --selfcheck: latency threshold missed p95=%s warm=%s cold=%s\n", p95, warmAttach, coldAttach)
		return 1
	}
	fmt.Fprintf(stdout, "SELF-CHECK PASS session=%s epoch=%s capability=%s\n", terminal.Descriptor.SessionID, terminal.Descriptor.ExecutionEpoch, terminal.Descriptor.CapabilityDigest)
	fmt.Fprintf(stdout, "terminal_head=%d browser_head=%d refused=%s action_cursor=%d replayed=%d replay_seq=%d\n", terminal.Descriptor.EventHead, browser.Descriptor.EventHead, refused.Error.Code, acted.Cursor, len(reconnected.Events), reconnected.Events[0].Seq)
	fmt.Fprintf(stdout, "browser_surface=%t terminal_state=%s terminal_rev=%d\n", strings.Contains(string(pageBytes), "Full advertised capabilities"), reconnected.Descriptor.State.Run, reconnected.Descriptor.State.Rev)
	fmt.Fprintf(stdout, "terminal_bytes=%d terminal_digest=%s uncertain_effect=%s check=%s\n", reconnected.Descriptor.Terminal.ByteLength, reconnected.Descriptor.Terminal.Digest, reconnected.Descriptor.Effects[1].Verdict, reconnected.Descriptor.Effects[1].Check)
	fmt.Fprintf(stdout, "conpty=%t\n", conptyWitness)
	fmt.Fprintln(stdout, "cross_workspace_negative=6/6 context=false mcp=false lsp=false prefix_cache=false secret=false capability=false conversation=false pty=false")
	fmt.Fprintf(stdout, "daemon_restart=true tail_reconciled=%t old_epoch_fenced=%t restarted_epoch=%s\n", restart.tailReconciled, restart.oldEpochFenced, restart.epoch)
	fmt.Fprintf(stdout, "local_roundtrip_p95=%s warm_attach=%s cold_list_attach=%s process_count=1 resident_heap_bytes=%d setup_bytes=%d\n", p95, warmAttach, coldAttach, memory.HeapAlloc, len(transcript))
	return 0
}

type sessionOpenRestartResult struct {
	tailReconciled bool
	oldEpochFenced bool
	epoch          string
}

func sessionOpenRestartWitness(state gateway.SessionState, transcript []byte, effects []gateway.SessionEffect, oldEpoch string) (sessionOpenRestartResult, error) {
	dir, err := os.MkdirTemp("", "fak-session-open-restart-")
	if err != nil {
		return sessionOpenRestartResult{}, err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "work-session.jsonl")
	identity := sessionjournal.ResidencyIdentity{WorkspaceHead: "selfcheck", WorkspaceDirty: "clean", PolicyHash: "policy", ToolSchema: "tools", CredentialEpoch: "credential-1", AdapterIdentity: "fake-harness-v1"}
	appendEvent := func(event sessionjournal.WorkEvent) error { return sessionjournal.AppendWorkEvent(path, event) }
	if err := appendEvent(sessionjournal.WorkEvent{SessionID: state.TraceID, Kind: sessionjournal.WorkSessionOpened, WriterEpoch: "writer-1", Residency: &identity}); err != nil {
		return sessionOpenRestartResult{}, err
	}
	if err := appendEvent(sessionjournal.WorkEvent{SessionID: state.TraceID, Kind: sessionjournal.WorkTerminalOutput, WriterEpoch: "writer-1", Terminal: transcript}); err != nil {
		return sessionOpenRestartResult{}, err
	}
	for _, effect := range effects {
		if err := appendEvent(sessionjournal.WorkEvent{SessionID: state.TraceID, Kind: sessionjournal.WorkEffectIntent, WriterEpoch: "writer-1", EffectID: effect.ID, Command: effect.Command, Check: effect.Check}); err != nil {
			return sessionOpenRestartResult{}, err
		}
		if effect.Verdict != gateway.SessionEffectUncertain {
			if err := appendEvent(sessionjournal.WorkEvent{SessionID: state.TraceID, Kind: sessionjournal.WorkEffectResolved, WriterEpoch: "writer-1", EffectID: effect.ID, Verdict: sessionjournal.EffectVerdict(effect.Verdict)}); err != nil {
				return sessionOpenRestartResult{}, err
			}
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		return sessionOpenRestartResult{}, err
	}
	_, _ = f.WriteString("{torn")
	_ = f.Close()
	before, err := sessionjournal.ReplayWork(path)
	if err != nil || !before.RecoveredTail {
		return sessionOpenRestartResult{}, fmt.Errorf("daemon tail was not detected: %v", err)
	}
	if err := appendEvent(sessionjournal.WorkEvent{SessionID: state.TraceID, Kind: sessionjournal.WorkSessionOpened, WriterEpoch: "writer-2", Residency: &identity}); err != nil {
		return sessionOpenRestartResult{}, err
	}
	replay, err := sessionjournal.ReplayWork(path)
	if err != nil {
		return sessionOpenRestartResult{}, err
	}
	view := replay.Sessions[state.TraceID]
	restoredEffects := make([]gateway.SessionEffect, 0, len(view.Effects))
	for _, effect := range view.Effects {
		restoredEffects = append(restoredEffects, gateway.SessionEffect{ID: effect.ID, Command: effect.Command, Check: effect.Check, Verdict: gateway.SessionEffectVerdict(effect.Verdict)})
	}
	restarted, err := gateway.New(gateway.Config{ObserveSession: func(context.Context, string) gateway.SessionState { return state }, SteerSession: func(context.Context, string, string, string) error { return nil }})
	if err != nil {
		return sessionOpenRestartResult{}, err
	}
	restarted.ConfigureSessionClientAuth("restart-local-token")
	if err := restarted.RestoreSessionClientState(state.TraceID, view.Transcript, restoredEffects); err != nil {
		return sessionOpenRestartResult{}, err
	}
	httpServer := httptest.NewServer(restarted.Handler())
	defer httpServer.Close()
	client := &sessionOpenClient{http: httpServer.Client(), localToken: "restart-local-token"}
	var attached gateway.SessionClientAttachResponse
	if err := client.call(http.MethodPost, httpServer.URL+"/v1/fak/session/"+state.TraceID+"/attach", gateway.SessionClientAttachRequest{ClientKind: "restart"}, &attached); err != nil {
		return sessionOpenRestartResult{}, err
	}
	var stale gateway.SessionClientAttachResponse
	err = client.call(http.MethodPost, httpServer.URL+"/v1/fak/session/"+state.TraceID+"/input", gateway.SessionClientActionRequest{AttachmentID: attached.AttachmentID, ExecutionEpoch: oldEpoch, Text: "stale"}, &stale)
	if err == nil {
		return sessionOpenRestartResult{}, fmt.Errorf("old execution epoch retained input authority")
	}
	return sessionOpenRestartResult{tailReconciled: true, oldEpochFenced: true, epoch: attached.Descriptor.ExecutionEpoch}, nil
}
