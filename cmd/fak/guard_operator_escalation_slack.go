package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const guardOperatorEscalationSlackSource = guardSessionThreadSource + ":operator-directed-escalate"

// routeGuardOperatorEscalationFailOpen is the Stop-hook disposition filter. Only the
// genuine human residue is routable; continue/warn/shadow remain silent.
func routeGuardOperatorEscalationFailOpen(sessionID string, disp guardStopDisposition, tr *guardStopTranscript) {
	if disp != stopDispOperatorDirectedEscalate {
		return
	}
	enqueueGuardOperatorEscalationFailOpen(sessionID, tr)
}

// enqueueGuardOperatorEscalationFailOpen routes a genuine HUMAN_RESIDUAL to the
// guarded session's existing Slack thread. It is deliberately best-effort: missing
// token, identity join, session row, or thread is a clean no-op on the Stop path.
func enqueueGuardOperatorEscalationFailOpen(sessionID string, tr *guardStopTranscript) {
	if strings.TrimSpace(sessionID) == "" || tr == nil || resolveGuardSessionsToken() == "" {
		return
	}
	regDir := resolveSweepRegDir("")
	traceByUUID, _ := resume.LoadIdentity(regDir)
	traceID := strings.TrimSpace(traceByUUID[strings.TrimSpace(sessionID)])
	if traceID == "" {
		return
	}
	res := guardsessions.Resolve(guardsessions.Load(regDir), traceID)
	if res.Matched != 1 || strings.TrimSpace(res.Row.Nonce) == "" {
		return
	}
	ob, err := openOutbox()
	if err != nil {
		return
	}
	// A single Stop can be re-fired by the harness. A deterministic nonce makes the
	// notification idempotent at the durable outbox boundary.
	sum := sha256.Sum256([]byte(sessionID + "\x00" + guardOperatorDirectedClassLabel(tr) + "\x00" + guardOperatorDirectedResolveText(tr)))
	nonce := "guard-escalate-" + hex.EncodeToString(sum[:12])
	if guardSlackOutboxHasNonce(ob, nonce) {
		return
	}
	_, err = ob.Enqueue(slackoutbox.Row{
		Nonce:       nonce,
		Channel:     resolveGuardSessionsChannel(),
		Text:        guardOperatorEscalationSlackText(tr),
		ParentNonce: res.Row.Nonce,
		Source:      guardOperatorEscalationSlackSource,
	})
	if err == nil {
		startGuardSessionThreadDrain()
	}
}

func guardOperatorEscalationSlackText(tr *guardStopTranscript) string {
	return fmt.Sprintf(":warning: *HUMAN_RESIDUAL* · `%s`\n%s",
		guardOperatorDirectedClassLabel(tr), guardOperatorDirectedResolveText(tr))
}

func guardSlackOutboxHasNonce(ob *slackoutbox.Outbox, nonce string) bool {
	snap, err := ob.Load()
	if err != nil {
		return false
	}
	for _, row := range snap.Rows {
		if row.Nonce == nonce {
			return true
		}
	}
	return false
}
