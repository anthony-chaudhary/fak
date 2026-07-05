package main

// accounts_rehome.go — `fak accounts rehome`, the operator "switch seat NOW" button on a
// LIVE guarded session. It POSTs /v1/fak/account/rehome on the guard's gateway (the addr
// printed in the guard banner), which forces the session onto the next available account —
// the on-demand form of the 403-triggered account failover in guard_account_failover.go.
// DISTINCT from a tombstone's rehome target (`fak accounts remove --rehome-to`): nothing in
// the registry changes here; only the live session's serving seat moves, and the swap takes
// effect on that session's next upstream request.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// runAccountsRehome POSTs the seat switch and renders the from->to result. Exit 0 on an
// applied swap, 1 on a transport error or a gateway refusal (404 no roster in force / 409
// no available sibling), matching the accounts-verb exit convention.
func runAccountsRehome(stdout, stderr io.Writer, addr, key, reason string, asJSON bool) int {
	payload, err := json.Marshal(map[string]string{"reason": strings.TrimSpace(reason)})
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts rehome: %v\n", err)
		return 1
	}
	url := strings.TrimRight(strings.TrimSpace(addr), "/") + "/v1/fak/account/rehome"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts rehome: %v\n", err)
		return 1
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts rehome: %v\n  is a `fak guard` session running at %s? Pass --addr with the gateway URL from the guard banner (or set FAK_ADDR).\n", err, addr)
		return 1
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxSessionRespBytes))
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts rehome: reading response: %v\n", err)
		return 1
	}
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(stderr, "fak accounts rehome: %s\n", rehomeErrorMessage(resp.StatusCode, raw))
		return 1
	}
	if asJSON {
		stdout.Write(raw)
		if !bytes.HasSuffix(raw, []byte("\n")) {
			fmt.Fprintln(stdout)
		}
		return 0
	}
	var res gateway.AccountRehome
	if err := json.Unmarshal(raw, &res); err != nil {
		fmt.Fprintf(stderr, "fak accounts rehome: unexpected response: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "rehomed: %s -> %s (reason %s)\n", rehomeSeatLabel(res.From, res.FromEmail), rehomeSeatLabel(res.To, res.ToEmail), res.Reason)
	fmt.Fprintln(stdout, "the swap takes effect on the session's next upstream request; the seat left is excluded from auto-reselection for the rest of the session")
	return 0
}

// rehomeSeatLabel renders one seat for the human line: "name (email)" when the email is
// known, else just the name.
func rehomeSeatLabel(name, email string) string {
	if email != "" {
		return name + " (" + email + ")"
	}
	return name
}

// rehomeErrorMessage extracts the gateway's error-envelope message for a non-200, falling
// back to the bare status when the body is not the envelope. The message is fak-authored
// operator guidance (the gateway never forwards upstream bytes on this route).
func rehomeErrorMessage(status int, raw []byte) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &env) == nil && strings.TrimSpace(env.Error.Message) != "" {
		return fmt.Sprintf("HTTP %d: %s", status, env.Error.Message)
	}
	return fmt.Sprintf("HTTP %d from the gateway", status)
}
