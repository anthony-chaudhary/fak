package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

const guardLoginHijackPollInterval = 2 * time.Second

// guardWatchLoginHijack watches the guarded seat's credential file for an in-harness /login.
// The OAuth write is the event boundary: after it changes, probe the credential actually served
// and compare it with the registry binding. This turns the library detector into an at-login
// warning rather than requiring a later manual `fak accounts check`.
func guardWatchLoginHijack(seatDir, registryPath string, probe accounts.IdentityProber, warn io.Writer, interval time.Duration) func() {
	if strings.TrimSpace(seatDir) == "" || strings.TrimSpace(registryPath) == "" || probe == nil {
		return func() {}
	}
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		return func() {}
	}
	var seat accounts.Home
	found := false
	wantDir := filepath.Clean(seatDir)
	for _, h := range reg.Homes {
		if h.Active() && strings.EqualFold(filepath.Clean(h.Dir), wantDir) {
			seat, found = h, true
			break
		}
	}
	if !found || strings.TrimSpace(seat.Identity.Email) == "" && strings.TrimSpace(seat.Identity.AccountUUID) == "" {
		return func() {}
	}
	credPath := filepath.Join(seatDir, ".credentials.json")
	baseline := loginCredentialStamp(credPath)
	if interval <= 0 {
		interval = guardLoginHijackPollInterval
	}
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		last := baseline
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				next := loginCredentialStamp(credPath)
				if next == last || next == "" {
					continue
				}
				last = next
				report := accounts.DetectLoginHijack(seat.Name, seatDir, seat.Identity, probe)
				if report.Hijacked() {
					fmt.Fprintf(warn, "fak guard: WARNING: login identity hijack: seat %s expected %s but now serves %s; run `fak accounts enroll-current --name <new-seat-name>` to enroll the new identity, then restore or rehome %s\n",
						seat.Name, loginIdentityLabel(report.Expected), loginProbedIdentityLabel(report.Actual), seat.Name)
				}
			}
		}
	}()
	return func() { close(stop); <-done }
}

func loginCredentialStamp(path string) string {
	st, err := os.Stat(path)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%d:%d", st.ModTime().UnixNano(), st.Size())
}

func loginProbedIdentityLabel(id accounts.ProbedIdentity) string {
	return loginIdentityLabel(accounts.Identity{Email: id.Email, AccountUUID: id.AccountUUID})
}

func loginIdentityLabel(id accounts.Identity) string {
	if strings.TrimSpace(id.Email) != "" && strings.TrimSpace(id.AccountUUID) != "" {
		return id.Email + " (" + id.AccountUUID + ")"
	}
	if strings.TrimSpace(id.Email) != "" {
		return id.Email
	}
	if strings.TrimSpace(id.AccountUUID) != "" {
		return id.AccountUUID
	}
	return "unknown identity"
}

func guardStartLoginHijackWatch(credPath string, warn io.Writer) func() {
	homeDir, _ := os.UserHomeDir()
	seatDir := filepath.Dir(credPath)
	registryPath := guardDefaultAccountsRegistryPath(homeDir)
	probe := func(token string) (accounts.ProbedIdentity, error) {
		return accounts.ProbeToken(nil, accounts.DefaultProfileURL, token)
	}
	return guardWatchLoginHijack(seatDir, registryPath, probe, warn, guardLoginHijackPollInterval)
}
