package main

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// guardCodexAuthHTTPTimeout bounds the owned client used for the ChatGPT
// subscription backend. It is generous because turns stream, but finite so a stalled
// endpoint cannot hang the guard forever.
const guardCodexAuthHTTPTimeout = 10 * time.Minute

// configureChatCodexSubscription binds an explicitly selected native Responses
// planner to the same account credential source used by the Codex guard adapter.
func configureChatCodexSubscription(p *agent.HTTPPlanner, codexHome string) error {
	if p == nil || p.Provider != agent.ProviderOpenAIResponses || strings.TrimRight(p.BaseURL, "/") != guardCodexChatGPTBackendBaseURL {
		return fmt.Errorf("--codex-auth requires --provider openai-responses and --base-url %s", guardCodexChatGPTBackendBaseURL)
	}
	cred, err := resolveCodexSubscriptionCredential(codexHome)
	if err != nil {
		return err
	}
	if cred.AuthMode != "chatgpt" {
		return fmt.Errorf("--codex-auth requires a ChatGPT subscription login")
	}
	// Pin the resolved home even when discovery selected it. Scheduled jobs must
	// refresh their own seat without silently moving onto a sibling account.
	p.APIKeyFunc, p.ExtraHeadersFunc = newCodexSubscriptionRefreshers(filepath.Dir(cred.Source), cred)
	p.APIKey = ""
	p.ExtraHeaders = nil
	p.ForceResponsesStream = true
	// A redirect must not carry the subscription credential to another endpoint.
	client := p.Client
	if client == nil {
		// Never inherit http.DefaultClient (no timeout); own a bounded clone so a stalled
		// subscription endpoint cannot hang the turn forever.
		client = &http.Client{Timeout: guardCodexAuthHTTPTimeout}
	}
	ownedClient := *client
	ownedClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	p.Client = &ownedClient
	return nil
}
