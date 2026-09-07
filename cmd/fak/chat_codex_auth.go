package main

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

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
	client := http.DefaultClient
	if p.Client != nil {
		client = p.Client
	}
	ownedClient := *client
	ownedClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	p.Client = &ownedClient
	return nil
}
