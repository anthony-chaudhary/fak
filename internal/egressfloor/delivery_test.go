package egressfloor

import (
	"strings"
	"testing"
)

// TestAdjudicatedDelivery is #2850's acceptance gate: an outbound message to a supported
// platform passes through the egress capability floor (allowlist-checked, redacted) and
// produces a witnessed delivery record — destination, redaction outcome, and verdict —
// before it is sent.
func TestAdjudicatedDelivery(t *testing.T) {
	// A policy permitting a named Slack channel and any Telegram chat, but no Discord.
	policy := DeliveryPolicy{Allow: map[Platform][]string{
		PlatformSlack:    {"C0ALLOWED"},
		PlatformTelegram: {AnyDestination},
	}}

	// The tokens the redactor must strip before anything leaves. The telegram token is
	// exactly <digits>:<35 url-safe chars>, matching the bot-token grammar.
	const slackTok = "xoxb-2411234567-abcdEFGH1234ijklMNOP"
	const tgTok = "1234567890:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw0"

	t.Run("allowed destination redacts and witnesses before send", func(t *testing.T) {
		rec := policy.Adjudicate(Delivery{
			Platform:    PlatformSlack,
			Destination: "C0ALLOWED",
			Message:     "deploy done, token " + slackTok + " fyi",
		})
		if !rec.Allowed {
			t.Fatalf("allowlisted destination must be allowed, got deny: %+v", rec)
		}
		if rec.Reason != 0 {
			t.Errorf("allowed record must carry no refusal code, got %d", rec.Reason)
		}
		if rec.RedactionCount == 0 || len(rec.Redactions) == 0 {
			t.Errorf("secret in message must be redacted, got count=%d classes=%v",
				rec.RedactionCount, rec.Redactions)
		}
		if strings.Contains(rec.RedactedMessage, slackTok) {
			t.Errorf("raw secret leaked into the message that would be sent: %q", rec.RedactedMessage)
		}
		if !strings.Contains(rec.RedactedMessage, "[REDACTED:slack-token]") {
			t.Errorf("expected a redaction placeholder, got %q", rec.RedactedMessage)
		}
		// The witness names WHERE, the VERDICT, and the REDACTION OUTCOME — and never the secret.
		w := rec.Witness()
		for _, want := range []string{"platform=slack", "dest=C0ALLOWED", "verdict=allow", "redactions=1"} {
			if !strings.Contains(w, want) {
				t.Errorf("witness %q missing %q", w, want)
			}
		}
		if strings.Contains(w, slackTok) {
			t.Errorf("witness leaked the raw secret: %q", w)
		}
	})

	t.Run("platform not on allowlist is denied but still redacted", func(t *testing.T) {
		rec := policy.Adjudicate(Delivery{
			Platform:    PlatformDiscord,
			Destination: "123",
			Message:     "here is my key " + slackTok,
		})
		if rec.Allowed {
			t.Fatalf("un-allowlisted platform must be denied, got allow: %+v", rec)
		}
		if rec.Reason != ReasonDeliveryBlock {
			t.Errorf("deny must cite ReasonDeliveryBlock (%d), got %d", ReasonDeliveryBlock, rec.Reason)
		}
		// Even a denied message must not carry the raw secret into the witness record.
		if strings.Contains(rec.RedactedMessage, slackTok) {
			t.Errorf("denied delivery still leaked the raw secret: %q", rec.RedactedMessage)
		}
		if !strings.Contains(rec.Witness(), "verdict=deny") {
			t.Errorf("witness must record the deny verdict: %q", rec.Witness())
		}
	})

	t.Run("allowed platform but destination off the list is denied", func(t *testing.T) {
		rec := policy.Adjudicate(Delivery{
			Platform:    PlatformSlack,
			Destination: "C0SECRET", // slack is allowed, but only for C0ALLOWED
			Message:     "hello",
		})
		if rec.Allowed {
			t.Fatalf("destination off the allowlist must be denied, got allow: %+v", rec)
		}
		if rec.Reason != ReasonDeliveryBlock {
			t.Errorf("deny must cite ReasonDeliveryBlock, got %d", rec.Reason)
		}
	})

	t.Run("wildcard destination permits any target on that platform", func(t *testing.T) {
		rec := policy.Adjudicate(Delivery{
			Platform:    PlatformTelegram,
			Destination: "any-chat-42",
			Message:     "bot token " + tgTok,
		})
		if !rec.Allowed {
			t.Fatalf("wildcard-permitted platform must allow any destination, got deny: %+v", rec)
		}
		if !strings.Contains(rec.RedactedMessage, "[REDACTED:telegram-bot-token]") {
			t.Errorf("telegram bot token must be redacted, got %q", rec.RedactedMessage)
		}
	})

	t.Run("deny by default: empty policy permits nothing", func(t *testing.T) {
		var empty DeliveryPolicy // nil Allow
		rec := empty.Adjudicate(Delivery{Platform: PlatformSlack, Destination: "C0ALLOWED", Message: "hi"})
		if rec.Allowed {
			t.Fatalf("nil-policy floor must deny by default, got allow: %+v", rec)
		}
	})

	t.Run("clean message on allowed destination sends unchanged", func(t *testing.T) {
		rec := policy.Adjudicate(Delivery{
			Platform:    PlatformSlack,
			Destination: "C0ALLOWED",
			Message:     "the nightly run is green",
		})
		if !rec.Allowed {
			t.Fatalf("clean allowed message must be allowed: %+v", rec)
		}
		if rec.RedactionCount != 0 || len(rec.Redactions) != 0 {
			t.Errorf("clean message must not be altered, got count=%d classes=%v",
				rec.RedactionCount, rec.Redactions)
		}
		if rec.RedactedMessage != "the nightly run is green" {
			t.Errorf("clean message body must pass through unchanged, got %q", rec.RedactedMessage)
		}
	})
}

// TestRedactSecretClasses pins each well-known secret shape the pre-send redactor strips,
// so a regression that stops matching a class is caught at origin rather than leaking.
func TestRedactSecretClasses(t *testing.T) {
	cases := []struct {
		name  string
		msg   string
		class string
	}{
		{"bearer", "Authorization: Bearer abcDEF123456ghiJKL", "bearer-token"},
		{"slack", "tok xoxb-2411234567-abcdEFGH1234ijklMNOP end", "slack-token"},
		{"github", "use ghp_" + strings.Repeat("a", 36) + " now", "github-token"},
		{"aws", "id AKIAIOSFODNN7EXAMPLE here", "aws-access-key-id"},
		{"openai", "key sk-" + strings.Repeat("A", 24) + " ok", "api-key"},
		{"pem", "-----BEGIN OPENSSH PRIVATE KEY-----", "private-key"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, count, classes := redactSecrets(tc.msg)
			if count == 0 {
				t.Fatalf("expected a redaction for %s, got none (out=%q)", tc.name, out)
			}
			found := false
			for _, c := range classes {
				if c == tc.class {
					found = true
				}
			}
			if !found {
				t.Errorf("expected class %q in %v", tc.class, classes)
			}
			if strings.Contains(out, "-----BEGIN") && tc.name != "pem" {
				t.Errorf("unexpected key header survived: %q", out)
			}
		})
	}

	if out, count, classes := redactSecrets("nothing secret here"); count != 0 || classes != nil || out != "nothing secret here" {
		t.Errorf("clean text must pass through: out=%q count=%d classes=%v", out, count, classes)
	}
}
