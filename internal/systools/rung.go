package systools

import (
	"context"
	"net/url"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Caps advertises no optional capabilities.
func (t *Toolset) Caps() []abi.Capability { return nil }

// Adjudicate decides one proposed call. It checks ownership, validates arguments and policy,
// pins the engine, and sets ReadOnlyHint = true.
func (t *Toolset) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungName}
	}
	engineID, mine := engineFor(c.Tool)
	if !mine {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungName}
	}
	if err := ctx.Err(); err != nil {
		return t.deny(refuse(CodeCanceled, err.Error()), c.Tool)
	}
	if r := t.admit(ctx, c); r != nil {
		return t.deny(r, c.Tool)
	}

	c.Engine = engineID
	if c.Meta == nil {
		c.Meta = make(map[string]string)
	}
	c.Meta["readOnlyHint"] = "true"
	if c.Tool != ToolGetTime {
		c.Meta["idempotentHint"] = "true"
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: RungName}
}

func (t *Toolset) admit(ctx context.Context, c *abi.ToolCall) *Refusal {
	body := bytesOf(ctx, c.Args)
	switch c.Tool {
	case ToolGetTime:
		var a GetTimeArgs
		if r := decodeArgs(body, &a); r != nil {
			return r
		}
		if r := a.Validate(); r != nil {
			return r
		}
	case ToolFetchWeb:
		var a FetchWebArgs
		if r := decodeArgs(body, &a); r != nil {
			return r
		}
		if r := a.Validate(); r != nil {
			return r
		}
		u, err := url.Parse(a.URL)
		if err != nil || u.Host == "" {
			return refuse(CodeMalformed, "fetch_web: invalid url: "+a.URL)
		}
		host := u.Hostname()
		if !t.domainAllowed(host) {
			return refuse(CodePolicyBlock, "fetch_web: domain "+host+" is not in the allowlist")
		}
		if !t.allowPrivateIPs {
			if r := t.checkSSRF(ctx, host); r != nil {
				return r
			}
		}
	case ToolWebSearch:
		var a WebSearchArgs
		if r := decodeArgs(body, &a); r != nil {
			return r
		}
		if r := a.Validate(); r != nil {
			return r
		}
	}

	if t.policy.Allow != nil && !t.policy.Allow[c.Tool] {
		return refuse(CodeDefaultDeny, "no policy admits tool "+c.Tool)
	}
	return nil
}

func (t *Toolset) deny(r *Refusal, tool string) abi.Verdict {
	r.Tool = tool
	reason := r.Reason
	if reason == abi.ReasonNone {
		reason = abi.ReasonPolicyBlock
	}
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: reason,
		By:     RungName,
		Meta:   map[string]string{"code": r.Code, "tool": tool, "detail": r.Detail},
	}
}
