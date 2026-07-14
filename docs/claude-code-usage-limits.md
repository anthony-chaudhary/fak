---
title: "Claude Code usage limits: shared limits, status, and resets"
description: "Direct answers about Claude Code usage limits, shared Claude capacity, the /status command, session and weekly resets, and what to do when usage is exhausted."
permalink: /claude-code-usage-limits/
---

# Claude Code usage limits

**Claude Code usage on a Pro or Max subscription shares capacity with Claude on the web,
desktop, and mobile.** Work done in any of those surfaces can reduce what remains for Claude
Code. Claude Code's `/status` command can show account and plan information, but the
authenticated usage page is the authoritative place to review current usage and reset times.

There is no trustworthy universal “messages per day” number. Usage varies with plan, model,
conversation and repository context, tool use, and concurrent Claude sessions. Anthropic can
also apply both shorter session limits and longer weekly limits. For the current provider
contract, see [Use Claude Code with your Pro or Max plan](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan)
and [Usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices).

## Does Claude Code have a usage limit?

Yes. Claude Code requests consume the allowance attached to the signed-in subscription or API
account. On Pro and Max subscriptions, the allowance is shared with ordinary Claude use rather
than reserved exclusively for the terminal. A busy Claude web conversation can therefore
leave less capacity for Claude Code, and a long coding session can leave less for Claude on the
web.

API-based Claude Code use follows API billing and rate-limit contracts instead of a Pro or Max
subscription allowance. First identify which authentication and billing path the client is
using; otherwise a quota diagnosis can mix two different products.

## How do I check Claude Code usage?

Run `/status` inside Claude Code to inspect the current client account and plan details. Then
open Claude's authenticated usage view for the exact account doing the work to see remaining
capacity and reset timestamps. The provider view is stronger evidence than a fixed allowance
quoted by a third-party page because account policies can change.

If Claude Code is wrapped by fak, the gateway summary reports requests and tool decisions, not
Anthropic's remaining quota. Use those local records to explain what the worker attempted, and
use Claude's usage view to explain how much provider capacity remains.

## Is Claude Code usage separate from Claude web usage?

No, not when Claude Code is authenticated through the same Pro or Max subscription. Claude on
web, desktop, mobile, and Claude Code draw from shared usage. Parallel sessions on those
surfaces can therefore make a Claude Code limit appear to arrive early.

A separately billed API account is a different capacity path. Do not assume that buying API
credits increases a subscription allowance or that a subscription reset changes API rate
limits.

## When does the Claude Code usage limit reset?

Use the reset timestamp displayed for the account. Claude may enforce more than one window,
including a shorter session window and longer weekly limits. A session window reopening does
not prove that a weekly model-specific or aggregate limit has reopened.

Record the exact observed timestamp, account, plan, and exhausted limit. Avoid hard-coding a
five-hour or weekly countdown into automation: provider policy and account state are the
source of truth.

## Why did Claude Code usage run out quickly?

Common causes include large repositories or files in context, long conversation history,
agentic tool loops, repeated failed builds, parallel Claude sessions, and use of a more
compute-intensive model. A message count misses all of these workload differences.

Before retrying, inspect the failure and stop loops that cannot succeed. Narrow the task,
preserve the objective and exact unfinished state, and resume after the displayed reset or
through an explicitly available capacity path. Repeatedly relaunching the same exhausted
account does not create capacity.

## What should I do when Claude Code reaches its usage limit?

1. Verify the signed-in account and billing path with `/status`.
2. Read the exact exhausted limit and reset time from Claude's usage view.
3. Stop automatic retries against that unavailable capacity.
4. Save the objective, changed files, test state, and next command.
5. Resume after reset, select another provider-offered model or capacity option, or use a
   separately enabled API/extra-usage path with its billing implications understood.

Fak can preserve that handoff and route from observed cooldown evidence, but it cannot reset
or bypass Anthropic's limit. See [Claude usage and usage limits](claude-usage-limits.md) for the
broader distinction between account usage and context length.

## Does fak reduce Claude Code usage?

Fak can reduce avoidable work by blocking destructive tool calls, stopping futile retries,
reusing safe setup, and preserving recoverable state. It does not guarantee a particular
quota saving, and it cannot alter the provider's meter. Any efficiency claim must compare the
same completed task, including retries and quality, against the real alternative.

## Quick answers

| Question | Answer |
|---|---|
| Does Claude Code have a usage limit? | Yes; it consumes the allowance of its subscription or API path. |
| Is Pro/Max Claude Code usage separate? | No; it is shared with Claude web, desktop, and mobile on that subscription. |
| How do I check it? | Use `/status` for client account details and Claude's authenticated usage page for capacity and reset times. |
| When does it reset? | At the timestamp shown for the exhausted limit; shorter and weekly windows can differ. |
| Can retries bypass it? | No. Stop futile retries and resume through an available capacity path. |
| Can fak reset it? | No. Fak can manage the workload around observed capacity, not change the quota. |

## Related guides

- [Claude usage and usage limits](claude-usage-limits.md)
- [Claude extra usage and spending limits](claude-extra-usage.md)
- [Claude Code integration](integrations/claude.md)
- [Managed-context continuous usage](managed-context-continuous-usage.md)
