# Slack helper ownership

Reusable Slack transport and control infrastructure is canonical in
[`anthony-chaudhary/slack-helpers`](https://github.com/anthony-chaudhary/slack-helpers).
Its ownership matrix is `docs/CANONICAL-OWNERSHIP.md` in that repository.

This public FAK repository intentionally remains canonical for FAK product
behavior: `fak slack`, guard escalation, cadence/trajectory/health commands,
the durable outbox, and FAK-specific watchdog digests. Those features use FAK
Go types and policy and are not generic helper-library code.

When adding a Slack-facing feature:

- put reusable API, file-transfer, reading/watching, formatting, report-delivery,
  or remote-control behavior in `slack-helpers`;
- keep FAK command semantics and policy here;
- keep credentials and channel IDs in deployment configuration;
- do not create another independent generic Slack transport in this repo.
