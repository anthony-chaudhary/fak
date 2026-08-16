# Tuned comparison baseline: create-mastra 1.25.0

This is the frozen next-best alternative for the harness-creation timing study. Mastra is a
maintained public agent framework with a one-command starter, user-owned TypeScript seam,
dependency lock, and production build. It is compared as a product-creation workflow, not
as a claim that its runtime or model quality is globally best.

## Pinned sources

- Generator: `create-mastra@1.25.0`
- Runtime selected by that generator: `@mastra/core@1.59.0`
- Source: <https://github.com/mastra-ai/mastra>
- Node prerequisite: `>=22.13.0`

## Frozen task card

The clock starts when the participant opens this card. On a clean npm cache:

1. Run `npx --yes create-mastra@1.25.0 product --empty --no-git --no-skills`.
2. Copy `user-harness.ts` and `selfcheck.ts` from this directory into
   `product/src/mastra/`; these are the stable task-card assets, equivalent to editing the
   generated fak product's public user-owned config seam.
3. Run `node --experimental-strip-types src/mastra/selfcheck.ts`.
4. Run `npm run build`.
5. Identify upgrade command
   `npm install --save-exact @mastra/core@1.59.0 && npm install --save-dev --save-exact mastra@1.25.0`.
6. Identify rollback: restore `package.json` and `package-lock.json`, reinstall with
   `npm ci`, and restore the two user-owned TypeScript files.

Success requires the exact `BASELINE_SELFCHECK ok` line, a successful production build,
and the two task-card files' hashes unchanged across an `npm install` rerun. Assistance,
clock, failure denominator, and transcript rules are identical to the fak task card.

The asset-copy step is intentionally explicit and timed. It gives both workflows a frozen,
reviewable customization rather than asking a participant to invent code under the clock.
It does not include an LLM call, API key, or network-dependent runtime selfcheck.
