# Frozen paired harness-creation task packet

This file is the canonical task packet for every eligible ten-minute comparison pair in
`study.json`. Hash these exact UTF-8 bytes as SHA-256 and record the result in both arm
receipts. The clock, assistance, failure-denominator, fresh-directory/cache, order, and
comparison-envelope rules are those in `participant-handoff.md` and apply to both arms.

The common need is fixed: create a read-only customer-support harness whose offline check
returns `readonly support ready: refund status` and reports that mutation is not allowed.
Do not substitute another need after assignment.

## fak arm

1. Retrieve the declared fak artifact.
2. Run `fak harness init` outside a fak checkout to create the product.
3. Edit only field values returned by `product.DefaultConfig` and, if needed, the body of
   `product.OfflineReply`; preserve those signatures.
4. Build the generated product and pass its offline `--selfcheck`.
5. Read `harness.lock.json` and record its exact upgrade command.

Success requires the generated product to build, its offline selfcheck to pass with the
fixed read-only support need represented, and the exact upgrade command to be recorded.

## tuned-baseline arm

Use the frozen `create-mastra@1.25.0` procedure and checked-in assets in
`baselines/create-mastra-1.25.0/README.md`. That card's exact `BASELINE_SELFCHECK ok` line,
production build, unchanged asset hashes, and upgrade command are the success boundary.

Both arms include artifact retrieval/install and customization time. A timeout, command
failure, abandoned run, wrong-file edit, failed selfcheck, or missing upgrade command is a
failed eligible attempt and remains in the denominator.
