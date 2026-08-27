# 100k-way micro-fleet economics spine — 2026-08-27

Issue #9253 now has a deterministic accounting primitive, not a performance result.

`internal/microfleeteconomics` compares equal-task, equal-quality receipts using **total micro-USD per accepted result**. Attempted branches are reported but never used as the denominator. The ledger separately includes useful work, branching, cache construction, cache hits, queue delay, verification, fan-in, cancellation, failures, retries, recovery, and stragglers.

Physical quantities remain visible: compute joules, machine milliseconds, network bytes, and storage byte-seconds. Caller-supplied integer rates convert those quantities to micro-USD; direct micro-USD can represent a priced resource outside that fixed list. Checked integer arithmetic rejects overflow instead of wrapping.

The JSON fixtures are synthetic reversal witnesses only. Under one declared rate/cost envelope, shared-cache 100,000-way fan-out is cheaper per accepted result than width 1 and width 1,000. Under another, coordination and low acceptance make width 1 cheaper than width 100,000. These fixtures prove the accounting can express both outcomes; they do **not** claim fak or any model reaches those costs, quality, acceptance, or throughput in production.

A real benchmark can populate the same receipts only after measuring the named physical resources and operation counts under a matched task and quality envelope. Native-inference evidence must name the fak-native engine and follow `docs/native-inference-goal.md`.
