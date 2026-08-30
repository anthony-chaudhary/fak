---
title: "Support and maturity terms"
description: "This map positions support-boundary names that otherwise look interchangeable. A support result describes one attempted outcome;"
---
# Support and maturity terms

This map positions support-boundary names that otherwise look interchangeable. A **support result** describes one attempted outcome; a **support descriptor** describes a capability surface; a **maturity rung** summarizes accumulated evidence; and a **coverage control** selects or bounds the evidence collection itself.

- **OutcomeUnsupported** is the per-attempt result that the requested outcome is outside the witnessed envelope. It is not the aggregate **SupportMaturity** rung.
- **ReasonUnsupportedActiveCacheCapability** is the cache-specific cause attached to an unsupported outcome. It is narrower than **OutcomeUnsupported**, which is the result class.
- **GuardSupport** describes compatibility at the guard integration boundary. It is not the product-wide M0-M7 **SupportMaturity** rung.
- **DefaultCoverage** selects fallback breadth before a scan. A **CoverageReport** records evidence after a scan.
- **SeatCoverageMaxAgeMin** bounds how old seat-coverage evidence may be. **DefaultCoverageScanCap** bounds how many items a scan may inspect.
- **SOTACoverageScorecard** is the evaluator that computes and explains state-of-the-art coverage. **sotacoverage** is the resulting coverage value.


### fak_native_receipt_signal_supported (latest-signal authority gauge)

fak_native_receipt_signal_supported is the labeled Prometheus gauge that reports whether the latest admitted native receipt carried authoritative support for each prefill, decode, queue, kernel, and transfer signal.

**Distinct from:** It reports per-signal support on the latest receipt; it is not the broad support concept and not the cumulative unsupported-receipt exclusion counter.


### fak_native_receipt_unsupported_total (excluded native-receipt counter)

fak_native_receipt_unsupported_total is the Prometheus counter of native receipt observations excluded from native performance totals because the receipt lacks the required supported or authoritative evidence.

**Distinct from:** It counts cumulative excluded receipt observations; it is not the latest-receipt per-signal support gauge, which reports a current labeled capability bit.
