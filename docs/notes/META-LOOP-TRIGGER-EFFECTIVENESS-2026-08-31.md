# Meta-loop trigger effectiveness spine — 2026-08-31

Issue: #10352. This spine adds measurement only: `fak superloop trigger` emits a deterministic receipt and never launches, schedules, or mutates a loop.

## Inventory and observed gap

FAK currently has operator-triggered super loops, scheduled scout/watchdog loops, in-session dispatch loops, scorecard RSI loops, and control-pane polling. Each has local gates, but there was no shared record of whether an evaluation was early, timely, overdue, stale, duplicated, capacity-blocked, or too costly. The live 2026-08-31 baseline had 12 loops, 1,615 fires, 1,023 admissions, 592 refusals, eight loops with zero admissions, and a maximum refusal streak of 415. That proves scheduling opportunity and useful execution must be measured separately.

## Receipt and precedence

Schema `fak-loop-trigger/1` records demand, freshness, ownership overlap, capacity, timing, expected value, wall/attention cost, and stable evidence references. It uses closed decisions `RUN`, `SKIP`, `DEFER`, `MERGE`, `REROUTE` and closed reasons. The initial fail-closed precedence is:

1. no demand → `SKIP/NO_DEMAND`;
2. stale evidence → `DEFER/INPUT_STALE`;
3. existing owner → `MERGE/ALREADY_OWNED`;
4. insufficient capacity → `DEFER/NO_CAPACITY`;
5. cooldown not elapsed → `DEFER/COOLDOWN`;
6. expected value below floor → `SKIP/BELOW_VALUE_FLOOR`;
7. service window missed → `RUN/DEADLINE`;
8. otherwise → `RUN/DEMAND_READY`.

## Effectiveness measures

- **Frequency:** empty-pass rate, cooldown suppressions, and useful outcomes per evaluation.
- **Timing:** overdue-demand seconds and later trigger regret (run below value floor, or defer past service window).
- **Observability:** attributable-decision rate: receipt has reason, evidence references, owner/capacity state, and a later outcome link.
- **Drift:** stale-input rate plus producer/consumer contract mismatch rate.
- **Heaviness:** coordination tax = setup + wait + polling + review wall/attention cost per witnessed useful outcome.
- **Routing:** overlap/merge and reroute rates, with effect identity kept separate from loop identity.

Cadence changes require a minimum sample and an evaluation horizon; one dramatic run is not enough. Prefer event + debounce when authoritative transitions exist, deadline-aware checks when silence becomes harmful, and bounded backoff/jitter where polling is unavoidable.

## Field mechanisms borrowed

- Temporal makes schedule overlap policy explicit and exports skip/catch-up/buffer metrics: https://docs.temporal.io/schedule and https://docs.temporal.io/references/cluster-metrics
- Kubernetes CronJob records scheduled time and exposes deadline/concurrency policy: https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/
- Airflow distinguishes event/asset scheduling from time scheduling: https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/event-scheduling.html and https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/asset-scheduling.html
- OpenTelemetry separates bounded metric identity from high-cardinality event evidence: https://opentelemetry.io/docs/specs/otel/metrics/data-model/

## Operating envelope and next tuning step

The spine accepts explicit witnessed inputs and classifies one proposed trigger evaluation. It does not yet derive those inputs from every scheduler or join decisions to later outcomes. Next, bind registered loops to trigger metadata, append receipts to an existing ledger, replay historical decisions, and only then tune cadence or consolidate overlapping loops.
