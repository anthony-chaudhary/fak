# Session-history refresh SLO — 2026-08-19

Command:

```text
fak vcache session-history benchmark --sizes 1000,10000,100000 --repetitions 3
```

Privacy-safe real-corpus summary: [`../_witnesses/session-history-real-corpus-summary-2026-08-19.json`](../_witnesses/session-history-real-corpus-summary-2026-08-19.json) (404 sessions / 406 files; aggregate counts only).`r`n`r`nMachine-readable synthetic benchmark witness: [`../_witnesses/session-history-refresh-benchmark-2026-08-19.json`](../_witnesses/session-history-refresh-benchmark-2026-08-19.json).

The benchmark generates privacy-safe normalized synthetic sessions and runs the production `MineIndexed` refresh path in three modes: cold index, unchanged index, and one changed source. Ordinary unit tests run only a 20-session structural sentinel and do not time-gate CI.

Default-on admission SLOs are intentionally loose rollback thresholds, not performance claims:

- unchanged p95 <= 5 s at every declared scale;
- one-change p95 <= 5 s;
- unchanged reuse >= 99%;
- index <= 4 KiB/session;
- changed-source end-to-index lag <= 60 s.

On Linux amd64 / Go 1.26.6, all scales passed. At 100K sessions: cold p95 1.900 s, unchanged p95 0.832 s, one-change p95 0.890 s, 99,999/100,000 files reused, 514 B/session, measured peak Go heap 280 MB, and 0.822 s change-to-index completion. These values describe this synthetic fixture and host only; they are not provider, billing, or universal hardware claims.

Re-run before changing default refresh cadence or storage format. Roll back default-on admission if any declared SLO fails twice on a representative host.

