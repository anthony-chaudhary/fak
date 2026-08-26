# Live dashboard LCD demo captured output

Command, run from the repository root:

```console
go run ./cmd/dashboarddemo -selfcheck
```

The block below is the complete deterministic stdout receipt consumed by the
package regression test. Gateway request diagnostics are emitted separately on
stderr.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{"schema":"fak.dashboarddemo-selfcheck/1","verdict":"pass","homepage_status":200,"health_status":200,"metrics_status":200,"refresh_seconds":5,"rich_dashboard_count":9,"rich_dashboards_lazy":true}
```
<!-- END SELFCHECK OUTPUT -->
