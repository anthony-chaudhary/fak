# `#grafana` channel link registry

[`links.json`](links.json) is the committed corpus the `#grafana` Slack surface
folds into a rollup card. It holds the **long-lived dashboard and debug links**
the channel re-posts — the twin of `experiments/benchmark/catalog.json` for the
bench surface. Snapshots (the other thing the channel carries) are posted
ad-hoc and are *not* listed here; they are exported from Grafana and posted with
`fak grafana post --snapshot`.

## What lands in `#grafana`

- **Snapshots** — `fak grafana post --snapshot --title … --url <snapshot> --dashboard … --range …`
  posts an exported Grafana snapshot link with its context. The operator exports
  the share/snapshot link from Grafana; fak does not pull it (inbound ingestion is
  a follow-on, [#1298](https://github.com/anthony-chaudhary/fak/issues/1298)).
- **A single dashboard link** — `fak grafana post --link <uid>` posts one entry
  from this registry.
- **A links rollup** — `fak grafana post --rollup [all|public-demo|debug]` folds
  this registry, grouped by category.

## Schema — `fak-grafana-links/1`

```json
{
  "schema": "fak-grafana-links/1",
  "base_url": "http://localhost:3000",
  "links": [
    {
      "title": "FAK Gateway Observability",
      "uid": "fak-gateway-observability",
      "category": "debug",
      "lifetime": "stack-local",
      "description": "one line: what the dashboard shows",
      "source": "tools/grafana/dashboards/fak-gateway-observability.json"
    }
  ]
}
```

| field         | meaning |
|---------------|---------|
| `base_url`    | Grafana base used to resolve a `uid`-only link into `base_url/d/<uid>`. Override per-post with `--base-url`. |
| `title`       | human label shown on the card. |
| `uid`         | Grafana dashboard uid. Resolved against `base_url` unless `url` is set. |
| `url`         | absolute link. Wins over `uid` — use it for a public-demo dashboard on a different host. |
| `category`    | `public-demo` (long-lived demo) · `debug` (triage view) · `rollup` (saved overview). Drives rollup grouping. |
| `lifetime`    | `long-lived` · `stack-local` · `ephemeral`. Advisory; shown on the card. |
| `description` | one-line summary. |
| `source`      | provenance — the dashboard JSON the entry came from. |

## Seeded from the real provisioned dashboards

The initial entries are the five dashboards
[`tools/grafana`](../../tools/grafana) provisions (real `uid`s, `category:
debug`, `lifetime: stack-local`) — they resolve against a local
`http://localhost:3000` stack (`tools/grafana/up.sh`, login `admin` / `fleet`).
**No URL here is fabricated.** To publish a long-lived public-demo dashboard,
add an entry with `category: public-demo`, `lifetime: long-lived`, and the
absolute `url` of the public Grafana — point `base_url` at a tailnet/public
Grafana, or set `url` per link.
