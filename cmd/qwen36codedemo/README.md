# qwen36codedemo

Browser coding-agent spine for epic #4762 / issue #4770.

- Backend mode runs beside the authenticated pure-fak Qwen3.6 gateway and requires both a gateway bearer and an independent edge secret.
- Edge mode runs on Cloud Run, serves the embedded browser UI over managed HTTPS, and proxies API calls with the edge secret injected server-side.
- Neither secret appears in browser JavaScript or HTML.

The cache panel reports observed counters only. It deliberately does not claim a performance advantage until the tuned benchmark artifact passes `fak claim-check`.
