# Actionable CI base-red dogfood readout — 2026-08-27

Issue: [#9466](https://github.com/anthony-chaudhary/fak/issues/9466)  
Command surface: `cmd/fak@r3454+g2f4dd8c1e`

## Run

This was a live repository run, not a fixture:

```console
$ fak release status --json --skip-cut-plan
```

The command exited 0 with parseable JSON and no stderr. The capture window was
2026-08-27T17:37Z–17:38Z. The relevant live result was:

| Field | Captured value |
|---|---|
| `ok` / `next_action.kind` | `false` / `fix_ci` |
| `rolling.ci_on_head.status` | `red` |
| decisive base-red run | `33083136896`, `a29d5cda785855a2e47ad35180bed51f59465e75`, `failure` |
| `rolling.ci_diagnosis.status` | `not_failed` |
| diagnosis reason | `head ci.yml run conclusion is queued` |
| diagnosis-selected run | `33099361387`, `3c3171ab2a9a6ec2a43e5613fb0fd2eaec9d6bf1`, `queued` |
| emitted package work units | 0 |

The run records are independently queryable with:

```console
$ gh run view 33083136896 --json databaseId,headSha,status,conclusion,url
$ gh run view 33099361387 --json databaseId,headSha,status,conclusion,url
```

At 2026-08-27T17:42:11Z those commands still reported the first run completed
with `failure` and the second run `queued`. The two public run records therefore
preserve the state distinction used by the live readout.

A second exact-command capture from clean detached head
`e0dd0298264d5c0b01face541ae1ef4de26bfbe7` exited 0 with no stderr. Its JSON was
79,980 bytes (`sha256:7f68d7b3f8a700167a35badd4106b16df4f88e9f60215d5ea88c0901175a9bd7`).
By then the selected fallback was decisive failed run `33084541915`, and the
diagnosis was `undifferentiated`; this later state did not invalidate the
queued-head ordering captured above.

## Outcome

The release verdict and diagnosis disagreed about the relevant run. The release
status correctly declared the inherited base red, but diagnosis treated the
newer queued run as proof that CI had not failed. Consequently it produced no
package work unit for the decisive failed ancestor.

Against #9466's target envelope, the shipped classifier's 17 workflow families
were present, but the live run produced 0 of the expected package work units and
0 correct exact bindings to the decisive red run. This is a product defect, not
a missing dogfood input.

Filed defect: [#9496](https://github.com/anthony-chaudhary/fak/issues/9496), marker
`fanout-releasestatus-spine-1e7d2968a994449e-dogfood-self-run-defect-queued-head`.
The defect owns the deterministic regression and implementation change; this
issue records only the real run and its finding.
