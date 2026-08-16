# Windows clean Mastra baseline calibration 1

Issue: #6969. Participant class: maintainer calibration; independent: false.

This calibration freezes the runnable tuned comparison workflow before any unfamiliar
participant timing. It does not count toward either promotional denominator.

## Result

- Generator: `create-mastra@1.25.0`; generated runtime: `@mastra/core@1.59.0`.
- Environment: Windows amd64, Node `v24.8.0`, empty dedicated npm cache.
- Install: 28.003 s.
- Offline selfcheck: 0.139 s.
- Production build: 14.155 s.
- Rerun/upgrade-shape preservation: 1.586 s.
- Total elapsed: 44.157 s.
- Outcome: success; owned files: 2; rebuilds: 1; failures/help requests: 0/0.
- User-owned SHA before/after rerun:
  `87f15ef122d82d9f7bd306f3522699758af4a799d828f16a9f5874b7c0ccd99a`.
- Closed transcript SHA-256:
  `c4532a3e7ba1aa407f12c578bf95e03873ce9bb4a5ef946b57450f62d8860e79`.

The generated dependency tree emitted engine warnings because some transitive packages
request Node `^22.18.0 || >=24.11.0` while this host has `v24.8.0`; generation, offline
execution, and production build nevertheless passed. Participants must use Node 24.11+
or current compatible Node 22 so this calibration warning is not normalized as expected
friction.

## Fairness boundary

Both workflows receive frozen task-card customization rather than being asked to invent
implementation code under the clock. Both install from a clean package/module cache,
preserve user-owned source across a dependency rerun, produce a build, execute an offline
selfcheck with no model/API key, identify exact upgrade and rollback commands, include
failures in the denominator, and exclude this maintainer calibration from claims.
