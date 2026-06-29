# Captured run — federated "what changed" feed + revoke

Captured against the on-disk `fak` **v0.34.0** kernel, two `fak serve --engine mock`
instances on `127.0.0.1:8431` (A) and `127.0.0.1:8432` (B), both started with
[`federated-policy.json`](federated-policy.json). Every block below is the raw wire
JSON returned by the route named above it. No model is in the loop — the feed and the
refutation are pure kernel state.

## Summary line

```
fak federated 'what changed' feed + revoke walkthrough
  instance A = http://127.0.0.1:8431
  instance B = http://127.0.0.1:8432

  ✓ A: a write lands on A's change feed              update_booking -> mutation seq=1 world_ver=1
  ✓ B: A's write did NOT auto-propagate (per-instance feed)   B feed has 0 event(s), none of them A's write
  ✓ B: B's own write lands on B's feed               cancel_order -> mutation on B
  ✓ A: revoke refutes the witness on A (trust_epoch advances)  evicted=0 trust_epoch=1
  ✓ B: the refutation propagates A -> B (revocation in BOTH feeds)   B trust_epoch=1
  ✓ cursor drain is monotone (nothing re-delivered past the cursor)   drained B since cursor=2 -> 0 new event(s)
  ✓ empty-witness revoke is refused (400)            HTTP 400: revoke requires a non-empty witness

summary: federated change-feed + revoke walkthrough passed
```

## Step 1 — A drives a write-shaped tool (it COMPLETES → a mutation)

`POST http://127.0.0.1:8431/v1/fak/syscall`
```json
{"tool":"update_booking","arguments":{"id":"SFO-JFK-7","seat":"14C"},"witness":"commit-aaa111"}
```
response:
```json
{"verdict":{"kind":"ALLOW","by":"monitor"},
 "result":{"status":"OK",
   "content":"{\"tool\":\"update_booking\",\"echo\":\"{\\\"id\\\":\\\"SFO-JFK-7\\\",\\\"seat\\\":\\\"14C\\\"}\",\"ok\":true}",
   "meta":{"engine":"mock","input_tokens":"57","output_tokens":"21"}},
 "trace_id":"gw-2"}
```

## Step 2 — A's "what changed" feed (the write is on it)

`GET http://127.0.0.1:8431/v1/fak/changes`
```json
{"events":[{"kind":"mutation","seq":1,"tool":"update_booking","tags":["*"],"world_ver":1,"trust_epoch":0}],
 "cursor":1}
```

## Step 3 — B's feed is INDEPENDENT (A's write did NOT auto-propagate)

`GET http://127.0.0.1:8432/v1/fak/changes`
```json
{"events":[],"cursor":0}
```
This is the honest, load-bearing fact: each instance observes its own coherence bus.
A's `update_booking` mutation is nowhere on B. Propagation is the host's job (Step 6).

## Step 4 — B drives a DIFFERENT write under its own witness

`POST http://127.0.0.1:8432/v1/fak/syscall`
```json
{"tool":"cancel_order","arguments":{"id":"ORD-99"},"witness":"commit-bbb222"}
```
B's feed afterward — `GET http://127.0.0.1:8432/v1/fak/changes`:
```json
{"events":[{"kind":"mutation","seq":1,"tool":"cancel_order","tags":["*"],"world_ver":1,"trust_epoch":0}],"cursor":1}
```

## Step 5 — A REFUTES B's witness (`commit-bbb222`)

`POST http://127.0.0.1:8431/v1/fak/revoke`
```json
{"witness":"commit-bbb222"}
```
response:
```json
{"witness":"commit-bbb222","evicted":0,"trust_epoch":1}
```
A's feed now carries the revocation — `GET http://127.0.0.1:8431/v1/fak/changes`:
```json
{"events":[
   {"kind":"mutation","seq":1,"tool":"update_booking","tags":["*"],"world_ver":1,"trust_epoch":0},
   {"kind":"revocation","seq":2,"witness":"commit-bbb222","world_ver":1,"trust_epoch":1}],
 "cursor":2}
```
The `trust_epoch` advanced `0 → 1`. `evicted=0` because no entry was pooled under that
witness in A's tier-2 cache on the mock path — the `trust_epoch` advance and the feed
broadcast are what carry the refutation.

## Step 6 — FEDERATION: the host replays A's revocation onto B

The host drains A's feed, finds the `revocation` for `commit-bbb222`, and replays it.
`POST http://127.0.0.1:8432/v1/fak/revoke`
```json
{"witness":"commit-bbb222"}
```
response:
```json
{"witness":"commit-bbb222","evicted":0,"trust_epoch":1}
```
B's feed now carries the **same** revocation — `GET http://127.0.0.1:8432/v1/fak/changes`:
```json
{"events":[
   {"kind":"mutation","seq":1,"tool":"cancel_order","tags":["*"],"world_ver":1,"trust_epoch":0},
   {"kind":"revocation","seq":2,"witness":"commit-bbb222","world_ver":1,"trust_epoch":1}],
 "cursor":2}
```
The refutation is now in **both** feeds — the peer-trust property.

## Step 7 — cursor drain is monotone

`GET http://127.0.0.1:8432/v1/fak/changes?since=2` (drain B past its cursor):
```json
{"events":[],"cursor":2}
```
Nothing is re-delivered past the cursor. The POST form is equivalent —
`POST http://127.0.0.1:8431/v1/fak/changes` with `{"since":1}`:
```json
{"events":[{"kind":"revocation","seq":2,"witness":"commit-bbb222","world_ver":1,"trust_epoch":1}],"cursor":2}
```

## Step 8 — an empty witness is refused at the wire

`POST http://127.0.0.1:8431/v1/fak/revoke` with `{"witness":""}` → HTTP `400`:
```json
{"error":{"code":null,"message":"revoke requires a non-empty witness","param":null,"type":"invalid_request_error"}}
```

## Note on the issue's framing (corrected)

Issue #340's deliverable narrates the feed "propagating A→B and B→A" as if the routes
gossip on their own. The **real** v0.34.0 behavior, verified above: each `fak serve`
has its own per-process coherence feed that does NOT auto-propagate (Step 3 proves it).
The federation is the host **draining one feed by cursor and replaying** the event onto
the other (Step 6). The walkthrough therefore shows propagation as an explicit replay,
not a hidden network protocol — the routes provide the verifiable ledger + the
refutation primitive; the host wires them between instances.
