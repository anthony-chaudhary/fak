# Billing quickstart: serve one model to multiple tenants

`fak serve --billing` mounts a multi-tenant billing chain in front of the
model upstreams a manifest declares. Every request that reaches the chain is
authenticated against a tenant key store, priced by a price book, and settled
against a per-tenant unit ledger. This guide walks that loop end to end with
two tenants: provision keys and balances, send one billed request per tenant,
then exhaust one tenant's budget and see the refusal the chain returns.

What you will set up:

- one model (`demo-model`) served through the billing chain
- two tenants (`tenant-a`, `tenant-b`), each with its own key and balance
- one billed request per tenant, then the `402` a tenant receives once its
  balance can no longer cover a reservation

All prices in this guide are synthetic examples that make the arithmetic easy
to check, not a quote. The ledger holds integer **ledger units**: one currency
unit of the price book equals 1,000,000 ledger units by default, and prices
are stated per million tokens.

## 1. Declare the billing manifest and price book

The manifest names the served models, the upstreams that run them, where
billing state lives, and the rate limits. Save this as `billing.json`:

```json
{
  "version": "fak-billing-manifest/v1",
  "state_dir": "./billing-state",
  "prices_path": "./prices.json",
  "models": [
    {
      "name": "demo-model",
      "upstream": "engine-a",
      "upstream_model": "demo-model-internal"
    }
  ],
  "upstreams": [
    {
      "name": "engine-a",
      "base_url": "http://127.0.0.1:29001/v1"
    }
  ],
  "limits": {
    "requests_per_minute": 60,
    "tokens_per_minute": 200000,
    "daily_token_cap": 2000000,
    "session_token_cap": 0
  }
}
```

- `state_dir` is where the key store (`keys.json`) and the ledger
  (`ledger.jsonl`) live. A fresh directory boots clean; the first mutation
  creates each file.
- `models[].name` is the name clients request; `upstream_model` is the name
  sent to the upstream. Valuation uses the served name.
- `base_url` points at an OpenAI-compatible upstream. For a real model, point
  it at your serving endpoint.

The price book states per-model rates. Save this as `prices.json` next to the
manifest:

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "demo-model",
      "version": 1,
      "source_revision": "example-rev-1",
      "source_time": "2026-01-01T00:00:00Z",
      "currency": "USD",
      "units": "per_million_tokens",
      "fresh_input_rate": 1.0,
      "cache_read_rate": null,
      "cache_write_rate": null,
      "output_rate": 2.0
    }
  ]
}
```

A `null` rate means that cache class is not priced for the model. With the
example rates, 100 fresh input tokens cost 100 ledger units and 20 output
tokens cost 40 ledger units — 140 units per request of this shape.

## 2. Mint tenant keys and top up balances

Tenant keys authenticate callers to the chain. Mint one key per tenant — the
full key is printed exactly once and cannot be recovered afterwards, so
capture it:

```bash
fak keys mint --manifest ./billing.json --tenant tenant-a
fak keys mint --manifest ./billing.json --tenant tenant-b
```

```text
WARNING: the full key is shown exactly once and cannot be recovered afterwards.
key: sk-fak-…
prefix: sk-fak-582d7
tenant: tenant-a
```

Top up each tenant's balance in ledger units:

```bash
fak keys topup --manifest ./billing.json --tenant tenant-a --units 10000000
fak keys topup --manifest ./billing.json --tenant tenant-b --units 1000000
```

```text
fak keys topup: tenant=tenant-a units=10000000 event=evt-3f7fde89…
balance: 10000000 ledger units
fak keys topup: tenant=tenant-b units=1000000 event=evt-2924fb7e…
balance: 1000000 ledger units
```

**Provision before you start the serve.** The serve reads the key store and
the ledger once at startup. Keys minted or topped up after the serve starts
are not visible to the running process — a request with such a key is refused
with `401 unknown_key`. Run this step before step 3, or restart the serve
after provisioning.

## 3. Start the serve with billing armed

```bash
fak serve --model mock --mock \
  --addr 127.0.0.1:29002 \
  --billing ./billing.json \
  --billing-addr 127.0.0.1:29000
```

`--billing` arms the billing chain on `--billing-addr`; the model listener
stays on `--addr`. `--model mock --mock` runs the deterministic mock planner
so this walkthrough needs no model weights. On startup the serve reports what
it armed:

```text
fak serve: billing armed:
  manifest   fak-billing-manifest/v1 (./billing.json)
  state_dir  ./billing-state
  models     1
  upstreams  1: engine-a
  limits     requests_per_minute=60 tokens_per_minute=200000 daily_token_cap=2000000 session_token_cap=0 (0 = uncapped)
  price_book sha256:d0986c98…

fak serve: billing chain listening on http://127.0.0.1:29000 (1 models: demo-model)
```

Clients talk to the **billing addr** (`:29000`), not the model listener.

## 4. Send one billed request per tenant

```bash
curl -X POST http://127.0.0.1:29000/v1/chat/completions \
  -H "Authorization: Bearer <tenant-a-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"hi"}]}'
```

A billed request returns the upstream completion with a billing request id:

```text
HTTP/1.1 200 OK
X-Fak-Billing-Request-Id: fak-ff8dfc192a40294470762d3da81775de
```

Send the same request with the tenant-b key; it also returns `200` with its
own request id. Settlement is asynchronous: the chain reserves the hold when
the request is admitted, proxies it, and settles the actual usage shortly
after. Wait a moment before reading balances.

## 5. Exhaust the budget and see the refusal

Tenant-b started with exactly 1,000,000 units. After the first request
settles 140 units, its available balance is 999,860 — less than the default
1,000,000-unit reservation hold. The next request is refused before any
reservation is taken:

```text
HTTP/1.1 402 Payment Required
```

```json
{
  "error": {
    "type": "billing_error",
    "code": "insufficient_balance",
    "message": "tenant \"tenant-b\" has insufficient available balance for the 1000000-unit hold"
  }
}
```

The refusal leaves the ledger untouched — no reservation, no settlement, and
the upstream is never called for a refused request.

Read both balances to confirm the settlements:

```bash
fak keys balance --manifest ./billing.json --tenant tenant-a
fak keys balance --manifest ./billing.json --tenant tenant-b
```

```text
fak keys balance: tenant=tenant-a
balance: 9999860 ledger units
fak keys balance: tenant=tenant-b
balance: 999860 ledger units
```

## How the units add up

Each request of this shape consumes 140 ledger units:

| Component | Tokens | Rate (per million) | Ledger units |
|---|---|---|---|
| Fresh input | 100 | 1.0 | 100 |
| Output | 20 | 2.0 | 40 |
| **Settled per request** | | | **140** |

- tenant-a: 10,000,000 − 140 = **9,999,860** — well above the hold, so every
  request is admitted.
- tenant-b: 1,000,000 − 140 = **999,860** — below the 1,000,000-unit hold, so
  the next request is refused with `402 insufficient_balance`.

The reservation hold is the chain's guard against concurrent spend: a request
must be able to cover the full hold up front, the hold is released when the
real usage settles, and the settle books the actual consumed units.

## What the chain does to each request

Every request admitted by the billing chain passes the same closed pipeline:

1. **Route check** — only served model names and supported methods (404/405).
2. **Auth** — bearer key lookup in the tenant key store (401 unknown, 403
   revoked).
3. **Scrub** — `Authorization`, `X-Api-Key`, `Cookie`, and `X-Fak-Session`
   are stripped before proxying.
4. **Shape** — a bounded JSON body naming a served model (400/413).
5. **Rate window** — the manifest's per-minute limits (429).
6. **Policy and reserve** — the billing policy admits, then the ledger takes
   the reservation hold (402 on insufficient balance).
7. **Proxy** — the request is forwarded to the manifest's upstream with the
   model rewritten to the upstream name.
8. **Settle** — actual usage is priced by the served model's rates and booked
   asynchronously.

Requests the chain refuses never reach the upstream. An upstream failure
releases the reservation and returns 502; a request whose usage cannot be
established leaves its reservation pending rather than settling to zero.
