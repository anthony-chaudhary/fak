# `fak info` overlay capture

The tabbed `fak info` overlay — the 20% status pane `fak guard --split` opens beside the
agent — is the fastest "what am I getting" read for an end user: cache savings, live
agents, account seats, and the safety floor, each on its own tab. It belongs in the
README, but the live overlay renders only from a running gateway, which is not
reproducible. This is the deterministic capture path that puts it in the docs and lets it
be regenerated as the overlay grows.

Source fixture: `visuals/info-overlay-live.json` — a single, payload-free `/debug/vars`
snapshot in the exact shape `fak info --json` emits.

## Render one tab (the offline renderer)

`fak info --from-fixture` is the offline twin of `fak console guard --journal`: instead of
polling a gateway it reads a recorded snapshot and draws one static overlay frame. It needs
no gateway, no agent, no key.

```powershell
go build -o fak ./cmd/fak
./fak info --from-fixture visuals/info-overlay-live.json --tab cache --width 118 --height 26
```

`--tab` is one of `overview`, `agents`, `accounts`, `cache`, `safety`. The frame is the
byte-identical text the live overlay would draw for that snapshot — the renderer is shared,
so a change to the overlay shows up in the capture automatically.

## Regenerate all the media

```powershell
go build -o fak ./cmd/fak
python tools/info_overlay_gen.py            # PNG per tab + hero screenshot + gif + mp4 + poster
python tools/info_overlay_gen.py --no-video # PNGs only
```

`tools/info_overlay_gen.py` drives the renderer above for every tab, paints each real frame
onto a light terminal canvas (DejaVu Sans Mono, matching the reference look), and assembles
the per-tab frames into a tab-cycle GIF + MP4 that walks through every tab so a reader sees
each tab's value in one short loop. Requires Pillow; the GIF/MP4 use `imageio-ffmpeg`
(the same encoder `tools/hero_video_gen.py` uses). The video step is skipped with a note if
the encoder is absent.

## Regenerate the fixture

The fixture is generated (not hand-maintained) from a checked-in, payload-free builder so it
tracks the `guardInfoVars` shape as the overlay grows:

```powershell
$env:FAK_UPDATE_INFO_FIXTURE = "1"
go test ./cmd/fak/ -run TestGenerateOverlayFixture
```

The builder is `demoOverlayVars()` in `cmd/fak/info_fixture_test.go`. To change what the
media shows (add a tab, a seat, a safety reason), edit the builder and rerun the two
commands above.

## Checked-in outputs

- `visuals/info-overlay-screenshot.png` — the hero (the `cache` tab), embedded in the README
- `visuals/info-overlay-overview.png` — every subsystem at once (the richest single frame)
- `visuals/info-overlay-agents.png` · `-accounts.png` · `-cache.png` · `-safety.png` — per tab
- `visuals/info-overlay-video.gif` · `-video.mp4` · `-video-poster.png` — the tab-cycle loop

## README front-door use

- `README.md` embeds `visuals/info-overlay-screenshot.png` (the cache tab) and links the
  tab-cycle `visuals/info-overlay-video.gif` / `visuals/info-overlay-video.mp4` in the
  `fak info` / token-savings section.

## Public-scrub note

The fixture is payload-free and uses only generic values: counters, gauges, block-element
bars, placeholder seat names (`seat-a`…`seat-c`), `@example.com` logins, an
`api.provider.example` serving endpoint, and generic safety reason codes (`DEFAULT_DENY`,
`SECRET_IN_ARGS`, `UNVERIFIED_RESULT`). It contains no prompt or result text, no local user
path, no real account tag, credential, hostname, or server-specific detail — the same
payload-free contract the live overlay itself honors (it renders only the `/debug/vars`
projection, never transcript text).
