---
title: "session-runtime names"
description: "This map positions the current session-runtime coverage backlog. Each entry names the exact repository symbol;"
---
# session-runtime names

This map positions the current `session-runtime` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

## Symbol map

- **`sessiondiag`** — the exact `session-runtime` symbol `sessiondiag`; use this spelling for that operation rather than the undifferentiated family name.
- **`validsession`** — the exact `session-runtime` symbol `validsession`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionresume`** — the exact `session-runtime` symbol `sessionresume`; use this spelling for that operation rather than the undifferentiated family name.
- **`wiplivesessions`** — the exact `session-runtime` symbol `wiplivesessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`defaultsessionstalewindow`** — the exact `session-runtime` symbol `defaultsessionstalewindow`; use this spelling for that operation rather than the undifferentiated family name.
- **`maxsessions`** — the exact `session-runtime` symbol `maxsessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionrecord`** — the exact `session-runtime` symbol `sessionrecord`; use this spelling for that operation rather than the undifferentiated family name.
- **`faksessionledgerdir`** — the exact `session-runtime` symbol `faksessionledgerdir`; use this spelling for that operation rather than the undifferentiated family name.
- **`gemma4session`** — the exact `session-runtime` symbol `gemma4session`; use this spelling for that operation rather than the undifferentiated family name.
- **`servesessiondurability`** — the exact `session-runtime` symbol `servesessiondurability`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionteleport`** — the exact `session-runtime` symbol `sessionteleport`; use this spelling for that operation rather than the undifferentiated family name.
- **`applycodexloopsessionmeta`** — the exact `session-runtime` symbol `applycodexloopsessionmeta`; use this spelling for that operation rather than the undifferentiated family name.
- **`cmdguardsessions`** — the exact `session-runtime` symbol `cmdguardsessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`cmdguardsessionstart`** — the exact `session-runtime` symbol `cmdguardsessionstart`; use this spelling for that operation rather than the undifferentiated family name.
- **`defaultfleetmetricsmaxsessions`** — the exact `session-runtime` symbol `defaultfleetmetricsmaxsessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`dumpservesessions`** — the exact `session-runtime` symbol `dumpservesessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`evsessionresume`** — the exact `session-runtime` symbol `evsessionresume`; use this spelling for that operation rather than the undifferentiated family name.
- **`guardsessionstarthint`** — the exact `session-runtime` symbol `guardsessionstarthint`; use this spelling for that operation rather than the undifferentiated family name.
- **`imagepadslots`** — the exact `session-runtime` symbol `imagepadslots`; use this spelling for that operation rather than the undifferentiated family name.
- **`newsessiongateway`** — the exact `session-runtime` symbol `newsessiongateway`; use this spelling for that operation rather than the undifferentiated family name.
- **`ownssessionloop`** — the exact `session-runtime` symbol `ownssessionloop`; use this spelling for that operation rather than the undifferentiated family name.
- **`pausesession`** — the exact `session-runtime` symbol `pausesession`; use this spelling for that operation rather than the undifferentiated family name.
- **`restoreservesessions`** — the exact `session-runtime` symbol `restoreservesessions`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionchangeevent`** — the exact `session-runtime` symbol `sessionchangeevent`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessiondurable`** — the exact `session-runtime` symbol `sessiondurable`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessiongateway`** — the exact `session-runtime` symbol `sessiongateway`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionidentity`** — the exact `session-runtime` symbol `sessionidentity`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionleases`** — the exact `session-runtime` symbol `sessionleases`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionpath`** — the exact `session-runtime` symbol `sessionpath`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionrecoverycheckpoint`** — the exact `session-runtime` symbol `sessionrecoverycheckpoint`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessionsubscribe`** — the exact `session-runtime` symbol `sessionsubscribe`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessiontag`** — the exact `session-runtime` symbol `sessiontag`; use this spelling for that operation rather than the undifferentiated family name.
- **`stopsession`** — the exact `session-runtime` symbol `stopsession`; use this spelling for that operation rather than the undifferentiated family name.
- **`tagservedsessionadmit`** — the exact `session-runtime` symbol `tagservedsessionadmit`; use this spelling for that operation rather than the undifferentiated family name.
- **`targetsession`** — the exact `session-runtime` symbol `targetsession`; use this spelling for that operation rather than the undifferentiated family name.
- **`validsessionid`** — the exact `session-runtime` symbol `validsessionid`; use this spelling for that operation rather than the undifferentiated family name.


### releaseWeightSession (model-weight hold release)

releaseWeightSession decrements the model weight-closer session hold after a native inference path finishes and completes deferred weight teardown when the last active holder leaves after CloseWeights begins.

**Distinct from:** It releases one model-weight lifetime hold; it is not the served Session record that persists run state across turns and it does not schedule or end an agent session.
