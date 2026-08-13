---
title: "fak — Installation & Erste Schritte (deutsche Einstiegsseite / German install guide)"
description: "Deutsche Installations-Einstiegsseite für fak: die eine Go-Binary aus einem sauberen Checkout bauen und laufen lassen — Tier 0 offline, Tier 1 vor ein echtes Modell, Tier 2 das fusionierte In-Kernel-Modell. DSGVO-taugliches Self-Hosting, kein Key/GPU/Netz für Tier 0."
---

# fak — Installation & Erste Schritte (deutsche Einstiegsseite)

> Dies ist eine **lokalisierte Einstiegsseite (entry point)**, keine vollständige
> Übersetzung der Dokumentation. Die kanonische Dokumentation ist Englisch — diese Seite
> bringt dich von einem sauberen Checkout zu einer laufenden Binary und reicht dich dann an
> die [englischen Docs](https://github.com/anthony-chaudhary/fak/blob/main/README.md) weiter.
> **Hinweis:** Diese Übersetzung ist maschinell erstellt und wartet auf ein natives Review —
> Korrekturen per Issue/PR sind willkommen.
>
> Der dichte Pitch (was fak ist und warum) steht in der [deutschen Einführung](./README.md);
> alle Sprachen im [i18n-Hub](../README.md).

Dies ist die **Installations- und Ausführungs-Vordertür**. Der dichte Pitch ist die
[README](./README.md). fak ist **eine Go-Binary**: ein einziges statisches Artefakt ohne
externe Abhängigkeiten — Gateway, KV-Cache- und Routing-Engine, Policy-Gate,
Result-Quarantäne und Audit-Log in einem Prozess. Weil fak **self-host-first** ist, bleiben
Daten auf deiner Infrastruktur (relevant für **DSGVO/GDPR** und das **EU AI Act**-Audit-Log);
den Token-Preis in **EUR** senkt fak, indem es geteilte Arbeit in langen Sessions
wiederverwendet — die Belege dazu stehen in der README und der
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).

## Voraussetzungen

- **Go 1.26+.** `fak/go.mod` deklariert `go 1.26`. Mit Go's Standard `GOTOOLCHAIN=auto`
  lädt ein älteres `go` die passende Toolchain beim ersten Build automatisch nach (einmalig
  Netz nötig); sonst Go 1.26 von <https://go.dev/dl/>. Prüfen mit `go version`.
- **Für Tier 0 ist das alles**: keine GPU, kein API-Key, kein Netz.
- **Tier 1** braucht zusätzlich irgendeinen OpenAI-kompatiblen Modell-Server (z. B. Ollama).

## Die vier Tiers auf einen Blick

Nichts Neues wird zwischen den Tiers installiert — dieselbe Binary, du fügst nur Flags hinzu.

| Tier | Was du bekommst | Setup |
|---|---|---|
| **0 — Kernel ausprobieren** | Die Adjudikationsgrenze offline laufen lassen/messen | `go build` |
| **1 — Vor ein echtes Modell** | Den Kernel vor ein Modell setzen, das du woanders servierst (Ollama / vLLM / llama.cpp / Cloud-Provider) | + ein laufender OpenAI-kompatibler Server |
| **1b — Lokales Modell in einem Befehl** | Ein lokales GGUF-Modell in-kernel mit deinem bestehenden Agenten — kein Key, kein Netz, kein zweites Terminal | `fak guard --gguf qwen2.5:7b -- claude` |
| **2 — Das fusionierte In-Kernel-Modell** | Der reine-Go-Forward-Pass, den der Kernel selbst besitzt | `go build` (+ echte Gewichte optional) |

Wenn du einfach **ein brauchbares Modell mit fak davor servieren** willst, willst du **Tier 1**.

## Die Binary holen

**Aus dem Clone bauen (Contributor):**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: -o fak.exe)
./fak help
```

**Mit Go installieren.** Der Modulpfad `github.com/anthony-chaudhary/fak` ist die
Repository-Wurzel und installiert direkt:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

## Der 60-Sekunden-Beweis (kein Key, kein Modell, keine GPU)

Das ist Tier 0: Die Capability-Floor verweigert einen Call strukturell (fail-closed), und die
Prompt-Injection wird geblockt, während der Task trotzdem fertig wird.

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

Du schreibst deinen Agenten nicht um — du richtest eine Base-URL auf `fak serve`, oder du
wickelst deinen bestehenden Agenten in einem einzigen Befehl ein:

```bash
fak manage claude      # wraps your existing agent in a single command
```

## Wohin als Nächstes

- [`docs/fak/tutorial.md`](../../fak/tutorial.md) — **die geführte erste Session** (~15 Min):
  jeder Befehl mit seiner echten, aufgezeichneten Ausgabe, offline, ohne Key und ohne GPU.
- [README (der volle Überblick)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — in 10 Minuten zum lokalen Modell](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — die englische Referenz zu dieser Seite](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — deinen Agenten anschließen](../../integrations/README.md)
- [Data residency & compliance — für DSGVO/GDPR und das EU AI Act](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — die Quelle jeder Zahl](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — was shipped/simuliert/Stub ist](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Lizenz: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
