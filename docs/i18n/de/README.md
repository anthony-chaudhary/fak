---
title: "fak — der Fused Agent Kernel (deutsche Einführung / German introduction)"
description: "Deutsche Einstiegsseite für fak: eine Go-Binary, die jeden Tool-Call vor der Ausführung prüft — derselbe Agent-Loop wird sicherer, günstiger, schneller; DSGVO-taugliches Self-Hosting und EU-AI-Act-Audit-Log."
---

# fak — der Fused Agent Kernel (deutsche Einführung)

> Dies ist eine **lokalisierte Einstiegsseite (entry point)**, keine vollständige
> Übersetzung der Dokumentation. Die kanonische Dokumentation ist Englisch — diese Seite
> gibt dir den Kern, den 60-Sekunden-Beweis und den Installationsweg und reicht dich dann
> an die [englischen Docs](../../../README.md) weiter.
> **Hinweis:** Diese Übersetzung ist maschinell erstellt und wartet auf ein natives
> Review — Korrekturen per Issue/PR sind willkommen.
>
> **En français :** [français](../fr/README.md) — alle Sprachen im
> [i18n-Hub](../README.md).

## fak in einer Zeile

**fak ist eine Go-Binary**, die zwischen deinem KI-Agenten und seinen Tool-Calls sitzt —
sie prüft jeden Tool-Call, *bevor* er läuft, und verwendet in langen Sessions die stabile
Arbeit wieder. Ergebnis: derselbe Agent-Loop wird **sicherer, günstiger und schneller**,
ohne dass du sonst etwas änderst.

Du schreibst deinen Agenten nicht um — du richtest eine Base-URL auf `fak` und jeder
Tool-Call passiert zuerst den Capability-Floor.

```bash
fak guard -- claude    # wickelt deinen bestehenden Agenten in einem einzigen Befehl ein
```

## Warum das für europäische Startups zählt

- **Daten bleiben auf deiner Infrastruktur (DSGVO/GDPR).** fak ist self-host-first: eine
  statische Binary, die vor einem **lokalen Modell** (`fak guard --gguf …`) oder einem
  Provider deiner Wahl sitzt — fail-closed auf jedem Backend, default-deny
  Capability-Floor, und ein manipulationsevidentes Audit-Log für jeden Tool-Call. Es gibt
  keinen „standardmäßig an einen Drittstaat weitergereicht"-Pfad, über den du nachdenken
  müsstest. Details:
  [Data residency & compliance](../../explainers/data-residency-and-compliance.md).
- **Das EU-AI-Act-Audit-Log ist schon da (Artikel 12, durchsetzbar ab 2. August 2026).**
  fak schreibt ein append-only, SHA-256-hash-verkettetes Entscheidungsjournal und prüft es
  offline mit `fak audit verify` — die Abbildung der Artikel-12-Pflichten auf den
  ausgelieferten Mechanismus steht in
  [EU AI Act Article 12 conformance](../../standards/eu-ai-act-article-12-conformance.md).
  Keine Rechtsberatung — aber der technische Baustein, nach dem ein Audit fragt.
- **Der Token-Preis ist ein Margen-Hebel.** fak verwendet in langen Sessions die geteilte
  Arbeit wieder (System-Prompt + Tool-Liste — der KV-Cache der bisherigen Arbeit): auf
  einem 50-Turn-×-5-Agent-Run **~4,1× weniger Arbeit** als ein getunter Warm-Cache-Stack
  (~60× gegenüber einem naiven Re-Send-Loop; die ehrliche Zahl ist 4,1×). Per-Aspect-Routing
  schickt die günstigen Anteile zusätzlich auf günstigere Modelle. Jede Zahl ist in
  [BENCHMARK-AUTHORITY](../../../BENCHMARK-AUTHORITY.md) belegt.
- **Apache-2.0, keine Beschaffungs-Hürde.** fak ist frei, quelloffen und self-hosted —
  kein Vendor-Vertrag, keine Kreditkarte, kein Account. `git clone` und `go install` sind
  der ganze Weg.
- **Eine statische Binary, null externe Abhängigkeiten.** Einfache Ops für ein kleines
  Team — kein Sidecar, kein separater Authorizer. Vom Laptop bis zur Flotte dasselbe
  Artefakt; du fügst Flags hinzu, keine Komponenten.

## Welche Probleme fak löst

- **Lange Sessions hören auf, teuer zu sein.** Der Prompt-Cache-Rabatt des Providers hält
  nur, solange das gecachte Präfix byte-für-byte identisch bleibt; fak wirft alte Turns ab
  und hält das Präfix trotzdem byte-identisch — der Rabatt reißt nicht ab.
- **Default-deny-Sicherheit.** Die Permission-Policy läuft *im* Kernel, auf demselben
  Call-Path. Eine irreversible Aktion zu verhindern hängt nicht davon ab, einen Angriff zu
  „erkennen" — der Hebel war nie verdrahtet. Das ist **fail-closed**, nicht fail-open.
- **Prompt-Injection / vergiftete Tool-Ergebnisse.** Verdächtige Tool-*Ergebnisse* kommen
  in eine Quarantäne, damit sie gar nicht erst in den Modell-Kontext gelangen — durch
  Struktur, nicht durch einen Classifier.

## Der 60-Sekunden-Beweis (kein Key, kein Modell, keine GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # Injection gestoppt, Task trotzdem erledigt
```

## Mit deinem Modell

fak ersetzt dein Modell nicht — es governt und cached es. **Qwen2/Qwen3 und GLM-MoE** sind
in der In-Kernel-Referenz-Engine bit-exakt nachgewiesen; alles andere (Mistral, DeepSeek,
jedes Open-Weights-Modell) wird über die OpenAI-kompatible Schnittstelle gefrontet — via
Ollama / vLLM / SGLang / llama.cpp / LM Studio oder jede OpenAI-kompatible API.

## Wohin als Nächstes

- [README (der volle Überblick)](../../../README.md)
- [START-HERE — in 10 Minuten zum lokalen Modell](../../../START-HERE.md)
- [Getting Started — die Binary installieren](../../../GETTING-STARTED.md)
- [Integrations — deinen Agenten anschließen](../../integrations/README.md)
- [Data residency & compliance — für DSGVO/GDPR](../../explainers/data-residency-and-compliance.md)
- [EU AI Act Artikel 12 — Audit-Log-Konformität](../../standards/eu-ai-act-article-12-conformance.md)
- [BENCHMARK-AUTHORITY — die Quelle jeder Zahl](../../../BENCHMARK-AUTHORITY.md)
- [CLAIMS — was shipped/simuliert/Stub ist](../../../CLAIMS.md)

Lizenz: [Apache-2.0](../../../LICENSE).
