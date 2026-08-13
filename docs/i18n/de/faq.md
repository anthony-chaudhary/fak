---
title: "fak — FAQ für den Erstkontakt (deutsche Einführung / German FAQ)"
description: "Die häufigsten Erstkontakt-Fragen zu fak auf Deutsch: eine Go-Binary, die jeden Tool-Call vor der Ausführung prüft — derselbe Agent-Loop wird sicherer, günstiger, schneller; DSGVO-taugliches Self-Hosting und EU-AI-Act-Audit-Log."
---

# fak — FAQ für den Erstkontakt

> Dies ist eine **lokalisierte Einstiegsseite (entry point)**, keine vollständige
> Übersetzung der Dokumentation. Die kanonische Dokumentation ist Englisch — diese Seite
> beantwortet die häufigsten Erstkontakt-Fragen und reicht dich dann an die
> [englischen Docs](https://github.com/anthony-chaudhary/fak/blob/main/README.md) weiter.
> **Hinweis:** Diese Übersetzung ist maschinell erstellt und wartet auf ein natives
> Review — Korrekturen per Issue/PR sind willkommen.
>
> **Mehr auf Deutsch:** [Einführung](./README.md) — alle Sprachen im
> [i18n-Hub](../README.md).

## Q1. Was ist fak?

Eine einzelne statische Go-Binary, die du **vor den KI-Agenten setzt, den du ohnehin
schon betreibst** (Claude Code, Codex, Cursor, jeder OpenAI-/Anthropic-/MCP-Client) —
indem du eine einzige Base-URL umlenkst, ohne Rewrite. Sie macht lange Sessions günstiger
(sie wirft alte Turns ab und hält dabei das Präfix des Provider-Prompt-Caches
byte-identisch), leitet jeden Tool-Call, kann lokale GGUF-Modelle in-process ausführen und
schreibt für jeden Call ein auditierbares Urteil.

## Q2. Muss ich das Modell wechseln oder meinen Agenten umschreiben?

Nein. fak governt und cached das Modell, das du bereits nutzt. Wickle es ein mit:

```bash
fak manage claude
```

oder richte eine Base-URL auf `fak serve`.

## Q3. Wohin gehen meine Daten — ist das konform?

Self-host-first: eine statische Binary vor einem lokalen Modell oder einem inländischen
Provider — mit fail-closed Residency, einem default-deny Capability-Floor und einem
manipulationsevidenten Audit-Log für jeden Tool-Call. Deine Daten verlassen deine Maschine
nicht.

Das ist der technische Baustein, nach dem die **DSGVO (GDPR)** und der **EU AI Act**
fragen: Datenhaltung in eigener Infrastruktur statt eines standardmäßigen Wegs in einen
Drittstaat, plus das append-only, hash-verkettete Entscheidungsjournal, das ein Audit
erwartet. Keine Rechtsberatung — aber der ausgelieferte Mechanismus. Details:
[Data residency & compliance](../../explainers/data-residency-and-compliance.md).

## Q4. Was kostet es? Ist es wirklich kostenlos?

**Apache-2.0**, frei, self-hosted. Keine Kreditkarte, keine grenzüberschreitende Rechnung,
keine juristische Person — also 0 € Lizenz- und Beschaffungskosten. `git clone` plus
`go install` sind der ganze Weg.

## Q5. Wie viel günstiger oder schneller ist es?

Auf einem gemessenen 50-Turn-×-5-Agent-Run **~4,1× weniger Arbeit** als ein *getunter*
Warm-Cache-Stack — das ist die ehrliche Kennzahl und der direkte Hebel auf deine Marge in
Euro. Die ~60×-Zahl (etwa 19 Stunden auf etwa 19 Minuten) gilt **nur gegenüber einem
naiven Re-Send-Loop**, der jedes Mal alles neu schickt — niemals als Schlagzeile. Der
Wiederverwendungs-Gewinn ist **self-host-only** und gilt für leseintensive Flotten.

## Q6. Welche Modelle funktionieren?

**Qwen2/Qwen3 und GLM-MoE** sind in der In-Kernel-Referenz-Engine bit-exakt nachgewiesen.
Alles andere (DeepSeek, Mistral, jedes Open-Weights-Modell) wird über die
OpenAI-kompatible Schnittstelle gefrontet: Ollama / vLLM / SGLang / llama.cpp / LM Studio.

## Q7. Wie stoppt es Prompt-Injection?

Zwei strukturelle Gates, kein Classifier: ein **default-deny Capability-Floor** (ein
gefährliches Tool steht nie auf der Allow-List) und **Result-Quarantine** (vergiftete
Tool-Ergebnisse erreichen den Modell-Kontext gar nicht erst). In Live-Tests erreichte die
Injection die ungeschützte Baseline 5/5, und fak wehrte sie 5/5 ab. Der Detektor, der
verdächtige Ergebnisse markiert, gilt bewusst als ~100 % umgehbar — ein Bonus, nie das
Fundament.

## Q8. Wie installiere ich es?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

oder aus einem Clone:

```bash
go build -o fak ./cmd/fak
```

Falls dein Netzwerk `proxy.golang.org` nicht erreicht, setze vorher einen `GOPROXY`, der
für dich funktioniert.

## Q9. Der 60-Sekunden-Beweis (kein Key, kein Modell, keine GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # Injection gestoppt, Task trotzdem erledigt
```

## Wohin als Nächstes

- [README (der volle Überblick)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — in 10 Minuten zum lokalen Modell](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — die Binary installieren](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — deinen Agenten anschließen](../../integrations/README.md)
- [Data residency & compliance — für DSGVO/GDPR und den EU AI Act](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — die Quelle jeder Zahl](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — was shipped/simuliert/Stub ist](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Lizenz: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
