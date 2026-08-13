---
title: "fak — Schnellstart (in 10 Minuten zum lokalen Modell)"
description: "Deutscher Schnellstart für fak: von null zu einer governten lokalen KI in etwa 10 Minuten — offline, ohne Key, ohne Cloud-Rechnung, Daten bleiben auf deiner Maschine. DSGVO-taugliches Self-Hosting, ein Befehl."
---

# fak — Schnellstart (in 10 Minuten zum lokalen Modell)

> Dies ist eine **lokalisierte Einstiegsseite (entry point)**, keine vollständige
> Übersetzung der Dokumentation. Die kanonische Dokumentation ist Englisch — diese Seite
> gibt dir kompakt den schnellsten Weg und den 60-Sekunden-Beweis und reicht dich dann an
> die [englischen Docs](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> weiter.
> **Hinweis:** Diese Übersetzung ist maschinell erstellt und wartet auf ein natives
> Review — Korrekturen per Issue/PR sind willkommen.
>
> Zurück zur [deutschen Einführung](./README.md) · alle Sprachen im
> [i18n-Hub](../README.md).

## Das Versprechen

Von null zu einer **governten lokalen KI in etwa 10 Minuten**: offline, ohne API-Key, ohne
Cloud-Rechnung. Deine Daten bleiben auf deiner Maschine — der self-host-first-Weg, den
DSGVO/GDPR und der EU AI Act belohnen. Für kleine Modelle genügt die **CPU**; eine GPU
brauchst du nicht.

Du schreibst deinen Agenten dabei nicht um. Du richtest entweder eine Base-URL auf
`fak serve` — oder du wickelst deinen bestehenden Agenten in einem einzigen Befehl ein.
So oder so passiert jeder Tool-Call zuerst den default-deny Capability-Floor *im* Kernel,
**fail-closed**: eine Aktion, die nie auf der Allow-Liste stand, läuft nicht — egal, wovon
das Modell überredet wurde.

## Der schnellste Weg

Ein lokales Modell hinter deinen bestehenden Coding-Agenten — ohne Key, ohne Netz, ein
Befehl:

```bash
fak manage --gguf qwen2.5:7b -- claude
```

`fak manage` startet deinen Agenten unverändert, schiebt aber ein lokales Modell davor und
prüft jeden Tool-Call, bevor er läuft.

## Der 60-Sekunden-Beweis (kein Key, kein Modell, keine GPU)

Ohne Download und ohne Modell siehst du die Tool-Call-Grenze als strukturelle
Entscheidung — ein DENY, ein ALLOW, und eine geblockte Prompt-Injection:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

Verdächtige Tool-*Ergebnisse* kommen dabei in eine Quarantäne, damit sie gar nicht erst in
den Modell-Kontext gelangen — durch Struktur, nicht durch einen Classifier. In Live-Tests
erreichte Prompt-Injection den ungeschützten Baseline-Agenten 5/5; fak hat sie 5/5
abgewehrt.

## Was ist fak?

**fak ist eine einzige statische Go-Binary**, die zwischen deinem KI-Agenten und den
Tools sitzt, die er aufruft. Sie prüft jeden Tool-Call *bevor* er läuft und verwendet in
langen Sessions die geteilte Arbeit (System-Prompt, Tool-Liste) wieder. Ergebnis:
derselbe Agent-Loop wird **sicherer, günstiger und schneller**, ohne Rewrite. fak
ersetzt dein Modell nicht — es governt und cached es.

## Wie schnell?

Auf einem gemessenen **50-Turn-×-5-Agent-Run** ist der ehrliche Gewinn gegenüber einem
*getunten* Warm-Cache-Stack **etwa 4,1× weniger Arbeit** — das ist die Kopfzahl. Der
Token-Preis ist damit ein Margen-Hebel, den du direkt in Euro spürst.

Die auffällige **~60×**-Zahl (etwa 19 Stunden auf etwa 19 Minuten) gilt **nur gegenüber
dem naiven Re-Send-Loop**, der die ganze wachsende Konversation jede Runde neu verarbeitet
— nie als Kopfzahl lesen. Der Wiederverwendungs-Gewinn ist **self-host only** und gilt für
read-heavy Flotten.

Der Prompt-Cache-Rabatt des Providers hält nur, solange das gecachte Präfix
byte-für-byte identisch bleibt. fak **garantiert** die Byte-Identität des Präfixes,
während es alte mittlere Turns abwirft; ob der Provider den Cache dann tatsächlich
wiederverwendet, ist die Entscheidung des Providers — fak reicht sie weiter, statt sie zu
behaupten. Jede Zahl ist in
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
bis auf Commit und Artefakt belegt.

## Wohin als Nächstes

- [Deutsche Einführung](./README.md) — Kern, Nutzen für EU-Teams, DSGVO/EU-AI-Act
- [README (der volle Überblick)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — in 10 Minuten zum lokalen Modell](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — die Binary installieren](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — deinen Agenten anschließen](../../integrations/README.md)
- [Data residency & compliance — für DSGVO/GDPR und den EU AI Act](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — die Quelle jeder Zahl](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — was shipped/simuliert/Stub ist](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Installation: **Apache-2.0**, frei, self-hosted — keine Kreditkarte, keine
grenzüberschreitende Rechnung, keine Rechtsentität. `git clone` und
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` sind der ganze Weg.

Lizenz: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
