---
title: "fak — preguntas frecuentes de primer contacto (FAQ en español / Spanish first-contact FAQ)"
description: "FAQ de entrada en español para fak: el binario Go que verifica cada tool call antes de ejecutarlo. Qué es, si hay que reescribir el agente, dónde quedan tus datos (RGPD y Reglamento de IA de la UE), cuánto cuesta y cuánto ahorra."
---

# fak — preguntas frecuentes de primer contacto

> Esta es una **página de entrada localizada (entry point)**, no una traducción completa de
> la documentación. La documentación canónica está en inglés — esta FAQ responde las primeras
> dudas y luego te deriva a la
> [documentación en inglés](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta página está generada automáticamente y está pendiente de revisión nativa —
> se agradecen correcciones por issue/PR.
>
> **Otros idiomas y páginas hermanas:** el [hub i18n](../README.md) ·
> [introducción en español](./README.md).

## Q1. ¿Qué es fak?

Un solo **binario Go estático** que colocas delante del agente de IA que ya ejecutas
(Claude Code, Codex, Cursor, cualquier cliente OpenAI/Anthropic/MCP): reapuntas una única
base URL, sin reescribir nada. Abarata las sesiones largas (descarta los turnos viejos
mientras mantiene el prefijo del prompt-cache del provider byte-idéntico), enruta cada tool
call, puede ejecutar modelos GGUF locales in-process y registra un veredicto auditable para
cada llamada.

## Q2. ¿Tengo que cambiar de modelo o reescribir mi agente?

No. fak gobierna y cachea el modelo que ya usas — no lo reemplaza. Lo envuelves con:

```bash
fak manage claude
```

o reapuntas una sola base URL hacia `fak serve`.

## Q3. ¿A dónde van mis datos? ¿Es esto compatible con la normativa?

Self-host primero: un binario estático delante de un modelo local o de un provider nacional,
con residencia fail-closed, un capability floor en default-deny y un registro de auditoría a
prueba de manipulación para cada tool call. Tus datos no salen de tu máquina. Esto es lo que
piden el **RGPD/GDPR** (residencia y minimización de datos) y el **Reglamento de IA de la UE**
(el Artículo 12 exige un registro de eventos a lo largo del ciclo de vida). fak entrega el
bloque técnico; no es asesoría legal. Detalles:
[Data residency & compliance](../../explainers/data-residency-and-compliance.md).

## Q4. ¿Cuánto cuesta? ¿De verdad es gratis?

**Apache-2.0**, libre, self-host. Sin tarjeta de crédito, sin factura transfronteriza (ni en
EUR ni en tu moneda local), sin entidad legal. `git clone` más `go install` es el camino completo.

## Q5. ¿Cuánto más barato o más rápido es?

En un run medido de 50 turnos × 5 agentes, **~4,1× menos trabajo** que un stack warm-cache
afinado (*tuned*). La cifra de **~60×** (de unas 19 horas a unos 19 minutos) solo se cumple
**frente a un bucle ingenuo que reenvía el contexto completo** — nunca como titular. La ganancia por
reutilización es **self-host únicamente**, para flotas con carga de lectura intensiva. En la
práctica: cada token que no reenvías es margen que no pagas, en EUR o en tu moneda local.
fak **garantiza** que el prefijo cacheado sea byte-idéntico; que el provider reutilice esa
caché es decisión del provider, que fak transmite en lugar de prometer. Cada cifra está
respaldada en
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).

## Q6. ¿Qué modelos funcionan?

**Qwen2/Qwen3 y GLM-MoE** están probados bit-exact en el motor de referencia in-kernel. El resto
(DeepSeek, Mistral, cualquier modelo open-weights) se sirve por la interfaz compatible
con OpenAI: Ollama / vLLM / SGLang / llama.cpp / LM Studio o cualquier API compatible con OpenAI.

## Q7. ¿Cómo detiene el prompt injection?

Con dos barreras estructurales, no con un classifier: un **capability floor en default-deny**
(una herramienta peligrosa nunca está en la allow-list, por más que se convenza al modelo) y
la **cuarentena de resultados** (los resultados de herramienta envenenados nunca llegan al
contexto del modelo). El detector que los marca se trata como ~100% evadible por diseño — un
extra, nunca el piso. En pruebas en vivo, la inyección alcanzó el baseline sin protección 5/5
y fak la aisló 5/5.

## Q8. ¿Cómo lo instalo?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

o, desde un clon:

```bash
go build -o fak ./cmd/fak
```

Si tu red bloquea el proxy de módulos por defecto, ajusta `GOPROXY` a un espejo alcanzable
antes de `go install`.

### La prueba de 60 segundos (sin key, sin modelo, sin GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # inyección detenida, tarea igualmente completada
```

## Q9. ¿Hacia dónde sigo?

- [README (la visión completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modelo local en 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — instalar el binario](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — la fuente de cada cifra](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — qué está shipped/simulado/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency & compliance — para RGPD/GDPR y el Reglamento de IA de la UE](../../explainers/data-residency-and-compliance.md)
- [Integrations — conectar tu agente](../../integrations/README.md)
- [Introducción en español](./README.md)

Licencia: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
