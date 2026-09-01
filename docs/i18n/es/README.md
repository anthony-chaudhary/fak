---
title: "fak — el Fused Agent Kernel (introducción en español / Spanish introduction)"
description: "Página de entrada en español para fak: un binario Go que verifica cada tool call antes de ejecutarlo — el mismo bucle de agente se vuelve más seguro, más barato y más rápido; self-host compatible con RGPD y registro de auditoría alineado con el Artículo 12 del Reglamento de IA de la UE."
---

# fak — el Fused Agent Kernel (introducción en español)

> Esta es una **página de entrada localizada (entry point)**, no una traducción completa de
> la documentación. La documentación canónica está en inglés — esta página te da el núcleo,
> la prueba de 60 segundos y la ruta de instalación, y luego te deriva a la
> [documentación en inglés](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta traducción está generada automáticamente y está pendiente de revisión
> nativa — se agradecen correcciones por issue/PR.
>
> **Auf Deutsch:** [Deutsch](../de/README.md) — todos los idiomas en el
> [hub i18n](../README.md).

## fak en una línea

**fak es un binario Go** que se sitúa entre tu agente de IA y sus tool calls — verifica
cada tool call *antes* de que se ejecute, y reutiliza el trabajo estable en las sesiones
largas. Resultado: el mismo bucle de agente se vuelve **más seguro, más barato y más
rápido**, sin que cambies nada más.

No reescribes tu agente — apuntas una base URL hacia `fak`, y cada tool call pasa primero
por el capability floor.

```bash
fak manage claude      # envuelve tu agente existente en un solo comando
```

## Por qué importa para este mercado

- **Los datos se quedan en tu infraestructura (RGPD/GDPR).** fak es self-host-first: un
  binario estático que se coloca delante de un **modelo local** (`fak manage --gguf …`) o del
  provider que elijas — fail-closed en cada backend, capability floor en default-deny, y un
  registro de auditoría a prueba de manipulación para cada tool call. No existe ninguna ruta
  de «reenviado por defecto a un tercer país» que tengas que analizar. En la UE (España
  incluida) esto es lo que pide el RGPD/GDPR. Detalles:
  [Data residency & compliance](../../explainers/data-residency-and-compliance.md).
- **El registro de auditoría del Reglamento de IA de la UE ya viene incluido (Artículo 12).**
  fak escribe un journal de decisiones append-only, encadenado por hash SHA-256, verificable
  offline con `fak audit verify` — la correspondencia entre las obligaciones del Artículo 12
  y el mecanismo entregado está documentada en
  [EU AI Act Article 12 conformance](../../standards/eu-ai-act-article-12-conformance.md).
  No es asesoría legal — pero sí el bloque técnico que un audit pide.
- **El precio del token es una palanca de margen (y de coste).** fak reutiliza el trabajo
  compartido de las sesiones largas (el system prompt + la lista de herramientas — el KV
  cache del trabajo ya hecho): en un run de 50 turnos × 5 agentes, **~4,1× menos trabajo**
  que un stack warm-cache afinado (~60× frente a un bucle ingenuo de reenvío; la cifra
  honesta es 4,1×). Sesiones largas más baratas por reutilización del prompt cache — lo que
  importa cuando eres sensible al coste, en España y en toda América Latina. Cada cifra está
  respaldada en
  [BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).
- **Apache-2.0, sin fricción de compra.** fak es libre, open source y self-hosted — sin
  contrato con proveedor, sin tarjeta, sin pago transfronterizo, sin cuenta. `git clone` y
  `go install` son el camino completo.
- **Un binario estático, cero dependencias externas.** Ops simples para un equipo pequeño —
  sin sidecar, sin authorizer aparte. Del portátil a la flota, el mismo artefacto; añades
  flags, no componentes.

## Qué problemas resuelve fak

- **Las sesiones largas dejan de ser caras.** El descuento de prompt-cache del provider solo
  se mantiene mientras el prefijo cacheado siga siendo idéntico byte por byte; fak descarta
  los turnos viejos y aun así mantiene el prefijo byte-idéntico — el descuento no se rompe.
- **Seguridad default-deny.** La política de permisos se ejecuta *dentro* del kernel, en el
  mismo call path. Impedir una acción irreversible no depende de «detectar» un ataque — la
  palanca nunca estuvo cableada. Es **fail-closed**, no fail-open.
- **Prompt injection / resultados de herramientas envenenados.** Los *resultados* de
  herramientas sospechosos entran en cuarentena para que ni siquiera lleguen al contexto del
  modelo — por estructura, no por un classifier.

## La prueba de 60 segundos (sin key, sin modelo, sin GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # inyección detenida, tarea igualmente completada
```

## Con tu modelo

fak no reemplaza tu modelo — lo gobierna y lo cachea. **Qwen2/Qwen3 y GLM-MoE** están
probados bit-exact en el motor de referencia in-kernel; el resto (Mistral, DeepSeek,
cualquier modelo open-weights) se sirve por la interfaz compatible con OpenAI — vía Ollama /
vLLM / SGLang / llama.cpp / LM Studio o cualquier API compatible con OpenAI.

## Hacia dónde seguir

- [Quickstart — una IA local gobernada en ~10 minutos](./quickstart.md)
- [Instalación — el binario y el mapa de tiers](./install.md)
- [FAQ — preguntas de primer contacto](./faq.md)
- [README (la visión completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modelo local en 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — instalar el binario](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — conectar tu agente](../../integrations/README.md)
- [Data residency & compliance — para RGPD/GDPR](../../explainers/data-residency-and-compliance.md)
- [EU AI Act Artículo 12 — conformidad del registro de auditoría](../../standards/eu-ai-act-article-12-conformance.md)
- [BENCHMARK-AUTHORITY — la fuente de cada cifra](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — qué está shipped/simulado/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licencia: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
