---
title: "fak — instalación y primeros pasos (install / getting started en español)"
description: "Página de entrada en español para instalar y ejecutar fak: un único binario Go que verifica cada tool call antes de ejecutarlo. Requisitos, mapa de tiers y los dos comandos de instalación; self-host compatible con RGPD y el Reglamento de IA de la UE."
---

# fak — instalación y primeros pasos

> Esta es una **página de entrada localizada (entry point)**, no una traducción completa de
> la documentación. La documentación canónica está en inglés — esta página te lleva de un
> checkout limpio a un kernel en marcha, con los dos comandos de instalación y el mapa de
> tiers, y luego te deriva a la
> [documentación en inglés](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta página está generada automáticamente y está pendiente de revisión nativa —
> se agradecen correcciones por issue/PR.
>
> Todos los idiomas en el [hub i18n](../README.md). El punto de partida en español es el
> [README](./README.md).

Esta es la **puerta de entrada de instalar-y-ejecutar**. El pitch denso está en el
[README](./README.md). Aquí vas de un checkout limpio a un kernel funcionando y a servir un
modelo por detrás, con comandos que se pueden copiar y pegar.

## 0. Requisitos

- **Go 1.26+.** `fak/go.mod` declara `go 1.26`. Con el `GOTOOLCHAIN=auto` por defecto de Go,
  un `go` más antiguo descargará el toolchain correcto de forma automática en el primer build
  (necesita red una vez); si no, instala Go 1.26 desde <https://go.dev/dl/>. Comprueba con
  `go version`.
- **Para el Tier 0, nada más:** sin GPU, sin API key, sin red.
- **El Tier 1** además necesita cualquier servidor de modelo compatible con OpenAI (p. ej.
  Ollama).

## 1. El mapa de tiers

`fak` es **un único binario Go**: un artefacto estático sin dependencias externas. Ese mismo
binario es toda la superficie — el gateway, el KV cache y el motor de enrutado, los
ahorradores de tokens y, en la misma costura, el capability floor, la cuarentena de
resultados y el registro de auditoría. Hay cuatro cosas que puedes hacer con él, en orden
creciente de coste de setup, y **entre una y otra no se instala nada nuevo**:

| Tier | Qué obtienes | Setup |
|---|---|---|
| **0 — Probar el kernel** | Ejecutar/medir la frontera de adjudicación offline | `go build` |
| **1 — Poner delante un modelo real** | El kernel delante de un modelo que sirves en otro sitio (Ollama / vLLM / llama.cpp / un proveedor cloud) | + un servidor compatible con OpenAI |
| **1b — Modelo local en un comando** | Un modelo GGUF local in-kernel con tu agente existente — sin key, sin red, sin segundo terminal | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — El modelo fusionado in-kernel** | La forward pass pura-Go que el kernel posee | + (pesos reales) export en Python |

Si solo quieres **servir un modelo útil con fak por delante**, quieres el **Tier 1**. El
modelo in-kernel del Tier 2 es una *forward pass de referencia* probada bit-a-bit contra
HuggingFace, no un motor de serving con calidad de chat.

## 2. Instalar el binario

**Con Go.** El module path `github.com/anthony-chaudhary/fak` es la raíz del repositorio, así
que se instala directamente:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

**Desde el clon (contribuidor):**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

> **Nota Windows.** Compila con `go build -o fak.exe ./cmd/fak`: `go build -o fak` (sin
> extensión) deja un fichero `fak` literal que cmd.exe / PowerShell no pueden lanzar por
> nombre. Luego invócalo como `.\fak.exe` donde esta guía escribe `./fak`.

No reescribes tu agente: apuntas una base URL hacia `fak`, o envuelves un agente existente en
un solo comando:

```bash
fak manage claude      # envuelve tu agente existente en un solo comando
```

## 3. La prueba de 60 segundos (sin key, sin modelo, sin GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # inyección detenida, tarea igualmente completada
```

El capability floor es **default-deny** y se comprueba *dentro* del kernel, en el mismo call
path (**fail-closed**): una acción que no está en la allow-list no puede ejecutarse, por más
que se convenza al modelo. En las pruebas en vivo, la prompt injection alcanzó la baseline sin
proteger 5/5; fak la aisló 5/5.

## Por qué importa para España, la UE y América Latina

- **Los datos se quedan en tu infraestructura (RGPD y Reglamento de IA de la UE).** fak es
  self-host-first: un binario estático delante de un **modelo local** o del provider que
  elijas, sin ninguna ruta de «reenviado por defecto a un tercer país» que analizar. El
  registro de auditoría append-only encadenado por hash es el bloque técnico que un audit del
  Reglamento de IA de la UE pide. Sin tarjeta, sin factura transfronteriza, sin entidad
  legal: `git clone` y `go install` son el camino completo.
- **El precio del token es una palanca de margen, en euros y en moneda local.** fak reutiliza
  el trabajo compartido de las sesiones largas: en un run de 50 turnos × 5 agentes, **~4,1×
  menos trabajo** que un stack warm-cache afinado. (La cifra de ~60×, de unas 19 horas a unos
  19 minutos, solo es cierta frente al patrón ingenuo de reenviar el contexto completo; nunca es el titular.)
  La reutilización es **solo en self-host** y aplica a flotas con mucha lectura. fak mantiene
  el prefijo cacheado idéntico byte por byte mientras descarta los turnos intermedios viejos,
  así que el descuento de prompt-cache del provider sobrevive — fak **garantiza** la identidad
  byte a byte del prefijo; que el provider reutilice la caché es decisión del provider, que
  fak transmite en lugar de afirmar.
- **fak gobierna y cachea tu modelo; no lo reemplaza.** **Qwen2/Qwen3 y GLM-MoE** están
  probados bit-exact en el motor de referencia in-kernel; el resto (DeepSeek, Mistral,
  cualquier modelo open-weights) se sirve por la interfaz compatible con OpenAI — vía Ollama /
  vLLM / SGLang / llama.cpp / LM Studio o cualquier API compatible con OpenAI.

## Hacia dónde seguir

- [README en español](./README.md) — el pitch completo y el porqué en tu idioma.
- [README (la visión completa, en inglés)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modelo local en 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — la referencia de instalación completa](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Data residency & compliance — para RGPD y el Reglamento de IA de la UE](../../explainers/data-residency-and-compliance.md)
- [Integrations — conectar tu agente](../../integrations/README.md)
- [BENCHMARK-AUTHORITY — la fuente de cada cifra](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — qué está shipped/simulado/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licencia: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
