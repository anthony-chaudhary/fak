---
title: "fak — quickstart: una IA local gobernada en 10 minutos (Spanish quickstart)"
description: "Guía rápida en español para fak: de cero a una IA local gobernada en ~10 minutos — offline, sin key, sin factura cloud, los datos se quedan en tu máquina; envuelve tu agente con un modelo local en un solo comando. Self-host compatible con RGPD y el Reglamento de IA de la UE."
---

# fak — quickstart: una IA local gobernada en ~10 minutos

> Esta es una **página de entrada localizada (entry point)**, no una traducción completa de
> la documentación. La documentación canónica está en inglés — esta página te lleva de cero a
> una IA local gobernada y luego te deriva a la
> [documentación en inglés](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta traducción está generada automáticamente y está pendiente de revisión
> nativa — se agradecen correcciones por issue/PR.
>
> Todos los idiomas en el [hub i18n](../README.md). En español: la
> [introducción a fak](./README.md) y esta guía rápida ([quickstart](./quickstart.md)).

## La promesa

De cero a una **IA local gobernada** en unos **10 minutos**: funciona **offline**, sin API key,
sin factura cloud, los **datos se quedan en tu máquina**, y una **CPU basta** para modelos
pequeños. No reescribes tu agente — lo envuelves. Cada tool call pasa primero por el
capability floor del kernel, y el trabajo estable de las sesiones largas se reutiliza.

## La ruta más rápida (un comando)

Envuelve tu agente existente con un modelo local en un solo comando:

```bash
fak manage claude
```

Esto pone a fak delante de tu agente: el modelo corre en local (nada sale de tu máquina) y
cada tool call se verifica *antes* de ejecutarse. En la UE (España incluida) y en toda América
Latina, mantener los datos en tu propia infraestructura es lo que exigen el **RGPD** y el
**Reglamento de IA de la UE** — sin ninguna ruta de «reenviado por defecto a un tercer país»
que tengas que analizar.

## La prueba de 60 segundos (sin key, sin modelo, sin GPU)

Antes de descargar nada, comprueba el límite de tool calls. Estos comandos no necesitan ni
key, ni modelo, ni GPU:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # inyección detenida, tarea igualmente completada
```

El `DENY` es **estructural**: una acción que no está en la allow-list no puede ejecutarse, por
más que se convenza al modelo — es **fail-closed**, decidido *dentro* del kernel en el mismo
call path. En pruebas en vivo, la inyección de prompt alcanzó la baseline sin protección 5/5;
fak la bloqueó 5/5.

## Qué es fak

**fak es un binario Go** que se sitúa entre tu agente de IA y las herramientas que llama.
Revisa cada tool call *antes* de que se ejecute y reutiliza el trabajo compartido a lo largo
de una sesión larga. Resultado: el mismo bucle de agente se vuelve **más seguro, más barato y
más rápido**, sin reescritura. No reemplaza tu modelo — lo gobierna y lo cachea. **Qwen2/Qwen3
y GLM-MoE** están probados bit-exact en el motor de referencia in-kernel; el resto
(DeepSeek, Mistral, cualquier modelo open-weights) se sirve por la interfaz compatible con
OpenAI — vía Ollama / vLLM / SGLang / llama.cpp / LM Studio o cualquier API compatible con
OpenAI.

## Qué tan rápido es (con honestidad)

En un run medido de **50 turnos × 5 agentes**, la mejora honesta frente a un stack warm-cache
**afinado** es de **~4,1× menos trabajo**. Esa es la cifra de titular. El **~60×** (de unas
**19 horas a unos 19 minutos**) es cierto **solo frente al patrón ingenuo** de reenviar el contexto completo
en cada turno — nunca es el titular. La ganancia por reutilización es **solo en self-host** y
aplica a flotas con mucha lectura. En euros y en las monedas locales de la región, esto es una
palanca directa de coste y margen. Cada cifra está trazada a su commit y artefacto en
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).

Sobre el prompt cache del provider: fak mantiene el prefijo cacheado **byte por byte idéntico**
mientras descarta los turnos viejos del medio, para que el descuento del provider sobreviva.
fak **garantiza** la identidad byte a byte del prefijo; que el provider reutilice de hecho la
caché es decisión del provider, que fak relaya en lugar de afirmar.

## Instalar y hacia dónde seguir

Sin tarjeta, sin factura transfronteriza, sin entidad legal — **Apache-2.0**, libre, self-host.
`git clone` más `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` es el camino completo.

- [README (la visión completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modelo local en 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — instalar el binario](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — conectar tu agente](../../integrations/README.md)
- [Data residency & compliance — para RGPD/GDPR y el Reglamento de IA de la UE](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — la fuente de cada cifra](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — qué está shipped/simulado/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licencia: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
