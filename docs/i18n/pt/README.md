---
title: "fak — o Fused Agent Kernel (introdução em português / Portuguese introduction)"
description: "Página de entrada em português para o fak: um binário Go que revisa cada tool call antes de executá-lo — o mesmo laço de agente fica mais seguro, mais barato e mais rápido; self-host alinhado à LGPD (Brasil) / RGPD (Portugal), Apache-2.0."
---

# fak — o Fused Agent Kernel (introdução em português)

> Esta é uma **página de entrada localizada (entry point)**, não uma tradução completa da
> documentação. A documentação canônica está em inglês — esta página lhe dá o núcleo do fak,
> a prova de 60 segundos e o caminho de instalação, e então o encaminha para a
> [documentação em inglês](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta tradução foi gerada por máquina e ainda aguarda revisão por falante nativo —
> correções são bem-vindas via issue/PR.
>
> **Noutros idiomas:** [Español](../es/README.md) · [Français](../fr/README.md) ·
> [Deutsch](../de/README.md) — todos os idiomas no [hub i18n](../README.md).

## fak em uma linha

**fak é um binário Go** que fica entre o seu agente de IA e os tool calls que ele faz —
revisa cada tool call *antes* de executá-lo e reutiliza o trabalho compartilhado repetido ao
longo de sessões longas. Resultado: o mesmo laço de agente fica **mais seguro, mais barato e
mais rápido**, sem reescrever nada.

Você não reescreve o seu agente — apenas aponta uma base URL para `fak serve`, e cada tool
call passa primeiro pelo capability floor. Ou envolve um agente existente com um só comando:

```bash
fak manage claude      # envolve o seu agente atual em um único comando
```

## Por que times no Brasil e em Portugal deveriam se importar

- **O token é cobrado em dólar; a margem é em BRL e EUR.** fak reutiliza o trabalho
  compartilhado das sessões longas (system prompt, lista de ferramentas — o KV cache do que
  já foi feito): num run de 50 turnos × 5 agentes, **~4,1× menos trabalho** que um stack
  warm-cache já afinado. (O número ~60× — de cerca de 19 horas para cerca de 19 minutos —
  vale **apenas versus o padrão ingênuo (naive)** de reenviar tudo a cada vez; não é a
  manchete.) Menos trabalho por token é uma alavanca direta de custo e de margem, em reais e
  em euros. O ganho por reutilização é **self-host only** e se aplica a frotas com muita
  leitura.
- **Os dados ficam no país (LGPD no Brasil / RGPD em Portugal).** fak é self-host primeiro:
  um binário estático que se coloca à frente de um **modelo local** ou do provider que você
  escolher — residency fail-closed em cada backend, capability floor em default-deny, e um
  log de auditoria à prova de adulteração para cada tool call. Não há um caminho de
  "reencaminhado por padrão para outro país" que você precise analisar — o que é exatamente o
  que a LGPD (Brasil) e o RGPD (Portugal) exigem.
- **Nenhum pagamento transfronteiriço.** fak é **Apache-2.0**, gratuito, self-host — sem
  cartão de crédito, sem invoice internacional, sem entidade jurídica. `git clone` mais
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` é o caminho inteiro.
- **Um binário estático, zero dependência externa.** Ops simples para um time pequeno — sem
  sidecar, sem authorizer separado. Do laptop à frota é o mesmo artefato; você adiciona
  flags, não componentes.

## Que problemas o fak resolve

- **Sessões longas continuam baratas.** O desconto de prompt-cache do provider só se mantém
  enquanto o prefixo cacheado permanecer byte a byte idêntico; fak descarta os turns antigos
  do meio e ainda assim mantém o prefixo **byte-idêntico**. fak **garante** a identidade
  byte a byte do prefixo; se o provider de fato reutiliza o cache é decisão do provider, que
  o fak repassa em vez de afirmar por conta própria.
- **Segurança default-deny.** A política de permissões roda *dentro* do kernel, no mesmo call
  path — **fail-closed**. Uma ação que nunca esteve na allow-list não pode rodar, não importa
  o quanto o modelo tenha sido convencido. Impedir uma ação irreversível não depende de
  "detectar" o ataque.
- **Quarentena de prompt injection.** *Resultados* de ferramentas suspeitos são retidos
  inteiramente fora do contexto do modelo — por estrutura, não por um classificador. O
  detector que os sinaliza é tratado como ~100% evadível por projeto: um bônus, nunca o piso.
  Em testes ao vivo, a injeção de prompt atingiu a baseline desprotegida 5/5; o fak a barrou
  5/5.

## A prova de 60 segundos (sem key, sem modelo, sem GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injeção bloqueada, tarefa ainda concluída
```

## Com o seu modelo

fak governa e faz cache do seu modelo — não o substitui. **Qwen2/Qwen3 e GLM-MoE** estão
provados bit-exact no motor de referência in-kernel; o restante (DeepSeek, Mistral,
qualquer modelo open-weights) é servido pela interface compatível com OpenAI — via Ollama /
vLLM / SGLang / llama.cpp / LM Studio ou qualquer API compatível com OpenAI.

## Para onde ir agora

- [Início rápido — um modelo local em 10 minutos](./quickstart.md)
- [Instalação — o binário e o mapa de tiers](./install.md)
- [FAQ — perguntas de primeiro contato](./faq.md)
- [README (a visão completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — um modelo local em 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — instalar o binário](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — conectar o seu agente](../../integrations/README.md)
- [Data residency & compliance — para a LGPD / RGPD](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — a fonte de cada número](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — o que é shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licença: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
