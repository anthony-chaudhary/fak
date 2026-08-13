---
title: "fak — установка и запуск (русскоязычная страница входа / Russian install guide)"
description: "Русскоязычная страница установки fak: одна статическая Go-бинарь, которая проверяет каждый tool call до запуска и переиспользует общую работу в длинной сессии — тот же цикл агента становится безопаснее, дешевле и быстрее. Self-host, соответствие 152-ФЗ, Apache-2.0."
---

# fak — установка и запуск

> Это **локализованная страница входа (entry point)**, а не полный перевод документации.
> Полная документация — на английском; эта страница даёт установку, карту уровней (tiers) и
> 60-секундную проверку, а затем направляет вас к
> [англоязычной документации](https://github.com/anthony-chaudhary/fak/blob/main/README.md) —
> она и есть источник истины.
> **Примечание:** этот перевод создан машиной и ожидает проверки носителем языка — нашли
> неточность, откройте issue/PR.
> Другие языки и полный список — на [i18n hub](../README.md).

Это входная дверь «установить и запустить». Плотная подача — в
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md).

**fak — это одна Go-бинарь**, которая садится между вашим AI-агентом и инструментами, которые
он вызывает: проверяет каждый tool call *до его запуска* и переиспользует повторяющуюся общую
работу в рамках длинной сессии. Результат: тот же самый цикл агента становится **безопаснее,
дешевле и быстрее**, без переписывания. Вы не переписываете агента — вы либо перенаправляете
один base URL на `fak serve`, либо оборачиваете существующего агента одной командой:

```bash
fak manage claude      # оборачивает вашего существующего агента одной командой
```

## Почему это важно для рынка России и СНГ

- **Счёт за токены растёт, а маржа считается в рублях.** fak переиспользует общую работу
  (system prompt, KV cache списка инструментов) в длинных сессиях: на прогоне 50 ходов × 5
  агентов это **~4.1× меньше работы**, чем у настроенного стека с тёплым кэшем (tuned
  warm-cache). Цифра ~60× (примерно 19 часов → примерно 19 минут) верна **только против
  наивного цикла, который каждый раз пересылает всё заново** — это не заголовочный показатель.
  Выигрыш от переиспользования доступен **только в self-host** и относится к read-heavy
  флотам. Провайдерский prompt-cache: fak держит кэшированный префикс **байт-в-байт**
  неизменным, сбрасывая старые средние ходы, поэтому скидка провайдера не теряется. fak
  **гарантирует** байтовую идентичность префикса; переиспользует ли провайдер кэш — решение
  провайдера, которое fak транслирует, а не заявляет.
- **Данные остаются в стране (152-ФЗ о локализации персональных данных).** fak — self-host в
  первую очередь: статическая бинарь перед **локальной моделью** или отечественным provider, с
  fail-closed residency на каждом backend, default-deny capability floor и tamper-evident
  журналом каждого tool call. Данные не покидают вашу машину.
- **Никаких трансграничных платежей.** fak — **Apache-2.0**, бесплатно, self-host: без карты,
  без cross-border-инвойса, без юрлица. `git clone` и `go install` — это весь путь.

## Требования

- **Go 1.26+.** `fak/go.mod` объявляет `go 1.26`. При стандартном `GOTOOLCHAIN=auto` более
  старый `go` сам скачает нужный toolchain при первой сборке (один раз нужна сеть); иначе
  поставьте Go 1.26 с <https://go.dev/dl/>. Проверка — `go version`.
- **Для Tier 0 больше ничего не нужно:** ни GPU, ни ключа, ни сети.
- **Tier 1** дополнительно требует любой OpenAI-совместимый сервер модели (например, Ollama).

## Уровни (tiers): от нулевой настройки к слитой модели

Между уровнями **ничего нового не устанавливается** — та же бинарь, только больше флагов.

| Tier | Что вы получаете | Настройка | Загрузки |
|---|---|---|---|
| **0 — Попробовать ядро** | Запуск/замер границы адъюдикации офлайн | `go build` | нет |
| **1 — Поставить перед реальной моделью** | Ядро перед моделью, которую вы обслуживаете (Ollama / vLLM / llama.cpp / облако) | + запущенный OpenAI-совместимый сервер | чат-модель |
| **1b — Локальная модель одной командой** | Локальная GGUF-модель in-kernel с вашим агентом — без ключа, без сети, без второго терминала | `fak manage --gguf qwen2.5:7b -- claude` | ~5 GB GGUF (кэшируется) |
| **2 — Слитая in-kernel модель** | Чистый Go-forward pass, которым владеет само ядро | + (реальные веса) экспорт Python | зависит от модели |

Модели: fak **управляет и кэширует** вашу модель, а не заменяет её. **Qwen2/Qwen3 и GLM-MoE**
проверены bit-exact в in-kernel reference engine; всё остальное (DeepSeek, Mistral, любая
open-weights модель) подключается по OpenAI-совместимому протоколу: Ollama / vLLM / SGLang /
llama.cpp / LM Studio или любой OpenAI-совместимый API.

## Установка бинари

**Собрать из клона (contributor):**

```bash
go build -o fak ./cmd/fak
```

**Установить через Go** (путь модуля — корень репозитория, ставится напрямую):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Замечание по установке для России и СНГ.** Где `proxy.golang.org` недоступен, может
> потребоваться `GOPROXY`. Подойдёт также self-hosted Go module proxy:
>
> ```bash
> export GOPROXY=https://<ваш-go-module-proxy>,direct
> go install github.com/anthony-chaudhary/fak/cmd/fak@latest
> ```

## 60-секундная проверка (без ключа, без модели, без GPU)

Default-deny capability floor проверяется **внутри ядра**, на том же call path — fail-closed:
действие, которого нет в allow-list, не запустится, как бы модель ни уговорили. Подозрительные
tool *results* удерживаются вне контекста модели целиком (карантин — это структура, а не
детектор). В живых тестах prompt injection достигал незащищённого baseline 5/5; fak отгородил
его 5/5.

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Куда дальше

- [README — полный обзор](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — локальная модель за 10 минут](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — установка и карта уровней](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — источник каждой цифры](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — что shipped / simulated / stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency и соответствие — для 152-ФЗ](../../explainers/data-residency-and-compliance.md)
- [Integrations — подключить вашего агента](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
