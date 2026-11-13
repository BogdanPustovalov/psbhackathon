# API для веб-интерфейса — предложение

> Статус: **черновик на согласование**. Автор: Алёна (фронтенд).
> Цель — договориться о формате до того, как бэкенд и фронтенд начнут писаться
> параллельно. Фронтенд пока работает на моке по этому формату и переключится
> на настоящий API, когда он будет готов.

## Зачем менять текущий API

Кейс требует, чтобы в интерфейсе можно было **посмотреть документы и фрагменты,
которые подтверждают ответ** — без этого команду не допускают к экспертной оценке.
Ещё 25% экспертной оценки — «контроль достоверности», 15% — «интерфейс».
Сейчас `QueryResponse` отдаёт только `answer` и `sources` с `text` и `score`.
Этого не хватает, чтобы:

1. вести диалог (нет истории сообщений в запросе);
2. показать, из какого документа, пункта и редакции взят фрагмент;
3. открыть полный текст документа и подсветить в нём фрагмент;
4. показать, насколько ответ надёжен и чего не хватает в базе;
5. задать уточняющий вопрос с вариантами (например, «Альфа-Трейд» или «Альфа-Трейдинг»).

## Принцип: только добавляем поля

Все изменения **обратно совместимы**: новые поля необязательные и со значениями
по умолчанию. Старые клиенты и тесты не ломаются; бэкенд может заполнять поля
постепенно, а интерфейс показывает только то, что пришло.

---

## 1. `POST /api/v1/query` — вопрос в диалоге

### Запрос

```json
{
  "question": "А по какому адресу везти партию по спецификации от 15 октября 2024?",
  "history": [
    {"role": "user", "content": "Какой срок поставки по договору с Альфа-Трейд?"},
    {"role": "assistant", "content": "10 рабочих дней с даты подписания Спецификации [1]."}
  ],
  "top_k": 5,
  "use_reranker": true,
  "filters": null
}
```

| Поле | Тип | Что это |
|---|---|---|
| `question` | string | Текущий вопрос (как сейчас) |
| `history` | `ChatMessage[]`, по умолчанию `[]` | **Новое.** Предыдущие сообщения диалога, от старых к новым. Бэкенд использует их, чтобы понять вопросы вроде «а по нему?» |
| `top_k`, `use_reranker`, `filters` | — | Как сейчас |

`ChatMessage = {role: "user" | "assistant", content: string}`.
Фронтенд хранит историю сам и шлёт последние ~10 сообщений; на бэкенде сессии не нужны.

> На платформе автооценки история приходит внутри поля `question` одной строкой.
> Это забота `solution.py`; веб-API принимает историю отдельным полем.

### Ответ

```json
{
  "query": "А по какому адресу везти партию по спецификации от 15 октября 2024?",
  "answer": "Партию нужно доставить по адресу: Московская область, г. Химки, ул. Заводская, д. 15, склад № 3 [1]. Этот адрес действует для спецификаций, подписанных после 1 октября 2024 г. [1]; исходный адрес в п. 3.2 договора заменён [2].",
  "sources": [
    {
      "n": 1,
      "id": "dop-1-p-17-2024#0",
      "text": "Изложить пункт 3.2 договора в следующей редакции: «3.2. Товар доставляется Поставщиком по адресу: ...»",
      "score": 7.4,
      "metadata": {
        "doc_id": "dop-1-p-17-2024",
        "doc_title": "Дополнительное соглашение № 1 к договору поставки № П-17/2024",
        "doc_type": "contract",
        "section": "п. 3.2 (новая редакция)",
        "edition": {"label": "от 03.09.2024", "valid_from": "2024-09-03", "valid_to": null, "status": "active"},
        "char_start": 412,
        "char_end": 655
      }
    },
    {
      "n": 2,
      "id": "p-17-2024#3",
      "text": "3.2. Товар доставляется Поставщиком по адресу: г. Москва, ул. Складская, д. 4, стр. 2.",
      "score": 5.1,
      "metadata": {
        "doc_id": "p-17-2024",
        "doc_title": "Договор поставки № П-17/2024 с ООО «Альфа-Трейд»",
        "doc_type": "contract",
        "section": "п. 3.2",
        "edition": {"label": "от 12.02.2024", "valid_from": "2024-02-12", "valid_to": null, "status": "amended"},
        "char_start": 1180,
        "char_end": 1266
      }
    }
  ],
  "confidence": {"zone": "confident", "best_score": 7.4},
  "missing": [],
  "clarification": null,
  "debug": {"retrieved": 20, "reranked": true, "timings_ms": {"retrieve": 120, "rerank": 640, "generate": 2100}}
}
```

| Поле | Тип | Что это |
|---|---|---|
| `answer` | string | Текст ответа. Ссылки на источники — **`[n]`**, где `n` совпадает с `sources[].n`. Допустим простой Markdown: списки (`1.`, `-`) и `**жирный**` — для пошаговых инструкций |
| `sources[]` | `SourceDocument[]` | Фрагменты, **переданные в LLM**, в порядке номеров `n` |
| `sources[].n` | int | **Новое.** Номер фрагмента, тот же, что в промпте LLM и в `[n]` в ответе |
| `sources[].metadata` | object | Ключи ниже — **договорённость**, а не жёсткая схема |
| `confidence` | `{zone, best_score}` \| null | **Новое.** Зона уверенности от реранкера (`Reranker.select()`): `no_answer` / `uncertain` / `confident` |
| `missing` | string[] | **Новое.** Чего не хватает в базе, чтобы ответить полностью. Пусто — ничего |
| `clarification` | `{question, options[]}` \| null | **Новое.** Если вопрос неоднозначный: что уточнить и варианты-кнопки |
| `debug` | object | Как сейчас; предлагаю добавить `timings_ms` — пригодится для критерия «скорость» |

#### Ключи `metadata` фрагмента

| Ключ | Пример | Зачем интерфейсу |
|---|---|---|
| `doc_id` | `"p-17-2024"` | Открыть документ через `GET /documents/{doc_id}` |
| `doc_title` | `"Договор поставки № П-17/2024 с ООО «Альфа-Трейд»"` | Заголовок карточки источника |
| `doc_type` | `"contract"` / `"law"` / `"internal"` | Иконка и группировка |
| `section` | `"п. 3.2"`, `"ст. 99"` | «Ссылаться на конкретные положения» |
| `edition` | `{label, valid_from, valid_to, status}` | Показать редакцию; `status`: `active` / `amended` (изменён доп. соглашением) / `expired` (утратил силу) |
| `char_start`, `char_end` | `1180`, `1266` | Подсветить фрагмент в полном тексте |

Эти же поля полезны и самому пайплайну: шапка с названием документа в каждом
фрагменте и фильтр по контрагенту до реранкера заметно улучшают поиск
(см. отчёт по реранкеру). Их удобно заполнять **при индексации** и хранить
в metadata Chroma.

Если `char_start`/`char_end` посчитать сложно, их можно не отдавать — интерфейс
найдёт фрагмент в тексте документа поиском по подстроке.

#### Примеры особых случаев

**Ответа нет в базе** — `confidence.zone = "no_answer"`:

```json
{
  "answer": "В базе нет документов о сроке исковой давности по договору аренды А-08/2024.",
  "sources": [],
  "confidence": {"zone": "no_answer", "best_score": 0.9},
  "missing": ["Условия о сроке исковой давности в договоре аренды А-08/2024"],
  "clarification": null
}
```

**Нужно уточнение** — интерфейс покажет вопрос и кнопки; нажатие кнопки
отправляет вариант как следующий `question`:

```json
{
  "answer": "В базе два похожих договора поставки. Уточните, о каком идёт речь.",
  "sources": [],
  "confidence": {"zone": "uncertain", "best_score": 4.2},
  "missing": [],
  "clarification": {
    "question": "С каким контрагентом договор?",
    "options": ["П-17/2024 с ООО «Альфа-Трейд»", "ПТ-05/2023 с ООО «Альфа-Трейдинг»"]
  }
}
```

---

## 2. `GET /api/v1/documents` — список документов

Нужен для панели «База знаний» в интерфейсе.

```json
[
  {
    "doc_id": "p-17-2024",
    "doc_title": "Договор поставки № П-17/2024 с ООО «Альфа-Трейд»",
    "doc_type": "contract",
    "edition": {"label": "от 12.02.2024", "valid_from": "2024-02-12", "valid_to": null, "status": "amended"},
    "related": ["dop-1-p-17-2024"]
  }
]
```

`related` — связанные документы (доп. соглашения, новая редакция регламента). Необязательно.

## 3. `GET /api/v1/documents/{doc_id}` — полный текст документа

Открывается по клику на источник; в нём подсвечивается фрагмент
`text[char_start:char_end]`.

```json
{
  "doc_id": "p-17-2024",
  "doc_title": "Договор поставки № П-17/2024 с ООО «Альфа-Трейд»",
  "doc_type": "contract",
  "edition": {"label": "от 12.02.2024", "valid_from": "2024-02-12", "valid_to": null, "status": "amended"},
  "related": ["dop-1-p-17-2024"],
  "text": "ДОГОВОР ПОСТАВКИ № П-17/2024\nг. Москва — 12 февраля 2024 г.\n..."
}
```

`text` — исходный текст документа без изменений, чтобы смещения `char_start`/`char_end`
указывали в правильное место. `404`, если документа нет.

---

## Набросок схем Pydantic

```python
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    filters: dict[str, Any] | None = None


class SourceDocument(BaseModel):
    id: str
    n: int | None = None
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Confidence(BaseModel):
    zone: Literal["no_answer", "uncertain", "confident"]
    best_score: float | None = None


class Clarification(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    confidence: Confidence | None = None
    missing: list[str] = Field(default_factory=list)
    clarification: Clarification | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class Edition(BaseModel):
    label: str
    valid_from: str | None = None
    valid_to: str | None = None
    status: Literal["active", "amended", "expired"] = "active"


class DocumentInfo(BaseModel):
    doc_id: str
    doc_title: str
    doc_type: Literal["contract", "law", "internal"]
    edition: Edition | None = None
    related: list[str] = Field(default_factory=list)


class DocumentDetail(DocumentInfo):
    text: str
```

## Что можно делать по шагам

Ничего не обязательно делать сразу; интерфейс показывает то, что пришло.

1. **Минимум для допуска:** `sources[].n`, `metadata.doc_id`/`doc_title`/`section`,
   `GET /documents/{doc_id}`.
2. **Диалог:** `history` в запросе.
3. **Достоверность:** `confidence`, `missing`, `metadata.edition`.
4. **Плюсы:** `clarification`, `GET /documents`, `debug.timings_ms`, `char_start`/`char_end`.

## Вопросы к команде

1. Кто берёт изменения схем и эндпоинты документов на бэкенде?
2. Как формировать `doc_id` — слаг из имени файла базы знаний (`p-17-2024`) подходит?
3. Где LLM будет возвращать `missing` и `clarification`: отдельным JSON-ответом модели
   (structured output) или отдельным коротким запросом-проверкой в зоне `uncertain`?
4. Переименовываем `docker-compose.yml` в `compose.yaml`, как требует кейс?
