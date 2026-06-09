# Дизайн: Облачный backend `stomserver` (Plan 2)

**Дата:** 2026-06-09
**Статус:** черновик дизайна (одобрен по секциям, ожидает финального ревью)
**Родительский дизайн:** `docs/superpowers/specs/2026-06-09-stom-cbct-segmentation-design.md` §5
**Зависит от:** пакет `stomcore` (Plan 1, готов)

## 1. Цель и контекст

Backend коммерческого продукта сегментации CBCT (аналог DiagnoCAT): принять
исследование, запустить **реальную** AI-сегментацию, вернуть маску. Гибридная
архитектура — этот backend обслуживает будущий десктоп-клиент (Plan 3).

**Модель сегментации:** DentalSegmentator (предобученная nnU-Net v2, лицензия
CC-BY 4.0, коммерческое использование разрешено при атрибуции). Сегментирует:
верхний череп/верхняя челюсть, нижняя челюсть, верхние зубы (группой), нижние
зубы (группой), нижнечелюстной канал. Инференс на CPU (GPU в окружении нет —
медленно, но корректно). Per-FDI нумерация зубов — будущий ML-план (обучение на
ToothFairy2), вне Plan 2.

**Ограничения окружения:** нет GPU/CUDA, нет Docker. 16 CPU, 19 ГБ RAM.

## 2. Границы Plan 2

**Входит:**
- Новый пакет `stomserver` (зависит от `stomcore`): API-сервер (FastAPI),
  воркер (RQ), слой БД (SQLAlchemy), слой `Storage`, конфиг.
- Полный пайплайн: загрузка NIfTI → задача в очереди → реальная сегментация
  DentalSegmentator → маска (`.nii.gz` + `label_map` JSON) → выдача.
- Авторизация статическим bearer-токеном, изоляция по `account_id`.
- Дополнение `stomcore`: `mask_io.py` (`save_mask_nifti`/`load_mask_nifti` +
  сериализация `label_map`) — нужно серверу и клиенту, живёт в общем ядре.
- Скрипт скачивания весов DentalSegmentator (веса не коммитятся в git).
- Файл атрибуции/цитирования (требование CC-BY).

**НЕ входит (отложено, интерфейсы заложены):** S3-реализация `Storage` (только
интерфейс + локальная), PostgreSQL в проде (dev на SQLite через ту же ORM),
регистрация/логин/роли, per-FDI нумерация зубов, GPU-оптимизация, автодетекция
патологий, интеграция с PACS.

## 3. Архитектура

Три запускаемых компонента + общие модули:
- **API-сервер** (FastAPI) — приём, постановка задач, статусы, выдача масок. Не
  считает.
- **Воркер** (RQ) — берёт задачи, гоняет DentalSegmentator на CPU, кладёт маску.
  Stateless: вся информация в задаче/БД/хранилище.
- **Очередь** (Redis + RQ; тесты — `fakeredis`) — развязка API и воркера.
- Общие: слой БД (SQLAlchemy), слой `Storage` (локальный диск / S3 позже),
  `config.py`.

**Границы:** API ↔ воркер общаются только через очередь + хранилище (без прямых
вызовов). `Storage` и БД спрятаны за интерфейсами; прод-реализации подключаются
сменой конфига.

**Раскладка пакета:**
```
src/stomcore/
  mask_io.py                 # НОВОЕ: save/load mask + label_map JSON
src/stomserver/
  config.py                  # настройки (env): DB_URL, STORAGE_DIR, REDIS_URL, MODEL_DIR, лимиты
  db/
    models.py                # SQLAlchemy: Account, ApiToken, Study, Job
    session.py               # engine/session factory
  storage/
    base.py                  # интерфейс Storage
    local.py                 # LocalFileStorage
  api/
    app.py                   # сборка FastAPI
    auth.py                  # зависимость: токен -> account
    deps.py                  # сессия БД, storage, очередь
    schemas.py               # Pydantic-схемы
    routes_studies.py
    routes_jobs.py
    errors.py                # глобальный обработчик -> {detail, code}
  segmentation/
    worker.py                # run_segmentation(job_id): оркестрация
    runner.py                # DentalSegmentatorRunner (прячет nnUNetv2_predict)
    labels.py                # DENTALSEGMENTATOR_LABELS (id -> {name,color,visible})
scripts/
  download_weights.py        # качает веса DentalSegmentator с Zenodo
  create_account.py          # админ-CLI: создать Account + выпустить ApiToken (печатает токен один раз)
tests/
  ... (см. §8)
NOTICE                       # атрибуция DentalSegmentator + nnU-Net (CC-BY)
```

## 4. Модель данных и хранилище

**БД (SQLAlchemy ORM; dev = SQLite, прод = PostgreSQL — та же ORM):**
- `Account` — `id`, `name`, `created_at`. Арендатор (клиника); основа изоляции.
- `ApiToken` — `id`, `token_hash` (хранится ХЭШ, не токен), `account_id`→Account,
  `created_at`.
- `Study` — `id`, `account_id`, `original_filename`, `storage_key`, `shape`,
  `spacing`, `created_at`.
- `Job` — `id`, `study_id`, `account_id`, `status`
  (`queued`/`running`/`done`/`failed`), `error`, `mask_storage_key`,
  `model_name` (`dentalsegmentator`), `created_at`, `updated_at`.

Все выборки фильтруются по `account_id` из токена; чужой `id` → `404`.

**Выпуск токенов:** `scripts/create_account.py` создаёт `Account`, генерирует
случайный токен (`secrets.token_urlsafe`), сохраняет его ХЭШ в `ApiToken` и
печатает сырой токен ОДИН раз в stdout. Это единственный способ получить токен
(нет регистрации). Тесты создают `Account`/`ApiToken` напрямую через ORM.
Хэширование токена — общая функция (`api/auth.py::hash_token`), переиспользуется
скриптом и зависимостью аутентификации (DRY).

**Слой `Storage` (интерфейс):** `put(key, data: bytes)`, `get(key) -> bytes`,
`exists(key) -> bool`, `delete(key)`. Реализация Plan 2 — `LocalFileStorage`
(корневая папка из конфига).

**Ключи хранилища:**
```
{account_id}/studies/{study_id}/volume.nii.gz
{account_id}/studies/{study_id}/mask.nii.gz
{account_id}/studies/{study_id}/mask_labels.json
```
Префикс `account_id` = изоляция и на уровне хранилища.

**Геометрический инвариант:** nnU-Net предсказывает в геометрии входного тома →
маска в той же геометрии. Воркер перед сохранением проверяет
`SegmentationMask.is_compatible_with(volume)`; рассогласование → `failed`.

## 5. API и авторизация

Каждый запрос: `Authorization: Bearer <token>`. Зависимость хэширует токен, ищет
`ApiToken.token_hash`, достаёт `account_id`. Нет/неверный → `401`.

| Метод | Путь | Действие | Ответ |
|---|---|---|---|
| `POST` | `/studies` | Загрузка тома (multipart `.nii.gz`); валидация через `stomcore.load_volume_nifti`; в `Storage` + `Study`. | `201 {study_id, shape, spacing}` |
| `POST` | `/studies/{id}/segment` | Ставит `Job` в очередь. | `202 {job_id, status}` |
| `GET` | `/jobs/{id}` | Статус. | `200 {job_id, status, error?}` |
| `GET` | `/studies/{id}/masks` | Если `done` — `mask.nii.gz` (`application/gzip`); иначе `409`. | `200`/`409` |
| `GET` | `/studies/{id}/masks/labels` | `mask_labels.json`. | `200`/`409` |
| `GET` | `/healthz` | Живость (без токена). | `200` |

Загрузка: файл во временный путь → `stomcore.load_volume_nifti` (это и
валидация; кривой NIfTI → `400`) → метаданные в `Study`.

## 6. Воркер сегментации и дополнение `stomcore`

**`stomcore/mask_io.py` (НОВОЕ):**
- `save_mask_nifti(mask, nifti_path, labels_path)` — целочисленный том маски в
  `.nii.gz` (через `sitk_interop`) + `label_map` в JSON-сайдкар
  (`{label_id: {name, color, visible}}`).
- `load_mask_nifti(nifti_path, labels_path) -> SegmentationMask` — обратное.
- Round-trip-тест с неединичной direction (закрывает тест-пробел из ревью Plan 1).

**Воркер (`stomserver/segmentation/`):**
- `worker.py::run_segmentation(job_id)`:
  1. Грузит `Job`/`Study`, ставит `running`.
  2. Качает том из `Storage` во временную папку.
  3. `DentalSegmentatorRunner.predict(volume) -> labels_array`.
  4. Оборачивает в `SegmentationMask` с `DENTALSEGMENTATOR_LABELS`, проверяет
     `is_compatible_with(volume)`.
  5. `save_mask_nifti` → `Storage`, пишет `mask_storage_key`, ставит `done`.
  6. Любая ошибка → `failed` + текст в `Job.error`; воркер берёт следующую.
- `runner.py::DentalSegmentatorRunner` — инкапсулирует `nnUNetv2_predict`
  (входная папка формата nnU-Net `*_0000.nii.gz`, запуск, чтение результата).
  Прячет детали за `predict(volume) -> labels_array`.
- `labels.py::DENTALSEGMENTATOR_LABELS` — фикс. маппинг id→{name, color, visible}
  для 5 структур.

**Веса:** `scripts/download_weights.py` качает
`Dataset112_DentalSegmentator_v100.zip` с Zenodo в `MODEL_DIR`, проверяет
контрольную сумму. Веса в git не коммитятся (gitignore). `NOTICE` с
атрибуцией DentalSegmentator + nnU-Net.

**Тесты без весов:** `DentalSegmentatorRunner` за интерфейсом → `FakeRunner`
возвращает детерминированную маску. Весь путь воркера тестируется без nnU-Net и
сети. Реальный прогон — отдельный `@pytest.mark.slow`, скипается без весов.

## 7. Обработка ошибок

- Загрузка: не-NIfTI/битый → `400`; превышен лимит → `413`.
- Авторизация: нет/неверный токен → `401`; чужой `study_id`/`job_id` → `404`
  (не `403` — не раскрываем существование).
- Постановка задачи: Redis недоступен → `503` (не 500-трейсбек).
- Воркер: любая ошибка (инференс/геометрия/хранилище/OOM) → `Job.failed` +
  текст; воркер продолжает.
- Выдача маски: не `done` → `409`; `failed` → `409` + текст; файл пропал → `404`.
- Глобальный обработчик FastAPI → единый JSON `{detail, code}` без трейсбеков.

## 8. Тестирование

- **Юнит:** `mask_io` (round-trip с неединичной direction); `LocalFileStorage`
  (put/get/exists/delete на `tmp_path`); `auth` (хэш токена, поиск аккаунта,
  отказ); `labels` (маппинг).
- **API** (FastAPI `TestClient`; БД = SQLite in-memory; очередь = `fakeredis`;
  storage = local в `tmp_path`): happy-path загрузки → study_id; постановка
  задачи; статусы; изоляция (чужой токен → 404); 401 без токена; 400 на битом
  NIfTI; 409 на незавершённой маске.
- **Воркер** (`FakeRunner`): queued→running→done; сохранение маски+labels;
  проверка геометрии; ошибка раннера → failed (очередь жива).
- **Интеграционный сквозной:** `TestClient` + синхронный прогон через
  `FakeRunner`: загрузил → сегментировал → скачал маску → `load_mask_nifti`
  совпадает по геометрии с томом.
- **Slow/опциональный:** реальный DentalSegmentator на крошечном томе,
  `@pytest.mark.slow`, скип без весов.
- **Стек:** `pytest`, `httpx`/`TestClient`, `fakeredis`, `SQLAlchemy`.

## 9. Открытые вопросы на будущее (вне Plan 2)

- S3-реализация `Storage` + миграция на PostgreSQL (Alembic).
- Полноценные аккаунты/логин/роли (отдельный auth-подпроект).
- Per-FDI нумерация зубов (обучение на ToothFairy2, GPU).
- GPU-инференс и очередь под нагрузкой (масштабирование воркеров).
- Точная проверка лицензионной атрибуции DentalSegmentator/nnU-Net юристом перед
  коммерческим релизом.
