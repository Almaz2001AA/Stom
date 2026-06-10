# Десктоп-клиент Stom (`stomclient`) — дизайн (Plan 3)

**Дата:** 2026-06-10
**Статус:** утверждён к реализации (бриф пройден)
**Предшественники:** Plan 1 (`stomcore`, готов), Plan 2 (`stomserver`, готов)

## 1. Цель

Десктоп-приложение для врача: импортировать CBCT (DICOM), отправить на
AI-сегментацию в облако (`stomserver`), посмотреть результат наложением масок
на срезах и сделать линейные измерения в мм. Это первый пользовательский
интерфейс продукта — аналог просмотрщика DiagnoCAT в MVP-объёме.

## 2. Объём MVP (и что отложено)

**Входит в MVP:**
- Импорт DICOM-серии CBCT → `Volume` (через `stomcore.DicomLoader`).
- Облачный цикл: upload (NIfTI) → segment → poll статуса → download маски.
- Просмотр: один большой 2D-срез с переключением плоскости
  (Axial/Sagittal/Coronal), скролл по срезам, window/level.
- Наложение масок: полупрозрачные цветные лейблы, список масок с
  переключением видимости и отображением цвета.
- Линейные измерения в мм (с учётом spacing геометрии).
- Экспорт: текущий срез в PNG; сохранение скачанной маски `.nii.gz` на диск.
- Настройки: диалог с URL сервера и API-токеном, сохранение в конфиг-файл.

**Отложено в fast-follow (вне Plan 3):**
- Ручная правка масок (`MaskEditor`: кисть/ластик/undo).
- 3D-объёмный рендер и мультиплоскостной «крест» 2×2 (тогда вводим VTK).
- Угловые измерения.
- Анонимизация DICOM перед отправкой.
- Переподключение к незавершённой облачной задаче по `job_id` после
  перезапуска приложения.

## 3. Архитектурный принцип

**Тонкий view / тестируемое ядро.** Вся логика (конфиг, сетевой клиент,
рендеринг срезов, математика измерений, машина состояний) живёт в чистых
Python-модулях без импортов Qt/VTK и тестируется headless через `pytest`.
Виджеты Qt — тонкие адаптеры: берут данные из ядра, рисуют, шлют события
обратно. `stomcore` — единственный источник правды о геометрии; маски обязаны
совпадать с томом (инвариант проверяется при загрузке маски).

**Решение по 2D-рендерингу (подход A):** срезы рендерятся через
`numpy → QImage` + `QPainter`, без VTK в MVP. VTK вводится позже только для
3D-fast-follow. Обоснование: 100% headless-тестируемость математики
(оконно-уровневое преобразование, композит наложения, измерения — чистые
функции над `ndarray`), отсутствие VTK-сложности (события, offscreen-рендер) в
MVP. Сознательно расходится с исходной заметкой «VTK для срезов» ради
скорости и тестируемости MVP.

## 4. Контракт с backend (`stomserver`)

Клиент общается строго через эти эндпоинты (Bearer-токен в заголовке
`Authorization`):

| Метод | Эндпоинт | Запрос → Ответ |
|---|---|---|
| POST | `/studies` | multipart-файл `.nii.gz` → `{study_id, shape, spacing}` (201) |
| POST | `/studies/{id}/segment` | → `{job_id, status, error}` (202) |
| GET | `/jobs/{id}` | → `{job_id, status, error}` (status: `queued/running/done/failed`) |
| GET | `/studies/{id}/masks` | → тело `.nii.gz` (application/gzip), 409 если не готово |
| GET | `/studies/{id}/masks/labels` | → `mask_labels.json` (application/json) |

Ошибки сервера приходят как `{detail, code}`. 401 несёт заголовок
`WWW-Authenticate: Bearer`. Эти форматы фиксируются контрактными тестами.

## 5. Компоненты (модули пакета `stomclient`)

### Ядро (без Qt/VTK)

- **`config.py`** — `ClientConfig(server_url: str, token: str | None)`.
  `load()` / `save()` в `~/.config/stom/client.toml` (путь
  переопределяется для тестов). При сохранении токена файл получает права
  `0600`. Опция «не сохранять токен» (тогда токен только в памяти сессии).

- **`cloud_client.py`** — `CloudClient(base_url, token, *, timeout, retries)`.
  Методы:
  - `upload_study(nifti_bytes: bytes, filename: str) -> StudyInfo`
  - `start_segmentation(study_id: int) -> str` (job_id)
  - `poll_status(job_id: int) -> JobStatus`
  - `download_mask(study_id: int) -> tuple[bytes, bytes]` (маска + labels JSON)

  Скрывает HTTP (библиотека `httpx`). Таймауты и ретраи с
  backoff на сетевых сбоях/5xx. Типизированные исключения:
  `AuthError` (401), `NotReady` (409), `CloudError` (прочие). Чистый —
  тестируется против мок-HTTP.

- **`slice_renderer.py`** — чистые функции:
  - `extract_slice(volume, plane, index) -> ndarray` (2D-срез по плоскости)
  - `apply_window_level(slice2d, center, width) -> ndarray[uint8]`
  - `composite_overlay(gray_uint8, mask_slice, label_map, alpha) -> ndarray`
    (RGB; учитывает цвет/видимость лейблов)

  Преобразование `ndarray → QImage` живёт в view, не здесь.

- **`measurement.py`** — `LinearMeasurement(p0, p1, plane, geometry)` →
  свойство `length_mm` (евклидово расстояние с учётом spacing активной
  плоскости). `MeasurementSet` — список измерений текущей сессии. Чистый.

- **`app_controller.py`** — `AppController`. Машина состояний сессии:
  `EMPTY → LOADED → UPLOADING → SEGMENTING → MASK_READY` (+ `FAILED`).
  Хранит: текущий `Volume`, `SegmentationMask` (если есть), активную
  плоскость/индекс/W-L, набор измерений, идентификаторы `study_id`/`job_id`,
  статус облака. Драйвит `CloudClient` и рендерер. Сетевые операции
  выполняет в worker-потоке; прогресс/смену состояния отдаёт наблюдателю
  (Qt-агностичный колбэк, к которому view цепляет сигналы). Тестируется с
  фейковым `CloudClient` и синтетическим `Volume`.

### View (тонкий, Qt; PySide6)

- **`main_window.py`** — `QMainWindow`. Левая панель: имя/форма исследования,
  статус облачной задачи, список масок (чекбокс видимости + плашка цвета),
  переключатель инструмента «Измерение». Центр — `SliceWidget`. Тулбар/меню:
  Open DICOM, Upload & Segment, переключатель плоскости, скролл срезов,
  Save PNG, Save Mask, Settings.

- **`slice_widget.py`** — `QWidget`, показывает `QImage` из `slice_renderer`.
  Обработка ввода (делегирует математику в ядро): колесо/стрелки — индекс
  среза; drag — window/level; click-drag в режиме измерения — линия; zoom/pan.

- **`settings_dialog.py`** — форма URL + токен, сохранение через `config.py`,
  кнопка «Проверить соединение» (пинг здоровья API).

- **`__main__.py`** / энтрипоинт `stom-client` — собирает `AppController` +
  `MainWindow`, запускает Qt-цикл.

### Переиспользование `stomcore`

`DicomLoader` (DICOM→`Volume`), `nifti_io.save_volume_nifti`/`load_volume_nifti`
(том→байты для upload и обратно), `mask_io.load_mask_nifti` (маска+labels из
скачанных байт), типы `Volume`/`Geometry`/`SegmentationMask`. Клиент не
дублирует ни геометрию, ни DICOM-логику.

## 6. Поток данных (сквозной)

1. **Open DICOM** → `DicomLoader.load(dir)` → `Volume` → `AppController`
   состояние `LOADED` → `SliceWidget` показывает axial-срез по центру.
2. **Upload & Segment** → `nifti_io` сериализует `Volume` в `.nii.gz` байты →
   `CloudClient.upload_study` → `study_id` → `start_segmentation` → `job_id` →
   состояние `SEGMENTING`.
3. **Poll** в worker-потоке: `poll_status(job_id)` с интервалом до `done`
   или `failed`. UI показывает статус, не блокируется.
4. **Download** → `download_mask(study_id)` → `mask_io` собирает
   `SegmentationMask` → **проверка `mask.is_compatible_with(volume)`**
   (геометрический инвариант: spacing/origin/direction/shape). Несовместимая
   маска отвергается с понятной ошибкой. Иначе состояние `MASK_READY`,
   наложение появляется во вьюере.
5. **Измерения**: клик-драг строит `LinearMeasurement`; длина в мм считается
   из spacing; линия и подпись рисуются поверх среза.
6. **Экспорт**: текущий кадр `SliceWidget` → PNG; скачанная маска → `.nii.gz`
   на диск.

## 7. Обработка ошибок

- **Импорт DICOM**: `DicomError` из `stomcore` (битая/неполная серия, не CBCT)
  → диалог с причиной, приложение не падает.
- **Сеть/облако**: `CloudClient` ретраит сетевые сбои/5xx; при исчерпании —
  `CloudError` → сообщение пользователю, действие можно повторить.
- **Аутентификация**: 401 → `AuthError` → предложение открыть Settings.
- **Сегментация упала** (`status == failed`): показать причину из `error`,
  предложить повтор (`start_segmentation` заново).
- **Геометрическое рассогласование** маски → отказ с понятным сообщением
  (используем `is_compatible_with`).
- **Нет/битый конфиг**: при старте без валидного конфига — открыть Settings.

## 8. Тестирование

- **Юнит (ядро, headless):** `config` (round-trip save/load, права файла),
  `cloud_client` (против мок-HTTP: успехи, 401, 409, 5xx-ретраи),
  `slice_renderer` (детерминированные массивы: окно/уровень, композит
  наложения с видимостью/цветом), `measurement` (длина в мм для известных
  точек/spacing), `app_controller` (переходы состояний с фейковым
  `CloudClient`, проверка отказа при геом-рассогласовании).
- **View (минимально):** smoke-тест сборки `MainWindow`/`SliceWidget` под
  `QT_QPA_PLATFORM=offscreen` через `pytest-qt`; если окружение без Qt —
  тест `skip`. Пиксели не проверяем, проверяем логику и проводку сигналов.
- **Фикстуры:** переиспользуем синтетический `Volume`/`SegmentationMask` из
  стиля тестов `stomcore`.
- **Стек:** `pytest`, мок-HTTP через `respx` (мокает `httpx`),
  `pytest-qt` (опционально).

## 9. Зависимости и упаковка

- Новый extra в `pyproject.toml`: `client` (PySide6, httpx) и
  `dev`-добавки (`pytest-qt`, `respx`). `stomcore` — уже в проекте.
- Запуск в MVP: консольный энтрипоинт `stom-client` (`python -m stomclient`).
  Установщик/сборка в один бинарь (PyInstaller) — вне Plan 3.

## 10. Безопасность медданных (MVP-уровень)

- Токен хранится в `~/.config/stom/client.toml` с правами `0600`; в Settings
  есть опция не сохранять токен на диск (держать только в памяти сессии).
- Транспорт: клиент должен поддерживать `https://` base-url (TLS на стороне
  деплоя сервера). Проверка сертификата включена по умолчанию.
- Анонимизация DICOM и полноценный аудит — отложены (см. §2).

## 11. Открытые вопросы (на будущее, вне MVP)

- VTK 3D-рендер и мультиплоскостной крест; общий слой геометрии срезов с 2D.
- Ручная правка масок и обратная отправка исправленной маски на сервер.
- Переподключение к незавершённым задачам, история исследований локально.
- Стриминговая/возобновляемая загрузка больших томов.
