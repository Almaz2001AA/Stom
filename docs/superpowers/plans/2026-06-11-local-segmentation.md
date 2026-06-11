# План: локальная сегментация на ПК (offline-режим)

**Дата:** 2026-06-11. **Причина:** аплоад большой студии (94 МБ DICOM → десятки МБ
gzip-NIfTI) рвётся на нестабильной связи через любой одиночный туннель (PuTTY/ngrok/
Cloudflare). Локальный инференс полностью убирает сеть из пайплайна.

**Решение по упаковке (выбор пользователя):** слим-GUI + догрузка движка и весов при
первом локальном запуске. torch/nnunet нельзя доставить внутрь замороженного
PyInstaller-приложения, поэтому «движок» — отдельный **engine-pack** (самостоятельный
процесс/exe с весами), который GUI скачивает один раз и дёргает как подпроцесс.

## Архитектура
`DentalSegmentatorRunner.predict(volume)` самодостаточен (stomcore + torch + nnunetv2 +
SimpleITK, без сервера/Redis/БД). Локальный режим = вместо upload→segment→poll→download
вызвать движок и отрисовать маску.

Вводим абстракцию `LocalEngine` с реализациями:
- `InProcessEngine(runner)` — вызывает runner в текущем процессе (dev/сервер/тесты).
- `SubprocessEngine(engine_exe)` — дёргает engine-pack как подпроцесс (боевой слим-клиент).

## Фаза A — общий движок + локальный режим в клиенте (проверяемо здесь, TDD)
1. Новый пакет `src/stomengine/`: перенести `runner.py` + `labels.py` из
   `stomserver/segmentation/`; добавить `engine.py` (`LocalEngine` протокол,
   `InProcessEngine` → возвращает готовый `SegmentationMask`).
2. Back-compat: `stomserver/segmentation/{runner,labels}.py` ре-экспортируют из stomengine
   (сервер/воркер/тесты не трогаем).
3. `AppController`: принимать опциональный `engine`; ветка локального режима в `submit()`
   (без upload/poll: SEGMENTING → predict в фоне → MASK_READY). Переиспользовать проверку
   геометрии и отрисовку.
4. UI: переключатель «Локальная сегментация» (Settings/чекбокс). Локальный путь не требует
   server URL/токена.
5. Тесты: контроллер в локальном режиме с фейковым движком; `InProcessEngine` с `FakeRunner`;
   импорт/лейблы stomengine. Прогнать весь сюит зелёным.

## Фаза B — провижининг движка (download on first run) — CI/Windows
6. CI собирает engine-pack (отдельная PyInstaller-сборка с torch CPU + nnunetv2 + весами 236 МБ)
   и публикует zip в Releases с checksum.
7. GUI при первом локальном запуске: скачать engine-pack в `%LOCALAPPDATA%/Stom/engine`,
   проверить checksum, распаковать; `SubprocessEngine` указывает на `stom-engine.exe`.
   Прогресс/отмена/повтор докачки.
8. Слим-GUI установщик: убрать тяжёлые зависимости; CI публикует два артефакта (GUI + engine-pack).

## Фаза C — скорость (CPU)
9. Env-тумблер отключения TTA-зеркалирования (`STOM_DISABLE_TTA`) → ускорение ~8×
   (≈1.5–2 мин вместо ~10). Применить и в локальном движке.

## Открытые вопросы
- Требования к ПК: ~6–8 ГБ свободной RAM на этот объём (100×700×700); на слабом ПК медленно/OOM.
  Уточнить характеристики целевого ПК.
