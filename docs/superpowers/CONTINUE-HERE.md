# Где я остановился — продолжить отсюда

**Последнее обновление:** 2026-06-09. Всё закоммичено и запушено в `origin/master`.

## Проект
Аналог DiagnoCAT — AI-сегментация стоматологической CBCT. Гибрид: десктоп-клиент + облако.
Репозиторий: github.com/Almaz2001AA/Stom (ветка `master`).

## Статус по планам
- ✅ **Plan 1 — Ядро `stomcore`** (Volume/Geometry/SegmentationMask, DICOM↔NIfTI, mask_io, CLI `stom-dicom2nifti`).
- ✅ **Plan 2 — Backend `stomserver`** (FastAPI API + RQ-воркер + SQLAlchemy/SQLite + LocalFileStorage + bearer-auth/изоляция; модель DentalSegmentator/nnU-Net за runner с FakeRunner). **66 тестов зелёные, 1 slow skipped.**
- ⬜ **Plan 3 — Десктоп-клиент** (VTK-вьюер, наложение/правка масок, измерения, `CloudClient` к API). ← СЛЕДУЮЩИЙ

## Как продолжить завтра
1. Открыть терминал в `/opt/almaz/test/Stom`.
2. Если пропал доступ на запись (папка принадлежит `www`): `sudo chown -R alex:alex /opt/almaz/test/Stom`.
3. Активировать окружение / прогнать тесты: `.venv/bin/python -m pytest -q` (ожидается 66 passed, 1 skipped).
4. В Claude Code продолжить эту сессию: `claude --continue` (последняя сессия) или `claude --resume` (выбрать из списка).
5. Сказать Клоду: «начни прорабатывать Plan 3» — пойдём тем же циклом: бриф → спек → план → реализация субагентами с ревью.

## Документы
- Спеки: `docs/superpowers/specs/2026-06-09-*.md`
- Планы: `docs/superpowers/plans/2026-06-09-*.md`
- Follow-ups (отложенное): `docs/superpowers/plans/FOLLOWUPS-*.md`

## Запуск backend (dev), если нужно проверить вручную
См. `src/stomserver/README.md`. Кратко: `pip install -e ".[dev,server]"`, `python scripts/create_account.py "Clinic A"` (выдаст токен), `redis-server`, `uvicorn "stomserver.api.app:create_app" --factory`, `rq worker segmentation`.

## Память Claude
Дорожная карта и решения сохранены в памяти проекта (`stom-project-roadmap`), подтянется автоматически в новой сессии.

## ВАЖНО (безопасность)
Если ещё не сделано — **отозвать Personal Access Token** на GitHub (Settings → Developer settings → Tokens), он засветился в переписке. `~/.git-credentials` на диске содержит токен в открытом виде.
