# Где я остановился — продолжить отсюда

**Последнее обновление:** 2026-06-10. Всё закоммичено и запушено в `origin/master`.

## Статус проекта — ВСЕ 3 ПЛАНА ГОТОВЫ
- ✅ **Plan 1 — `stomcore`** (Volume/Geometry/Mask, DICOM↔NIfTI, mask_io, CLI).
- ✅ **Plan 2 — `stomserver`** (FastAPI + RQ + SQLite + auth; DentalSegmentator за runner).
- ✅ **Plan 3 — `stomclient`** (PySide6-десктоп: вьюер срезов, наложение масок, измерения мм,
  CloudClient, Settings, экспорт PNG/маски). Влит в master. **122 теста зелёные.**
- ✅ **Windows-установщик** собирается в GitHub Actions и лежит в Releases:
  https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.0 (`StomClientSetup.exe`).
- ✅ **Настоящая сегментация работает** на CPU: веса DentalSegmentator скачаны в
  `./models/`, runner поправлен (`use_folds=(0,)`), добавлена гармонизация интенсивностей
  (`harmonize_to_model_domain`) — без неё CBCT этого аппарата давал пустую маску.
  Проверено на реальном снимке: все 5 структур (зубы/челюсти/канал) выделены.

## ЕДИНСТВЕННЫЙ незакрытый момент — туннель ПК → сервер
Серверная часть полностью готова и крутится. Не доделан **сетевой доступ** из
установленного приложения на Windows к API сервера.

**Сетевая картина:**
- Сервер за NAT: внешний IP `185.46.68.46`, SSH-порт **`1975`** (НЕ 22), пользователь `alex`,
  **только ключевая аутентификация** (пароль отключён, есть fail2ban — не плодить
  неудачные попытки входа, иначе временный бан IP).
- LAN-IP сервера `192.168.0.131` — из сети пользователя НЕдоступен (все таймауты были из-за этого).
- ПК пользователя (внешний IP `178.206.229.84`) заходит на сервер через PuTTY с **сохранённой
  рабочей сессией** (в ней прописан ключ).

**Что сделать завтра, чтобы подключить приложение:**
1. В **уже открытом** рабочем окне PuTTY: правый клик по заголовку → **Change Settings…** →
   Connection → SSH → Tunnels → Source `8010`, Destination `127.0.0.1:8010` → **Add** → **Apply**.
   (Так туннель добавляется в живое соединение — без нового логина и без fail2ban.)
   Альтернатива: Load рабочей сессии → добавить тот же туннель → Save → Open.
2. В приложении **Stom CBCT Viewer → Settings**:
   - Server URL: `http://localhost:8010`
   - Token: `7oIvkH3GO0d7BsjkROpRR0CHZ03Lue-TF64Id_Bsazk` (лежит в `stom.db`, действует)
3. **Open DICOM** → **Upload & Segment**. На CPU полный CBCT считается **~10 мин**
   (боевой runner использует TTA 8×; можно сделать отключаемым через env для скорости ~1.5 мин).

## ВАЖНО: фоновые сервисы могли умереть за ночь — перезапуск
API/воркер/Redis были запущены как фоновые процессы сессии и, скорее всего, к завтрашнему
дню остановятся. Поднять заново:

```bash
cd /opt/almaz/test/Stom
# 1) Redis (бинарь из redislite, без sudo)
RS=.venv/lib/python3.13/site-packages/redislite/bin/redis-server
"$RS" --port 6379 --bind 127.0.0.1 --save "" --appendonly no &
# 2) общий env
export STOM_DB_URL="sqlite:////opt/almaz/test/Stom/stom.db"
export STOM_STORAGE_DIR=/opt/almaz/test/Stom/storage
export STOM_REDIS_URL=redis://localhost:6379/0
export STOM_MODEL_DIR=/opt/almaz/test/Stom/models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres
export STOM_MAX_UPLOAD_BYTES=1073741824
# 3) API на всех интерфейсах (порт 8000 занят чужим процессом → используем 8010)
.venv/bin/uvicorn "stomserver.api.app:create_app" --factory --host 0.0.0.0 --port 8010 &
# 4) воркер с реальной моделью
OMP_NUM_THREADS=14 .venv/bin/rq worker segmentation --url redis://localhost:6379/0 &
```
Проверка: `curl -s http://localhost:8010/healthz` → `{"status":"ok"}`.
Если `stom.db` пропал — новый токен: `STOM_DB_URL=... .venv/bin/python scripts/create_account.py "Clinic A"`.

## Полезные пути и факты
- Тестовый реальный CBCT: `/opt/almaz/test/Stom/20241021-...ГалееваЛяйсан...CT...` (100×700×700, 0.2мм).
- Веса модели: `models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres`.
- GPU нет → инференс на CPU. Демо-скрипт сегментации с рендером: `/tmp/seg_real3.py` (могу пересоздать).
- Память Клода: `stom-project-roadmap`, `dentalsegmentator-intensity-harmonization` — подтянутся сами.

## Открытые follow-ups (по желанию)
- Сделать TTA-зеркалирование отключаемым через env (ускорить CPU-инференс ~8×).
- Гармонизацию сейчас применяю всегда; при необходимости сделать «умной» (только если вход далёк от домена).
- Прочие отложенные — в `docs/superpowers/plans/FOLLOWUPS-*.md`.

## Как продолжить
Открыть Claude Code в `/opt/almaz/test/Stom`, `claude --continue`, сказать:
«подними бэкенд по CONTINUE-HERE и доведём туннель/приложение».
