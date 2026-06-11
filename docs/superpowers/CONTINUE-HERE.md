# Где я остановился — продолжить отсюда

**Последнее обновление:** 2026-06-11 (вечер). Всё закоммичено и запушено в `origin/master`.
Ветка: `master` (HEAD `9f8daef`). Локальная ветка `feat/local-segmentation` уже влита (PR #1).

## Что сделал сегодня — ЛОКАЛЬНАЯ (on-device) СЕГМЕНТАЦИЯ, выпущена в v0.1.3
Причина фичи: большие CBCT-аплоады рвались через любой туннель → перенесли инференс на ПК
пользователя, сеть из пайплайна убрана. План: `docs/superpowers/plans/2026-06-11-local-segmentation.md`.

- ✅ **Phase A** — пакет `stomengine` (runner+labels+`LocalEngine`/`InProcessEngine`, без server/Redis/DB),
  локальный режим в `AppController.submit()`, чекбокс «Local (on-device)» в UI.
- ✅ **Phase B** — `stomclient.engine_pack` (манифест → download+SHA-256+extract), `SubprocessEngine`
  (инференс через бинарь engine-pack), CLI `stom-engine predict`, PyInstaller-спека + CI-workflow.
- ✅ **Phase C** — env-тумблер `STOM_DISABLE_TTA=1` → ~8× быстрее на CPU (`tta_enabled()`,
  `DentalSegmentatorRunner(use_tta=)` → `use_mirroring=` в nnUNetPredictor).
- ✅ **Багфикс (v0.1.3)** — чекбокс «Local» был навсегда серым на свежей установке (engine-pack
  никто из UI не качал). Добавлена кнопка **«Install local engine…»** в `main_window.py`:
  фоновая загрузка с прогрессом → `provision_local_engine()` → `AppController.set_engine()` →
  чекбокс активируется. **155 тестов зелёные.**

**Релизы (CI обе сборки зелёные, артефакты в Releases):**
- `v0.1.2` — первый engine-pack релиз (БЕЗ кнопки, не использовать).
- `v0.1.3` — **актуальный**: https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.3
  - `StomClientSetup.exe` (183 МБ, слим-GUI с кнопкой)
  - `stom-engine-pack-win64.zip` (524 МБ, torch CPU+nnU-Net+веса) + `engine-pack-manifest.json`
  - Манифест по `releases/latest/download/` резолвится в v0.1.3 (клиент тянет его сам).
- Repo variable `WEIGHTS_URL` = Zenodo (`zenodo.org/records/10829675/.../Dataset112_DentalSegmentator_v100.zip?download=1`).
- Теги v0.1.0/v0.1.1 заняты старыми коммитами (предшествуют workflow).

## НЕЗАКРЫТЫЙ ХВОСТ — дождаться smoke-теста на Windows
CI smoke-тест на чистой `windows-latest` (`.github/workflows/smoke-windows-install.yml` +
`.github/scripts/smoke_engine_pack.py`): скачать+тихо поставить `StomClientSetup.exe`, затем
реальный first-run (манифест → download+verify+extract engine-pack → `SubprocessEngine` инференс).
- **Run id `27345327941`** (запущен против tag v0.1.2; engine-pack тянет latest = v0.1.3).
- На вечер 11.06 статус: тихая установка ✅; шаг «engine-pack + inference» висел **`in_progress` >40 мин**
  (подозрительно долго для 32³ с TTA-off → возможно медленная загрузка 524 МБ / зависание).
- **Завтра первым делом** проверить итог:
  ```bash
  TOKEN=$(grep -o '://[^:]*:[^@]*@github.com' ~/.git-credentials | head -1 | sed 's#://[^:]*:##; s#@github.com##')
  curl -s -H "Authorization: token $TOKEN" https://api.github.com/repos/Almaz2001AA/Stom/actions/runs/27345327941/jobs \
    | jq -r '.jobs[].steps[] | "\(.status)\t\(.conclusion//"-")\t\(.name)"'
  ```
  Если упал/завис — скачать логи шага и разобрать; при необходимости перезапустить
  `workflow_dispatch` (workflow id `293836682`, input `tag=v0.1.3`). Фоновый поллинг прошлой
  сессии (`btx6oysp9`) до завтра НЕ доживёт.

## Установка у конечного пользователя (v0.1.3)
1. Поставить `StomClientSetup.exe` (SmartScreen → «Подробнее → Выполнить в любом случае»).
2. В приложении нажать **«Install local engine…»** (один раз, ~0.5 ГБ → `%LOCALAPPDATA%\Stom\engine`).
3. Галка **«Local (on-device)»** активируется → работа локально, без сервера/токена.
   Опц. системная переменная `STOM_DISABLE_TTA=1` → ускорение ~8×.
   Ручной фолбэк: распаковать engine-pack так, чтобы был `%LOCALAPPDATA%\Stom\engine\stom-engine.exe`.

## Облачный путь (по-прежнему рабочий, опционально) — туннель ПК → сервер
Локальный режим теперь основной, но облако осталось. Если нужно поднять бэкенд:
- Сервер за NAT: внешний `185.46.68.46`, SSH-порт **`1975`** (НЕ 22), user `alex`, только ключ,
  есть fail2ban (не плодить неудачные логины). LAN `192.168.0.131` из сети юзера НЕдоступен.
- ПК юзера заходит через PuTTY с сохранённой сессией. Туннель в живое окно:
  Change Settings… → SSH → Tunnels → Source `8010`, Dest `127.0.0.1:8010` → Add → Apply.
- В приложении Settings: URL `http://localhost:8010`, Token `7oIvkH3GO0d7BsjkROpRR0CHZ03Lue-TF64Id_Bsazk`.
- Перезапуск бэкенда (фоновые сервисы за ночь умирают):
  ```bash
  cd /opt/almaz/test/Stom
  RS=.venv/lib/python3.13/site-packages/redislite/bin/redis-server
  "$RS" --port 6379 --bind 127.0.0.1 --save "" --appendonly no &
  export STOM_DB_URL="sqlite:////opt/almaz/test/Stom/stom.db"
  export STOM_STORAGE_DIR=/opt/almaz/test/Stom/storage
  export STOM_REDIS_URL=redis://localhost:6379/0
  export STOM_MODEL_DIR=/opt/almaz/test/Stom/models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres
  export STOM_MAX_UPLOAD_BYTES=1073741824
  .venv/bin/uvicorn "stomserver.api.app:create_app" --factory --host 0.0.0.0 --port 8010 &
  OMP_NUM_THREADS=14 .venv/bin/rq worker segmentation --url redis://localhost:6379/0 &
  ```
  Проверка: `curl -s http://localhost:8010/healthz` → `{"status":"ok"}`.

## Полезные пути и факты
- Тесты: `.venv/bin/python -m pytest -q` → **155 passed**. Python только в `.venv` (нет системного `python`).
- `gh` CLI НЕ установлен → PR/releases/variables/dispatch делаю через GitHub API + токен из `~/.git-credentials`.
- Веса: `models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres`. GPU нет → CPU.
- Тестовый реальный CBCT: `/opt/almaz/test/Stom/20241021-...ГалееваЛяйсан...CT...` (100×700×700, 0.2мм).
- Незакоммиченный мусор в рабочем дереве (НЕ коммитить): `QQ.png`, `stom.db`, `storage/`.
- Память Клода: `stom-project-roadmap`, `dentalsegmentator-intensity-harmonization` — подтянутся сами.

## Открытые follow-ups
- Дождаться/починить Windows smoke-тест (см. выше) — главный незакрытый пункт.
- RAM целевого ПК (~6–8 ГБ на объём 100×700×700) не подтверждён на реальном железе.
- Прочие отложенные — в `docs/superpowers/plans/FOLLOWUPS-*.md`.

## Как продолжить
Открыть Claude Code в `/opt/almaz/test/Stom`, `claude --continue`, сказать:
«проверь итог smoke-теста (run 27345327941) по CONTINUE-HERE и доведём релиз v0.1.3».
