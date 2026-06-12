# Где я остановился — продолжить отсюда

**Последнее обновление:** 2026-06-12 (день). Всё закоммичено и запушено в `origin/master`.
Ветка: `master` (HEAD `082869c`). Локальная ветка `feat/local-segmentation` уже влита (PR #1).

## ✅ v0.1.7 — живой прогресс сегментации в процентах (выпущено, smoke зелёный)
Причина: при локальной сегментации не было обратной связи во время долгого CPU-инференса —
было непонятно, началась она вообще или нет. Теперь GUI показывает «Сегментация… NN%»,
процент растёт по мере прохождения тайлов nnU-Net (а до первого тайла — «Сегментация… (идёт расчёт)»).

Сделано (коммит `6f53d66`, выпущено в v0.1.7):
- **runner:** `DentalSegmentatorRunner.predict(progress=)` направляет tile-loop nnU-Net в колбэк
  `(done, total)`, подменяя символ `tqdm` в модуле инференса на крошечный счётчик-шим (без форка
  и без копирования версионных внутренностей). `FakeRunner` шлёт один шаг — чтобы проводка
  проверялась end-to-end.
- **engine:** `SubprocessEngine` теперь **стримит** stdout engine-pack в потоке-читателе (Popen,
  stderr слит в stdout, чтобы не было дедлока на полном пайпе) и парсит строки `PROGRESS d t`
  вживую — вместо `subprocess.run(capture_output)`, который блокировал до выхода процесса.
  Непрогрессный вывод по-прежнему копится для диагностики ошибок. `InProcessEngine` пробрасывает
  progress, когда задан.
- **cli:** `stom-engine predict` печатает flush-нутые строки `PROGRESS d t` по тайлам.
- **app_controller/main_window:** `submit(progress=)` → Qt-сигнал `_SubmitWorker` → статус-лейбл,
  маршалится в UI-поток.
- **Важно:** прогресс задействует ОБЕ половины — стриминг-ридер на стороне клиента (.exe) и эмиттер
  `PROGRESS` на стороне engine-pack. Поэтому v0.1.7 пересобирала и публиковала обе сборки.
- **Тесты:** 185 passed (+6). Шим проверен против реальной модели (32³ → 1 тайл: `(0,1)`,`(1,1)`;
  на полном CBCT тайлов много → плавный процент). Обе CI-сборки v0.1.7 зелёные, релиз опубликован
  (все 3 артефакта), манифест `releases/latest/download/` резолвится в **v0.1.7**.
- **Smoke зелёный против v0.1.7** (run `27411914427`, `TAG: v0.1.7` → `SMOKE TEST PASSED`).
- **v0.1.7 — актуальный релиз:** https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.7
  (`StomClientSetup.exe` ~193 МБ, `stom-engine-pack-win64.zip` ~551 МБ, `engine-pack-manifest.json`).

## ✅ v0.1.6 — in-process инференс (чинит краш `0xC000013A`, выпущено, smoke зелёный)
Причина: реальный CBCT-прогон у пользователя падал с `local engine failed: exit code 3221225786`
(`0xC000013A` = `STATUS_CONTROL_C_EXIT`) и пустым stderr. Smoke в CI проходил, т.к. там движок
стартует из консольного `python.exe`; GUI-клиент (`stom-client.exe`) оконный → запуск консольного
движка выделяет временную консоль, чьё закрытие шлёт `CTRL_CLOSE` воркерам `multiprocessing.Pool`
nnU-Net → код выхода воркера всплывает наверх.

Корневой фикс (коммит `600dc44`, выпущен в v0.1.6):
- `DentalSegmentatorRunner` теперь использует `nnUNetPredictor.predict_single_npy_array`
  (`read_images` + предсказание по одному массиву + `export_prediction_from_logits`) вместо
  `predict_from_files`. Работает полностью в процессе, НЕ плодит дочерних процессов — убирает
  весь класс падений frozen-Windows на multiprocessing (этот краш и прошлый freeze_support-висяк).
- Защита в глубину: `SubprocessEngine` запускает движок с `CREATE_NO_WINDOW` на Windows (не привязан
  к временной консоли GUI, консоль не мигает); ошибки декодируют известные NTSTATUS-коды
  (`0xC000013A/05/0135`) и фолбэчат на stdout, когда stderr пуст — вместо непрозрачного числа.
- **Тесты:** 179 passed. Обе CI-сборки v0.1.6 зелёные, релиз опубликован (все 3 артефакта),
  манифест `releases/latest/download/` резолвится в **v0.1.6**.
- **Smoke зелёный против v0.1.6** (run `27410249842`, `TAG: v0.1.6` → `SMOKE TEST PASSED`, ~2 мин).
  Важно: smoke гоняет движок из консольного python и синтетику 32³ — сам краш `0xC000013A`
  (только в оконном GUI на реальном CBCT) он НЕ воспроизводит. Нужна проверка на машине юзера.
- **v0.1.6 — актуальный релиз:** https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.6
  (`StomClientSetup.exe` ~192 МБ, `stom-engine-pack-win64.zip` ~551 МБ, `engine-pack-manifest.json`).

## ✅ v0.1.5 — автообновление + русский интерфейс (выпущено, smoke зелёный)
Спека: `docs/superpowers/specs/2026-06-12-autoupdate-and-russian-ui-design.md`.
Причина: пользователь прислал скрин ошибки `freeze_support` — у него на диске остался
старый сломанный engine-pack (клиент не отслеживал версию → не обновлял). Плюс запросил
автообновление и понятный русский UI. Сделано (коммит `f6ed818`):
- **Версионирование engine-pack** (чинит баг): `engine_pack` пишет маркер `installed.json`;
  `installed_version()`/`engine_update_available()` видят устаревший/без-маркерный (legacy,
  сломанный v0.1.2/v0.1.3) пакет; `provision(clean=True)` ставит начисто.
- **Автообновление (спрашивает юзера):** движок — фоновая проверка версии при старте →
  «Обновить движок?»; приложение — `stomclient/updates.py` тянет последний релиз с GitHub →
  «Скачать и установить?» → качает+запускает установщик. Inno: `CloseApplications=yes`.
  Всё fail-soft (нет сети → проверки молча пропускаются).
- **Русский UI:** `ui/strings.py` (все надписи централизованы); панель сгруппирована
  (Исследование/Сегментация/Инструменты/Экспорт/Маски); статусы по-русски; ошибки —
  короткий текст + «Подробности», при `freeze_support` подсказка «Обновить движок…».
- **Тесты:** 178 passed (было 156). Релиз **v0.1.5** — обе сборки зелёные, smoke на чистой
  Windows VM пройден (run `27402668105`, `SMOKE TEST PASSED`).
- **Важно:** автообновление работает только начиная с v0.1.5 (старые версии о нём не знают) —
  v0.1.5 надо поставить вручную один раз; дальше движок/клиент обновляются сами.

## ✅ ХВОСТ ЗАКРЫТ — smoke-тест зелёный на v0.1.4
Прошлый smoke (run `27345327941`, против v0.1.2/engine-pack v0.1.3) **завис на 6 ч** и был
отменён по дефолтному job-таймауту. Причина: engine-pack бинарь форк-бомбил сам себя —
`packaging/engine_launch.py` (frozen-точка входа) НЕ звал `multiprocessing.freeze_support()`,
а `nnUNetPredictor.predict_from_files` плодит `multiprocessing.Pool`. В one-folder PyInstaller
каждый воркер ре-exec'ит `stom-engine.exe` → снова `main()` → снова `predict` → ещё воркеры,
до бесконечности. Даже 32³ висел вечно.

**Фикс (коммит `83d4a2f`, выпущен в v0.1.4):**
- `packaging/engine_launch.py`: `multiprocessing.freeze_support()` первым делом в `__main__` — это и есть фикс.
- `stomengine.SubprocessEngine(timeout=)`: зависший движок теперь кидает `RuntimeError`, а не висит.
- smoke-workflow: `timeout-minutes: 45`; smoke-скрипт зовёт движок с `timeout=1200`.
- bump `0.1.3 → 0.1.4` (engine-pack обязательно пере-собрать+пере-выложить, чтобы фикс уехал).
- тесты: фейковые `run()` принимают `timeout`; добавлен тест таймаута → **156 passed**.

**Результат:** обе сборки v0.1.4 зелёные, smoke (run `27400087507`, `tag=v0.1.4`) — `success`
за ~2 мин: manifest→download+verify+extract engine-pack→реальный инференс через
`SubprocessEngine` → `SMOKE TEST PASSED` (mask 32³, TTA off). **Релиз v0.1.4 доведён.**

**v0.1.4 — актуальный релиз:** https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.4
(`StomClientSetup.exe` 183 МБ, `stom-engine-pack-win64.zip` 525 МБ, `engine-pack-manifest.json`;
манифест `releases/latest/download/` резолвится в v0.1.4). **v0.1.2/v0.1.3 НЕ использовать** —
их engine-pack без freeze_support, инференс зависает.

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
- `v0.1.2`/`v0.1.3` — **НЕ использовать**: engine-pack без `freeze_support()`, инференс зависает.
- `v0.1.4` — рабочий, но без автообновления и русского UI.
- `v0.1.5` — рабочий (автообновление + русский UI), но движок падал на реальном CBCT (`0xC000013A`).
- `v0.1.6` — рабочий (чинит `0xC000013A`), но без индикатора прогресса сегментации.
- `v0.1.7` — **актуальный**: https://github.com/Almaz2001AA/Stom/releases/tag/v0.1.7
  - `StomClientSetup.exe` (~193 МБ, русский слим-GUI + автообновление + живой прогресс %)
  - `stom-engine-pack-win64.zip` (~551 МБ, torch CPU+nnU-Net+веса) + `engine-pack-manifest.json`
  - Манифест по `releases/latest/download/` резолвится в v0.1.7 (клиент тянет его сам).
  - Smoke-тест на чистой Windows VM пройден (run `27411914427`, `SMOKE TEST PASSED`).
- Repo variable `WEIGHTS_URL` = Zenodo (`zenodo.org/records/10829675/.../Dataset112_DentalSegmentator_v100.zip?download=1`).
- Теги v0.1.0/v0.1.1 заняты старыми коммитами (предшествуют workflow).

Smoke-тест (`.github/workflows/smoke-windows-install.yml` + `.github/scripts/smoke_engine_pack.py`)
перезапускается через `workflow_dispatch` с input `tag` (напр. `tag=v0.1.5`); теперь с
`timeout-minutes: 45` на job и `timeout=1200` на инференс — зависание падает быстро, а не за 6 ч.

## Установка у конечного пользователя (v0.1.7)
1. Поставить `StomClientSetup.exe` (SmartScreen → «Подробнее → Выполнить в любом случае»).
2. В приложении нажать **«Установить движок…»** (один раз, ~0.5 ГБ → `%LOCALAPPDATA%\Stom\engine`).
   На старом сломанном движке приложение само предложит **«Обновить движок…»** до v0.1.7.
3. Галка **«Локально (на этом ПК)»** активируется → работа локально, без сервера/токена.
   Опц. системная переменная `STOM_DISABLE_TTA=1` → ускорение ~8×.
   Ручной фолбэк: распаковать engine-pack так, чтобы был `%LOCALAPPDATA%\Stom\engine\stom-engine.exe`.
4. Дальше движок и само приложение обновляются автоматически (с подтверждением).

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
- Тесты: `.venv/bin/python -m pytest -q` → **185 passed** (UI-тесты: `QT_QPA_PLATFORM=offscreen`).
  Python только в `.venv` (нет системного `python`).
- `gh` CLI НЕ установлен → PR/releases/variables/dispatch делаю через GitHub API + токен из `~/.git-credentials`.
- Веса: `models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres`. GPU нет → CPU.
- Тестовый реальный CBCT: `/opt/almaz/test/Stom/20241021-...ГалееваЛяйсан...CT...` (100×700×700, 0.2мм).
- Незакоммиченный мусор в рабочем дереве (НЕ коммитить): `aa.png`, `stom.db`, `storage/`.
- Память Клода: `stom-project-roadmap`, `dentalsegmentator-intensity-harmonization` — подтянутся сами.

## Открытые follow-ups
- ✅ Windows smoke-тест — закрыт (v0.1.7 зелёный, run `27411914427`).
- ✅ Автообновление + русский UI — выпущено в v0.1.5.
- ✅ Краш `0xC000013A` у пользователя — корневой фикс выпущен в v0.1.6 (in-process инференс).
- ✅ Индикатор прогресса сегментации (проценты) — выпущен в v0.1.7.
- **ГЛАВНОЕ — проверить у пользователя:** поставить v0.1.7 поверх его установки → приложение должно
  само предложить «Обновить движок…» → локальная сегментация на реальном CBCT должна (1) пройти БЕЗ
  краша `0xC000013A` и (2) показывать растущий процент «Сегментация… NN%». Это подтверждение фиксов
  v0.1.6/v0.1.7 на реальном железе (smoke ловит engine-pack путь, но не оконный GUI на полном CBCT).
- Авто-обновление .exe (скачивание+запуск установщика, `CloseApplications=yes`) на реальном
  Windows ещё не проверено end-to-end (smoke его не покрывает — только engine-pack путь).
- RAM целевого ПК (~6–8 ГБ на объём 100×700×700) не подтверждён на реальном железе.
- Реальный инференс на полном CBCT (100×700×700) через выпущенный engine-pack на Windows ещё
  не гоняли — smoke проверяет только синтетику 32³. Стоит проверить на железе юзера.
- Прочие отложенные — в `docs/superpowers/plans/FOLLOWUPS-*.md`.

## Как продолжить
Релиз v0.1.7 доведён и проверен (185 тестов + smoke зелёные). Главное незакрытое — проверка на
машине пользователя: автообновление до v0.1.7 + реальный CBCT-прогон без краша `0xC000013A` и
с растущим процентом прогресса (см. follow-ups). Открыть Claude Code в `/opt/almaz/test/Stom`,
`claude --continue`.
