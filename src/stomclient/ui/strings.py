"""Russian UI strings, centralized so the interface reads in one voice.

The client ships Russian-only (no language switch — YAGNI). Keeping every visible
label here, rather than scattered through ``main_window``, makes wording easy to
review and lets tests assert on user-facing text without hard-coding it twice.
"""

from __future__ import annotations

from ..app_controller import State

WINDOW_TITLE = "Stom — просмотр и сегментация КЛКТ"

# Section headers in the left panel.
SECTION = {
    "study": "Исследование",
    "segmentation": "Сегментация",
    "tools": "Инструменты",
    "export": "Экспорт",
    "masks": "Маски:",
}

# Buttons and controls.
BTN = {
    "settings": "Настройки…",
    "open": "Открыть DICOM…",
    "segment_cloud": "Загрузить и сегментировать",
    "segment_local": "Сегментировать (локально)",
    "local_checkbox": "Локально (на этом ПК)",
    "install_engine": "Установить движок…",
    "update_engine": "Обновить движок…",
    "measure": "Измерение",
    "clear_measure": "Очистить измерения",
    "save_png": "Сохранить PNG…",
    "save_mask": "Сохранить маску…",
}

# Plane names for the orientation combo (keys are the renderer's plane ids).
PLANE = {
    "axial": "Аксиальная",
    "coronal": "Корональная",
    "sagittal": "Сагиттальная",
}

# Tooltips.
TIP = {
    "local_on": "Сегментация на этом ПК — без загрузки на сервер.",
    "local_off": "Локальный движок не установлен.",
    "install_engine": "Скачать движок сегментации (~0.5 ГБ).",
    "update_engine": "Доступна новая версия движка — обновить.",
}

# Status-line text for each controller state.
STATUS = {
    State.EMPTY: "Нет исследования",
    State.LOADED: "Исследование загружено",
    State.UPLOADING: "Загрузка на сервер…",
    State.SEGMENTING: "Сегментация… (идёт расчёт)",
    State.MASK_READY: "Готово — маска получена",
    State.FAILED: "Ошибка",
}

# Live segmentation progress shown in place of the SEGMENTING status while
# inference runs (percentage of the sliding-window tiles done).
STATUS_SEG_PROGRESS = "Сегментация… {pct}%"

# Dialog titles and messages.
MSG = {
    "dicom_error_title": "Ошибка DICOM",
    "no_server_title": "Сервер не задан",
    "no_server_body": "Откройте «Настройки» и укажите адрес сервера.",
    "no_study_title": "Нет исследования",
    "no_study_body": "Сначала откройте серию DICOM.",
    "no_mask_title": "Нет маски",
    "no_mask_body": "Маска ещё не получена.",
    "seg_error_title": "Ошибка сегментации",
    "cloud_error_title": "Ошибка сервера",
    "seg_failed_title": "Сегментация не удалась",
    "install_progress": "Загрузка движка сегментации…",
    "install_title": "Установка движка",
    "install_done_title": "Локальный движок",
    "install_done_body": "Движок для работы на этом ПК установлен.",
    "install_failed_title": "Не удалось установить",
    "update_progress": "Обновление движка сегментации…",
    "update_title": "Обновление движка",
    "update_done_body": "Движок обновлён.",
    "engine_update_title": "Доступно обновление движка",
    "engine_update_body": (
        "Доступна новая версия локального движка сегментации. "
        "Обновить сейчас? (~0.5 ГБ)"
    ),
    "engine_outdated_hint": (
        "\n\nПохоже, установлен устаревший движок. Нажмите «Обновить движок…», "
        "чтобы скачать исправленную версию."
    ),
    "client_update_title": "Доступна новая версия",
    "client_update_body": (
        "Вышла новая версия приложения {version}. "
        "Скачать и установить сейчас?"
    ),
    "client_update_progress": "Загрузка установщика…",
    "client_update_ready_title": "Установка обновления",
    "client_update_ready_body": (
        "Установщик загружен. Приложение закроется и запустится установка."
    ),
    "client_update_failed_title": "Не удалось обновить приложение",
    "details_button": "Подробности",
}
