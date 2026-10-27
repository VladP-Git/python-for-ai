import itertools
import sys
import time
import threading
from contextlib import contextmanager


def _format_time(seconds: float) -> str:
    """Превращает секунды в формат ММ:СС.д"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    tenths = int((seconds - int(seconds)) * 10)
    return f"{minutes:02d}:{secs:02d}.{tenths}"


def _format_total_time(seconds: float) -> str:
    """Превращает итоговые секунды в формат ММ:СС.сс"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    hundredths = int((seconds - int(seconds)) * 100)
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"


def _spinner_worker(stop_event, start_time, message, style):
    """Фоновое выполнение анимации с красивым таймером ММ:СС"""
    if style == 'dots':
        dots = 0
        while not stop_event.is_set():
            dots = (dots + 1) % 4
            elapsed = float(time.time() - start_time)
            time_str = _format_time(elapsed)
            sys.stdout.write(f"\r{message}{'.' * dots}{' ' * (3 - dots)} [{time_str}]\x1b[K")
            sys.stdout.flush()
            time.sleep(0.4)
    else:  # 'spin'
        spinner = itertools.cycle(['-', '\\', '|', '/'])
        while not stop_event.is_set():
            elapsed = float(time.time() - start_time)
            time_str = _format_time(elapsed)
            sys.stdout.write(f"\r{message} {next(spinner)} [{time_str}]\x1b[K")
            sys.stdout.flush()
            time.sleep(0.3)


@contextmanager
def loading_animation(message="Выполняется", style="spin"):
    """Контекстный менеджер для управления анимацией и замера времени"""
    start_time = time.time()
    stop_event = threading.Event()

    loader_thread = threading.Thread(
        target=_spinner_worker,
        args=(stop_event, start_time, message, style),
        daemon=True
    )

    loader_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        loader_thread.join()

        total_time = float(time.time() - start_time)
        time_str = _format_total_time(total_time)
        # Красивый финальный вывод с минутами и секундами
        sys.stdout.write(f"\r{message} ... Готово! (Затрачено: {time_str})\x1b[K\n")
        sys.stdout.flush()
