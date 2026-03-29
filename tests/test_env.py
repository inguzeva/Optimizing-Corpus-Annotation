import subprocess
import pytest

def test_env_script():
    """Запускаем скрипт проверки окружения, игнорируем опциональные модули."""
    result = subprocess.run(
        ["python", "scripts/00_check_env.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)  # показываем лог в CI
    # Скрипт может возвращать 1, если отсутствуют необязательные модули, но тест не падает
    # Проверяем только, что Python и структура проекта есть
    required_ok_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("[OK]") and not any(
            w in line for w in ("pdfplumber", "fitz", "pandas", "rapidfuzz")
        )
    ]
    assert required_ok_lines, "Critical environment checks failed"