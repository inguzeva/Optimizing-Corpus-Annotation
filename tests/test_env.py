import subprocess

def test_env_script():
    result = subprocess.run(
        ["python", "scripts/00_check_env.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0