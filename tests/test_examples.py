import os
import subprocess
import sys

import pytest

EXAMPLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../example'))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

EXAMPLE_DIRS = sorted(
    name
    for name in os.listdir(EXAMPLE_ROOT)
    if os.path.isfile(os.path.join(EXAMPLE_ROOT, name, 'Makefile'))
)


@pytest.mark.parametrize('example_name', EXAMPLE_DIRS)
def test_example(example_name):
    target_dir = os.path.join(EXAMPLE_ROOT, example_name)

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{PROJECT_ROOT}/src:{env.get('PYTHONPATH', '')}"
    env['PATH'] = f"{os.path.dirname(sys.executable)}:{env.get('PATH', '')}"

    result = subprocess.run(
        ['make', 'all'], cwd=target_dir, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f'Make all failed with stderr:\n{result.stderr}\nStdout:\n{result.stdout}')
