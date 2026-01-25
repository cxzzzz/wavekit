import os
import subprocess
from contextlib import contextmanager

import pytest


# Helper context manager to change working directory
@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


@pytest.fixture
def example_dir():
    # Assuming tests are run from project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../example'))


@pytest.fixture
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run_make_all(cwd, project_root):
    # Set PYTHONPATH to include the project root's src directory
    env = os.environ.copy()
    src_path = os.path.join(project_root, 'src')
    env['PYTHONPATH'] = f"{src_path}:{env.get('PYTHONPATH', '')}"

    # Run 'make all'
    result = subprocess.run(['make', 'all'], cwd=cwd, env=env, capture_output=True, text=True)
    return result


def test_scoreboard_verify(example_dir, project_root):
    target_dir = os.path.join(example_dir, 'scoreboard')

    result = run_make_all(target_dir, project_root)

    # Check for success
    if result.returncode != 0:
        pytest.fail(f'Make all failed with stderr:\n{result.stderr}\nStdout:\n{result.stdout}')


def test_fifo_occupancy(example_dir, project_root):
    target_dir = os.path.join(example_dir, 'fifo_occupancy')

    result = run_make_all(target_dir, project_root)

    if result.returncode != 0:
        pytest.fail(f'Make all failed with stderr:\n{result.stderr}\nStdout:\n{result.stdout}')


def test_fifo_latency(example_dir, project_root):
    target_dir = os.path.join(example_dir, 'fifo_latency')

    result = run_make_all(target_dir, project_root)

    if result.returncode != 0:
        pytest.fail(f'Make all failed with stderr:\n{result.stderr}\nStdout:\n{result.stdout}')
