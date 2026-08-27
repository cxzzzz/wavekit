# 安装

Python 3.9 及以上版本可直接从 PyPI 安装 wavekit：

```console
python -m pip install wavekit
```

开发 wavekit 时，先克隆仓库并安装 Poetry 环境：

```console
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

## FSDB 支持

读取 FSDB 文件需要 Verdi NPI 运行时（`libNPI.so`）。

在通常的 Verdi 环境中，`VERDI_HOME` 已经配置好，无需额外设置。如果没有配置，可以通过以下任一方式让系统找到 `libNPI.so`：

- `WAVEKIT_NPI_LIB`：直接指定 `libNPI.so` 的路径；
- `VERDI_HOME`：Verdi 安装目录；
- `LD_LIBRARY_PATH`：包含 `libNPI.so` 的库搜索路径。

打开 FSDB 文件前，可以先检查运行时是否可用：

```python
from wavekit import has_fsdb_support

print(f"FSDB support: {has_fsdb_support()}")
```

## 验证安装

```console
python -c "import wavekit; print(wavekit.__version__)"
```

然后继续阅读 [FIFO 水位教程](first-waveform.md)。
