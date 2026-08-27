# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://pepy.tech/badge/wavekit)](https://pepy.tech/project/wavekit)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

[English](README.md) | 中文

**使用 Python 进行高层次数字波形分析。**

波形文件通常暴露时间戳和值变化，而硬件工程师通常按时钟周期、信号关系和多周期行为进行思考。wavekit 通过灵活的信号查询，以及周期级和事务级分析，弥合了这种抽象层次差异。

> **AI 集成：** [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) 通过 MCP 工具提供 wavekit 分析能力，支持 AI 辅助的波形分析。

## 特性

- **灵活的信号查询：**通过灵活的路径匹配，从层次化波形数据中查找并批量加载相关信号。
- **周期级分析：**使用丰富的 Waveform 操作处理时钟采样数据，用于分析接口反压、FIFO 占用量等周期级行为。
- **事务级分析：**使用时序模式匹配描述跨多个时钟周期的信号关系，用于协议分析、事务提取和延迟测量。
- **多格式波形支持：**通过相同的 reader 和 `Waveform` API 加载并分析 VCD、FST 和 FSDB 文件。

## 安装

```console
python -m pip install wavekit
```

wavekit 支持 Python 3.9 及以上版本。FSDB 支持需要 Verdi NPI 运行时（`libNPI.so`）；具体配置请参阅 [Reader 配置指南](docs/guides/reader.md)。

## 快速上手

从一个 VCD 波形中计算 FIFO 的占用量：

```python
import numpy as np

from wavekit import VcdReader

with VcdReader('simulation.vcd') as r:
    clock = 'tb.clk'
    depth = 16

    w_ptr = r.load_waveform(
        'tb.u_fifo.w_ptr',
        clock=clock,
    )
    r_ptr = r.load_waveform(
        'tb.u_fifo.r_ptr',
        clock=clock,
    )

    occupancy = (w_ptr + depth - r_ptr) % depth

    print('Average occupancy:', np.mean(occupancy.value))
```

完整、可运行的 FIFO 示例请参阅 [第一个波形教程](docs/getting-started/first-waveform.md)。

## 文档

完整的参考文档和指南请参阅[文档](docs/index.md)：

- [安装](docs/getting-started/installation.md)
- [第一个波形](docs/getting-started/first-waveform.md)
- [Reader 与 FSDB 配置](docs/guides/reader.md)
- [信号查询](docs/guides/signal-query.md)
- [波形分析](docs/guides/waveform-analysis.md)
- [模式匹配](docs/guides/pattern-matching.md)
- [API 参考](docs/reference/api.md)
- [完整示例](docs/examples.md)

## 开发

使用 Poetry 安装开发依赖并运行检查：

```console
poetry install
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
```

使用 Python 3.10 或更高版本在本地构建文档：

```console
poetry install --with docs
poetry run zensical build --clean --strict
```

贡献方式请参阅[贡献指南](docs/contributing.md)。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
