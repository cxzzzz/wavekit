# <img src="docs/zh/images/logo.svg" alt="" width="30" height="30" valign="middle"> wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://pepy.tech/badge/wavekit)](https://pepy.tech/project/wavekit)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

[English](README.md) | 中文

**Wavekit 是一个用于高层次数字波形分析的 Python 库。** Wavekit 支持常见的数字波形格式，借助批量信号查询以及周期级与事务级分析，帮助用户从波形记录的信号变化中提取周期状态、时序关系等更高层次的信息。

> **AI 集成：** [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) 提供 MCP Server，让 AI 工具可以调用
> wavekit 进行波形分析。

## wavekit 主要功能

- **批量信号查询：** 支持多种路径匹配方式，从层次化波形数据中查找并批量加载相关信号。
- **周期级分析：** 使用多种 Waveform 操作处理时钟采样数据，分析接口反压、FIFO 水位等周期级行为。
- **事务级分析：** 使用时序模式匹配描述跨多个时钟周期的信号关系，分析协议行为、提取事务并测量延迟。
- **多格式波形支持：** 使用同一套 API 加载和分析 VCD、FST 和 FSDB 文件。

## 安装

```console
python -m pip install wavekit
```

wavekit 支持 Python 3.9 及以上版本。FSDB 支持需要 Verdi NPI 运行时（`libNPI.so`）；配置方法请参阅[安装说明](https://cxzzzz.github.io/wavekit/zh/getting-started/installation/)。

## 快速上手

根据 VCD 波形计算 FIFO 水位：

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

完整的可运行 FIFO 示例请参阅 [第一个波形教程](https://cxzzzz.github.io/wavekit/zh/getting-started/first-waveform/)。

## 文档

完整的参考文档和指南请参阅[文档](https://cxzzzz.github.io/wavekit/zh/)：

- [安装](https://cxzzzz.github.io/wavekit/zh/getting-started/installation/)
- [第一个波形](https://cxzzzz.github.io/wavekit/zh/getting-started/first-waveform/)
- [Reader 使用指南](https://cxzzzz.github.io/wavekit/zh/guides/reader/)
- [信号查询](https://cxzzzz.github.io/wavekit/zh/guides/signal-query/)
- [波形分析](https://cxzzzz.github.io/wavekit/zh/guides/waveform-analysis/)
- [模式匹配](https://cxzzzz.github.io/wavekit/zh/guides/pattern-matching/)
- [API 参考](https://cxzzzz.github.io/wavekit/zh/reference/api/)
- [完整示例](https://cxzzzz.github.io/wavekit/zh/examples/)

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

有关贡献方式，请参阅[贡献指南](https://cxzzzz.github.io/wavekit/zh/contributing/)。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
