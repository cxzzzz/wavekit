# 参与贡献

欢迎为 wavekit 的库代码、示例和文档贡献内容。

## 开发环境

```console
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

修改文档时，还需要安装文档依赖：

```console
poetry install --with docs
```

## 质量检查

从仓库根目录运行完整检查：

```console
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest
```

示例测试会使用 Icarus Verilog 编译 HDL 测试文件，然后运行分析脚本。运行示例测试前请安装 `iverilog`。

## 构建文档

文档工具链需要 Python 3.10 或更高版本。使用严格模式构建文档，这样缺少页面或内部链接失效时会直接报告错误：

```console
poetry run zensical build --clean --strict
```

## 文档约定

- 每个页面都要有明确的 H1 标题。
- 文档页面之间使用相对链接。
- 完整的可运行示例放在 `example/` 中，Markdown 只链接到示例，不复制示例源码。
- 公共 API 使用 NumPy 风格 docstring。
- 添加完整示例及其验证测试时，同时更新手动维护的[示例索引](examples.md)。

## 提交 Pull Request

保持每个 pull request 的范围清晰，并说明涉及的用户行为、API 或示例变化。提交前运行相关检查。
