# 示例

完整示例位于 [`example/`](../example/) 目录下，每个目录包含 HDL 测试文件、分析脚本和 Makefile。命令请从仓库根目录执行。

| 示例 | 用途 | 涉及的 wavekit 功能 | 运行命令 |
| --- | --- | --- | --- |
| [FIFO 水位](../example/fifo_occupancy/) | 根据采样到的指针计算 FIFO 水位 | Reader 加载；Waveform 运算；与 NumPy 交互 | `make -C example/fifo_occupancy all` |
| [FIFO 反压](../example/fifo_latency/) | 统计写请求因 FIFO 已满而被阻塞的时长 | Waveform 操作；边沿检测 | `make -C example/fifo_latency all` |
| [AXI-Lite 读延迟](../example/axi_lite_read_latency/) | 测量 AXI-Lite 读响应延迟 | 声明式模式匹配；事件消费 | `make -C example/axi_lite_read_latency all` |
| [AXI ID 匹配](../example/axi_id_matching/) | 根据事务 ID 将读响应与请求匹配 | 声明式模式匹配；基于已捕获值的条件判断 | `make -C example/axi_id_matching all` |
| [DMA 命令流](../example/dma_command_stream/) | 提取长度可变的读写命令 | 编程式模式；Python 控制流 | `make -C example/dma_command_stream all` |
| [Scoreboard](../example/scoreboard/) | 检查 FIFO 读写数据的完整性和顺序 | 波形过滤；索引提取；与 NumPy 交互 | `make -C example/scoreboard all` |
