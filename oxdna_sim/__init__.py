"""oxDNA 粗粒化分子动力学模拟管道

用于验证发夹阻断引物的三态竞争热力学机制：
  · hairpin-folded state (茎区稳定折叠，3' 端封闭)
  · unfolded state       (单链伸展，准备结合目标)
  · transition state     (折叠/展开过渡态能垒)

模拟由 6 个温度点的独立 MD 构成，覆盖发夹 Tm 上下范围，
从单链轨迹中提取茎区碱基对数(order parameter) → 构建自由能面(FES) →
与 primer3 预测 Tm 对比验证。

依赖：
  oxDNA 模拟引擎  https://github.com/lorenzo-rovigatti/oxDNA
                  (需独立安装，提供 `oxDNA` 可执行文件)
  numpy, scipy, matplotlib  (分析与作图，已包含在 requirements.txt)
"""
