# mq-env：量子计算软件与编程 · 实验材料

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MindQuantum-HiQ/mq-env/HEAD)
[![notebooks](https://github.com/MindQuantum-HiQ/mq-env/actions/workflows/notebooks.yml/badge.svg)](https://github.com/MindQuantum-HiQ/mq-env/actions/workflows/notebooks.yml)

「量子计算软件与编程」课程的配套实验，基于 [MindSpore Quantum](https://www.mindspore.cn/mindquantum) 0.12。每个 Notebook 对应讲义的一个部分，练习带有即时检查，可以在浏览器中一键运行。

## 实验目录

| Notebook | 内容 | 对应讲义 | 时长 |
|---|---|---|---|
| [00.环境检查](00.环境检查.ipynb) | 检查依赖，运行第一个量子程序 | 03 环境准备 | 5 分钟 |
| [01.编程基础](01.编程基础.ipynb) | 五个动词：准备、演化、组合、测量、估计；Bernstein–Vazirani；课后挑战：IQP 编码线路 | 04 编程基础 | 40 分钟 |
| [02.Grover搜索](02.Grover搜索.ipynb) | Oracle、扩散算子、最优迭代次数、多目标搜索 | 实验 A | 30 分钟 |
| [03.QFT与相位估计](03.QFT与相位估计.ipynb) | 用 DFT 验证 QFT；挑战：用相位估计破解神秘量子门 | 实验 B | 45 分钟 |
| [04.QAOA与VQE](04.QAOA与VQE.ipynb) | QAOA 求解 Max-Cut 并与暴力枚举对照；课后作业：VQE 计算氢分子势能曲线 | 实验 C | 45 分钟 |
| [05.量子神经网络](05.量子神经网络.ipynb) | 编码线路 + 可训练线路，鸢尾花分类，与逻辑回归对照 | 实验 D | 45 分钟 |
| [06.量子启发式算法](06.量子启发式算法.ipynb) | 模拟分叉求解 2,000 个顶点的 Max-Cut，与贪心搜索对照 | 实验 E | 20 分钟 |
| [07.噪声与纠错](07.噪声与纠错.ipynb) | 噪声信道、重复码、阈值、症状与解码器 | 实验 F | 30 分钟 |

按课时选做：90 分钟讲座完成 00–02；半天工作坊再加 04；全天实训完成全部。

## 运行方式

**在线，零安装**：点击上方 Binder 徽章，1–3 分钟后进入 JupyterLab。MyBinder 是临时环境，约 10 分钟无操作就会被回收，修改请及时下载保存。

**本地**（Python 3.10 或 3.11）：

```bash
git clone https://github.com/MindQuantum-HiQ/mq-env.git
cd mq-env
python -m venv .venv
source .venv/bin/activate                      # Windows：.venv\Scripts\activate
pip install -r requirements.txt jupyterlab     # 国内可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
jupyter lab                                    # 打开 00.环境检查.ipynb，全部运行
```

实验 C 的 VQE 作业还需要 `pip install openfermion openfermionpyscf`（仅支持 Linux 与 macOS；Windows 用户请使用 MyBinder）。

## 怎么做练习

练习单元格中的 `...` 需要替换成你的代码。运行后，紧随其后的 `check` 会立即给出反馈：

- ✅ 通过：答案正确
- ❌ 未通过：附一条提示，只指出问题，不给出答案
- ⏳ 尚未完成：还有 `...` 没有替换

参考答案与完整运行结果在 [`solutions/`](solutions/) 目录，建议先独立完成。

## 目录结构

```
00–07.*.ipynb    学生版 Notebook（由参考答案生成）
solutions/       参考答案，含运行结果
mqlab/           环境检查与练习的即时检查
data/G22.txt     Gset 基准图 G22（实验 E）
images/          Notebook 中的线路图
tools/           生成学生版、执行全部 Notebook 的脚本
```

## 维护

只修改 `solutions/` 中的参考答案。学生需要填写的代码用 `### BEGIN SOLUTION` 与 `### END SOLUTION` 标出，规则见 [`tools/make_student.py`](tools/make_student.py)。修改后运行：

```bash
python tools/make_student.py                       # 生成学生版
pip install nbclient nbformat ipykernel
python tools/run_notebooks.py --write-solutions    # 执行全部 Notebook，并更新参考答案中的运行结果
```

每次推送时，GitHub Actions 会执行全部 Notebook：参考答案的所有检查必须通过，学生版必须能从头运行到尾。

依赖版本固定在 [`requirements.txt`](requirements.txt)，均已实测：mindquantum 0.12.0、MindSpore 2.11.0、NumPy 1.26。NumPy 2 与 MindSpore 2.11 不兼容，训练时会报错。

## 许可

代码与 Notebook 采用 Apache License 2.0。G22 数据来自 Gset 基准集（Y. Ye，Stanford University）。
