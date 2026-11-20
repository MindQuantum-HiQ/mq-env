"""环境检查与数据路径。"""
import os
import sys
from importlib import metadata

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (包名, 用途, 是否必需)
_PACKAGES = [
    ("mindquantum", "量子编程框架", True),
    ("mindspore", "实验 C、D 的训练", True),
    ("numpy", "数值计算", True),
    ("scipy", "实验 E 的稀疏矩阵", True),
    ("networkx", "实验 C 的图", True),
    ("scikit-learn", "实验 D 的数据与经典基线", True),
    ("matplotlib", "画图", True),
    ("openfermionpyscf", "实验 C 的 VQE 作业（仅 Linux / macOS）", False),
]


def data_path(name):
    """返回 data/ 目录下文件的绝对路径（无论 Notebook 位于哪一级目录）。"""
    return os.path.join(_ROOT, "data", name)


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _major_minor(v):
    return tuple(int(x) for x in v.split(".")[:2])


def check_env():
    """打印关键依赖的版本，并给出结论。"""
    problems, notes = [], []
    py = sys.version_info
    rows = [("Python", f"{py.major}.{py.minor}.{py.micro}", "解释器")]
    if (py.major, py.minor) not in [(3, 10), (3, 11)]:
        problems.append("Python 需要 3.10 或 3.11（mindquantum 0.12 与 MindSpore 2.11 的交集）")

    for pkg, use, required in _PACKAGES:
        v = _version(pkg)
        rows.append((pkg, v or "未安装", use))
        if v is None:
            (problems if required else notes).append(f"{pkg} 未安装：{use}")
    v = _version("mindquantum")
    if v and _major_minor(v) < (0, 12):
        problems.append(f"mindquantum {v} 过旧，请升级到 0.12 或更高版本")
    v = _version("numpy")
    if v and _major_minor(v) >= (2, 0):
        problems.append("NumPy 2 与 MindSpore 2.11 不兼容（训练时报错），请安装 numpy<2")

    width = max(len(r[0]) for r in rows) + 2
    for name, ver, use in rows:
        print(f"{name:<{width}}{ver:<12}{use}")
    print()
    for n in notes:
        print(f"提示：{n}")
    if problems:
        for p in problems:
            print(f"❌ {p}")
        print("\n请先解决以上问题，或使用 README 中的 MyBinder 在线环境。")
    else:
        print("✅ 环境就绪")
