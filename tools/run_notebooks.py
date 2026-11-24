"""从头执行全部 Notebook，确认实验材料可用。

- 参考答案（solutions/）：必须无报错，且所有检查都显示 ✅；
- 学生版（根目录）：必须能从头运行到尾，练习显示 ⏳（尚未完成），不得出现报错或 ❌。

用法：
    python tools/run_notebooks.py                       # 全部执行（CI 使用）
    python tools/run_notebooks.py --write-solutions     # 同时把运行结果写回参考答案
    python tools/run_notebooks.py 03                    # 只执行编号以 03 开头的 Notebook
需要：pip install nbclient nbformat ipykernel
"""
import glob
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def outputs_text(nb):
    parts = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif "data" in out and "text/plain" in out["data"]:
                parts.append(out["data"]["text/plain"])
    return "\n".join(parts)


def run(path, write=False):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path)}})
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        return f"报错：{str(e).splitlines()[-1] if str(e) else e}"
    text = outputs_text(nb)
    is_solution = os.path.basename(os.path.dirname(path)) == "solutions"
    bad = ["❌", "⏳"] if is_solution else ["❌"]
    found = [b for b in bad if b in text]
    if found:
        return "检查未通过：输出中出现 " + "、".join(found)
    if write and is_solution:
        for cell in nb.cells:
            cell.get("metadata", {}).pop("execution", None)
        nbformat.write(nb, path)
    print(f"  {time.time() - t0:5.1f} s")
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write-solutions" in sys.argv
    paths = sorted(glob.glob(os.path.join(ROOT, "solutions", "*.ipynb"))) + sorted(glob.glob(os.path.join(ROOT, "*.ipynb")))
    if args:
        paths = [p for p in paths if any(os.path.basename(p).startswith(a) for a in args)]
    failures = []
    for p in paths:
        rel = os.path.relpath(p, ROOT)
        print(rel, flush=True)
        err = run(p, write)
        if err:
            print("  " + err)
            failures.append(rel)
    if failures:
        print("\n未通过：" + "、".join(failures))
        sys.exit(1)
    print(f"\n全部 {len(paths)} 个 Notebook 通过")


if __name__ == "__main__":
    main()
