"""从 solutions/ 中的参考答案生成学生版 Notebook（写到仓库根目录）。

只需维护 solutions/ 中的参考答案，学生版由本脚本生成。规则：

- 代码单元中 `### BEGIN SOLUTION` 与 `### END SOLUTION` 之间的内容替换为 `...`（保持缩进）；
  块内若有 `### STUDENT: <代码>` 行，则改用这些行作为学生版代码；
- 含有 “仅参考答案需要” 的代码行删除；以 `<!-- 仅参考答案 -->` 开头的 Markdown 单元格删除；
- Markdown 中的 `../images/` 改为 `images/`；
- 清空输出与执行计数。

用法：
    python tools/make_student.py           # 生成学生版
    python tools/make_student.py --check   # 只检查学生版是否与参考答案同步（CI 使用）
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = re.compile(r"^(\s*)### BEGIN SOLUTION\s*$")
END = re.compile(r"^\s*### END SOLUTION\s*$")
STUDENT = re.compile(r"^(\s*)### STUDENT: ?(.*)$")
SOLUTION_ONLY = "仅参考答案需要"
SOLUTION_ONLY_MD = "<!-- 仅参考答案 -->"


def _lines(source):
    return "".join(source).split("\n") if isinstance(source, list) else source.split("\n")


def strip_code(source):
    out, block, indent = [], None, ""
    for line in _lines(source):
        if block is None:
            m = BEGIN.match(line)
            if m:
                block, indent = [], m.group(1)
            elif SOLUTION_ONLY not in line:
                out.append(line)
        elif END.match(line):
            out.extend(block if block else [indent + "..."])
            block = None
        else:
            m = STUDENT.match(line)
            if m:
                block.append(m.group(1) + m.group(2))
    if block is not None:
        raise ValueError("缺少 ### END SOLUTION")
    return "\n".join(out)


def student_notebook(nb):
    nb = json.loads(json.dumps(nb))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb["cells"] = [c for c in nb["cells"]
                   if not (c["cell_type"] == "markdown" and "".join(c["source"]).lstrip().startswith(SOLUTION_ONLY_MD))]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = strip_code(cell["source"])
            cell["outputs"] = []
            cell["execution_count"] = None
        elif cell["cell_type"] == "markdown":
            cell["source"] = "\n".join(_lines(cell["source"])).replace("](../images/", "](images/")
        cell.get("metadata", {}).pop("execution", None)
    return nb


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main(check=False):
    stale = []
    for path in sorted(glob.glob(os.path.join(ROOT, "solutions", "*.ipynb"))):
        with open(path, encoding="utf-8") as f:
            text = dumps(student_notebook(json.load(f)))
        target = os.path.join(ROOT, os.path.basename(path))
        old = open(target, encoding="utf-8").read() if os.path.exists(target) else None
        if old != text:
            stale.append(os.path.basename(target))
            if not check:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(text)
    if check and stale:
        print("学生版与参考答案不同步，请运行 python tools/make_student.py：", ", ".join(stale))
        sys.exit(1)
    print("已更新：" + ", ".join(stale) if stale and not check else "学生版与参考答案同步")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
