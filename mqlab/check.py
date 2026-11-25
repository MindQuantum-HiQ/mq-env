"""练习的即时检查。

每个检查函数打印一行结果：
  ✅ 通过：答案正确
  ❌ 未通过：附一条提示（只指出问题，不给出答案）
  ⏳ 尚未完成：还有 ... 没有替换成你的代码

比较量子态与线路时忽略全局相位；多个正确写法都能通过。
"""
import numpy as np
from mindquantum import Circuit, H, RZ, Simulator, UnivMathGate, X

__all__ = [
    # 01 编程基础
    "bell_states", "iqp",
    # 02 Grover
    "grover_iterations",
    # 03 QFT 与相位估计
    "mystery_gate", "mystery_phase",
    # 04 QAOA 与 VQE
    "vqe_ansatz", "vqe_curve",
    # 07 噪声与纠错
    "decoder",
]


# ---------------------------------------------------------------- 通用工具

def _ok(msg):
    print(f"✅ 通过：{msg}")


def _fail(msg):
    print(f"❌ 未通过：{msg}")


def _todo(msg="把 ... 替换为你的代码，再运行一次"):
    print(f"⏳ 尚未完成：{msg}")


def _unfinished(x):
    return x is Ellipsis or x is None or (isinstance(x, Circuit) and len(x) == 0)


def _call(fn, *args):
    """调用学生写的函数；返回 (结果, 错误信息)。"""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001  学生代码可能抛出任何异常
        return None, f"{type(e).__name__}: {e}"


def _state(circ, pr=None):
    c = circ.remove_measure()
    sim = Simulator("mqvector", c.n_qubits)
    sim.apply_circuit(c, pr=pr)
    return sim.get_qs()


def _same_state(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    return a.shape == b.shape and abs(np.vdot(a / np.linalg.norm(a), b / np.linalg.norm(b))) > 1 - 1e-6


def _same_unitary(u, v):
    if u.shape != v.shape:
        return False
    k = np.unravel_index(np.argmax(np.abs(v)), v.shape)
    if abs(u[k]) < 1e-9:
        return False
    return np.allclose(u * (v[k] / u[k]), v, atol=1e-6)


def _ket(psi, n_terms=4):
    """把态矢量写成狄拉克记号（只显示最大的几项）。"""
    psi = np.asarray(psi)
    n = int(np.log2(len(psi)))
    idx = [i for i in np.argsort(-np.abs(psi)) if abs(psi[i]) > 1e-6][:n_terms]
    terms = []
    for i in sorted(idx):
        a = psi[i]
        a = a.real if abs(a.imag) < 1e-9 else a
        terms.append(f"({np.round(a, 3)})|{i:0{n}b}⟩")
    more = " + …" if np.sum(np.abs(psi) > 1e-6) > n_terms else ""
    return " + ".join(terms) + more


# ---------------------------------------------------------------- 01 编程基础

_BELL = {
    "phi+": ("Φ+", [1, 0, 0, 1]),
    "phi-": ("Φ−", [1, 0, 0, -1]),
    "psi+": ("Ψ+", [0, 1, 1, 0]),
    "psi-": ("Ψ−", [0, 1, -1, 0]),
}


def bell_states(fn):
    """练习：bell(kind) 返回制备对应 Bell 态的线路，kind 取 'phi+'、'phi-'、'psi+'、'psi-'。"""
    for kind, (name, vec) in _BELL.items():
        circ, err = _call(fn, kind)
        if err:
            return _fail(f"bell('{kind}') 运行出错：{err}")
        if _unfinished(circ):
            return _todo(f"bell('{kind}') 返回的线路是空的")
        if circ.n_qubits != 2:
            return _fail(f"bell('{kind}') 用了 {circ.n_qubits} 个比特，应为 2 个")
        psi = _state(circ)
        if _same_state(psi, vec):
            continue
        got = [k for k, (_, v) in _BELL.items() if _same_state(psi, v)]
        if got:
            hint = "差一个相对相位 −1" if got[0][:3] == kind[:3] else "两个比特的取值关系不对：Φ 态两比特相同，Ψ 态两比特相反"
            return _fail(f"bell('{kind}') 得到的是 {_BELL[got[0]][0]}，不是 {name}：{hint}")
        return _fail(f"bell('{kind}') 的末态是 {_ket(psi)}，不是 {name}")
    _ok("四个 Bell 态全部正确")


def _iqp_reference():
    c = Circuit([H.on(i) for i in range(4)])
    c += Circuit([RZ(f"alpha{i}").on(i) for i in range(4)])
    for i in range(3):
        c += X.on(i + 1, i)
        c += RZ(f"alpha{4 + i}").on(i + 1)
        c += X.on(i + 1, i)
    return c


def iqp(circ):
    """课后挑战：按图搭建 4 比特 IQP 编码线路，参数名为 alpha0 … alpha6。"""
    if _unfinished(circ):
        return _todo()
    want = [f"alpha{i}" for i in range(7)]
    if circ.n_qubits != 4:
        return _fail(f"线路用了 {circ.n_qubits} 个比特，应为 4 个")
    if sorted(circ.params_name) != want:
        return _fail(f"参数应为 alpha0 … alpha6，现在是 {circ.params_name}")
    ref, rng = _iqp_reference(), np.random.default_rng(0)
    c = circ.remove_measure()
    for _ in range(3):
        pr = dict(zip(want, rng.uniform(0, 2 * np.pi, 7)))
        if not _same_unitary(c.matrix(pr=pr), ref.matrix(pr=pr)):
            return _fail("线路与图中不一致：检查每个 RZ 的位置与参数编号，以及 CNOT 的控制比特和目标比特")
    _ok("与图中的 IQP 编码线路一致")


# ---------------------------------------------------------------- 02 Grover

def grover_iterations(k):
    """练习：4 个比特、1 个目标时的最优迭代次数。"""
    if _unfinished(k):
        return _todo()
    theta = np.arcsin(1 / 4)
    p = {kk: np.sin((2 * kk + 1) * theta) ** 2 for kk in range(8)}
    k = int(k)
    if k == 3:
        return _ok(f"N = 16 时最优迭代 3 次，成功率 {p[3]:.1%}")
    msg = f"迭代 {k} 次的成功率为 {p[k]:.1%}。" if k in p else ""
    _fail(msg + "提示：sin θ = 1/√N，最优迭代次数 k ≈ π/(4θ) − 1/2")


# ---------------------------------------------------------------- 03 QFT 与相位估计

_MYSTERY_PHI = 13 / 32


def mystery_gate():
    """课后挑战：返回单比特门 U，U|1⟩ = e^{2πiφ}|1⟩，φ 未知（用相位估计求出）。"""
    return UnivMathGate("U", np.diag([1, np.exp(2j * np.pi * _MYSTERY_PHI)]))


def mystery_phase(phi):
    """课后挑战：神秘门的相位 φ。"""
    if _unfinished(phi):
        return _todo()
    try:
        phi = float(np.real(phi))
    except (TypeError, ValueError):
        return _fail("phi 应是一个 0 到 1 之间的数")
    if abs(phi - _MYSTERY_PHI) < 1e-9:
        return _ok("φ = 13/32 = 0.40625：要 5 个计数比特，采样结果才完全确定")
    for n in range(5, 9):                      # 5 位以上才能精确表示 φ
        bits = f"{round(_MYSTERY_PHI * 2 ** n):0{n}b}"
        if abs(phi - int(bits[::-1], 2) / 2 ** n) < 1e-9:
            return _fail("比特串读反了：MindSpore Quantum 按 q(n−1)…q0 书写，读成 φ 的二进制之前要先反转")
    if abs(phi - _MYSTERY_PHI) < 1 / 8:
        return _fail(f"{phi} 只是近似值：在这个计数比特数下，采样结果不止一种。"
                     "增加计数比特，直到 1,000 次采样全部相同")
    return _fail(f"{phi} 与 φ 相差较远：检查线路是否与第 3 节的 qpe(U, n) 相同，以及读数时是否除以 2^n")


# ---------------------------------------------------------------- 04 QAOA 与 VQE

def _vqe_reference():
    from mindquantum import RY
    c = Circuit([X.on(0), X.on(1)])
    c += Circuit([RY(f"a{i}").on(i) for i in range(4)])
    c += Circuit([X.on(i + 1, i) for i in range(3)])
    c += Circuit([RY(f"b{i}").on(i) for i in range(4)])
    return c


def vqe_ansatz(circ):
    """作业：按图搭建 VQE 线路，参数名为 a0 … a3、b0 … b3。"""
    if _unfinished(circ):
        return _todo()
    want = sorted([f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)])
    if circ.n_qubits != 4:
        return _fail(f"线路用了 {circ.n_qubits} 个比特，应为 4 个")
    if sorted(circ.params_name) != want:
        return _fail(f"参数应为 a0 … a3、b0 … b3，现在是 {circ.params_name}")
    ref, rng = _vqe_reference(), np.random.default_rng(1)
    for _ in range(3):
        pr = dict(zip(want, rng.uniform(0, 2 * np.pi, 8)))
        if not _same_state(_state(circ, pr), _state(ref, pr)):
            return _fail("线路与图中不一致：检查初始的 X 门、RY 的参数编号与 CNOT 的方向")
    _ok("与图中的线路一致")


def vqe_curve(energies, fci):
    """作业：VQE 能量与 FCI 的最大偏差是否在化学精度（1.6 mHa）以内。"""
    if _unfinished(energies) or len(energies) == 0:
        return _todo("energies 还是空的：完成上面的训练循环")
    e, f = np.ravel(np.asarray(energies, dtype=float)), np.ravel(np.asarray(fci, dtype=float))
    if e.shape != f.shape:
        return _fail(f"energies 有 {e.size} 个值，FCI 有 {f.size} 个，应一一对应")
    dev = np.max(np.abs(e - f)) * 1000
    if dev < 1.6:
        return _ok(f"所有键长都达到化学精度：与 FCI 的最大偏差 {dev:.2g} mHa < 1.6 mHa")
    _fail(f"与 FCI 的最大偏差为 {dev:.2f} mHa，超过化学精度 1.6 mHa：增加训练步数，或检查线路")


# ---------------------------------------------------------------- 07 噪声与纠错

def decoder(fn):
    """练习：decode(s0, s1) 根据症状返回需要翻转的比特编号（0、1、2），无需纠正时返回 None。"""
    table = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
    for (s0, s1), want in table.items():
        got, err = _call(fn, s0, s1)
        if err:
            return _fail(f"症状 ({s0}, {s1}) 时运行出错：{err}")
        if got is Ellipsis:
            return _todo()
        if got != want:
            w = "不翻转（None）" if want is None else f"翻转 q{want}"
            return _fail(f"症状 (s0, s1) = ({s0}, {s1}) 时应{w}，你的解码器返回 {got!r}")
    _ok("解码器正确：四种症状各自对应唯一的纠正操作")
