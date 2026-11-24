"""练习的即时检查。

每个检查函数打印一行结果：
  ✅ 通过：答案正确
  ❌ 未通过：附一条提示（只指出问题，不给出答案）
  ⏳ 尚未完成：还有 ... 没有替换成你的代码

比较量子态与线路时忽略全局相位；多个正确写法都能通过。
"""
from math import comb

import numpy as np
from mindquantum import Circuit, H, PhaseShift, RZ, Simulator, X
from mindquantum import qft as _qft

__all__ = [
    # 01 编程基础
    "minus_state", "bell_states", "ghz", "bell_correlators", "gradient_descent", "bv", "iqp",
    # 02 Grover
    "grover_iterations", "oracle_multi",
    # 03 QFT 与相位估计
    "inverse_qft_reference", "qpe",
    # 04 QAOA 与 VQE
    "brute_force_maxcut", "vqe_ansatz", "vqe_curve",
    # 06 量子启发式算法
    "cut_value",
    # 07 噪声与纠错
    "repetition_theory", "decoder",
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


def _data_marginal(circ, n_data):
    """线路末态在低位 n_data 个比特上的边缘分布（高位比特视为辅助比特）。"""
    probs = np.abs(_state(circ)) ** 2
    return probs.reshape(-1, 2 ** n_data).sum(axis=0)


# ---------------------------------------------------------------- 01 编程基础

def minus_state(circ):
    """练习：在 q0 上制备 |−⟩ = (|0⟩ − |1⟩)/√2。"""
    if _unfinished(circ):
        return _todo()
    if circ.n_qubits != 1:
        return _fail(f"线路用了 {circ.n_qubits} 个比特，本题只需要 q0")
    psi = _state(circ)
    if _same_state(psi, [1, -1]):
        return _ok("得到 |−⟩ = (|0⟩ − |1⟩)/√2")
    if _same_state(psi, [1, 1]):
        return _fail("得到的是 |+⟩：两个振幅大小对了，还差一个相对相位 −1")
    if _same_state(psi, [0, 1]) or _same_state(psi, [1, 0]):
        return _fail("末态是一个基态：还需要用 H 制造叠加")
    return _fail(f"末态是 {_ket(psi)}，不是 |−⟩")


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


def ghz(fn):
    """练习：ghz(n) 返回制备 n 比特 GHZ 态 (|0…0⟩ + |1…1⟩)/√2 的线路。"""
    for n in (2, 3, 5):
        circ, err = _call(fn, n)
        if err:
            return _fail(f"ghz({n}) 运行出错：{err}")
        if _unfinished(circ):
            return _todo(f"ghz({n}) 返回的线路是空的")
        if circ.n_qubits != n:
            return _fail(f"ghz({n}) 用了 {circ.n_qubits} 个比特，应为 {n} 个")
        target = np.zeros(2 ** n)
        target[0] = target[-1] = 1
        psi = _state(circ)
        if not _same_state(psi, target):
            return _fail(f"ghz({n}) 的末态是 {_ket(psi)}；应只有 |{'0' * n}⟩ 和 |{'1' * n}⟩ 两项，振幅相同")
    _ok("n = 2、3、5 的 GHZ 态都正确")


def bell_correlators(xx, yy):
    """练习：Φ+ 态上 ⟨X0 X1⟩ 与 ⟨Y0 Y1⟩ 的值。"""
    if _unfinished(xx) or _unfinished(yy):
        return _todo()
    try:
        xx, yy = float(np.real(xx)), float(np.real(yy))
    except (TypeError, ValueError):
        return _fail("请给出数值（get_expectation 返回复数，可以取 .real）")
    if abs(xx - 1) < 1e-6 and abs(yy + 1) < 1e-6:
        return _ok("⟨X0 X1⟩ = +1，⟨Y0 Y1⟩ = −1：Φ+ 在 X、Y 方向上也完全相关")
    _fail(f"得到 ⟨X0 X1⟩ = {xx:+.3f}、⟨Y0 Y1⟩ = {yy:+.3f}。检查线路是否制备了 Φ+，哈密顿量是否写成 'X0 X1'、'Y0 Y1'")


def gradient_descent(update):
    """练习：update(theta, grad, eta) 返回梯度下降一步后的 θ。

    检查方法：对 E(θ) = ⟨Z⟩（线路 RX(θ)|0⟩）从 θ = 0.5 出发、学习率 0.2 迭代 60 步，看能否到达最小值 −1。
    """
    from mindquantum import Hamiltonian, QubitOperator, RX
    circ = Circuit([RX("theta").on(0)])
    grad_ops = Simulator("mqvector", 1).get_expectation_with_grad(Hamiltonian(QubitOperator("Z0")), circ)
    theta = 0.5
    for _ in range(60):
        _, g = grad_ops(np.array([theta]))
        new, err = _call(update, theta, float(g.real.item()), 0.2)
        if err:
            return _fail(f"运行出错：{err}")
        if _unfinished(new):
            return _todo()
        theta = float(np.real(new))
    e = np.cos(theta)
    if e < -0.999:
        return _ok(f"60 步后 θ = {theta:.3f}，⟨Z⟩ = {e:.4f}：找到了最小值 −1")
    if e < np.cos(0.5):
        return _fail(f"60 步后 ⟨Z⟩ = {e:.3f}：方向对了但收敛太慢，检查步长是否乘了学习率 eta")
    _fail(f"60 步后 ⟨Z⟩ = {e:.3f}，没有下降：梯度下降应沿梯度的反方向走，θ ← θ − η·dE/dθ")


def bv(fn):
    """练习：bv_circuit(s) 返回 Bernstein–Vazirani 线路（数据比特 q0…q(n−1)，辅助比特 qn）。"""
    for s in ["1011", "0110", "1", "111", "10000000"]:
        circ, err = _call(fn, s)
        if err:
            return _fail(f"bv_circuit('{s}') 运行出错：{err}")
        if _unfinished(circ):
            return _todo(f"bv_circuit('{s}') 返回的线路是空的")
        n = len(s)
        if circ.n_qubits != n + 1:
            return _fail(f"bv_circuit('{s}') 用了 {circ.n_qubits} 个比特，应为 len(s) + 1 = {n + 1} 个（含辅助比特）")
        p = _data_marginal(circ, n)
        if p[int(s, 2)] > 0.999:
            continue
        if p[int(s[::-1], 2)] > 0.999:
            return _fail(f"bv_circuit('{s}') 读出了 '{s[::-1]}'：比特序反了，s 的最右一位对应 q0")
        if p[0] > 0.999:
            return _fail(f"bv_circuit('{s}') 读出全 0：Oracle 没有起作用，检查辅助比特是否先用 X 翻到 |1⟩、再用 H 变成 |−⟩")
        if p.max() < 0.5:
            return _fail(f"bv_circuit('{s}') 的结果是随机的：检查测量前数据比特是否再作用了一层 H")
        return _fail(f"bv_circuit('{s}') 最可能读出 '{int(np.argmax(p)):0{n}b}'，应为 '{s}'")
    _ok("5 个秘密串（含 1 位和 8 位）都只需一次运行")


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


def oracle_multi(fn):
    """练习：oracle_multi(targets) 翻转每个目标态的符号，其余基态不变。"""
    for targets in (["101", "011"], ["000", "111", "010"], ["0110"]):
        circ, err = _call(fn, targets)
        if err:
            return _fail(f"oracle_multi({targets}) 运行出错：{err}")
        if _unfinished(circ):
            return _todo(f"oracle_multi({targets}) 返回的线路是空的")
        n = len(targets[0])
        if circ.n_qubits != n:
            return _fail(f"oracle_multi({targets}) 用了 {circ.n_qubits} 个比特，应为 {n} 个")
        u = circ.matrix()
        d = np.diag(u)
        if not np.allclose(u, np.diag(d), atol=1e-8):
            return _fail("Oracle 只能改变相位：线路的矩阵应为对角阵")
        want = np.ones(2 ** n)
        for t in targets:
            want[int(t, 2)] = -1
        ratio = d / want
        if not np.allclose(ratio, ratio[0], atol=1e-8):
            ref = next(i for i in range(2 ** n) if want[i] > 0)
            flipped = [f"{i:0{n}b}" for i in range(2 ** n) if np.real(d[i] / d[ref]) < 0]
            return _fail(f"oracle_multi({targets}) 翻转了 {flipped} 的符号，应翻转 {targets}")
    _ok("多目标 Oracle 正确：每个目标态的符号都被翻转，其余基态不变")


# ---------------------------------------------------------------- 03 QFT 与相位估计

def inverse_qft_reference(fn):
    """练习：用 NumPy 写出与 qft(range(n)).hermitian() 等价的经典计算。"""
    rng = np.random.default_rng(3)
    for n in (2, 3, 4):
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        ref, err = _call(fn, psi.copy())
        if err:
            return _fail(f"n = {n} 时运行出错：{err}")
        if _unfinished(ref):
            return _todo()
        sim = Simulator("mqvector", n)
        sim.set_qs(psi)
        sim.apply_circuit(_qft(range(n)).hermitian())
        if not np.allclose(np.asarray(ref), sim.get_qs(), atol=1e-8):
            return _fail(f"n = {n} 时与 qft(range(n)).hermitian() 的结果不一致。"
                         "提示：正变换对应 np.fft.ifft，逆变换对应 np.fft.fft；比特反转仍然需要")
    _ok("n = 2、3、4 的随机态都与逆 QFT 线路一致")


def qpe(fn):
    """挑战：qpe(U, n) 返回相位估计线路（计数比特 q0…q(n−1)，目标比特 qn）。"""
    n, conventions = 3, set()
    for phi in (0.625, 0.375, 0.125, 0.875):
        circ, err = _call(fn, PhaseShift(2 * np.pi * phi), n)
        if err:
            return _fail(f"φ = {phi} 时运行出错：{err}")
        if _unfinished(circ):
            return _todo()
        if circ.n_qubits != n + 1:
            return _fail(f"线路用了 {circ.n_qubits} 个比特，应为 n + 1 = {n + 1} 个")
        p = _data_marginal(circ, n)
        bits = f"{round(phi * 2 ** n):0{n}b}"
        if p[int(bits[::-1], 2)] > 0.99:
            conventions.add("reverse")
        elif p[int(bits, 2)] > 0.99:
            conventions.add("direct")
        elif p[0] > 0.99:
            return _fail("测得的相位总是 0：目标比特要先制备到 U 的本征态 |1⟩")
        elif p.max() < 0.9:
            return _fail("结果不确定：检查受控 U 的幂次与计数比特的对应关系"
                         "（最上面的计数比特控制最高次幂），以及是否用了逆 QFT（.hermitian()）")
        else:
            return _fail(f"φ = {phi} 时最可能的测量结果是 '{int(np.argmax(p)):0{n}b}'，与 φ 不对应")
    if conventions == {"reverse"}:
        return _ok("四个相位都被精确估计；按本题约定，读结果时要反转比特串")
    if conventions == {"direct"}:
        return _ok("四个相位都被精确估计；你的线路直接按 q2 q1 q0 输出 φ 的二进制，读结果时不需要反转")
    _fail("不同相位下比特顺序不一致，检查受控门与逆 QFT 的比特顺序")


# ---------------------------------------------------------------- 04 QAOA 与 VQE

def _maxcut_bruteforce(graph):
    nodes = sorted(graph.nodes)
    best = 0
    for mask in range(2 ** len(nodes)):
        side = {v: (mask >> k) & 1 for k, v in enumerate(nodes)}
        best = max(best, sum(side[a] != side[b] for a, b in graph.edges))
    return best


def brute_force_maxcut(fn):
    """练习：brute_force_maxcut(graph) 返回最大割的边数。"""
    import networkx as nx
    graphs = [nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2)])]
    graphs += [nx.gnm_random_graph(n, m, seed=s) for n, m, s in [(6, 9, 1), (7, 12, 2), (8, 14, 3)]]
    for g in graphs:
        got, err = _call(fn, g)
        if err:
            return _fail(f"运行出错：{err}")
        if _unfinished(got):
            return _todo()
        if isinstance(got, (tuple, list)):
            got = got[0]
        want = _maxcut_bruteforce(g)
        if int(got) != want:
            return _fail(f"{g.number_of_nodes()} 个顶点、{g.number_of_edges()} 条边的图：得到 {got}，应为 {want}")
    _ok("4 个图（含 3 个随机图）的最大割全部正确")


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


# ---------------------------------------------------------------- 06 量子启发式算法

def cut_value(fn):
    """练习：cut_value(A, x) 返回分组 x（元素为 ±1）对应的割边数，A 为对称邻接矩阵。"""
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(1)
    for n in (5, 30, 200):
        upper = np.triu((rng.random((n, n)) < 0.15).astype(float), 1)
        a = csr_matrix(upper + upper.T)
        x = rng.choice([-1, 1], n)
        got, err = _call(fn, a, x.copy())
        if err:
            return _fail(f"n = {n} 时运行出错：{err}")
        if _unfinished(got):
            return _todo()
        i, j = np.nonzero(upper)
        want = int(np.sum(x[i] != x[j]))
        if abs(float(got) - want) > 1e-6:
            hint = "（每条边被数了两次？对称矩阵中每条边出现两次）" if abs(float(got) - 2 * want) < 1e-6 else ""
            return _fail(f"{n} 个顶点的随机图：得到 {float(got):g}，应为 {want}{hint}")
    _ok("3 个随机图的割边数全部正确")


# ---------------------------------------------------------------- 07 噪声与纠错

def _rep_theory(d, p):
    return sum(comb(d, k) * p ** k * (1 - p) ** (d - k) for k in range(d // 2 + 1, d + 1))


def repetition_theory(fn):
    """练习：码距 d 的重复码在独立比特翻转概率 p 下的逻辑错误率（多数表决）。"""
    for d in (1, 3, 5, 7):
        for p in (0.01, 0.1, 0.3):
            got, err = _call(fn, d, p)
            if err:
                return _fail(f"d = {d}、p = {p} 时运行出错：{err}")
            if _unfinished(got):
                return _todo()
            want = _rep_theory(d, p)
            if abs(float(got) - want) > 1e-9 + 1e-6 * want:
                return _fail(f"d = {d}、p = {p}：得到 {float(got):.6g}，应为 {want:.6g}。"
                             "提示：超过一半的比特翻转时，多数表决才会出错")
    _ok("理论公式正确")


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
