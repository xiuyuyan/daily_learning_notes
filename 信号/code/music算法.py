import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------- 信号生成函数 ---------------------------
def gen_signal(fre, t_0, theta, speed, numbers, space):
    """
    生成单个阵元在特定时间点的接收信号（复数形式）

    参数：
        fre (float): 信号频率（Hz）
        t_0 (float): 当前时间点（秒）
        theta (float): 信号入射角度（弧度，范围[0, π]，0为阵元法线方向，π为反法线方向）
        speed (float): 信号传播速度（如声速340m/s）
        numbers (int): 阵元数量
        space (float): 相邻阵元间距（米）

    返回：
        np.ndarray: 单个阵元在时间t_0的接收信号（复数数组，长度为numbers）
    """
    res = []
    for i in range(numbers):
        # 计算当前阵元（第i个，i从0开始）的接收信号相位
        # 相位由两部分组成：时间相关相位 + 空间相关相位
        time_phase = 2j * np.pi * fre * t_0  # 时间相位：随时间t_0线性变化（j为虚数单位）
        # 空间相位：由阵元位置引起的波程差导致
        # 波程差 = i * space * cos(theta)（theta为入射角，cos(theta)是信号在阵元排列方向的投影）
        # 相位差 = 2π * 波程差 / 波长，波长 = speed / fre → 相位差 = 2π * fre * 波程差 / speed
        space_phase = -2j * np.pi * fre * i * space * np.cos(theta) / speed  # 负号表示波从远场传来
        total_phase = time_phase + space_phase
        res.append(np.exp(total_phase))  # 复数信号形式：exp(j*相位)
    return np.array(res)  # 转换为numpy数组


# --------------------------- 方向矢量生成函数 ---------------------------
def steer_vector(fre, theta, speed, numbers, space):
    """
    生成特定入射角度的方向矢量（Steering Vector）
    方向矢量表示理想情况下（无噪声），信号在各阵元的相对相位

    参数：
        fre (float): 信号频率（Hz）
        theta (float): 信号入射角度（弧度，范围[0, π]，0为阵元法线方向，π为反法线方向）
        speed (float): 信号传播速度（m/s）
        numbers (int): 阵元数量
        space (float): 相邻阵元间距（米）

    返回：
        np.ndarray: 方向矢量（列向量，形状为(numbers, 1)）
    """
    alphas = []
    for i in range(numbers):
        # 参考阵元（i=0）的相位设为0，其他阵元的相位由空间位置决定
        # 与gen_signal中的空间相位计算一致（省略时间相位，因为方向矢量与时间无关）
        phase = -2j * np.pi * fre * i * space * np.cos(theta) / speed
        alphas.append(np.exp(phase))
    return np.array(alphas).reshape(-1, 1)  # 转换为列向量（形状：[numbers, 1]）


# --------------------------- MUSIC算法核心函数 ---------------------------
def cal_music(fre, speed, numbers, space, signals, method='noise'):
    """
    MUSIC算法实现：通过噪声子空间与方向矢量的正交性估计信号角度

    参数：
        fre (float): 信号频率（Hz）
        speed (float): 信号传播速度（m/s）
        numbers (int): 阵元数量（M）
        space (float): 相邻阵元间距（米）
        signals (np.ndarray): 接收信号矩阵（形状：[N, M]，N为快拍数，M为阵元数）
        method (str): 计算功率谱的方法，'signal'（信号子空间投影）或'noise'（噪声子空间正交性）

    返回：
        thetas (np.ndarray): 扫描的角度范围（度数，形状：[180]）
        P (np.ndarray): 各角度对应的功率谱值（形状：[180]）
    """
    N = signals.shape[0]  # 快拍数（时间点数，即接收信号的时间维度长度）
    M = signals.shape[1]  # 阵元数（接收信号的阵元维度长度）

    # --------------------------- 步骤1：计算协方差矩阵 ---------------------------
    # 协方差矩阵R_x反映阵元间信号的相关性，形状为[M, M]
    # 公式：R_x = (1/N) * signalsᴴ * signals（ᴴ表示共轭转置）
    # 作用：通过平均快拍数据降低噪声影响，捕捉信号的统计特性
    R_x = np.matmul(np.conjugate(signals.T), signals) / N  # 等价于 (signals.conj().T @ signals) / N

    # --------------------------- 步骤2：特征分解 ---------------------------
    # 对协方差矩阵进行特征分解，得到特征值lamda和特征向量u
    # 特征向量按特征值降序排列（大特征值对应信号子空间，小特征值对应噪声子空间）
    lamda, u = np.linalg.eig(R_x)  # lamda是特征值数组，u是特征向量矩阵（每列是一个特征向量）
    idx = np.argsort(lamda)[::-1]  # 对特征值索引降序排序（从大到小）
    u = u[:, idx]  # 特征向量矩阵按特征值降序重新排列（每列对应排序后的特征值）

    # --------------------------- 步骤3：分离信号子空间与噪声子空间 ---------------------------
    # 信号子空间：由最大特征值对应的特征向量张成（通常取前K个，K为信号源数量）
    # 这里假设只有1个信号源（实际可根据特征值差异判断信号源数量）
    u_s = u[:, 0].reshape(-1, 1)  # 取最大特征值对应的特征向量（列向量，形状：[M, 1]）
    # 噪声子空间：由剩余特征向量张成（排除最大特征值对应的特征向量）
    u_n = u[:, 1:]  # 形状：[M, M-1]（M为阵元数，M-1为噪声子空间维度）

    # --------------------------- 步骤4：角度扫描与功率谱计算 ---------------------------
    # 扫描角度范围：0°到180°（转换为弧度）
    thetas = np.linspace(0, np.pi, 180)  # 180个均匀分布的角度点（0到π弧度）
    P = []  # 存储各角度的功率谱值

    for _theta in thetas:
        # 生成当前角度_theta的方向矢量（列向量，形状：[M, 1]）
        _alphas = steer_vector(fre, _theta, speed, numbers, space).reshape(-1, 1)

        if method == 'signal':
            # ---------------------- 方法1：信号子空间投影法 ----------------------
            # 投影矩阵：将向量投影到噪声子空间（I - P_s，P_s是信号子空间投影矩阵）
            # P_s = u_s @ u_sᴴ（信号子空间投影矩阵），因此投影矩阵为 I - P_s
            projection = np.eye(M) - u_s @ np.conjugate(u_s.T)  # 形状：[M, M]
            # 功率谱公式：1 / (aᴴ * (I - P_s) * a)，a是方向矢量
            # 分子是方向矢量在噪声子空间的正交投影的模长平方的倒数
            numerator = np.matmul(np.conjugate(_alphas).T, np.matmul(projection, _alphas))  # aᴴ*(I-P_s)*a
            P_x = 1 / numerator  # 避免除零错误（实际中方向矢量不在噪声子空间时不为零）

        elif method == 'noise':
            # ---------------------- 方法2：噪声子空间正交性法 ----------------------
            # 功率谱公式：1 / (aᴴ * u_n * u_nᴴ * a)
            # 原理：方向矢量a在信号子空间时，与噪声子空间正交（u_nᴴ * a = 0），此时分母趋近于0，功率谱极大
            # 分步计算：aᴴ * u_n → [1, M-1]；u_nᴴ * a → [M-1, 1]；最终乘积为标量
            numerator = np.matmul(np.conjugate(_alphas).T, u_n)  # aᴴ * u_n → [1, M-1]
            numerator = np.matmul(numerator, np.conjugate(u_n.T))  # (aᴴ * u_n) * u_nᴴ → [1, M]
            numerator = np.matmul(numerator, _alphas)  # (aᴴ * u_n * u_nᴴ) * a → 标量
            P_x = 1 / numerator  # 分母为0时功率谱无穷大（实际中取倒数）

        else:
            print(f'未知方法: {method}')
            break

        P.append(P_x)  # 存储当前角度的功率谱值

    # 转换为一维数组，并将角度从弧度转换为度数
    P = np.array(P).flatten()  # 展平为一维数组（形状：[180]）
    return thetas / np.pi * 180, P  # 角度转换为度数（范围：0°到180°）


# --------------------------- 主程序：参数初始化与数据生成 ---------------------------
if __name__ == "__main__":
    # --------------------------- 系统参数设置 ---------------------------
    fs = 20000  # 采样率（Hz），决定时间分辨率（时间间隔dt=1/fs=0.00005秒）
    fre = 200 # 信号频率（Hz）
    # 时间轴：0到0.01秒（共10毫秒），包含200个时间点（快拍数N=200）
    t = np.arange(0, 0.01, 1 / fs)  # 形状：[200]
    theta1 = np.pi / 3  # 信号1入射角度（60°，转换为弧度）
    theta2 = 2 * np.pi / 3  # 信号2入射角度（120°，转换为弧度）
    # theta3 = 3 * np.pi / 6  # 信号3入射角度（90°，转换为弧度）
    speed = 340  # 信号传播速度（声速，m/s）
    numbers = 32  # 阵元数量（M=32）
    space = 1  # 相邻阵元间距（米），需满足space ≤ λ/2（λ=speed/fre=1.7米，半波长0.85米）

    # --------------------------- 生成模拟快拍数据 ---------------------------
    # 接收信号矩阵：形状为[N, M]（N=200时间点，M=32阵元）
    signals = []
    for t_0 in t:  # 遍历每个时间点生成快拍数据
        # 生成信号1在第t_0时刻的32个阵元接收信号
        signal1 = gen_signal(fre, t_0, theta1, speed, numbers, space)
        # 生成信号2在第t_0时刻的32个阵元接收信号
        signal2 = gen_signal(fre, t_0, theta2, speed, numbers, space)
        # signal3 = gen_signal(fre, t_0, theta3, speed, numbers, space)

        # combined_signal = signal1 + signal2 + signal3
        combined_signal = signal1 + signal2

        signals.append(combined_signal)  # 添加到快拍列表
    # 转换为numpy数组（形状：[200, 32]）
    signals = np.array(signals)

    # --------------------------- 运行MUSIC算法 ---------------------------
    # thetas_deg, power_spectrum = cal_music(fre, speed, numbers, space, signals, method='noise')
    thetas_deg, power_spectrum = cal_music(fre, speed, numbers, space, signals, method='signal')

    # --------------------------- 绘制结果 ---------------------------
    plt.figure(figsize=(10, 4))  # 创建画布（宽10英寸，高4英寸）
    # 绘制功率谱（转换为dB刻度，避免对数运算错误）
    plt.plot(thetas_deg, 10 * np.log10(np.abs(power_spectrum) + 1e-10), linewidth=1.5)
    plt.xlim(0, 180)  # 横轴范围：0°到180°
    plt.xlabel('角度 (度)')  # 横轴标签（中文显示）
    plt.ylabel('功率 (dB)')  # 纵轴标签（中文显示）
    plt.title('MUSIC算法测向结果（0-180°）')  # 图标题（中文显示）
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
    plt.tight_layout()  # 自动调整子图布局
    plt.show()  # 显示图形