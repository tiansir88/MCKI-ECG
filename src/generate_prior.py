import numpy as np
import os


def generate_jaccard_prior(y_train_mh_path: str):
    if not os.path.exists(y_train_mh_path):
        raise FileNotFoundError(f"找不到标签文件: {y_train_mh_path}")

    print(f"正在加载标签数据: {y_train_mh_path} ...")
    y = np.load(y_train_mh_path).astype(np.float32)

    # 1. 计算共现矩阵 (交集): co_occur[i, j] = |L_i ∩ L_j|
    # 这代表同时拥有标签 i 和 标签 j 的样本数量
    co_occur = np.dot(y.T, y)

    # 2. 计算每个类别的总样本数: nums[i] = |L_i|
    nums = np.sum(y, axis=0)

    # 3. 计算并集矩阵: |L_i ∪ L_j| = |L_i| + |L_j| - |L_i ∩ L_j|
    # 利用 numpy 的广播机制，快速生成所有类别对的并集数量
    union = nums[:, np.newaxis] + nums[np.newaxis, :] - co_occur

    # 4. 计算 Jaccard 相似度: 交集 / 并集
    # 加上 1e-8 是为了防止分母为 0 导致数值溢出
    similarity = co_occur / (union + 1e-8)

    # 保证对角线（类别与其自身的相似度）严格为 1.0
    np.fill_diagonal(similarity, 1.0)

    # 5. 格式化打印结果，方便你直接复制到模型代码中
    print("\n" + "=" * 60)
    print("✅ 这是基于【Jaccard 相似度 (交并比)】的相关性矩阵:")
    print("💡 这是一个严格对称的无向图矩阵 (S[i,j] == S[j,i])")
    print("=" * 60 + "\n")

    print("DEFAULT_S_HIERARCHY = [")
    for row in similarity:
        # 将每一行的数值格式化为保留 4 位小数
        formatted_row = ", ".join([f"{val:.4f}" for val in row])
        print(f"    [{formatted_row}],")
    print("]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # ⚠️ 请确保这里的路径与你服务器上真实的标签文件路径一致
    LABEL_PATH = "/root/MCKI_Project/data/processed_v3/y_train_mh.npy"

    generate_jaccard_prior(LABEL_PATH)