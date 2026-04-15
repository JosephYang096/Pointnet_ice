"""
一个极简 CLIP 教学脚本（仅依赖 numpy）。
目标：用最小代码理解 CLIP 的核心意义——把图像和文本映射到同一语义空间，并通过对比学习对齐。
"""

import numpy as np


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def tokenize(sentence: str):
    return sentence.lower().replace(",", " ").replace(".", " ").split()


def build_bow(texts):
    vocab = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    bow = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in tokenize(text):
            bow[i, vocab[token]] += 1.0
    return bow, vocab


def retrieval_top1(logits: np.ndarray):
    img_to_text = np.argmax(logits, axis=1)
    text_to_img = np.argmax(logits, axis=0)
    n = logits.shape[0]
    i2t_acc = np.mean(img_to_text == np.arange(n))
    t2i_acc = np.mean(text_to_img == np.arange(n))
    return i2t_acc, t2i_acc, img_to_text, text_to_img


def main():
    np.random.seed(42)

    # 6 对“图像-文本”样本：图像先用手工特征模拟（不是像素），让核心逻辑更清晰
    # 维度含义：[animal, cold, hot, vehicle]
    image_features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # cat
            [1.0, 0.0, 0.0, 0.0],  # dog
            [0.0, 1.0, 0.0, 0.0],  # ice
            [0.0, 0.0, 1.0, 0.0],  # fire
            [0.0, 0.0, 0.0, 1.0],  # car
            [0.0, 0.0, 0.0, 1.0],  # bus
        ],
        dtype=np.float32,
    )

    texts = [
        "a small cat",
        "a loyal dog",
        "cold ice cube",
        "hot fire flame",
        "a fast car",
        "a big bus",
    ]

    text_bow, vocab = build_bow(texts)
    n_samples = len(texts)

    # 两个编码器（线性层）：图像->共享空间，文本->共享空间
    embed_dim = 8
    w_img = 0.1 * np.random.randn(image_features.shape[1], embed_dim).astype(np.float32)
    w_txt = 0.1 * np.random.randn(text_bow.shape[1], embed_dim).astype(np.float32)

    lr = 0.2
    tau = 0.1
    epochs = 400

    # 训练前检索效果
    img_emb = image_features @ w_img
    txt_emb = text_bow @ w_txt
    logits = (img_emb @ txt_emb.T) / tau
    i2t_before, t2i_before, _, _ = retrieval_top1(logits)

    eye = np.eye(n_samples, dtype=np.float32)
    for epoch in range(epochs):
        # 前向
        img_emb = image_features @ w_img
        txt_emb = text_bow @ w_txt
        logits = (img_emb @ txt_emb.T) / tau

        # CLIP 常用的双向对比损失（图->文 + 文->图）
        p_row = softmax(logits, axis=1)  # 每张图匹配哪个文本
        p_col = softmax(logits, axis=0)  # 每段文本匹配哪个图

        loss_i2t = -np.mean(np.log(p_row[np.arange(n_samples), np.arange(n_samples)] + 1e-12))
        loss_t2i = -np.mean(np.log(p_col[np.arange(n_samples), np.arange(n_samples)] + 1e-12))
        loss = 0.5 * (loss_i2t + loss_t2i)

        # 反向（线性映射下的简化梯度）
        d_logits = 0.5 * ((p_row - eye) / n_samples + (p_col - eye) / n_samples)
        d_img_emb = (d_logits @ txt_emb) / tau
        d_txt_emb = (d_logits.T @ img_emb) / tau

        d_w_img = image_features.T @ d_img_emb
        d_w_txt = text_bow.T @ d_txt_emb

        w_img -= lr * d_w_img
        w_txt -= lr * d_w_txt

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1:3d} | loss = {loss:.4f}")

    # 训练后检索效果
    img_emb = image_features @ w_img
    txt_emb = text_bow @ w_txt
    logits = (img_emb @ txt_emb.T) / tau
    i2t_after, t2i_after, i2t_idx, t2i_idx = retrieval_top1(logits)

    print("\n=== 结果对比 ===")
    print(f"训练前 图->文 Top1 准确率: {i2t_before:.2f}")
    print(f"训练前 文->图 Top1 准确率: {t2i_before:.2f}")
    print(f"训练后 图->文 Top1 准确率: {i2t_after:.2f}")
    print(f"训练后 文->图 Top1 准确率: {t2i_after:.2f}")

    print("\n=== 图像检索文本示例 ===")
    for i in range(n_samples):
        print(f"图像{i} -> 文本{i2t_idx[i]} | GT={i} | text='{texts[i2t_idx[i]]}'")

    print("\n=== 你需要抓住的 CLIP 意义 ===")
    print("1) 不再只学“图像分类头”，而是学“图像-文本对齐”的通用语义空间。")
    print("2) 一旦对齐好，文本本身就能当分类器（零样本分类的基础）。")
    print("3) 同一个空间支持跨模态检索：图找文、文找图。")
    print("4) 这让模型从“固定标签任务”走向“自然语言驱动任务”。")
    print(f"\n词表大小（用于文本编码）: {len(vocab)}")


if __name__ == "__main__":
    main()
