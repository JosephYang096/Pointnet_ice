"""
一个极简 CLIP 教学脚本（纯 Python，无第三方依赖）。
目标：理解 CLIP 的核心——把图像和文本映射到同一语义空间，并通过对比学习对齐。
"""

import math
import random

EPSILON = 1e-12
WEIGHT_INIT_SCALE = 0.1
PUNCTUATION_TO_SPACE_TRANSLATOR = str.maketrans(
    {ch: " " for ch in [",", ".", "?", "!", ";", ":", "(", ")", "\"", "'"]}
)


def tokenize(sentence):
    cleaned = sentence.lower().translate(PUNCTUATION_TO_SPACE_TRANSLATOR)
    return cleaned.split()


def build_bow(texts):
    vocab = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    bow = []
    for text in texts:
        vec = [0.0] * len(vocab)
        for token in tokenize(text):
            vec[vocab[token]] += 1.0
        bow.append(vec)
    return bow, vocab


def matmul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out


def transpose(m):
    return [list(col) for col in zip(*m)]


def softmax_row(row):
    m = max(row)
    ex = [math.exp(x - m) for x in row]
    s = sum(ex) + EPSILON
    return [v / s for v in ex]


def softmax_axis1(logits):
    return [softmax_row(row) for row in logits]


def softmax_axis0(logits):
    t = transpose(logits)
    t_soft = [softmax_row(col) for col in t]
    return transpose(t_soft)


def retrieval_top1(logits):
    n = len(logits)
    img_to_text = [max(range(n), key=lambda j: logits[i][j]) for i in range(n)]
    text_to_img = [max(range(n), key=lambda i: logits[i][j]) for j in range(n)]
    i2t_acc = sum(1 for i, p in enumerate(img_to_text) if i == p) / n
    t2i_acc = sum(1 for i, p in enumerate(text_to_img) if i == p) / n
    return i2t_acc, t2i_acc, img_to_text


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def scale(m, v):
    return [[x * v for x in row] for row in m]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sub(a, b):
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def main():
    random.seed(42)

    # 图像手工特征维度: [animal, cold, hot, vehicle]
    image_features = [
        [1.0, 0.0, 0.0, 0.0],  # cat
        [1.0, 0.0, 0.0, 0.0],  # dog
        [0.0, 1.0, 0.0, 0.0],  # ice
        [0.0, 0.0, 1.0, 0.0],  # fire
        [0.0, 0.0, 0.0, 1.0],  # car
        [0.0, 0.0, 0.0, 1.0],  # bus
    ]

    texts = [
        "a small cat",
        "a loyal dog",
        "cold ice cube",
        "hot fire flame",
        "a fast car",
        "a big bus",
    ]

    text_bow, vocab = build_bow(texts)
    n = len(texts)

    embed_dim = 8
    image_projection_weights = [
        [WEIGHT_INIT_SCALE * (random.random() * 2 - 1) for _ in range(embed_dim)]
        for _ in range(len(image_features[0]))
    ]
    text_projection_weights = [
        [WEIGHT_INIT_SCALE * (random.random() * 2 - 1) for _ in range(embed_dim)]
        for _ in range(len(text_bow[0]))
    ]

    lr = 0.2
    tau = 0.1
    epochs = 400

    # 训练前
    img_emb = matmul(image_features, image_projection_weights)
    txt_emb = matmul(text_bow, text_projection_weights)
    logits = scale(matmul(img_emb, transpose(txt_emb)), 1.0 / tau)
    i2t_before, t2i_before, _ = retrieval_top1(logits)

    eye = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for epoch in range(epochs):
        img_emb = matmul(image_features, image_projection_weights)
        txt_emb = matmul(text_bow, text_projection_weights)
        logits = scale(matmul(img_emb, transpose(txt_emb)), 1.0 / tau)

        p_row = softmax_axis1(logits)
        p_col = softmax_axis0(logits)

        loss_i2t = -sum(math.log(p_row[i][i] + EPSILON) for i in range(n)) / n
        loss_t2i = -sum(math.log(p_col[i][i] + EPSILON) for i in range(n)) / n
        loss = 0.5 * (loss_i2t + loss_t2i)

        d_logits = zeros(n, n)
        for i in range(n):
            for j in range(n):
                d_logits[i][j] = 0.5 * (((p_row[i][j] - eye[i][j]) / n) + ((p_col[i][j] - eye[i][j]) / n))

        d_img_emb = scale(matmul(d_logits, txt_emb), 1.0 / tau)
        d_txt_emb = scale(matmul(transpose(d_logits), img_emb), 1.0 / tau)
        d_w_img = matmul(transpose(image_features), d_img_emb)
        d_w_txt = matmul(transpose(text_bow), d_txt_emb)

        image_projection_weights = sub(image_projection_weights, scale(d_w_img, lr))
        text_projection_weights = sub(text_projection_weights, scale(d_w_txt, lr))

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1:3d} | loss = {loss:.4f}")

    # 训练后
    img_emb = matmul(image_features, image_projection_weights)
    txt_emb = matmul(text_bow, text_projection_weights)
    logits = scale(matmul(img_emb, transpose(txt_emb)), 1.0 / tau)
    i2t_after, t2i_after, i2t_idx = retrieval_top1(logits)

    print("\n=== 结果对比 ===")
    print(f"训练前 图->文 Top1 准确率: {i2t_before:.2f}")
    print(f"训练前 文->图 Top1 准确率: {t2i_before:.2f}")
    print(f"训练后 图->文 Top1 准确率: {i2t_after:.2f}")
    print(f"训练后 文->图 Top1 准确率: {t2i_after:.2f}")

    print("\n=== 图像检索文本示例 ===")
    for i in range(n):
        print(f"图像{i} -> 文本{i2t_idx[i]} | Ground Truth={i} | text='{texts[i2t_idx[i]]}'")

    print("\n=== 你需要抓住的 CLIP 意义 ===")
    print("1) 学的是图文对齐空间，而不是单一任务标签头。")
    print("2) 文本提示词可直接变成分类器（零样本能力基础）。")
    print("3) 支持跨模态检索：图找文、文找图。")
    print("4) 让模型从固定类目走向自然语言驱动。")
    print(f"5) 本示例词表大小: {len(vocab)}")


if __name__ == "__main__":
    main()
