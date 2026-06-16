# 基于可穿戴传感器的人体活动识别项目

这是一个适合上传 GitHub 的 MLT 课程项目整理版。项目基于背部和大腿传感器数据，完成人体活动识别任务。

## 项目内容

项目主要包括：

1. 数据读取与探索性分析
2. 删除 Cycling 相关标签
3. 合并 Stairs ascending 和 Stairs descending
4. 对异常的 S007 数据进行清洗
5. 生成 2 秒窗口、1 秒重叠的滑动窗口
6. 提取 ENMO 与统计特征
7. 对比 K-Means、Random Forest 和 1D CNN
8. 输出 Accuracy、Macro Recall、分类报告和混淆矩阵

## 数据集放置方式

数据集不要直接上传 GitHub。你需要在本地放成下面这种结构：

```text
MLT-CW-Dataset/
├── S001.csv
├── S002.csv
├── S007.csv
├── S007_cleaned.csv
└── test-set/
    ├── xxx.csv
    └── ...
```

每个 CSV 至少需要包含：

```text
timestamp, label, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z
```

## 推荐运行顺序

```bash
pip install -r requirements.txt
pip install -e .

python scripts/prepare_data.py --data-dir MLT-CW-Dataset --clean-s007
python scripts/run_random_forest.py --data-dir MLT-CW-Dataset --mode Fine-Tuned
python scripts/run_kmeans.py --data-dir MLT-CW-Dataset --mode Fine-Tuned
python scripts/run_cnn.py --data-dir MLT-CW-Dataset --mode Fine-Tuned
```

## 简历表达建议

可以写成：

> Built a human activity recognition pipeline using wearable sensor data, including data cleaning, sliding-window segmentation, ENMO/statistical feature extraction, and model comparison across K-Means, Random Forest, and 1D CNN. The fine-tuned Random Forest achieved 91.76% accuracy and 82.03% macro recall on the test set.

## 上传 GitHub 注意事项

- 不要上传原始数据集。
- 不要上传学校未授权材料。
- 可以上传代码、README、requirements、项目结构和 notebook。
- 简历里建议放 GitHub 仓库链接，而不是只放 GitHub 主页。
