import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# ==========================================
# 全局配置 (Global Configuration)
# ==========================================
@dataclass
class Config:
    # 数据文件路径（默认与脚本同目录）
    TRAIN_DATA_PATH: str = "KDDTrain+.txt"
    TEST_DATA_PATH: str = "KDDTest+.txt"

    # RandomForest 参数
    RF_N_ESTIMATORS: int = 100
    RF_RANDOM_STATE: int = 42
    RF_N_JOBS: int = -1

    # 默认阈值（用于将概率转为 0/1；默认 0.5）
    THRESHOLD: float = 0.5

    # 额外：在约束 FPR 下选阈值（IDS 更常用）
    FPR_CONSTRAINT: float = 0.05  # 5%

    # 输出目录与文件名
    OUTPUT_DIR: str = "outputs"
    CONFUSION_MATRIX_FILE: str = "confusion_matrix.png"
    ROC_CURVE_FILE: str = "roc_curve.png"
    PR_CURVE_FILE: str = "pr_curve.png"
    THRESHOLD_SWEEP_FILE: str = "threshold_sweep.png"
    FEATURE_IMPORTANCE_FILE: str = "feature_importance_top15.png"
    METRICS_REPORT_FILE: str = "metrics_report.txt"
    FPR_SELECTION_REPORT_FILE: str = "threshold_by_fpr.txt"

    # NSL-KDD 数据集 Schema
    COLUMNS = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "class",
        "difficulty",
    ]


def ensure_output_dir() -> str:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    return os.path.abspath(Config.OUTPUT_DIR)


# ==========================================
# 数据处理 (ETL / Preprocessing)
# ==========================================
def data_ingestion_and_preprocessing(train_path: str, test_path: str):
    """
    负责数据的抽取、转换与加载 (ETL)。
    主要步骤:
    1. 加载原始数据
    2. 标签二值化处理 (Normal vs Attack)
    3. 特征空间对齐与编码 (Feature Space Alignment & Encoding)
    """
    print("[System] Initializing data ingestion pipeline...")
    print(f"  > Training source: {train_path}")
    print(f"  > Testing source : {test_path}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Critical Error: Data files not found. Put KDDTrain+.txt and KDDTest+.txt next to a.py.")

    # 1) Data Loading
    train_df = pd.read_csv(train_path, names=Config.COLUMNS)
    test_df = pd.read_csv(test_path, names=Config.COLUMNS)

    # 2) Label Binarization
    print("[System] Performing target label binarization (Normal vs Attack)...")
    train_df["class"] = train_df["class"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)
    test_df["class"] = test_df["class"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)

    # 标记来源，方便合并后再拆分
    train_df["dataset_type"] = "train"
    test_df["dataset_type"] = "test"

    full_dataset = pd.concat([train_df, test_df], axis=0)

    # 移除与模型无关的元数据列
    if "difficulty" in full_dataset.columns:
        full_dataset.drop(["difficulty"], axis=1, inplace=True)

    # 3) Consistent Encoding for categorical features
    print("[System] Encoding categorical features (consistent mapping)...")
    le = LabelEncoder()
    for col in full_dataset.columns:
        if full_dataset[col].dtype == "object" and col != "dataset_type":
            full_dataset[col] = le.fit_transform(full_dataset[col].astype(str))

    # Restore split
    train_final = full_dataset[full_dataset["dataset_type"] == "train"].drop("dataset_type", axis=1)
    test_final = full_dataset[full_dataset["dataset_type"] == "test"].drop("dataset_type", axis=1)

    X_train = train_final.drop("class", axis=1)
    y_train = train_final["class"].astype(int)
    X_test = test_final.drop("class", axis=1)
    y_test = test_final["class"].astype(int)

    print("[System] Preprocessing completed.")
    print(f"  > Train Set Shape: {X_train.shape}")
    print(f"  > Test Set Shape : {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ==========================================
# 模型训练 (Model Training)
# ==========================================
def train_classifier(X_train, y_train) -> RandomForestClassifier:
    print("\n[System] Starting model training phase...")
    print(f"  > Algorithm: Random Forest (n_estimators={Config.RF_N_ESTIMATORS})")

    model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        random_state=Config.RF_RANDOM_STATE,
        n_jobs=Config.RF_N_JOBS,
    )

    start_ts = time.time()
    model.fit(X_train, y_train)
    end_ts = time.time()

    print(f"[System] Model successfully trained in {end_ts - start_ts:.4f} seconds.")
    return model


# ==========================================
# 指标计算 (Metrics)
# ==========================================
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + tnr) / 2.0
    mcc = matthews_corrcoef(y_true, y_pred)

    out = {
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Specificity(TNR)": float(tnr),
        "Balanced_Accuracy": float(bal_acc),
        "MCC": float(mcc),
    }

    if y_score is not None:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["ROC_AUC"] = float("nan")
        try:
            out["Avg_Precision(AP)"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["Avg_Precision(AP)"] = float("nan")

    return out


def print_and_save_report(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    sample_count: int,
    infer_seconds: float,
    throughput: float,
    avg_latency_ms: float,
    threshold: float,
):
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"{'PERFORMANCE METRICS REPORT':^60}")
    report_lines.append("=" * 60)
    report_lines.append(f"{'Metric':<30} | {'Value':>25}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Total Samples Processed':<30} | {sample_count:>25,}")
    report_lines.append(f"{'Inference Threshold':<30} | {threshold:>25.2f}")
    report_lines.append(f"{'Model Accuracy':<30} | {metrics['Accuracy']*100:>24.2f}%")
    report_lines.append(f"{'Precision':<30} | {metrics['Precision']*100:>24.2f}%")
    report_lines.append(f"{'Recall':<30} | {metrics['Recall']*100:>24.2f}%")
    report_lines.append(f"{'F1-score':<30} | {metrics['F1']*100:>24.2f}%")
    report_lines.append(f"{'False Positive Rate (FPR)':<30} | {metrics['FPR']*100:>24.2f}%")
    report_lines.append(f"{'False Negative Rate (FNR)':<30} | {metrics['FNR']*100:>24.2f}%")
    report_lines.append(f"{'Specificity (TNR)':<30} | {metrics['Specificity(TNR)']*100:>24.2f}%")
    report_lines.append(f"{'Balanced Accuracy':<30} | {metrics['Balanced_Accuracy']*100:>24.2f}%")
    report_lines.append(f"{'MCC':<30} | {metrics['MCC']:>25.3f}")
    if "ROC_AUC" in metrics:
        report_lines.append(f"{'ROC-AUC':<30} | {metrics['ROC_AUC']:>25.4f}")
    if "Avg_Precision(AP)" in metrics:
        report_lines.append(f"{'PR-AUC (AP)':<30} | {metrics['Avg_Precision(AP)']:>25.4f}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Inference Time (total)':<30} | {infer_seconds:>25.4f} s")
    report_lines.append(f"{'Throughput (TPS)':<30} | {throughput:>25,.0f} req/s")
    report_lines.append(f"{'Avg Latency per Request':<30} | {avg_latency_ms:>25.4f} ms")
    report_lines.append("=" * 60)

    text = "\n".join(report_lines)
    print(text)

    # classification_report（文本）
    cr = classification_report(y_true, y_pred, target_names=["Normal", "Attack"], digits=4)
    print("\n[System] Classification Report:\n" + cr)

    # 保存到文件
    report_path = os.path.join(output_dir, Config.METRICS_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text + "\n\n")
        f.write("[Classification Report]\n")
        f.write(cr + "\n")
    print(f"[System] Metrics report saved to: {report_path}")


# ==========================================
# 可视化 (Plots)
# ==========================================
def plot_confusion_matrix(y_true, y_pred, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.title("Confusion Matrix: Traffic Classification Performance")
    plt.xlabel("Predicted Traffic Type")
    plt.ylabel("Actual Traffic Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_score, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true, y_score, output_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_sweep(y_true, y_score, output_path: str) -> Tuple[float, float]:
    """
    扫描阈值，画 Precision / Recall / F1 / FPR 随阈值变化曲线。
    返回：最佳阈值（按 F1 最大）与对应 F1
    """
    thresholds = np.linspace(0.0, 1.0, 201)
    precisions = []
    recalls = []
    f1s = []
    fprs = []

    for thr in thresholds:
        y_pred_thr = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr).ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        fprs.append(fpr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    fprs = np.array(fprs)

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    plt.figure(figsize=(11, 7))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, fprs, label="FPR")
    plt.axvline(best_thr, linestyle="--", label=f"Best F1 thr={best_thr:.2f}")
    plt.title("Threshold Sweep (based on Attack Probability)")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return best_thr, best_f1


def plot_feature_importance(model: RandomForestClassifier, feature_names, output_path: str, top_k: int = 15):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    idx = np.argsort(importances)[::-1][:top_k]
    top_features = np.array(feature_names)[idx]
    top_importances = importances[idx]

    plt.figure(figsize=(12, 7))
    plt.bar(range(len(top_features)), top_importances)
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha="right")
    plt.title(f"Top-{top_k} Feature Importances (Random Forest)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================
# 阈值策略：F1 最优 vs 误报率约束 (IDS 常用)
# ==========================================
def select_threshold_by_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.05):
    """
    在 FPR <= max_fpr 的约束下，选择 Recall 最大的阈值（更符合 IDS：先控误报，再尽量提高检出率）
    返回：best_thr, best_metrics(dict)
    """
    thresholds = np.linspace(0.0, 1.0, 2001)  # 更细
    best_thr = 0.5
    best_rec = -1.0
    best_metrics = None

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if fpr <= max_fpr and rec > best_rec:
            best_rec = rec
            best_thr = float(thr)
            best_metrics = {
                "threshold": best_thr,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            }

    return best_thr, best_metrics


def save_fpr_selection_report(output_dir: str, max_fpr: float, best_metrics: Dict):
    path = os.path.join(output_dir, Config.FPR_SELECTION_REPORT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[Threshold Selection Under FPR Constraint]\n")
        f.write(f"Constraint: FPR <= {max_fpr:.4f}\n\n")
        if best_metrics is None:
            f.write("No threshold satisfies the constraint.\n")
        else:
            f.write(f"Best Threshold: {best_metrics['threshold']:.4f}\n")
            f.write(f"Precision    : {best_metrics['precision']:.4f}\n")
            f.write(f"Recall       : {best_metrics['recall']:.4f}\n")
            f.write(f"F1           : {best_metrics['f1']:.4f}\n")
            f.write(f"FPR          : {best_metrics['fpr']:.4f}\n")
            f.write(f"Confusion    : TN={best_metrics['tn']} FP={best_metrics['fp']} "
                    f"FN={best_metrics['fn']} TP={best_metrics['tp']}\n")
    print(f"[System] FPR-threshold selection report saved to: {path}")


# ==========================================
# 评估主流程 (Evaluation)
# ==========================================
def evaluate_model_performance(model, X_test, y_test, threshold: float, output_dir: str):
    print("\n[System] Initiating inference and evaluation...")

    sample_count = len(X_test)

    # 1) 预测标签（用于吞吐/延迟统计：基于 predict 的推理）
    start_ts = time.time()
    y_pred_default = model.predict(X_test)
    end_ts = time.time()

    infer_seconds = end_ts - start_ts
    throughput = sample_count / infer_seconds if infer_seconds > 0 else 0.0
    avg_latency_ms = (infer_seconds / sample_count) * 1000.0

    # 2) 预测概率（用于 ROC/PR/阈值扫描）
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred_thr = (y_score >= threshold).astype(int)
    else:
        y_pred_thr = y_pred_default

    # 指标（用“阈值化后的预测”更贴近 IDS）
    metrics = compute_metrics(y_test.values, y_pred_thr, y_score)

    # 输出 & 保存报告
    print_and_save_report(
        metrics=metrics,
        y_true=y_test.values,
        y_pred=y_pred_thr,
        output_dir=output_dir,
        sample_count=sample_count,
        infer_seconds=infer_seconds,
        throughput=throughput,
        avg_latency_ms=avg_latency_ms,
        threshold=threshold,
    )

    return y_pred_thr, y_score, metrics, throughput, avg_latency_ms


# ==========================================
# 程序入口 (Main)
# ==========================================
if __name__ == "__main__":
    try:
        out_dir = ensure_output_dir()

        # Step 1: ETL
        X_train, y_train, X_test, y_test = data_ingestion_and_preprocessing(
            Config.TRAIN_DATA_PATH, Config.TEST_DATA_PATH
        )

        # Step 2: Train
        trained_model = train_classifier(X_train, y_train)

        # Step 3: Evaluate (default threshold)
        y_pred, y_score, metrics, tps, lat_ms = evaluate_model_performance(
            trained_model, X_test, y_test, threshold=Config.THRESHOLD, output_dir=out_dir
        )

        # Step 4: Confusion Matrix
        cm_path = os.path.join(out_dir, Config.CONFUSION_MATRIX_FILE)
        plot_confusion_matrix(y_test.values, y_pred, cm_path)
        print(f"\n[System] Confusion matrix saved to: {cm_path}")

        # Step 5: ROC / PR / Threshold sweep（需要概率）
        if y_score is not None:
            roc_path = os.path.join(out_dir, Config.ROC_CURVE_FILE)
            pr_path = os.path.join(out_dir, Config.PR_CURVE_FILE)
            thr_path = os.path.join(out_dir, Config.THRESHOLD_SWEEP_FILE)

            plot_roc_curve(y_test.values, y_score, roc_path)
            plot_pr_curve(y_test.values, y_score, pr_path)
            best_thr, best_f1 = plot_threshold_sweep(y_test.values, y_score, thr_path)

            print(f"[System] ROC curve saved to: {roc_path}")
            print(f"[System] PR curve saved to : {pr_path}")
            print(f"[System] Threshold sweep saved to: {thr_path}")
            print(f"[System] Best threshold by F1: {best_thr:.2f} (F1={best_f1:.4f})")

            # Step 6: IDS 更常用的阈值策略：在 FPR 约束下最大化 Recall
            fpr_thr, fpr_metrics = select_threshold_by_fpr(
                y_test.values, y_score, max_fpr=Config.FPR_CONSTRAINT
            )
            if fpr_metrics is None:
                print(f"[System] No threshold satisfies FPR <= {Config.FPR_CONSTRAINT:.2%}")
            else:
                print(f"[System] Best threshold under FPR<={Config.FPR_CONSTRAINT:.0%}: {fpr_thr:.3f}")
                print(f"[System]  -> Precision={fpr_metrics['precision']:.4f}, "
                      f"Recall={fpr_metrics['recall']:.4f}, F1={fpr_metrics['f1']:.4f}, "
                      f"FPR={fpr_metrics['fpr']:.4f}")
                print(f"[System]  -> Confusion: TN={fpr_metrics['tn']} FP={fpr_metrics['fp']} "
                      f"FN={fpr_metrics['fn']} TP={fpr_metrics['tp']}")
            save_fpr_selection_report(out_dir, Config.FPR_CONSTRAINT, fpr_metrics)

        # Step 7: Feature importance
        fi_path = os.path.join(out_dir, Config.FEATURE_IMPORTANCE_FILE)
        plot_feature_importance(trained_model, X_train.columns.tolist(), fi_path, top_k=15)
        print(f"[System] Feature importance saved to: {fi_path}")

        print("\n[System] Process finished successfully.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Execution failed: {str(e)}")
