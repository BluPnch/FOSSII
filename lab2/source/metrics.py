import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Матрица неточностей.

    matrix[i, j] - сколько объектов класса i
    сеть отнесла к классу j.

    Строки - истинный класс, столбцы - предсказанный.
    """

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    np.add.at(matrix, (y_true, y_pred), 1)

    return matrix


def safe_divide(numerator, denominator):
    """
    Деление, которое возвращает 0 там,
    где знаменатель равен 0.
    """

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )


def classification_metrics(matrix):
    """
    Все показатели считаются по матрице неточностей.

    Для класса k (один против остальных):

        TP = matrix[k, k]
        FP = сумма столбца k без диагонали
        FN = сумма строки k без диагонали

        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2 * precision * recall / (precision + recall)

    accuracy = сумма диагонали / сумма всей матрицы
    """

    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "accuracy": tp.sum() / matrix.sum(),

        "precision": precision,
        "recall": recall,
        "f1": f1,

        # Макро-усреднение: среднее по классам
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
    }


def roc_curve(is_positive, scores):
    """
    ROC-кривая для бинарной задачи.

    is_positive: True, если объект принадлежит классу
    scores:      вероятность класса по мнению сети

    Порог опускается от самой большой вероятности
    к самой маленькой, на каждом шаге считаем:

        TPR = TP / P
        FPR = FP / N
    """

    order = np.argsort(-scores, kind="stable")

    is_positive = np.asarray(is_positive, dtype=bool)[order]
    scores = scores[order]

    tp = np.cumsum(is_positive)
    fp = np.cumsum(~is_positive)

    # Одинаковые вероятности - это один порог,
    # поэтому оставляем только последнюю точку из группы
    last_in_group = np.append(np.diff(scores) != 0, True)

    tp = tp[last_in_group]
    fp = fp[last_in_group]

    tpr = safe_divide(tp, tp[-1])
    fpr = safe_divide(fp, fp[-1])

    # Кривая начинается в точке (0, 0)
    return (
        np.concatenate([[0.0], fpr]),
        np.concatenate([[0.0], tpr]),
    )


def auc(fpr, tpr):
    """
    Площадь под ROC-кривой методом трапеций.
    """

    return np.sum(
        np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2
    )


def roc_one_vs_rest(y_true, probabilities):
    """
    ROC-кривая и AUC для каждого класса
    по схеме "один против остальных".

    probabilities: (N, num_classes)

    Возвращает список (fpr, tpr, auc) по классам.
    """

    curves = []

    for k in range(probabilities.shape[1]):

        fpr, tpr = roc_curve(y_true == k, probabilities[:, k])

        curves.append((fpr, tpr, auc(fpr, tpr)))

    return curves
