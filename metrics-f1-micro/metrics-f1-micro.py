def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    n = len(y_pred)
    TP = 0
    for i in range(n):
        if y_pred[i] == y_true[i]:
            TP+=1

    return (TP/n)