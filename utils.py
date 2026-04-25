from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fnr = fn / (fn + tp)

    return acc, fnr