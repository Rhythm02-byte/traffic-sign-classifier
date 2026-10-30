import torch

from tsc.evaluate import NUM_CLASSES, confusion_matrix, per_class_accuracy


def test_per_class_accuracy_counts_each_class():
    y_true = torch.arange(NUM_CLASSES).repeat(2)  # every class appears twice
    y_pred = y_true.clone()
    y_pred[0] = 1  # one image of class 0 is predicted as class 1

    rows = per_class_accuracy(y_true, y_pred)

    assert rows[0] == (0, 2, 1, 0.5)
    assert rows[1] == (1, 2, 2, 1.0)
    assert len(rows) == NUM_CLASSES


def test_confusion_matrix_counts_true_and_predicted_pairs():
    y_true = torch.tensor([0, 0, 1])
    y_pred = torch.tensor([0, 1, 1])

    cm = confusion_matrix(y_true, y_pred)

    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert cm[0, 0] == 1  # true 0, predicted 0
    assert cm[0, 1] == 1  # true 0, predicted 1
    assert cm[1, 1] == 1  # true 1, predicted 1
    assert cm.sum() == 3
