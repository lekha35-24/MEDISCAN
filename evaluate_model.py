import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from counterfeit_model import load_model, predict_counterfeit

model = load_model()
data_dir = "dataset/test"
classes = ["Fake", "Real"]

y_true, y_pred, y_scores = [], [], []

for label_name in classes:
    folder = os.path.join(data_dir, label_name)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        true_label = 0 if label_name == "Fake" else 1
        pred_label_name, conf = predict_counterfeit(model, img)
        pred_label = 0 if pred_label_name == "Fake" else 1

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(conf if pred_label == 1 else 1 - conf)  # score = P(Real)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print("Accuracy:", round(accuracy * 100, 2), "%")

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print("AUC Score:", round(roc_auc, 3))

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()