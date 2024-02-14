import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier

from config.config import RESULTS_DIR
from src.dataloader import dataloader


def main() -> None:
    test_stamp = None

    # gather all metrics across folds
    baccs = []
    aurocs = []

    # folds are going to be iterated from the last to earliest,
    # but this has no effect since model is retrained everytime
    # it's just for data convenience
    for i in range(5):
        print(f"Starting training fold {i}")
        train, test, test_stamp = dataloader(test_stamp)
        clf = XGBClassifier(random_state=0).fit(
            train.drop(columns=["client_id", "churn"]), train["churn"]
        )
        preds = (
            clf.predict_proba(test.drop(columns=["client_id", "churn"]))[:, 1]
            > 0.2
        )
        preds_prob = clf.predict_proba(
            test.drop(columns=["client_id", "churn"])
        )[:, 1]

        b_acc = balanced_accuracy_score(
            test["churn"],
            preds,
        )
        baccs.append(b_acc)
        print(f" BACC: {b_acc:.2f}")

        fpr, tpr, _ = roc_curve(
            test["churn"],
            preds_prob,
        )

        auroc = auc(fpr, tpr)
        aurocs.append(auroc)
        print(f" AUROC: {auroc:.2f}")

        if i == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            cm = confusion_matrix(test["churn"], preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title("Confusion matrix - last fold")
            plt.savefig(RESULTS_DIR / "confusion_matrix.png", transparent=True)
            plt.close()

            plt.rcParams.update({"font.size": 18})
            plt.figure(figsize=(7, 7))
            # create ROC curve
            plt.plot(fpr, tpr, color="#29BA74")

            # Add a random prediction line (diagonal line)
            random_line = np.linspace(0, 1, num=100)
            plt.plot(
                random_line,
                random_line,
                linestyle="--",
                label="Random Prediction Line",
                color="grey",
            )

            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.title("ROC Curve - last fold")
            plt.savefig(RESULTS_DIR / "ROC.png", transparent=True)
            plt.close()

    print("**************************")
    print(f"Avg. BACC {np.mean(baccs):.2f}")
    print(f"Avg. AUROC {np.mean(aurocs):.2f}")


if __name__ == "__main__":
    main()
