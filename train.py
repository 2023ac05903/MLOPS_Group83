import mlflow
import mlflow.sklearn
from models import model1, model2  # Adjust the import path as needed

def train_and_log(run_name, run_func):
    with mlflow.start_run(run_name=run_name):
        model, acc = run_func()

        mlflow.log_param("model_name", run_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, run_name)
        print(f"{run_name} Accuracy: {acc:.2f}")

if __name__ == "__main__":
    train_and_log("LogisticRegression", model1.run_model)
    train_and_log("RandomForest", model2.run_model)