from niid_bench.utils import calculate_trend
import torch
import torch.nn as nn
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
# === Model Definition ===
class AccuracyPredictor(nn.Module):
    def __init__(self):
        super(AccuracyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),  # ← includes trend!
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Load Trained Model ===
# model = AccuracyPredictor()
# model.load_state_dict(torch.load("accuracy_predictor.pt"))
# model.eval()

# === Prediction Function ===
def simulate_accuracy(round, fraction_fit, fraction_fit_prev, accuracy_prev, trend):

    model = AccuracyPredictor()
    model.load_state_dict(torch.load("accuracy_predictor_with_trends.pt"))
    model.eval()
    
    X = np.array([[round, fraction_fit, fraction_fit_prev, accuracy_prev, trend]], dtype=np.float32)
    
    # Normalize round to [0, 1] over 60 rounds
    X[:, 0] = (X[:, 0] - 1) / 59.0

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    return y_pred[0][0]


# === Multi-objective Optuna Objective ===
def make_multi_objective(round_now, performance_dataset, h=5):
    def objective(trial):
        acc_total = 0.0
        fraction_fit_total = 0.0
        r = round_now

        acc = performance_dataset[r - 1]["centralized_accuracy"]
        frac_prev = performance_dataset[r - 1]["fraction_fit"]
        trend = calculate_trend(performance_dataset, metric="centralized_accuracy", window=5)
        if trend is None:
            return float("inf"), float("inf")

        first_predicted_accuracy = None

        for step in range(h):
            frac = trial.suggest_float(f"fraction_fit_{step}", 0.1, 1.0)
            acc_pred = simulate_accuracy(r + step, frac, frac_prev, acc, trend)

            if step == 0:
                first_predicted_accuracy = acc_pred  # 🟩 Store prediction for next round

            if acc is not None and acc_pred < 0.8 * acc:
                return float("inf"), float("inf")

            acc_total += acc_pred
            fraction_fit_total += frac
            acc = acc_pred
            frac_prev = frac

        # 🟢 Save prediction in trial object
        trial.set_user_attr("first_predicted_accuracy", first_predicted_accuracy)

        return -acc_total / h, fraction_fit_total / h

    return objective

# === Trade-off Selection Function ===
def select_best_tradeoff(study, alpha=0.8, beta=0.2):
    best_score = float("inf")
    best_trial = None

    for t in study.best_trials:
        acc = -t.values[0]   # Un-negate accuracy
        frac = t.values[1]
        score = -alpha * acc + beta * frac

        if score < best_score:
            best_score = score
            best_trial = t

    return best_trial

# === Optimization Runner ===
def optimize_fraction_fit_moo(round_now, performance_dataset, h=5, n_trials=100, alpha=0.4, beta=0.6):
    objective_fn = make_multi_objective(round_now, performance_dataset, h=h)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.NSGAIISampler(seed=42))
    study.optimize(objective_fn, n_trials=n_trials)

    # Trade-off selection
    best_trial = select_best_tradeoff(study, alpha, beta)
    best_fractions = [best_trial.params[f"fraction_fit_{i}"] for i in range(h)]

    # Get predicted accuracy for round_now + 1
    predicted_next_accuracy = best_trial.user_attrs.get("first_predicted_accuracy", None)

    print(f"\n🎯 Best Trade-off → Accuracy (avg): {-best_trial.values[0]:.4f} | Fraction Fit (avg): {best_trial.values[1]:.2f}")
    for i, frac in enumerate(best_fractions):
        print(f"  ➤ Round {round_now + i}: fraction_fit = {frac:.3f}")

    return best_fractions, predicted_next_accuracy, best_fractions[0]

# === Example Usage ===
if __name__ == "__main__":
    my_performance_dataset = {}
    # Example round 31 simulation
    best_plan, total_accuracy, total_fraction = optimize_fraction_fit_moo(
        round_now=31,
        performance_dataset=my_performance_dataset,  # should be a dict like: {1: {...}, 2: {...}, ...}
        h=5,
        n_trials=200,
        alpha=0.6,  # maximize accuracy
        beta=0.4    # minimize cost
    )
