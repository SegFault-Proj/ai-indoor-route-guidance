from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).with_name("congestion_model.joblib")


def create_simulated_dataset(sample_count: int = 5000, seed: int = 42):
    """
    실제 행사 로그가 없기 때문에 초기 MVP는 시뮬레이션 데이터로 학습합니다.
    향후 실제 인원 계수, 통로 폭, 시간대, 이동 기록으로 교체해야 합니다.
    """
    rng = np.random.default_rng(seed)

    people_count = rng.integers(0, 181, sample_count)
    corridor_width_m = rng.uniform(1.8, 6.0, sample_count)
    hour = rng.integers(9, 19, sample_count)
    event_phase = rng.integers(0, 4, sample_count)
    booth_zone = rng.integers(0, 2, sample_count)
    recent_inflow = rng.integers(0, 101, sample_count)

    density = people_count / corridor_width_m
    peak_hour = ((hour >= 12) & (hour <= 15)).astype(float)
    phase_penalty = np.select(
        [event_phase == 1, event_phase == 2, event_phase == 3],
        [0.18, 0.28, 0.58],
        default=0.0,
    )

    multiplier = (
        1.0
        + density / 48.0
        + recent_inflow / 145.0
        + booth_zone * 0.18
        + peak_hour * 0.12
        + phase_penalty
        + rng.normal(0, 0.07, sample_count)
    )
    multiplier = np.clip(multiplier, 1.0, 3.5)

    features = np.column_stack(
        [
            people_count,
            corridor_width_m,
            hour,
            event_phase,
            booth_zone,
            recent_inflow,
        ]
    )

    return features, multiplier


def train_and_save():
    X, y = create_simulated_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)

    payload = {
        "model": model,
        "feature_names": [
            "people_count",
            "corridor_width_m",
            "hour",
            "event_phase",
            "booth_zone",
            "recent_inflow",
        ],
        "training_type": "simulated",
        "validation_mae": float(mae),
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Simulation validation MAE: {mae:.4f}")


if __name__ == "__main__":
    train_and_save()
