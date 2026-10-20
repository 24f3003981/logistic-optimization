"""
Smart Fulfilment & Delivery Optimisation - Week 1 prototype
Runs end-to-end on synthetic data so the pipeline can be tested before real
datasets (Olist, DataCo Smart Supply Chain, M5) are plugged in.
Usage:  python logistics_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# --- [1. DATA COLLECTION] ---
def load_orders(n=5000):
    """Synthetic stand-in for an orders table. Replace with pd.read_csv(...)."""
    df = pd.DataFrame({
        "order_id": np.arange(n),
        "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
        "lat": rng.normal(12.9, 0.35, n),
        "lon": rng.normal(77.6, 0.35, n),
        "weight_kg": rng.gamma(2.0, 3.0, n),
        "promised_days": rng.choice([2, 3, 5], n, p=[0.3, 0.5, 0.2]),
        "in_full": rng.random(n) > 0.06,
    })
    depot = (12.9, 77.6)
    df["dist_km"] = 111 * np.hypot(df.lat - depot[0], (df.lon - depot[1]) * np.cos(np.radians(12.9)))
    df["actual_days"] = (1 + df.dist_km / 25 + df.weight_kg / 30
                         + rng.normal(0, 0.6, n)).clip(lower=1).round(1)
    df.loc[rng.random(n) < 0.01, "weight_kg"] = np.nan      # inject dirty data
    return pd.concat([df, df.sample(20, random_state=1)])     # inject duplicates

# --- [2. DATA CLEANING] ---
def clean_orders(df):
    df = df.drop_duplicates(subset="order_id").copy()
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    q1, q3 = df["actual_days"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[df["actual_days"].between(q1 - 3 * iqr, q3 + 3 * iqr)]   # drop extreme outliers
    df["month"] = df["order_date"].dt.month
    df["dow"] = df["order_date"].dt.dayofweek
    return df.reset_index(drop=True)

# --- [3. KPI CALCULATION] ---
def compute_kpis(df, cost_per_km=14.0, base_cost=40.0):
    on_time = df["actual_days"] <= df["promised_days"]
    kpis = {
        "OTIF_%": float(round(100 * (on_time & df["in_full"]).mean(), 1)),
        "On_time_%": float(round(100 * on_time.mean(), 1)),
        "Avg_cost_per_delivery_INR": float(round((base_cost + cost_per_km * df["dist_km"]).mean(), 1)),
        "Avg_delivery_days": float(round(df["actual_days"].mean(), 2)),
    }
    return kpis

# --- [4. CLUSTERING] ---
def build_zones(df, k=5):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["zone"] = km.fit_predict(df[["lat", "lon"]])
    return df, km.cluster_centers_

# --- [5. REGRESSION] ---
def train_delivery_time_model(df):
    features = ["dist_km", "weight_kg", "promised_days", "month", "dow", "zone"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[features], df["actual_days"], test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
    return model, mean_absolute_error(y_te, model.predict(X_te))

# --- [6. DEMAND FORECAST AND INVENTORY POLICY] ---
def forecast_and_reorder(lead_time_days=4, service_z=1.65):
    days = pd.date_range("2025-01-01", periods=365)
    demand = (50 + 10 * np.sin(np.arange(365) * 2 * np.pi / 7)
              + rng.normal(0, 6, 365)).clip(min=0)
    s = pd.DataFrame({"demand": demand}, index=days)
    for lag in (1, 7, 14):
        s[f"lag{lag}"] = s["demand"].shift(lag)
    s["roll7"] = s["demand"].shift(1).rolling(7).mean()
    s = s.dropna()
    train, test = s.iloc[:-60], s.iloc[-60:]
    m = GradientBoostingRegressor(random_state=0).fit(train.drop(columns="demand"), train["demand"])
    pred = m.predict(test.drop(columns="demand"))
    wape = np.abs(test["demand"] - pred).sum() / test["demand"].sum()
    sigma = (test["demand"] - pred).std()
    safety = service_z * sigma * np.sqrt(lead_time_days)
    reorder_point = pred.mean() * lead_time_days + safety
    return wape, reorder_point

# --- [7. ROUTE OPTIMISATION] ---
def route_cost(order, dist):
    return sum(dist[a][b] for a, b in zip(order[:-1], order[1:]))

def nearest_neighbour_route(dist):
    """Baseline heuristic: always visit the closest unvisited stop."""
    unvisited, route = set(range(1, len(dist))), [0]
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[route[-1]][j])
        route.append(nxt)
        unvisited.remove(nxt)
    return route + [0]

def two_opt(route, dist):
    """Improve a route by reversing segments while that shortens it."""
    best, improved = route, True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_cost(cand, dist) < route_cost(best, dist) - 1e-9:
                    best, improved = cand, True
    return best

def demo_routing(n_stops=25):
    pts = rng.random((n_stops + 1, 2)) * 30                  # depot = index 0
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    base = nearest_neighbour_route(dist)
    better = two_opt(base, dist)
    return route_cost(base, dist), route_cost(better, dist)

# --- [8. MAIN PIPELINE] ---
if __name__ == "__main__":
    orders = clean_orders(load_orders())
    print("Orders after cleaning:", len(orders))
    print("Baseline KPIs:", compute_kpis(orders))

    orders, centres = build_zones(orders)
    print("Orders per zone:\n", orders["zone"].value_counts().sort_index().to_string())

    _, mae = train_delivery_time_model(orders)
    print(f"Delivery-time model MAE: {mae:.2f} days")

    wape, rop = forecast_and_reorder()
    print(f"Demand forecast WAPE: {wape:.1%} | Reorder point: {rop:.0f} units")

    base, opt = demo_routing()
    print(f"Route length: baseline {base:.1f} km -> 2-opt {opt:.1f} km "
          f"({100 * (base - opt) / base:.1f}% shorter)")
