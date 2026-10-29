import contextlib
import io
import random
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import main as solver


st.set_page_config(
    page_title="Hybrid CPU-GPU Logistics Optimizer",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Indigenous GPU-Accelerated Vehicle Routing Optimization Solver")
st.caption("Vehicle Routing Problem (CVRP) · Genetic Algorithm · Adaptive CPU/GPU evaluation")

# Match the reproducible 20-customer dataset used by main.py.
@st.cache_data
def create_problem():
    rng = np.random.default_rng(42)
    locations = np.vstack((
        np.array([[50, 50]], dtype=np.int32),
        rng.integers(0, 101, size=(20, 2), dtype=np.int32),
    ))
    demands = np.concatenate((
        np.array([0], dtype=np.int32),
        rng.integers(1, 5, size=20, dtype=np.int32),
    ))
    capacity = 10
    delta = locations[:, None, :] - locations[None, :, :]
    distance_matrix = np.sqrt(np.sum(delta.astype(np.float64) ** 2, axis=2))
    return locations, demands, capacity, distance_matrix


def greedy_routes(demands, capacity):
    routes, route, load = [], [0], 0
    for customer in range(1, len(demands)):
        demand = int(demands[customer])
        if load + demand > capacity:
            route.append(0)
            routes.append(route)
            route, load = [0], 0
        route.append(customer)
        load += demand
    route.append(0)
    routes.append(route)
    return routes


def route_summary(routes, demands, distance_matrix, capacity):
    records = []
    served = []
    for number, route in enumerate(routes, start=1):
        customers = [int(c) for c in route if c != 0]
        served.extend(customers)
        load = sum(int(demands[c]) for c in customers)
        distance = solver.calculate_route_distance(route, distance_matrix)
        records.append({
            "Vehicle": number,
            "Route": " → ".join(map(str, route)),
            "Customers": len(customers),
            "Load": f"{load}/{capacity}",
            "Distance": round(float(distance), 2),
            "Capacity OK": load <= capacity,
        })
    coverage_ok = sorted(served) == list(range(1, len(demands)))
    capacity_ok = all(row["Capacity OK"] for row in records)
    return records, coverage_ok, capacity_ok


def draw_routes(locations, routes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[locations[0, 0]], y=[locations[0, 1]],
        mode="markers+text", text=["Depot"], textposition="top center",
        marker=dict(size=18, symbol="square", color="#dc2626"),
        name="Warehouse",
    ))
    fig.add_trace(go.Scatter(
        x=locations[1:, 0], y=locations[1:, 1],
        mode="markers+text",
        text=[f"C{i}" for i in range(1, len(locations))],
        textposition="top center",
        customdata=np.column_stack((np.arange(1, len(locations)),
                                    locations[1:, 0], locations[1:, 1])),
        hovertemplate="Customer %{customdata[0]}<br>x=%{x}, y=%{y}<extra></extra>",
        marker=dict(size=10, color="#2563eb"),
        name="Customers",
    ))
    palette = ["#7c3aed", "#059669", "#d97706", "#db2777",
               "#0891b2", "#65a30d", "#9333ea", "#ea580c",
               "#0f766e", "#be123c", "#4f46e5"]
    for idx, route in enumerate(routes):
        coords = locations[np.asarray(route, dtype=int)]
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="lines+markers",
            line=dict(width=2.5, color=palette[idx % len(palette)]),
            marker=dict(size=5),
            name=f"Vehicle {idx + 1}",
            hovertemplate="x=%{x}, y=%{y}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        legend_title="Legend",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


locations, demands, capacity, distance_matrix = create_problem()
baseline_routes = greedy_routes(demands, capacity)
baseline_distance = sum(
    solver.calculate_route_distance(route, distance_matrix)
    for route in baseline_routes
)
total_demand = int(demands[1:].sum())
minimum_vehicles = int(np.ceil(total_demand / capacity))

with st.sidebar:
    st.header("Optimization settings")
    population = st.select_slider(
        "Population size", options=[20, 80, 100, 160, 320, 500, 1000], value=1000
    )
    generations = st.slider("Generations", min_value=5, max_value=200, value=10, step=5)
    mutation_rate = st.slider(
        "Mutation rate", min_value=0.05, max_value=0.50, value=0.20, step=0.05
    )
    backend_choice = st.selectbox(
        "Evaluation backend", ["Auto (adaptive)", "CPU", "GPU"], index=0
    )
    st.caption("GPU use requires a working OpenCL device. The solver has CPU fallback.")
    run_button = st.button("▶ Run optimization", type="primary", use_container_width=True)
    st.divider()
    st.subheader("CPU vs GPU benchmark")
    bench_population = st.selectbox("Benchmark population", [80, 100, 160, 320, 1000], index=1)
    bench_generations = st.slider("Benchmark generations", 5, 100, 20, step=5)
    bench_trials = st.selectbox("Paired trials", [1, 3, 5], index=1)
    benchmark_button = st.button("⚖ Compare CPU and GPU", use_container_width=True)

tab_overview, tab_routes, tab_benchmark, tab_about = st.tabs(
    ["Overview", "Routes & constraints", "CPU vs GPU", "How it works"]
)

with tab_overview:
    st.subheader("Problem overview")
    a, b, c, d = st.columns(4)
    a.metric("Customers", len(demands) - 1)
    b.metric("Total demand", total_demand)
    c.metric("Vehicle capacity", capacity)
    d.metric("Demand-based minimum vehicles", minimum_vehicles)
    st.info(
        "The depot is location 0. Customer coordinates and demands are generated "
        "with a fixed seed so the same problem can be reproduced."
    )
    st.write("**Baseline:** a simple capacity-feasible greedy assignment. "
             "The genetic algorithm searches for a shorter total fleet route.")
    if "optimization" not in st.session_state:
        st.warning("Choose settings in the sidebar and click **Run optimization** to generate live results.")
    else:
        result = st.session_state["optimization"]
        improvement = (baseline_distance - result["distance"]) / baseline_distance * 100
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Greedy distance", f"{baseline_distance:.2f}")
        x2.metric("Optimized distance", f"{result['distance']:.2f}")
        x3.metric("Distance reduction", f"{improvement:.2f}%")
        x4.metric("Vehicles used", len(result["routes"]))
        st.caption(
            f"Run configuration: population {result['population']}, "
            f"{result['generations']} generations, backend {result['backend']}."
        )
        st.plotly_chart(
    draw_routes(locations, result["routes"]),
    use_container_width=True,
    key="overview_routes_chart"
)

with tab_routes:
    st.subheader("Delivery routes")
    st.write("Warehouse/depot is marked in red. Each vehicle's route is shown in a different color.")
    if "optimization" not in st.session_state:
        st.info("Run an optimization first to display the optimized routes.")
    else:
        result = st.session_state["optimization"]
        records, coverage_ok, capacity_ok = route_summary(
            result["routes"], demands, distance_matrix, capacity
        )
        st.plotly_chart(
    draw_routes(locations, result["routes"]),
    use_container_width=True,
    key="details_routes_chart"
)
        st.dataframe(records, use_container_width=True, hide_index=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("Every customer visited exactly once", "PASS" if coverage_ok else "FAIL")
        v2.metric("All vehicle capacities respected", "PASS" if capacity_ok else "FAIL")
        v3.metric("Overall validation", "VALID" if coverage_ok and capacity_ok else "INVALID")
        with st.expander("View coordinates and demands"):
            st.dataframe(
                [{"Location": "Depot" if i == 0 else f"Customer {i}",
                  "ID": i, "X": int(locations[i, 0]), "Y": int(locations[i, 1]),
                  "Demand": int(demands[i])} for i in range(len(demands))],
                use_container_width=True, hide_index=True
            )

with tab_benchmark:
    st.subheader("End-to-end CPU vs GPU comparison")
    st.write(
        "Paired trials use the same random seed for CPU and GPU. The measured time includes "
        "solver setup and GPU evaluator initialization; results depend on your hardware."
    )
    if "benchmark" not in st.session_state:
        st.info("Choose benchmark settings in the sidebar and click **Compare CPU and GPU**.")
    else:
        data = st.session_state["benchmark"]
        left, right, ratio_col = st.columns(3)
        left.metric("CPU median", f"{data['cpu_median']:.3f} s")
        right.metric("GPU median", f"{data['gpu_median']:.3f} s")
        ratio_col.metric("CPU/GPU speedup", f"{data['speedup']:.2f}×")
        faster = "GPU" if data["gpu_median"] < data["cpu_median"] else "CPU"
        st.success(f"{faster} was faster in this benchmark run.")
        st.dataframe(data["rows"], use_container_width=True, hide_index=True)
        st.bar_chart(
            {"CPU time (s)": [row["CPU time (s)"] for row in data["rows"]],
             "GPU time (s)": [row["GPU time (s)"] for row in data["rows"]]}
        )
        st.caption("Do not generalize one run to all workloads; repeat trials for stronger evidence.")

with tab_about:
    st.subheader("Project workflow")
    st.markdown("""
    1. **Input:** depot/customer coordinates, customer demands, vehicle capacity.
    2. **Distance matrix:** Euclidean distance between every pair of locations.
    3. **Genetic algorithm:** chromosome population, fitness evaluation, tournament selection,
       order crossover, mutation, and elitism.
    4. **Hybrid evaluation:** CPU or OpenCL GPU fitness evaluation; adaptive mode uses the
       benchmark policy already implemented in `main.py`.
    5. **Validation:** every customer is served once and no vehicle exceeds capacity.
    6. **Evaluation:** compare route distance against the greedy baseline and benchmark CPU/GPU time.
    """)
    st.warning(
        "The displayed coordinates are a synthetic test instance, not live GPS or a real delivery map. "
        "Speedup is workload- and device-dependent."
    )

if run_button:
    run_succeeded = False
    with st.spinner("Running the genetic algorithm…"):
        backend_arg = {
            "Auto (adaptive)": "auto",
            "CPU": False,
            "GPU": True,
        }[backend_choice]
        random.seed(42)
        np.random.seed(42)
        log_buffer = io.StringIO()
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(log_buffer):
                chromosome, cost, history = solver.genetic_algorithm(
                    demands=demands,
                    vehicle_capacity=capacity,
                    distance_matrix=distance_matrix,
                    population_size=int(population),
                    generations=int(generations),
                    mutation_rate=float(mutation_rate),
                    use_gpu=backend_arg,
                )
            elapsed = time.perf_counter() - start
            routes = solver.decode_chromosome(chromosome, demands, capacity)
            rows, coverage_ok, capacity_ok = route_summary(
                routes, demands, distance_matrix, capacity
            )
            log = log_buffer.getvalue()
            chosen_backend = backend_choice
            if "Adaptive backend: GPU" in log:
                chosen_backend = "GPU"
            elif "Adaptive backend: CPU" in log or "switching to CPU" in log or "falling back to CPU" in log:
                chosen_backend = "CPU"
            st.session_state["optimization"] = {
                "chromosome": [int(x) for x in chromosome],
                "distance": float(cost),
                "routes": routes,
                "history": [float(x) for x in history],
                "elapsed": elapsed,
                "backend": chosen_backend,
                "population": int(population),
                "generations": int(generations),
                "valid": bool(coverage_ok and capacity_ok),
                "route_rows": rows,
                "log": log,
            }
            st.session_state.pop("benchmark", None)
            run_succeeded = True
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
        if log_buffer.getvalue():
            with st.expander("Solver diagnostic log"):
                st.code(log_buffer.getvalue())
        if run_succeeded:
            st.rerun()

if benchmark_button:
    cpu_times, gpu_times, rows = [], [], []
    cpu_distances, gpu_distances = [], []
    validity = []
    progress = st.progress(0, text="Starting paired CPU/GPU trials…")
    failed = None
    for trial in range(bench_trials):
        seed = [101, 202, 303, 404, 505][trial]
        trial_row = {"Trial": trial + 1, "Seed": seed}
        for label, use_gpu in [("CPU", False), ("GPU", True)]:
            random.seed(seed)
            np.random.seed(seed)
            log_buffer = io.StringIO()
            start = time.perf_counter()
            try:
                with contextlib.redirect_stdout(log_buffer):
                    chromosome, cost, _ = solver.genetic_algorithm(
                        demands=demands,
                        vehicle_capacity=capacity,
                        distance_matrix=distance_matrix,
                        population_size=int(bench_population),
                        generations=int(bench_generations),
                        mutation_rate=float(mutation_rate),
                        use_gpu=use_gpu,
                    )
                elapsed = time.perf_counter() - start
                routes = solver.decode_chromosome(chromosome, demands, capacity)
                _, coverage_ok, capacity_ok = route_summary(
                    routes, demands, distance_matrix, capacity
                )
                trial_row[f"{label} time (s)"] = round(elapsed, 4)
                trial_row[f"{label} distance"] = round(float(cost), 2)
                trial_row[f"{label} valid"] = bool(coverage_ok and capacity_ok)
                validity.append(bool(coverage_ok and capacity_ok))
                if label == "CPU":
                    cpu_times.append(elapsed)
                    cpu_distances.append(float(cost))
                else:
                    gpu_times.append(elapsed)
                    gpu_distances.append(float(cost))
            except Exception as exc:
                failed = f"{label} run failed: {exc}"
                break
        if failed:
            break
        rows.append(trial_row)
        progress.progress((trial + 1) / bench_trials, text=f"Completed trial {trial + 1}/{bench_trials}")
    progress.empty()
    if failed:
        st.error(failed)
        st.info("Check that PyOpenCL and an OpenCL device are available for the GPU benchmark.")
    elif rows:
        cpu_median = float(np.median(cpu_times))
        gpu_median = float(np.median(gpu_times))
        st.session_state["benchmark"] = {
            "rows": rows,
            "cpu_median": cpu_median,
            "gpu_median": gpu_median,
            "speedup": cpu_median / gpu_median if gpu_median > 0 else float("inf"),
            "cpu_median_distance": float(np.median(cpu_distances)),
            "gpu_median_distance": float(np.median(gpu_distances)),
            "all_valid": all(validity),
        }
        with st.expander("Benchmark validation details", expanded=True):
            st.write(f"All benchmark solutions valid: **{all(validity)}**")
            st.write(f"Median CPU distance: **{np.median(cpu_distances):.2f}**")
            st.write(f"Median GPU distance: **{np.median(gpu_distances):.2f}**")
        st.rerun()
