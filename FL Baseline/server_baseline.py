
import time
from typing import List, Tuple

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg


SERVER_HOST = "127.0.0.1"
FL_PORT = 8080

# ---- latency tracking ----
round_t0 = {}  # round -> perf_counter start


def _weighted_avg(metric_list: List[Tuple[int, float]]) -> float:
    total = sum(n for n, _ in metric_list)
    if total == 0:
        return 0.0
    return sum(n * v for n, v in metric_list) / total


class FedAvgBaseline(FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        round_t0[server_round] = time.perf_counter()
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # pass round number to clients 
        for (_, fit_ins) in fit_instructions:
            fit_ins.config["server_round"] = server_round

        return fit_instructions

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # accuracy (from clients)
        acc_list = []
        for _, eval_res in results:
            if eval_res.metrics and "accuracy" in eval_res.metrics:
                acc_list.append((eval_res.num_examples, float(eval_res.metrics["accuracy"])))

        avg_acc = _weighted_avg(acc_list) if acc_list else 0.0

        total_round_time = None
        if server_round in round_t0:
            total_round_time = time.perf_counter() - round_t0[server_round]

        if total_round_time is None:
            print(
                f"\n[RESULT][BASELINE] Round {server_round:02d} | "
                f"loss={aggregated_loss:.4f} | accuracy={avg_acc:.4f} ({avg_acc*100:.2f}%)\n"
            )
        else:
            print(
                f"\n[RESULT][BASELINE] Round {server_round:02d} | "
                f"loss={aggregated_loss:.4f} | accuracy={avg_acc:.4f} ({avg_acc*100:.2f}%) | "
                f"round_time={total_round_time:.3f}s\n"
            )

        return aggregated_loss, aggregated_metrics


def main():
    print("[BASELINE SERVER] starting Flower server ...")

    strategy = FedAvgBaseline(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
    )

    fl.server.start_server(
        server_address=f"{SERVER_HOST}:{FL_PORT}",
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
