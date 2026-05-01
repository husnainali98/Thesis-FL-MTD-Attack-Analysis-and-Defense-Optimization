import socket
import pickle
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from model_utils import Net

SERVER_HOST = "127.0.0.1"
FL_PORT = 8080

ASS_SERVER_PORT = 6000
EXPECTED_CLIENTS = 10

NODE_HOST = "127.0.0.1"
BASE_NODE_PORT = 5000
NUM_NODES = 3

# MPC helper node ports: 5000, 5001, 5002
NODE_PORTS = [BASE_NODE_PORT + i for i in range(NUM_NODES)]

# Stores server-side shares received from clients, indexed by round number
ass_server_shares: Dict[int, List[List[np.ndarray]]] = {}
ass_lock = threading.Lock()

# Timing trackers for performance measurement
round_t0: Dict[int, float] = {}
secureagg_wait_share_t: Dict[int, float] = {}
secureagg_wait_node_t: Dict[Tuple[int, int], float] = {}
secureagg_total_t: Dict[int, float] = {}


def _weighted_avg(metric_list: List[Tuple[int, float]]) -> float:
    total = sum(n for n, _ in metric_list)
    if total == 0:
        return 0.0
    return sum(n * v for n, v in metric_list) / total


def ass_listener():
    """
    Listens for server-side shares sent by clients.
    Each client splits its local model update (delta) into additive shares.
    One share is sent here; the remaining shares go to MPC helper nodes.
    """
    print(f"[SERVER] ASS listener on {SERVER_HOST}:{ASS_SERVER_PORT} ...", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((SERVER_HOST, ASS_SERVER_PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            conn.close()

            try:
                msg = pickle.loads(data)
            except Exception:
                continue

            if msg.get("type") != "share_server":
                continue

            r = int(msg["round"])
            share_list = msg["share"]

            with ass_lock:
                ass_server_shares.setdefault(r, []).append(share_list)
                print(
                    f"[SERVER] got share_server for round={r} from {addr}, "
                    f"count={len(ass_server_shares[r])}/{EXPECTED_CLIENTS}",
                    flush=True,
                )


def _request_node_sum_once(round_num: int, node_port: int, timeout_sec: float = 3.0):
    """
    Sends a get_sum request to one MPC helper node.
    The node replies with the sum of all node-shares it has collected for this round.
    Returns (True, summed_shares) if ready, or (False, None) if not ready yet.
    """
    reply_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reply_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    reply_sock.bind((SERVER_HOST, 0))
    reply_sock.listen(1)
    reply_sock.settimeout(timeout_sec)
    reply_port = reply_sock.getsockname()[1]

    req = {
        "type": "get_sum",
        "round": round_num,
        "reply_host": SERVER_HOST,
        "reply_port": reply_port,
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_sec)
            s.connect((NODE_HOST, node_port))
            s.sendall(pickle.dumps(req))
    except Exception:
        reply_sock.close()
        return False, None

    try:
        conn, _ = reply_sock.accept()
    except Exception:
        reply_sock.close()
        return False, None

    data = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        conn.close()
        reply_sock.close()

    try:
        msg = pickle.loads(data)
    except Exception:
        return False, None

    if not msg.get("ready", False):
        return False, None

    return True, msg["sum"]


def get_node_sum(round_num: int, node_port: int, timeout_sec: float = 30.0, poll_interval: float = 0.5):
    """
    Polls a MPC helper node until it has collected shares from all expected clients.
    Raises RuntimeError if the node does not respond within timeout_sec seconds.
    """
    t0 = time.perf_counter()
    start_wall = time.time()

    while True:
        ready, s = _request_node_sum_once(round_num, node_port=node_port, timeout_sec=3.0)
        if ready:
            secureagg_wait_node_t[(round_num, node_port)] = time.perf_counter() - t0
            return s

        if time.time() - start_wall > timeout_sec:
            raise RuntimeError(f"Timeout: Node {node_port} not ready")

        time.sleep(poll_interval)


class FedAvgWithASS(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = Net()
        self._global_nds = None

    def initialize_parameters(self, client_manager):
        nds = [p.detach().cpu().numpy() for p in self._model.parameters()]
        self._global_nds = nds
        print("[SERVER] Initialized global parameters locally.", flush=True)
        return ndarrays_to_parameters(nds)

    def configure_fit(self, server_round, parameters, client_manager):
        round_t0[server_round] = time.perf_counter()
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        for (_, fit_ins) in fit_instructions:
            fit_ins.config["server_round"] = server_round
        return fit_instructions

    def aggregate_fit(self, server_round, results, failures):
        """
        Pure Secure Aggregation using Multi-Party Additive Secret Sharing.

            Each client splits its local model update (delta) into additive shares:
                server_share + node_share_1 + node_share_2 + node_share_3 = delta

            Server reconstructs the sum of all client deltas by combining:
                - Its own shares from all clients (collected via ass_listener)
                - Summed node-shares from each MPC helper node (collected via get_node_sum)

            Average delta is applied to the global model:
                new_global_model = old_global_model + (sum_of_deltas / num_clients)

            The server never observes any plaintext model update from any client.
        """
        secure_start = time.perf_counter()

        # Wait until all expected clients have submitted their server-side shares
        while True:
            with ass_lock:
                cnt = len(ass_server_shares.get(server_round, []))
            if cnt >= EXPECTED_CLIENTS:
                break
            time.sleep(0.2)

        with ass_lock:
            shares_list = ass_server_shares[server_round]

        # Sum server-side shares across all clients, layer by layer
        server_sum = [arr.copy() for arr in shares_list[0]]
        for client_layers in shares_list[1:]:
            for i in range(len(server_sum)):
                server_sum[i] = server_sum[i] + client_layers[i]

        # Collect summed node-shares from each MPC helper node
        node_sums = [get_node_sum(server_round, p) for p in NODE_PORTS]

        # Reconstruct full delta sum, compute average, and update global model
        for i in range(len(self._global_nds)):
            delta_sum = server_sum[i]
            for node_sum in node_sums:
                delta_sum = delta_sum + node_sum[i]

            delta_avg = delta_sum / float(EXPECTED_CLIENTS)
            self._global_nds[i] = (
                self._global_nds[i].astype(np.float64) + delta_avg.astype(np.float64)
            ).astype(np.float32)

        secureagg_total_t[server_round] = time.perf_counter() - secure_start

        print(f"[SERVER] round={server_round} applied secure update.", flush=True)
        return ndarrays_to_parameters(self._global_nds), {}

    def aggregate_evaluate(self, server_round, results, failures):

        loss_list = []
        acc_list = []

        for _, res in results:
            loss_list.append((res.num_examples, float(res.loss)))
            if res.metrics and "accuracy" in res.metrics:
                acc_list.append((res.num_examples, float(res.metrics["accuracy"])))

        avg_loss = _weighted_avg(loss_list)
        avg_acc = _weighted_avg(acc_list)

        total_round_time = time.perf_counter() - round_t0.get(server_round, 0)

        print(
            f"\n[RESULT] Round {server_round} | "
            f"loss={avg_loss:.4f} | accuracy={avg_acc:.4f} ({avg_acc*100:.2f}%) | "
            f"time={total_round_time:.2f}s\n",
            flush=True,
        )

        return avg_loss, {}


def main():
    threading.Thread(target=ass_listener, daemon=True).start()

    strategy = FedAvgWithASS(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=EXPECTED_CLIENTS,
        min_available_clients=EXPECTED_CLIENTS,
    )

    fl.server.start_server(
        server_address=f"{SERVER_HOST}:{FL_PORT}",
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()