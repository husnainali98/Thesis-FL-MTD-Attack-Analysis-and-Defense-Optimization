import socket
import pickle
import threading
import time
import random
from math import ceil
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from flwr.server.strategy import FedAvg

from model_utils import Net

SERVER_HOST = "127.0.0.1"
FL_PORT = 8080

ASS_SERVER_PORT = 6000

# -------------------------------------------------------
# Total clients 
# -------------------------------------------------------
TOTAL_CLIENTS = 20

# -------------------------------------------------------
# Node configuration
# -------------------------------------------------------
NODE_HOST = "127.0.0.1"
BASE_NODE_PORT = 5000
MAX_NUM_NODES = 3  # change this when you add more nodes

ALL_NODE_PORTS = [BASE_NODE_PORT + i for i in range(MAX_NUM_NODES)]

# -------------------------------------------------------
# Fixed number of selected clients per round
# Clients change randomly, but count stays fixed
# -------------------------------------------------------
SELECTED_CLIENTS_PER_ROUND = 10

# -------------------------------------------------------
# WARMUP CONFIGURATION (NEW!)
# -------------------------------------------------------
WARMUP_ROUNDS = 8  # No detection during first 8 rounds (model learning period)

# -------------------------------------------------------
# Truly dynamic adaptive node policy
# Safe and risk counts depend on total node pool size
# -------------------------------------------------------
SAFE_NODE_RATIO = 0.25
RISK_NODE_RATIO = 0.75

MIN_SAFE_NODES = 2
MIN_RISK_NODES = 3

# -------------------------------------------------------
# Detection thresholds
# -------------------------------------------------------
SUSPICIOUS_LOSS_THRESHOLD = 2.0
SUSPICIOUS_ACC_THRESHOLD = 0.80
STABLE_RECOVERY_ROUNDS = 3

# -------------------------------------------------------
# Trust settings 
# -------------------------------------------------------
MIN_TRUST_SCORE = 0.1
TRUST_PENALTY_FACTOR = 0.7  # Trust Penalty
TRUST_BAN_THRESHOLD = 0.6   # Ban Threshold
SUSPICIOUS_EVENTS_BAN = 10   # Suspicious event threshold

# In risk mode, we do not sample from all clients blindly.
# Instead, build a trusted candidate pool first.
# Example:
# if 10 clients exist and ratio=0.7 -> top 7 trusted clients are candidate pool
RISK_SELECTION_POOL_RATIO = 0.7

# Keep at least this many clients in candidate pool
MIN_RISK_POOL_SIZE = SELECTED_CLIENTS_PER_ROUND

# -------------------------------------------------------
# Runtime state
# -------------------------------------------------------
# round -> selected client count
round_selected_clients: Dict[int, int] = {}

# round -> selected active node ports
round_active_node_ports: Dict[int, List[int]] = {}

# round -> selected client IDs
round_selected_client_ids: Dict[int, List[str]] = {}

# round -> list of server-side shares from selected clients
ass_server_shares: Dict[int, List[List[np.ndarray]]] = {}
ass_lock = threading.Lock()

# timing
round_t0: Dict[int, float] = {}
secureagg_wait_share_t: Dict[int, float] = {}
secureagg_wait_node_t: Dict[Tuple[int, int], float] = {}
secureagg_total_t: Dict[int, float] = {}

# adaptive detector state
current_risk_state = False
stable_rounds_counter = 0

# -------------------------------------------------------
# Trust state
# -------------------------------------------------------
client_selected_count: Dict[str, int] = {}
client_suspicious_count: Dict[str, int] = {}
client_trust_score: Dict[str, float] = {}
banned_clients: Set[str] = set()  # Track banned clients

# -------------------------------------------------------
# Previous round accuracy tracking
# -------------------------------------------------------
previous_round_accuracy: Optional[float] = None


def _weighted_avg(metric_list: List[Tuple[int, float]]) -> float:
    total = sum(n for n, _ in metric_list)
    if total == 0:
        return 0.0
    return sum(n * v for n, v in metric_list) / total


def ensure_client_exists(cid: str):
    if cid not in client_selected_count:
        client_selected_count[cid] = 0
        client_suspicious_count[cid] = 0
        client_trust_score[cid] = 1.0


def recompute_trust(cid: str):
    ensure_client_exists(cid)

    selected = client_selected_count[cid]
    suspicious = client_suspicious_count[cid]

    if selected <= 0:
        client_trust_score[cid] = 1.0
        return

    suspicious_ratio = suspicious / float(selected)
    trust = 1.0 - TRUST_PENALTY_FACTOR * suspicious_ratio
    client_trust_score[cid] = max(MIN_TRUST_SCORE, trust)


def compute_active_node_count(total_nodes: int, risky: bool) -> int:
    """
    Dynamic node scaling:
    - Safe mode uses a smaller fraction of total nodes
    - Risk mode uses a larger fraction of total nodes
    """
    if total_nodes <= 0:
        raise RuntimeError("No MPC nodes configured.")

    if risky:
        count = int(ceil(RISK_NODE_RATIO * total_nodes))
        count = max(MIN_RISK_NODES, count)
    else:
        count = int(ceil(SAFE_NODE_RATIO * total_nodes))
        count = max(MIN_SAFE_NODES, count)

    return min(total_nodes, count)


def choose_active_node_ports(num_active_nodes: int) -> List[int]:
    """
    Randomly choose active helper nodes from the full node pool.
    """
    if num_active_nodes > len(ALL_NODE_PORTS):
        raise RuntimeError(
            f"Requested {num_active_nodes} active nodes, "
            f"but only {len(ALL_NODE_PORTS)} are configured."
        )

    chosen = random.sample(ALL_NODE_PORTS, num_active_nodes)
    chosen.sort()
    return chosen


# Added server_round parameter for warmup
def detect_suspicious_behavior(loss_value: float, acc_value: float, previous_acc: Optional[float] = None, server_round: int = 0) -> bool:
    """
    Simple detector with WARMUP period.
    During warmup, NEVER flag as suspicious (model is learning).
    After warmup, use thresholds normally.
    """
    # WARMUP: First N rounds are NEVER suspicious
    if server_round <= WARMUP_ROUNDS:
        return False
    
    # After warmup, use original logic
    if loss_value > SUSPICIOUS_LOSS_THRESHOLD:
        return True
    if acc_value < SUSPICIOUS_ACC_THRESHOLD:
        return True
    if previous_acc is not None and acc_value < previous_acc:
        return True
    return False


def ass_listener():
    """
    Background listener.
    Clients send their server-side additive shares here.
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
                expected = round_selected_clients.get(r, "?")
                print(
                    f"[SERVER] got share_server for round={r} from {addr}, "
                    f"count={len(ass_server_shares[r])}/{expected}",
                    flush=True,
                )


def _request_node_sum_once(round_num: int, node_port: int, timeout_sec: float = 3.0):
    """
    Ask one helper node for its summed shares for a round.
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
        return False, None, 0

    try:
        conn, _ = reply_sock.accept()
    except Exception:
        reply_sock.close()
        return False, None, 0

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
        return False, None, 0

    return msg.get("ready", False), msg.get("sum", None), int(msg.get("count", 0))


def get_node_sum(
    round_num: int,
    node_port: int,
    expected_count: int,
    timeout_sec: float = 30.0,
    poll_interval: float = 0.5,
):
    """
    Poll one node until it has enough shares for the selected clients of this round.
    """
    t0 = time.perf_counter()
    start_wall = time.time()

    while True:
        ready, s, count = _request_node_sum_once(
            round_num,
            node_port=node_port,
            timeout_sec=3.0,
        )

        if ready and count >= expected_count:
            secureagg_wait_node_t[(round_num, node_port)] = time.perf_counter() - t0
            return s

        if time.time() - start_wall > timeout_sec:
            secureagg_wait_node_t[(round_num, node_port)] = time.perf_counter() - t0
            raise RuntimeError(
                f"Timeout: Node {node_port} has only {count}/{expected_count} shares for round {round_num}"
            )

        time.sleep(poll_interval)


class FedAvgWithAdaptiveMPC(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = Net()
        self._global_nds: Optional[List[np.ndarray]] = None

    def initialize_parameters(self, client_manager):
        nds = [p.detach().cpu().numpy() for p in self._model.parameters()]
        self._global_nds = nds
        print("[SERVER] Initialized global parameters locally.", flush=True)
        return ndarrays_to_parameters(nds)

    def _get_active_clients(self, client_manager):
        """Get all active (non-banned) clients"""
        all_clients_dict = client_manager.all()
        active_clients = []
        for client in all_clients_dict.values():
            cid = str(client.cid)
            if cid not in banned_clients:
                active_clients.append(client)
        return active_clients

    def _random_sample_clients(self, client_manager, selected_count: int):
        """
        Safe mode: Random selection from ACTIVE (non-banned) clients only
        """
        active_clients = self._get_active_clients(client_manager)
        
        if len(active_clients) < selected_count:
            raise RuntimeError(
                f"Not enough active clients for sampling: "
                f"need {selected_count}, have {len(active_clients)} "
                f"(banned: {len(banned_clients)})"
            )
        
        # Ensure all active clients exist in trust tracking
        for client in active_clients:
            cid = str(client.cid)
            ensure_client_exists(cid)
        
        selected_clients = random.sample(active_clients, selected_count)
        
        print(
            f"[SERVER][SAFE] Selected from {len(active_clients)} active clients "
            f"(banned: {len(banned_clients)})",
            flush=True,
        )
        
        return selected_clients

    def _trust_based_sample_clients(self, client_manager, selected_count: int):
        """
        Risk mode:
        1. Look at all ACTIVE (non-banned) clients
        2. Sort by trust descending
        3. Create candidate pool from most trusted clients
        4. Randomly sample selected_count from that pool
        """
        active_clients = self._get_active_clients(client_manager)
        
        if len(active_clients) < selected_count:
            raise RuntimeError(
                f"Not enough active clients for trust-based sampling: "
                f"need {selected_count}, have {len(active_clients)} "
                f"(banned: {len(banned_clients)})"
            )
        
        for client in active_clients:
            cid = str(client.cid)
            ensure_client_exists(cid)

        sorted_clients = sorted(
            active_clients,
            key=lambda c: client_trust_score.get(str(c.cid), 1.0),
            reverse=True,
        )

        candidate_pool_size = max(
            MIN_RISK_POOL_SIZE,
            int(ceil(RISK_SELECTION_POOL_RATIO * len(sorted_clients)))
        )
        candidate_pool_size = min(len(sorted_clients), candidate_pool_size)

        candidate_pool = sorted_clients[:candidate_pool_size]

        selected_clients = random.sample(candidate_pool, selected_count)

        print(
            f"[SERVER][TRUST] Risk-mode client selection from top {candidate_pool_size}/{len(sorted_clients)} trusted clients.",
            flush=True,
        )
        print(
            f"[SERVER][TRUST] Banned clients so far: {len(banned_clients)}",
            flush=True,
        )
        print(
            "[SERVER][TRUST] Candidate pool trust scores:",
            [(str(c.cid), round(client_trust_score.get(str(c.cid), 1.0), 3)) for c in candidate_pool],
            flush=True,
        )

        return selected_clients

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Safe mode:
            - fixed-size random client selection from ACTIVE clients only
        Risk mode:
            - trust-aware client selection from ACTIVE clients only
        Dynamic node count based on total node pool and risk state.
        Random node identity selection each round.
        """
        global current_risk_state

        round_t0[server_round] = time.perf_counter()

        available = client_manager.num_available()

        # Wait for all TOTAL_CLIENTS (20) only in the first round
        if server_round == 1:
            print(f"[SERVER] Waiting for all {TOTAL_CLIENTS} clients to connect before starting Round 1...", flush=True)
            while available < TOTAL_CLIENTS:
                print(
                    f"[SERVER] Waiting for clients... "
                    f"Need all {TOTAL_CLIENTS}, currently have {available}",
                    flush=True,
                )
                time.sleep(2)
                available = client_manager.num_available()
            print(f"[SERVER] All {TOTAL_CLIENTS} clients connected! Starting federated learning.", flush=True)
        else:
            # For subsequent rounds, wait only for SELECTED_CLIENTS_PER_ROUND
            while available < SELECTED_CLIENTS_PER_ROUND:
                print(
                    f"[SERVER] Waiting for clients... "
                    f"Need at least {SELECTED_CLIENTS_PER_ROUND}, currently have {available}",
                    flush=True,
                )
                time.sleep(2)
                available = client_manager.num_available()

        selected_count = SELECTED_CLIENTS_PER_ROUND
        round_selected_clients[server_round] = selected_count

        active_node_count = compute_active_node_count(
            total_nodes=len(ALL_NODE_PORTS),
            risky=current_risk_state,
        )
        active_node_ports = choose_active_node_ports(active_node_count)
        round_active_node_ports[server_round] = active_node_ports

        config = {
            "server_round": str(server_round),
            "active_node_ports": ",".join(str(p) for p in active_node_ports),
        }

        fit_ins = FitIns(parameters=parameters, config=config)

        if current_risk_state:
            selected_clients = self._trust_based_sample_clients(client_manager, selected_count)
        else:
            selected_clients = self._random_sample_clients(client_manager, selected_count)

        selected_ids = []
        for client in selected_clients:
            cid = str(client.cid)
            ensure_client_exists(cid)
            selected_ids.append(cid)

        round_selected_client_ids[server_round] = selected_ids

        # Calculate malicious ratio in selected clients (for monitoring)
        malicious_selected = 0
        # Note: need to know which clients are malicious
        # This is just for logging if have that info
        if hasattr(self, 'malicious_client_ids'):
            malicious_selected = len([cid for cid in selected_ids if cid in self.malicious_client_ids])

        print(
            f"[SERVER] Round {server_round}: selected "
            f"{selected_count} client(s) from {available} available client(s)",
            flush=True,
        )
        print(
            f"[SERVER] Round {server_round}: selected client IDs = {selected_ids}",
            flush=True,
        )
        print(
            f"[SERVER] Round {server_round}: using ACTIVE MPC nodes = {active_node_ports}",
            flush=True,
        )
        print(
            f"[SERVER] Round {server_round}: Banned clients so far = {len(banned_clients)}",
            flush=True,
        )

        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(self, server_round, results, failures):
        secure_start = time.perf_counter()

        selected_count = round_selected_clients.get(server_round, len(results))
        active_node_ports = round_active_node_ports.get(server_round, [])

        # Wait for all selected clients' server shares
        wait_share_t0 = time.perf_counter()
        start_wall = time.time()

        while True:
            with ass_lock:
                cnt = len(ass_server_shares.get(server_round, []))
            if cnt >= selected_count:
                break
            if time.time() - start_wall > 30:
                secureagg_wait_share_t[server_round] = time.perf_counter() - wait_share_t0
                raise RuntimeError(
                    f"Timeout: missing share_server for round {server_round} "
                    f"(have {cnt}/{selected_count})"
                )
            time.sleep(0.2)

        secureagg_wait_share_t[server_round] = time.perf_counter() - wait_share_t0

        # Sum server shares
        with ass_lock:
            shares_list = ass_server_shares[server_round]

        server_sum = [arr.copy() for arr in shares_list[0]]
        for client_layers in shares_list[1:]:
            for i in range(len(server_sum)):
                server_sum[i] = server_sum[i] + client_layers[i]

        # Get sums only from active nodes for this round
        node_sums = []
        for node_port in active_node_ports:
            node_sum = get_node_sum(
                server_round,
                node_port=node_port,
                expected_count=selected_count,
                timeout_sec=30.0,
                poll_interval=0.5,
            )
            node_sums.append(node_sum)

        # Reconstruct full delta and apply average update
        # Pure MPC: update global model with computed delta (no FedAvg addition)
        for i in range(len(self._global_nds)):
            delta_sum = server_sum[i]

            for node_sum in node_sums:
                delta_sum = delta_sum + node_sum[i]

            delta_avg = delta_sum / float(selected_count)

            self._global_nds[i] = (
                self._global_nds[i].astype(np.float64)
                + delta_avg.astype(np.float64)
            ).astype(np.float32)

        secureagg_total_t[server_round] = time.perf_counter() - secure_start

        print(
            f"[SERVER] round={server_round} applied FULL-ASS-secure update "
            f"to ALL layers using ACTIVE nodes {active_node_ports} "
            f"and {selected_count} selected client(s).",
            flush=True,
        )

        return ndarrays_to_parameters(self._global_nds), {}

    def aggregate_evaluate(self, server_round, results, failures):
        global current_risk_state
        global stable_rounds_counter
        global banned_clients
        global previous_round_accuracy

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        acc_list = []
        for _, eval_res in results:
            if eval_res.metrics and "accuracy" in eval_res.metrics:
                acc_list.append((eval_res.num_examples, float(eval_res.metrics["accuracy"])))
        avg_acc = _weighted_avg(acc_list) if acc_list else 0.0

        total_round_time = None
        if server_round in round_t0:
            total_round_time = time.perf_counter() - round_t0[server_round]

        sa_total = secureagg_total_t.get(server_round, 0.0)
        sa_wait_share = secureagg_wait_share_t.get(server_round, 0.0)

        active_node_ports = round_active_node_ports.get(server_round, [])
        node_waits = []
        for node_port in active_node_ports:
            node_waits.append(secureagg_wait_node_t.get((server_round, node_port), 0.0))

        overhead_pct = 0.0
        if total_round_time and total_round_time > 1e-9:
            overhead_pct = (sa_total / total_round_time) * 100.0

        node_wait_str = ", ".join(
            [f"port{active_node_ports[idx]}_wait={node_waits[idx]:.3f}s" for idx in range(len(node_waits))]
        )

        selected_count = round_selected_clients.get(server_round, "?")
        selected_ids = round_selected_client_ids.get(server_round, [])

        # Pass server_round to detect_suspicious_behavior for warmup
        suspicious = detect_suspicious_behavior(
            loss_value=float(aggregated_loss),
            acc_value=float(avg_acc),
            previous_acc=previous_round_accuracy,
            server_round=server_round,  # <-- ADDED: passes current round number
        )

        # Update previous accuracy for next round
        previous_round_accuracy = float(avg_acc)

        # -------------------------------------------------------
        # Update trust history
        # -------------------------------------------------------
        newly_banned = []
        for cid in selected_ids:
            ensure_client_exists(cid)
            client_selected_count[cid] += 1

        if suspicious:
            print(
                f"[SERVER][DETECTOR]  Suspicious global behavior detected in round {server_round}!",
                flush=True,
            )
            for cid in selected_ids:
                client_suspicious_count[cid] += 1
                recompute_trust(cid)
                
                # Check if client should be banned (MORE REALISTIC: both conditions)
                if (client_suspicious_count[cid] >= SUSPICIOUS_EVENTS_BAN and 
                    client_trust_score[cid] < TRUST_BAN_THRESHOLD):
                    if cid not in banned_clients:
                        banned_clients.add(cid)
                        newly_banned.append(cid)
                        print(
                            f"[SERVER][BAN]  BANNED client {cid} | "
                            f"trust={client_trust_score[cid]:.3f} | "
                            f"suspicious_events={client_suspicious_count[cid]} | "
                            f"selected={client_selected_count[cid]}",
                            flush=True,
                        )
                elif client_suspicious_count[cid] >= SUSPICIOUS_EVENTS_BAN:
                    print(
                        f"[SERVER][WARN]  Client {cid} has {client_suspicious_count[cid]} suspicious events "
                        f"but trust={client_trust_score[cid]:.3f} (not banned yet - needs trust < {TRUST_BAN_THRESHOLD})",
                        flush=True,
                    )
        else:
            # GOOD ROUND LOGIC - trust increases in good rounds
            for cid in selected_ids:
                recompute_trust(cid)

        # -------------------------------------------------------
        # Adaptive detector state
        # -------------------------------------------------------
        if suspicious:
            print(
                f"[SERVER][DETECTOR] Suspicious global behavior after round {server_round}: "
                f"loss={aggregated_loss:.4f}, acc={avg_acc:.4f}",
                flush=True,
            )
            stable_rounds_counter = 0
            current_risk_state = True
        else:
            print(
                f"[SERVER][DETECTOR]  No suspicious global behavior after round {server_round}.",
                flush=True,
            )

            if current_risk_state:
                stable_rounds_counter += 1
                if stable_rounds_counter >= STABLE_RECOVERY_ROUNDS:
                    print(
                        f"[SERVER][DETECTOR] System stable again -> returning to SAFE node scaling.",
                        flush=True,
                    )
                    current_risk_state = False
                    stable_rounds_counter = 0
            else:
                stable_rounds_counter = 0

        next_active_node_count = compute_active_node_count(
            total_nodes=len(ALL_NODE_PORTS),
            risky=current_risk_state,
        )

        # -------------------------------------------------------
        # Print trust info
        # -------------------------------------------------------
        print("[SERVER][TRUST] Current trust scores:", flush=True)
        
        # Separate into active and banned
        active_clients = []
        banned_list = []
        for cid in sorted(client_trust_score.keys(), key=lambda x: int(x) if x.isdigit() else x):
            if cid in banned_clients:
                banned_list.append(cid)
            else:
                active_clients.append(cid)
        
        if active_clients:
            print("  [ACTIVE CLIENTS]", flush=True)
            for cid in active_clients:
                print(
                    f"    Client {cid}: trust={client_trust_score[cid]:.3f}, "
                    f"selected={client_selected_count[cid]}, "
                    f"suspicious={client_suspicious_count[cid]}",
                    flush=True,
                )
        
        if banned_list:
            print("  [BANNED CLIENTS]", flush=True)
            for cid in banned_list:
                print(
                    f"    Client {cid}: trust={client_trust_score[cid]:.3f}, "
                    f"selected={client_selected_count[cid]}, "
                    f"suspicious={client_suspicious_count[cid]}",
                    flush=True,
                )

        print(
            f"\n[RESULT][ADAPTIVE-MPC] Round {server_round:02d} | "
            f"selected_clients={selected_count} | "
            f"selected_ids={selected_ids} | "
            f"active_nodes_used={len(active_node_ports)} | "
            f"ports_used={active_node_ports} | "
            f"loss={aggregated_loss:.4f} | "
            f"accuracy={avg_acc:.4f} ({avg_acc*100:.2f}%) | "
            f"round_time={total_round_time:.3f}s | "
            f"banned_so_far={len(banned_clients)}\n"
            f"         SecureAgg: total={sa_total:.3f}s "
            f"(shares_wait={sa_wait_share:.3f}s, {node_wait_str}) "
            f"=> overhead={overhead_pct:.1f}%\n"
            f"         NEXT active node count = {next_active_node_count}\n",
            flush=True,
        )

        return aggregated_loss, aggregated_metrics


def main():
    print(
        f"[SERVER] starting adaptive MPC server with max {MAX_NUM_NODES} node(s) "
        f"and TRUST-BASED client selection with BANNING...",
        flush=True,
    )
    print(
        f"[SERVER] Safe mode uses about {int(SAFE_NODE_RATIO * 100)}% of nodes "
        f"(min {MIN_SAFE_NODES}), risk mode uses about {int(RISK_NODE_RATIO * 100)}% "
        f"of nodes (min {MIN_RISK_NODES}).",
        flush=True,
    )
    print(
        f"[MTD] Banning Configuration (REALISTIC):",
        flush=True,
    )
    print(f"  - Each suspicious event reduces trust by: {TRUST_PENALTY_FACTOR}", flush=True)
    print(f"  - Trust threshold for banning: {TRUST_BAN_THRESHOLD}", flush=True)
    print(f"  - Ban requires BOTH: ≥{SUSPICIOUS_EVENTS_BAN} events AND trust < {TRUST_BAN_THRESHOLD}", flush=True)
    print(f"  - Minimum trust score: {MIN_TRUST_SCORE}", flush=True)
    print(f"  - Warmup rounds: {WARMUP_ROUNDS} (no detection during warmup)", flush=True)

    threading.Thread(target=ass_listener, daemon=True).start()

    strategy = FedAvgWithAdaptiveMPC(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=SELECTED_CLIENTS_PER_ROUND,
        min_evaluate_clients=SELECTED_CLIENTS_PER_ROUND,
        min_available_clients=SELECTED_CLIENTS_PER_ROUND,
    )

    fl.server.start_server(
        server_address=f"{SERVER_HOST}:{FL_PORT}",
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()



    