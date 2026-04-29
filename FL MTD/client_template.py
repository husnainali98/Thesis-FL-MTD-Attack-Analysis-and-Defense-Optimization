import socket
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from flwr.client import Client
from flwr.common import (
    GetParametersIns, GetParametersRes,
    FitIns, FitRes,
    EvaluateIns, EvaluateRes,
    Status, Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from model_utils import Net
from mpc_utils import split_additive_multi

# -------------------------------------------------------
# Main server listener for server-side shares
# -------------------------------------------------------
FL_SERVER_ASS_HOST = "127.0.0.1"
FL_SERVER_ASS_PORT = 6000


def send_share_server(round_num: int, share_server_list: List[np.ndarray]) -> None:
    """
    Send the server share list to the main server.
    """
    msg = {
        "type": "share_server",
        "round": round_num,
        "share": share_server_list,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((FL_SERVER_ASS_HOST, FL_SERVER_ASS_PORT))
        s.sendall(pickle.dumps(msg))


def send_share_node(round_num: int, host: str, port: int, share_list: List[np.ndarray]) -> None:
    """
    Send one node-share list to one helper node.
    """
    msg = {
        "type": "share",
        "round": round_num,
        "share": share_list,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(pickle.dumps(msg))


def parse_active_node_ports(config_value: str) -> List[Tuple[str, int]]:
    """
    Convert config string like '5000,5001' into:
        [('127.0.0.1', 5000), ('127.0.0.1', 5001)]
    """
    if not config_value:
        return []

    ports: List[Tuple[str, int]] = []
    for item in config_value.split(","):
        item = item.strip()
        if item:
            ports.append(("127.0.0.1", int(item)))
    return ports


class FLClient(Client):
    def __init__(
        self,
        trainloader,
        testloader,
        is_malicious: bool = True,
        attack_type: str = "flip",
        attack_strength: float = 2.0,
    ):
        self.model = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.opt = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Attack settings
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_strength = attack_strength

    def _get_model_ndarrays(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def _set_model_ndarrays(self, nds: List[np.ndarray]):
        for p, w in zip(self.model.parameters(), nds):
            p.data = torch.tensor(w, dtype=p.data.dtype)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        nds = self._get_model_ndarrays()
        return GetParametersRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(nds),
        )

    def fit(self, ins: FitIns) -> FitRes:
        round_num = int(ins.config.get("server_round", 0))

        # ---------------------------------------------------
        # active helper nodes are received dynamically from server
        # ---------------------------------------------------
        active_node_targets = parse_active_node_ports(ins.config.get("active_node_ports", ""))
        num_active_nodes = len(active_node_targets)

        print(
            f"[CLIENT] round={round_num} received active helper nodes: {active_node_targets}",
            flush=True,
        )

        if num_active_nodes == 0:
            raise RuntimeError(
                f"[CLIENT] round={round_num}: server sent 0 active helper nodes. "
                f"MPC mode requires at least 1 active node."
            )

        # ---------------------------------------------------
        # 1) Load global model sent by Flower server
        # ---------------------------------------------------
        global_nds = parameters_to_ndarrays(ins.parameters)
        self._set_model_ndarrays(global_nds)

        # ---------------------------------------------------
        # 2) Local training
        # ---------------------------------------------------
        self.model.train()
        last_loss = 0.0

        for images, labels in self.trainloader:
            self.opt.zero_grad()
            out = self.model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            self.opt.step()
            last_loss = float(loss.item())

        # ---------------------------------------------------
        # 3) Get updated local model
        # ---------------------------------------------------
        new_nds = self._get_model_ndarrays()

        # ---------------------------------------------------
        # 4) Full-layer additive secret sharing using ONLY
        #    the active nodes sent by the server
        # ---------------------------------------------------
        share_server_list: List[np.ndarray] = []
        node_share_lists: List[List[np.ndarray]] = [[] for _ in range(num_active_nodes)]

        for i in range(len(new_nds)):
            delta = new_nds[i].astype(np.float64) - global_nds[i].astype(np.float64)

            # ---------------------------------------------------
            #  MODEL POISONING ATTACK(flip)
            # ---------------------------------------------------
            if self.is_malicious:
                if self.attack_type == "scale":
                    delta = delta * self.attack_strength
                elif self.attack_type == "flip":
                    delta = -delta * self.attack_strength
                elif self.attack_type == "noise":
                    delta = np.random.normal(
                        loc=0.0,
                        scale=self.attack_strength,
                        size=delta.shape
                    ).astype(np.float64)
                else:
                    raise ValueError(f"Unknown attack_type: {self.attack_type}")

                print(
                    f"[CLIENT][ATTACK] round={round_num}, layer={i}, "
                    f"type={self.attack_type}, strength={self.attack_strength}",
                    flush=True,
                )

            # Split into:
            #   server_share + active_node_shares = delta
            server_share, node_shares = split_additive_multi(
                delta,
                num_node_shares=num_active_nodes
            )

            # ---------------- DEBUG CHECK ----------------
            reconstructed = server_share.copy()
            for sh in node_shares:
                reconstructed = reconstructed + sh

            max_err = np.max(np.abs(reconstructed - delta))
            print(
                f"[CLIENT][DEBUG] round={round_num}, layer={i}, "
                f"active_nodes={num_active_nodes}, "
                f"max reconstruction error={max_err:.12f}",
                flush=True,
            )
            # --------------------------------------------

            share_server_list.append(server_share)

            for node_idx in range(num_active_nodes):
                node_share_lists[node_idx].append(node_shares[node_idx])

        # ---------------------------------------------------
        # 5) Send server share
        # ---------------------------------------------------
        try:
            send_share_server(round_num, share_server_list)
        except Exception as e:
            print(
                f"[CLIENT] WARNING: could not send share_server for round {round_num}: {e}",
                flush=True,
            )

        # ---------------------------------------------------
        # 6) Send shares ONLY to active nodes
        # ---------------------------------------------------
        for node_idx, (host, port) in enumerate(active_node_targets):
            try:
                send_share_node(round_num, host, port, node_share_lists[node_idx])
                print(
                    f"[CLIENT] round={round_num} sent node-share to {host}:{port}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[CLIENT] WARNING: could not send share to node {port} for round {round_num}: {e}",
                    flush=True,
                )

        # ---------------------------------------------------
        # 7) Return SAFE params to Flower
        # ---------------------------------------------------
        safe_nds = [arr.copy() for arr in global_nds]

        return FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(safe_nds),
            num_examples=len(self.trainloader.dataset),
            metrics={"loss": last_loss},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        nds = parameters_to_ndarrays(ins.parameters)
        self._set_model_ndarrays(nds)

        self.model.eval()
        correct, total = 0, 0
        loss_sum = 0.0

        with torch.no_grad():
            for images, labels in self.testloader:
                out = self.model(images)
                loss = F.cross_entropy(out, labels)
                loss_sum += float(loss.item()) * labels.size(0)

                pred = out.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = correct / total if total else 0.0
        avg_loss = loss_sum / total if total else 0.0

        return EvaluateRes(
            status=Status(code=Code.OK, message="ok"),
            loss=avg_loss,
            num_examples=total,
            metrics={"accuracy": acc},
        )