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


# ---------------- SERVER ----------------
FL_SERVER_ASS_HOST = "127.0.0.1"
FL_SERVER_ASS_PORT = 6000


# ---------------- STATIC MPC NODES ----------------
NODE_HOST = "127.0.0.1"
BASE_NODE_PORT = 5000
NUM_NODES = 3

NODE_TARGETS = [
    (NODE_HOST, BASE_NODE_PORT + i) for i in range(NUM_NODES)
]


def send_share_server(round_num: int, share_server_list):
    msg = {"type": "share_server", "round": round_num, "share": share_server_list}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((FL_SERVER_ASS_HOST, FL_SERVER_ASS_PORT))
        s.sendall(pickle.dumps(msg))


def send_share_node(round_num: int, host, port, share_list):
    msg = {"type": "share", "round": round_num, "share": share_list}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(pickle.dumps(msg))


class FLClient(Client):
    def __init__(
        self,
        trainloader,
        testloader,
        is_malicious=True,
        attack_type="flip",
        attack_strength=2.0,   
    ):
        self.model = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.opt = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_strength = attack_strength

    def _get_model_ndarrays(self):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def _set_model_ndarrays(self, nds):
        for p, w in zip(self.model.parameters(), nds):
            p.data = torch.tensor(w)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        nds = self._get_model_ndarrays()
        return GetParametersRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(nds),
        )

    def fit(self, ins: FitIns) -> FitRes:
        round_num = int(ins.config.get("server_round", 0))

        # ---------------- STATIC NODES ----------------
        active_nodes = NODE_TARGETS
        num_nodes = NUM_NODES

        # LOAD GLOBAL MODEL
        global_nds = parameters_to_ndarrays(ins.parameters)
        self._set_model_ndarrays(global_nds)

        # LOCAL TRAINING
        self.model.train()
        last_loss = 0.0

        for images, labels in self.trainloader:
            self.opt.zero_grad()
            out = self.model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            self.opt.step()
            last_loss = float(loss.item())

        new_nds = self._get_model_ndarrays()

        # SHARE STORAGE
        share_server_list = []
        node_share_lists = [[] for _ in range(num_nodes)]

        for i in range(len(new_nds)):

            delta = new_nds[i].astype(np.float64) - global_nds[i].astype(np.float64)

            # ---------- ATTACK ----------
            if self.is_malicious:
                if self.attack_type == "scale":
                    delta = delta * self.attack_strength
                elif self.attack_type == "flip":
                    delta = -delta * self.attack_strength
                elif self.attack_type == "noise":
                    delta = np.random.normal(0.0, self.attack_strength, size=delta.shape)

                print(f"[CLIENT] malicious update | round={round_num}, layer={i}")

            # SPLIT INTO SHARES
            server_share, node_shares = split_additive_multi(
                delta,
                num_node_shares=num_nodes
            )

            share_server_list.append(server_share)

            for node_idx in range(num_nodes):
                node_share_lists[node_idx].append(node_shares[node_idx])

        # SEND SHARES
        send_share_server(round_num, share_server_list)

        for node_idx, (host, port) in enumerate(active_nodes):
            send_share_node(round_num, host, port, node_share_lists[node_idx])

        # RETURN SAFE MODEL
        safe_nds = list(global_nds)

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