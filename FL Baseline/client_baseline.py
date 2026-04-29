from typing import List

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

        # Attack config
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_strength = attack_strength

    # -------------------------------
    # Model helpers
    # -------------------------------
    def _get_model_ndarrays(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def _set_model_ndarrays(self, nds: List[np.ndarray]):
        for p, w in zip(self.model.parameters(), nds):
            p.data = torch.tensor(w, dtype=p.data.dtype)

    # -------------------------------
    # Flower methods
    # -------------------------------
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        nds = self._get_model_ndarrays()
        return GetParametersRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(nds),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Load global model
        global_nds = parameters_to_ndarrays(ins.parameters)
        self._set_model_ndarrays(global_nds)

        # -------------------------------
        # Local training
        # -------------------------------
        self.model.train()
        last_loss = 0.0

        for images, labels in self.trainloader:
            self.opt.zero_grad()
            out = self.model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            self.opt.step()
            last_loss = float(loss.item())

        # Get updated model
        new_nds = self._get_model_ndarrays()

        # -------------------------------
        #  SAME ATTACK AS MTD (DELTA-BASED)
        # -------------------------------
        for i in range(len(new_nds)):
            delta = new_nds[i].astype(np.float64) - global_nds[i].astype(np.float64)

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
                    f"[BASELINE CLIENT][ATTACK] layer={i}, "
                    f"type={self.attack_type}, strength={self.attack_strength}",
                    flush=True,
                )

            # Apply update
            new_nds[i] = global_nds[i] + delta

        # Return updated model
        return FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(new_nds),
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