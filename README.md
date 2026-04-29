# Master's Thesis: Federated Learning with Moving Target Defense: Attack Analysis and Defense Optimization

**Author:** Husnain Ali (BCYW9R)  
**Supervisor:** Aya Khedda  
**Institution:** Eötvös Loránd University, Faculty of Informatics

---

## Overview

This repository contains the complete implementation for my Master's thesis on defending Federated Learning against sign-flipping poisoning attacks using Moving Target Defense (MTD) with Multi-Party Computation (MPC).

Three configurations are implemented and compared:

| Configuration | Folder | Purpose |
|---------------|--------|---------|
| Baseline | FL Baseline | Standard FedAvg with no defense |
| MPC | FL MPC | Secure aggregation only (privacy, no robustness) |
| MTD (Proposed) | FL MTD | Privacy and robustness (trust scoring with banning) |

---

## Key Results

After 200 rounds with 30 percent malicious clients and sign-flipping attack strength of 2.0:

| Configuration | Final Accuracy | Final Loss | Malicious Clients Banned |
|---------------|----------------|------------|--------------------------|
| Baseline | 50.67% | 5,112,840 | Not applicable |
| MPC | 45.74% | 11,067,662 | Not applicable |
| MTD (Proposed) | 96.44% | 0.13 | 5 out of 6 (83%) |

No honest clients were banned.

---

## How to Run Each Configuration

### Prerequisites

Python 3.10 or higher required.

### Installation

Install the required packages:

`pip install flwr torch torchvision numpy`

The MNIST dataset will download automatically when you run any client for the first time.

---

### 1. Baseline FL (No Defense)

#### Navigate to the baseline folder:

`cd FL Baseline`

#### Start the server in one terminal:

`python server_baseline.py`

#### Start each client in separate terminals:

`python client1.py`

`python client2.py`

`python client3.py attack`

`python client4.py attack`

`python client5.py attack`

`python client6.py`

`python client7.py`

`python client8.py`

`python client9.py`

`python client10.py`

---

### 2. MPC-based FL (Privacy Only)

##### Navigate to the MPC folder:

`cd FL MPC`

##### Start the server in one terminal:

`python server.py`

##### Start 3 MPC helper nodes in separate terminals:

`python node_run.py 0`

`python node_run.py 1`

`python node_run.py 2`

#### Start 10 clients in separate terminals:

`python client1.py`

`python client2.py`

`python client3.py attack`

`python client4.py attack`

`python client5.py attack`

`python client6.py`

`python client7.py`

`python client8.py`

`python client9.py`

`python client10.py`

---

### 3. MTD (Proposed Defense)

#### Navigate to the MTD folder:

`cd FL MTD`

#### Start the server in one terminal:

`python server.py`

#### Start three MPC helper nodes in separate terminals:

`python node_run.py 0`

`python node_run.py 1`

`python node_run.py 2`

#### Start 20 clients in separate terminals:

`python client1.py`

`python client2.py`

`python client3.py attack`

`python client4.py attack`

`python client5.py attack`

`python client6.py attack`

`python client7.py attack`

`python client8.py attack`

`python client9.py`

`python client10.py`

`python client11.py`

`python client12.py`

`python client13.py`

`python client14.py`

`python client15.py`

`python client16.py`

`python client17.py`

`python client18.py`

`python client19.py`

`python client20.py`

---

## Attack Configuration

The sign-flipping attack is configured in client_baseline.py for Baseline and client_template.py for MPC and MTD.

#### Parameters:

- attack_type = flip (Type of attack: flip, scale, or noise)
- attack_strength = 2.0 (Multiplier for attack magnitude)

#### The attack transforms an honest update delta into a poisoned update:

`delta_prime = -2.0 * delta`

---

## Repository Structure

Thesis-FL-MTD-Attack-Analysis-and-Defense-Optimization/

- FL Baseline/
  - server_baseline.py
  - client_baseline.py
  - client1.py to client10.py
  - data_utils.py
  - model_utils.py

- FL MPC/
  - server.py
  - client_template.py
  - client1.py to client10.py
  - node_template.py
  - node_run.py
  - mpc_utils.py
  - data_utils.py
  - model_utils.py

- FL MTD/
  - server.py
  - client_template.py
  - client1.py to client20.py
  - node_template.py
  - node_run.py
  - mpc_utils.py
  - data_utils.py
  - model_utils.py

- README.md

## MTD Defense Parameters

- TRUST_BAN_THRESHOLD = 0.6 (Minimum trust for banning)
- SUSPICIOUS_EVENTS_BAN = 10 (Minimum suspicious events for banning)
- TRUST_PENALTY_FACTOR = 0.7 (Trust decrease per suspicious event)
- WARMUP_ROUNDS = 8 (Rounds before detection activates)
- STABLE_RECOVERY_ROUNDS = 3 (Good rounds needed to exit risk mode)
- SUSPICIOUS_ACC_THRESHOLD = 0.80 (Accuracy below this triggers suspicion)
- SUSPICIOUS_LOSS_THRESHOLD = 2.0 (Loss above this triggers suspicion)

#### Trust score formula:

`trust = 1.0 - 0.7 * (suspicious_count / selected_count)`

`trust = max(0.1, trust)`

---

## Contact

Author: Husnain Ali  
Supervisor: Aya Khedda  
Institution: ELTE Faculty of Informatics
