import sys
import flwr as fl
from client_baseline import FLClient
from data_utils import load_data_for_client

def main():
    attack_enabled = len(sys.argv) > 1 and sys.argv[1].lower() == "attack"

    trainloader, testloader = load_data_for_client(cid=2)

    client = FLClient(
        trainloader,
        testloader,
        is_malicious=attack_enabled,  
        attack_type="flip",
        attack_strength=2.0,
    )

    if attack_enabled:
        print("[CLIENT] Running in ATTACK mode", flush=True)
    else:
        print("[CLIENT] Running in NORMAL mode", flush=True)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    main()