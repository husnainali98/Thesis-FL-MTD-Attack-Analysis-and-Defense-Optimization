import flwr as fl
from client_baseline import FLClient
from data_utils import load_data_for_client

def main():
    trainloader, testloader = load_data_for_client(cid=0)

    client = FLClient(
        trainloader,
        testloader,
        is_malicious=False   # NORMAL
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    main()