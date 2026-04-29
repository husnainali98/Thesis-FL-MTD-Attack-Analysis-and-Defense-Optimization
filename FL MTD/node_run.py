import sys
from node_template import MPCNode

HOST = "127.0.0.1"
BASE_PORT = 5000


def main():
    # node_id = 0, 1, 2, ...
    node_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # node ports will be: 5000, 5001, 5002, ...
    port = BASE_PORT + node_id

    node = MPCNode(
        host=HOST,
        port=port,
    )

    node.start()


if __name__ == "__main__":
    main()