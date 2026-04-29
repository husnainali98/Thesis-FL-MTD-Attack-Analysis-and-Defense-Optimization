import socket
import pickle
import threading


class MPCNode:
    def __init__(self, host: str, port: int, expected_clients: int):
        self.host = host
        self.port = port
        self.expected_clients = expected_clients

        
        self.round_state = {}
        self.lock = threading.Lock()

    def handle_client(self, conn, addr):
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        conn.close()

        msg = pickle.loads(data)

        # -----------------------------------------
        # Client sends one node-share for this round
        # -----------------------------------------
        if msg.get("type") == "share":
            r = int(msg["round"])
            share_list = msg["share"]

            with self.lock:
                if r not in self.round_state:
                    self.round_state[r] = {"count": 0, "sum": None}

                if self.round_state[r]["sum"] is None:
                    self.round_state[r]["sum"] = share_list
                else:
                    for i in range(len(share_list)):
                        self.round_state[r]["sum"][i] += share_list[i]

                self.round_state[r]["count"] += 1

                print(
                    f"[NODE:{self.port}] round={r} "
                    f"count={self.round_state[r]['count']}/{self.expected_clients}",
                    flush=True,
                )

        # -----------------------------------------
        # Server asks this node for summed share
        # -----------------------------------------
        elif msg.get("type") == "get_sum":
            r = int(msg["round"])

            with self.lock:
                st = self.round_state.get(r, None)

            reply = {"type": "sum_reply", "round": r, "ready": False, "sum": None}

            if st and st["count"] >= self.expected_clients:
                reply["ready"] = True
                reply["sum"] = st["sum"]

            rh, rp = msg["reply_host"], int(msg["reply_port"])

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((rh, rp))
                s.sendall(pickle.dumps(reply))

    def start(self):
        print(f"[NODE:{self.port}] Listening on {self.host}:{self.port} ...", flush=True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()

            while True:
                conn, addr = s.accept()
                threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True,
                ).start()