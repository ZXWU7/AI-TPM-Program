import socket


def start_server(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen(1)
    print(f"Server listening on {ip}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        data = client_socket.recv(1024)
        if data:
            print(f"Received: {data.decode()}")
            client_socket.sendall(data)  # Echo back the received data
        client_socket.close()


if __name__ == "__main__":
    IP = "127.0.0.1"
    PORT = 6501
    start_server(IP, PORT)
