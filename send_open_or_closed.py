import socket
import time

# Configurația către cine trimitem (IP-ul fix al lui Pi 2)
IP_DESTINATIE = "192.168.1.11"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def trimite_stare(stare):
    # Trimitem flag-ul prin rețea
    sock.sendto(stare.encode('utf-8'), (IP_DESTINATIE, PORT))