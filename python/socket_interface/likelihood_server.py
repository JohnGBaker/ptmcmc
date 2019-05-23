# A server that accepts connections on a socket, reads parameters from it,
# passes the parameters to a likelihood function, and sends the result back to
# the socket

import os
import socket
import struct
import math


def log_likelihood(x):
    return math.log(math.exp(-0.5 * x * x) + math.exp(-0.5 * (x - 10) * (x - 10)))


n_params = 1

socket_addr = '/tmp/likelihood.socket'

if os.path.exists(socket_addr):
    os.unlink(socket_addr)

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.bind(socket_addr)
    while True:
        print('Waiting for connection')
        s.listen(1)
        conn, addr = s.accept()
        print('Connection accepted')
        with conn:
            while True:
                in_packet = conn.recv(n_params * 8)
                if len(in_packet) < n_params * 8:
                    print('No more parameters given, closing connection')
                    break
                params = struct.unpack('{:d}d'.format(n_params), in_packet)
                llh = log_likelihood(*params)
                out_packet = struct.pack('d', llh)
                conn.sendall(out_packet)
