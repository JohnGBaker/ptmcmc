# A server that accepts connections on a socket, reads parameters from it,
# passes the parameters to a likelihood function, and sends the result back to
# the socket

import os
import socket
import struct
import math


# log-likelihood function (modify according to problem)

def log_likelihood(x):
    return math.log(math.exp(-0.5 * x * x) + math.exp(-0.5 * (x - 10) * (x - 10)))


# definition of parameters and priors:
# name, boundary min, boundary max, prior center, prior scale
params = [('x', -100, 100, 0, 20)]
n_params = len(params)

# path to Unix domain socket
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
            # send parameter definition
            out_packet = struct.pack('I', n_params)
            for name, min_bound, max_bound, center, scale in params:
                out_packet += struct.pack('128sdddd', name.encode('ascii'),
                                          min_bound, max_bound, center, scale)
            conn.sendall(out_packet)

            while True:
                in_packet = conn.recv(n_params * 8)
                if len(in_packet) < n_params * 8:
                    print('No more parameters given, closing connection')
                    break
                params = struct.unpack('{:d}d'.format(n_params), in_packet)
                llh = log_likelihood(*params)
                out_packet = struct.pack('d', llh)
                conn.sendall(out_packet)
