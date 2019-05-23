Using a likelihood written in a non-C language
==============================================

This directory contains an example showing how to use ptmcmc with a likelihood
function written in a non-C language, such as Python. The method consists in
having ptmcmc send the parameter vector over a Unix domain socket to a separate
process. The separate process then computes the likelihood and returns it back
to ptmcmc over the same socket.

Although this is a basic example, it can be extended to have many identical
likelihood processes, perhaps running on different machines and accepting
connections via regular network sockets, in order to parallelize the
computation.


Running
-------

First build `socket_sampler` using the Makefile in this directory.

Start `likelihood_server.py`. Note that it requires Python 3.
This opens the socket and listens for a connection.

In a separate shell, run this:
```bash
OMP_NUM_THREADS=1 ./socket_sampler --socket=/tmp/likelihood.socket --pt=10 --nsteps=200000
```
This process connects to the socket and uses ptmcmc to drive the sampling.
Forcing OpenMP to a single thread makes it faster.


Speed considerations
--------------------

Using a socket and a Python sender and receiver, I can send 10 double-precision
parameters and get a double-precision result back in ~20 μs.

Using the `socket_sampler` with a single parameter and 10 temperatures, I can
generate 2000000 samples in ~6 minutes of wall clock time, so ~18 μs per
likelihood, close to the earlier test.  Both the sampler and the server
processes use ~70% of the CPU.  In comparison, implementing the same likelihood
in the sampler process runs in ~1.5 minutes of wall clock time at 100% CPU, so
~4.5 μs per likelihood. In both tests I forced a single OpenMP thread.
