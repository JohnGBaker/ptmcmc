// Using John's PTMCMC sampler with a likelihood provided by a socket connection

#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <complex>
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include "omp.h"
#include "options.hh"
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"

using namespace std;

class socket_likelihood : public bayes_likelihood
{
    //int idx_x;
    int socket_fd;

public:
    socket_likelihood():bayes_likelihood(nullptr,nullptr,nullptr) {};

    virtual void setup(const string &socket_path) {
        haveSetup();

        // set up the socket connection
        struct ::sockaddr_un addr;

        socket_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            perror("cannot create socket");
            exit(-1);
        }

        ::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        ::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(socket_fd, (struct ::sockaddr*)&addr, sizeof(addr)) == -1) {
            perror("cannot connect to socket");
            exit(-1);
        }

        // set up parameter space
        // TODO could be configurable through the socket
        int npar = 1;
        stateSpace space(npar);
        string names[] = {"x"};
        space.set_names(names);
        space.set_bound(0, boundary(boundary::limit, boundary::limit, -100, 100));
        cout << "Parameter space:\n" << space.show() << endl;

        nativeSpace = space;
        defWorkingStateSpace(nativeSpace);
        best = state(&space,space.size());

        // configure the prior
        // TODO could be configurable through the socket
        const int uni = mixed_dist_product::uniform;
        valarray<double> centers((initializer_list<double>){0});
        valarray<double> scales((initializer_list<double>){20});
        valarray<int> types((initializer_list<int>){uni});
        setPrior(new mixed_dist_product(&nativeSpace, types, centers, scales));
    };

    void defWorkingStateSpace(const stateSpace &sp) {
        checkSetup(); //Call this assert whenever we need options to have been processed.
        //idx_x = sp.requireIndex("x");
        haveWorkingStateSpace();
    };

    int size() const { return 0; };

    double evaluate_log(state &s) {
        double result = NAN;

        // the communication must be marked as critical
        // or the multiple threads will corrupt it.
        // In principle there could be one server per thread though
        #pragma omp critical
        {
          int rc;
          valarray<double>params = s.get_params();
          double param_buf[params.size()];

          // send params via socket
          for (size_t i = 0; i < params.size(); i++) {
              param_buf[i] = params[i];
          }
          rc = params.size() * sizeof(double);
          if (::write(socket_fd, param_buf, rc) != rc) {
              if (rc > 0) {
                  perror("partial write");
              } else {
                  perror("write error");
                  exit(-1);
              }
          }

          // read back log likelihood from socket
          rc = ::read(socket_fd, &result, sizeof(double));
          if (rc != sizeof(double)) {
              perror("partial read");
              exit(1);
          }

          double post = result;
          post += nativePrior->evaluate_log(s); //May need a mechanism to check that prior is set
          if (post>best_post) {
              best_post = post;
              best = state(s);
          }
        }
        return result;
    };
};

shared_ptr<Random> globalRNG;//used for some debugging... 

int main(int argc, char*argv[])
{
  Options opt(true);
  //Create the sampler
  ptmcmc_sampler mcmc;
  bayes_sampler *s0 = &mcmc;
  //Create the model components and likelihood;
  //bayes_data *data=new GRBpop_z_only_data();
  //bayes_signal *signal=new GRBpop_one_break_z_signal();
  socket_likelihood *like = new socket_likelihood();
  
  //prep command-line options
  s0->addOptions(opt);
  //data->addOptions(opt);
  //signal->addOptions(opt);
  like->addOptions(opt);

  //Add some command more line options
  opt.add(Option("nchains", "Number of consequtive chain runs. Default 1", "1"));
  opt.add(Option("seed", "Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)", "-1"));
  opt.add(Option("precision", "Set output precision digits. (Default 13).", "13"));
  opt.add(Option("outname", "Base name for output files (Default 'mcmc_output').", "mcmc_output"));
  opt.add(Option("socket", "Path to Unix domain socket for connecting to the likelihood server.", "/tmp/likelihood.socket"));
  
  int Nlead_args = 1;

  bool parseBAD = opt.parse(argc,argv);
  if (parseBAD) {
    cout << "Usage:\n mcmc [-options=vals] " << endl;
    cout << opt.print_usage() << endl;
    return 1;
  }

  cout << "flags=\n" << opt.report() << endl;

  like->setup(opt.value("socket"));

  double seed;
  int Nchain, output_precision;
  int Nsigma = 1;
  int Nbest = 10;
  string outname;
  ostringstream ss("");
  istringstream(opt.value("nchains")) >> Nchain;
  istringstream(opt.value("seed")) >> seed;
  if (seed < 0) {
    seed = fmod(time(NULL) / 3.0e7, 1);
  }
  istringstream(opt.value("precision")) >> output_precision;
  istringstream(opt.value("outname")) >> outname;

  //report
  cout.precision(output_precision);
  cout << "\noutname = '" << outname << "'" << endl;
  cout << "seed=" << seed << endl; 
  cout << "Running on " << omp_get_max_threads() << " thread" << (omp_get_max_threads() > 1 ? "s" : "") << "." << endl;

  //Should probably move this to ptmcmc/bayesian
  ProbabilityDist::setSeed(seed);
  globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.

  //Get the space/prior for use here
  stateSpace space;
  shared_ptr<const sampleable_probability_function> prior;  
  space = *like->getObjectStateSpace();
  cout << "like.nativeSpace=\n" << space.show() << endl;
  prior=like->getObjectPrior();
  cout << "Prior is:\n" << prior->show() << endl;
  valarray<double> scales;prior->getScales(scales);

  //Read Params
  int Npar = space.size();
  cout << "Npar=" << Npar << endl;
  
  //Bayesian sampling [assuming mcmc]:
  //Set the proposal distribution
  int Ninit;
  proposal_distribution *prop = ptmcmc_sampler::new_proposal_distribution(Npar,Ninit,opt,prior.get(),&scales);
  cout << "Proposal distribution is:\n" << prop->show() << endl;
  //set up the mcmc sampler (assuming mcmc)
  //mcmc.setup(Ninit,*like,*prior,*prop,output_precision);
  mcmc.setup(*like, *prior, output_precision);
  mcmc.select_proposal();

  //Prepare for chain output
  ss << outname;
  string base = ss.str();

  for (int ic = 0; ic < Nchain; ic++) {
    bayes_sampler *s = s0->clone();
    s->initialize();
    s->run(base, ic);
    delete s;
  }

  cout << "best_post " << like->bestPost() << ", state=" << like->bestState().get_string() << endl;
  delete like;
}

