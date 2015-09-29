//bambi.cc
//Code for interfacing with BAMBI Multinest sampler.
//
#include "bayesian.hh"

int bambi_sampler::run(const string & base, int ic=0){
  if(!have_init){
    cout<<"bambi_sampler::run.  Must call initialize() before running!"<<endl;
    exit(1);
  }

  cout<<"\nRunning BAMBI... "<<ic<<endl;


};

int bambi_sampler::dump_posterior_samples(int &nSamples, int &nPar, double **posterior, double **paramConstr){
  int np=space->size();
  ostringstream ss;
  ss<<base<<"_c"<<ic<<".dat";
  ofstream os(ss.str().c_str(),mode);
  //os.precision(output_precision);
  
  //In ptmcmc we continue to append as the chain evolves.  We might mock up something simlar by appending new sets of posterior samples
  //but the statistics on that might be confusing.  Only the last would be really correct....
  //ios_base::openmode mode=ios_base::out;
  //if(ic>0)mode=mode|ios_base::app;
  
  os<<"#eval: log(posterior) log(likelihood) unused(0) unused(0): ";
  for(int i=0;i<np;i++)os<<space->get_name(i)<<" ";
  os<<endl;
  for(int i=0;i<nSamples;i++){
    valarray<double> pars(Npar);
    for(int j=0;j<Npar;j++)pars[j] = posterior[0][j * nSamples + i];
    state s(space,pars);
    double llike= posterior[0][(Npars+1) * nSamples + i];
    double lprior=prior->evaluate_log(s); 
    os<<i<<" "<<llike+lprior<<" "<<llike<<" "<<0<<" "<<0<<": ";
    for(int j=0;j<np;j++)os<<pars[j]<<" ";
    os<<endl;
  }
};

///Convert from unit hypercube to the physical params that likelihood fn needs.
///
///For ( eg Gaussian or uniform ) 1-D priors we can use the prior's CDF to convert to a uniform prior.
///For the conversion from unit interval parameter to the nominal parameter is then the inverse CDF.
///
///BAMBI comment:
///Convert from unit hypercube to the physical params that likelihood fn needs.
///ndim physical params
///npar>-ndim tracked params.
///for npar>ndim, these are derived or input params.
///values between 0 and 1 for first npar of cub vals.
///convert to phys in place, code will save the results.
///Output here will ignore elems beyond ndim
///Also gets called for setting training data for NNs
void bambi_sampler::getphysparams(double *Cube, int &ndim, int &nPar)const
{
  state phys_state=prior->invcdf(state(*Cube));
  for(int i = 0; i < ndim; i++) Cube[i] = phys_state.get_param(i);
}



//**********************************************************************************************************************
//  Code below here is modified from the eggbox.cc example included with BAMBI
//**********************************************************************************************************************


/******************************************** getphysparams routine ****************************************************/

///Convert from unit hypercube to the physical params that likelihood fn needs.
///ndim physical params
///npar>-ndim tracked params.
///for npar>ndim, these are derived or input params.
///values between 0 and 1 for first npar of cub vals.
///convert to phys in place, code will save the results.
///Output here will ignore elems beyond ndim
///Also gets called for setting training data for NNs
void getphysparams(double *Cube, int &ndim, int &nPar, void *context)
{
  bambi_sampler * bbs=dynamic_cast<bambi_sampler>(context);
  bbs->getphysparams(Cube, ndim, nPar);
}

/******************************************** getallparams routine ****************************************************/
///Here can add dervied params, or re-sort the params as desired.
void getallparams(double *Cube, int &ndim, int &nPar, void *context)
{
	getphysparams(Cube,ndim,nPar,context);
}

/******************************************** loglikelihood routine ****************************************************/

// Input arguments
// ndim 						= dimensionality (total number of free parameters) of the problem
// npars 						= total number of free plus derived parameters
//
// Input/Output arguments
// Cube[npars] 						= on entry has the ndim parameters in unit-hypercube
//	 						on exit, the physical parameters plus copy any derived parameters you want to store with the free parameters
//	 
// Output arguments
// lnew 						= loglikelihood

void getLogLike(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
  bambi_sampler * bbs=dynamic_cast<bambi_sampler>(context);
  getallparams(Cube,ndim,npars,context);
  state phys_state=prior->invcdf(state(*Cube));
  lnew=llike->evaluate_log(phys_state);
}


/************************************************* dumper routine ******************************************************/

// The dumper routine will be called every updInt*10 iterations
// MultiNest does not need to the user to do anything. User can use the arguments in whichever way he/she wants
//
//
// Arguments:
//
// nSamples 						= total number of samples in posterior distribution
// nlive 						= total number of live points
// nPar 						= total number of parameters (free + derived)
// physLive[1][nlive * (nPar + 1)] 			= 2D array containing the last set of live points (physical parameters plus derived parameters) along with their loglikelihood values
// posterior[1][nSamples * (nPar + 2)] 			= posterior distribution containing nSamples points. Each sample has nPar parameters (physical + derived) along with the their loglike value & posterior probability
// paramConstr[1][4*nPar]:
// paramConstr[0][0] to paramConstr[0][nPar - 1] 	= mean values of the parameters
// paramConstr[0][nPar] to paramConstr[0][2*nPar - 1] 	= standard deviation of the parameters
// paramConstr[0][nPar*2] to paramConstr[0][3*nPar - 1] = best-fit (maxlike) parameters
// paramConstr[0][nPar*4] to paramConstr[0][4*nPar - 1] = MAP (maximum-a-posteriori) parameters
// maxLogLike						= maximum loglikelihood value
// logZ							= log evidence value
// logZerr						= error on log evidence value
// context						void pointer, any additional information

///example Phil prints a file with evidence and error on evidence.
void dumper(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &logZerr, void *context)
{
  // convert the 2D Fortran arrays to C++ arrays
  // the posterior distribution
  // postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
  //int i, j;
  //double postdist[nSamples][nPar + 2];
  //for( i = 0; i < nPar + 2; i++ )
  //  for( j = 0; j < nSamples; j++ )
  //    postdist[j][i] = posterior[0][i * nSamples + j];
  // last set of live points
  // pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
  //double pLivePts[nlive][nPar + 1];
  //for( i = 0; i < nPar + 1; i++ )
  //  for( j = 0; j < nlive; j++ )
  //    pLivePts[j][i] = physLive[0][i * nlive + j];
  bambi_sampler * bbs=dynamic_cast<bambi_sampler>(context);
  cout<<"BAMBI Update:"<<endl;
  cout<<"mean values of the parameters:"<<endl;
  for(int i=0;i<np;i++)cout<<"   "<<bbs->getParName(i)<<" = "<<paramConstr[0][i];
  cout<<"standard devations of the parameters:"<<endl;
  for(int i=0;i<np;i++)cout<<"   "<<bbs->getParName(i)<<" = "<<paramConstr[0][nPar+i];
  cout<<"Maximum likelihood point: log-like="<<maxLogLike<<endl;
  for(int i=0;i<np;i++)cout<<"   "<<bbs->getParName(i)<<" = "<<paramConstr[0][2*nPar+i];
  cout<<"MAP (Maximum-a-posteriori) parameters:"<<endl;
  for(int i=0;i<np;i++)cout<<"   "<<bbs->getParName(i)<<" = "<<paramConstr[0][3*nPar+i];
  cout<<"Evidence = "<<logZ<<" +/- "<<logZerr<<endl;
};

/***********************************************************************************************************************/




/************************************************** Main program *******************************************************/



int main(int argc, char *argv[])
{
#ifdef PARALLEL
 	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
	
	// set the MultiNest sampling parameters
	
	///These can be command line params...	
	int mmodal = 0;					// do mode separation?
	
	int ceff = 0;					// run in constant efficiency mode?
	
	int nlive = 4000;				// number of live points
	
	double efr = 0.8;				// set the required efficiency
	
	double tol = 0.5;				// tol, defines the stopping criteria
	
	int ndims = 2;					// dimensionality (no. of free parameters)
	
	int nPar = 2;					// total no. of parameters including free & derived parameters
	
	int nClsPar = 2;				// no. of parameters to do mode separation on
	
	int updInt = 4000;				// after how many iterations feedback is required & the output files should be updated
							// note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	
	double Ztol = -1E90;				// all the modes with logZ < Ztol are ignored
	
	int maxModes = 1;				// expected max no. of modes (used only for memory allocation)
	
	int pWrap[ndims];				// which parameters to have periodic boundary conditions?
	for(int i = 0; i < ndims; i++) pWrap[i] = 0;
	
	strcpy(root, "chains/eggboxC++_");			// root for output files
	strcpy(networkinputs, "example_eggbox_C++/eggbox_net.inp");			// file with input parameters for network training
	
	int seed = -1;					// random no. generator seed, if < 0 then take the seed from system clock
	
	int fb = 1;					// need feedback on standard output?
	
	resume = 0;					// resume from a previous job?
	
	int outfile = 1;				// write output files?
	
	int initMPI = 0;				// initialize MPI routines?, relevant only if compiling with MPI
							// set it to F if you want your main program to handle MPI initialization
	
	logZero = -1E90;				// points with loglike < logZero will be ignored by MultiNest
	
	int maxiter = 0;				// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
							// has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	
	void *context = 0;				// not required by MultiNest, any additional information user wants to pass
	
	doBAMBI = 1;					// BAMBI?

	useNN = 0;
	
	// calling MultiNest

	nested::run(mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, pWrap, fb, resume, outfile, initMPI,
	logZero, maxiter, LogLike, dumper, bambi, context);
	
#ifdef PARALLEL
 	MPI_Finalize();
#endif
}

/***********************************************************************************************************************/
