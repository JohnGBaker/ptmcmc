///State and state space structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2014)

#include "states.hh"

bool boundary::enforce(double &x)const{
  //cout<<"boundary::enforce: testing value "<<x<<" in range "<<show()<<endl;//debug
    //check wrapping first
  if((lowertype==wrap) ^ (uppertype==wrap)){//can't have just one wrap 
      cout<<"boundary::enforce: Inconsistent wrap."<<endl;
      return false;
    } else if (lowertype==wrap) {
      //cout<<"boundary::enforce: testing value "<<x<<" in range "<<show()<<endl;//debug
      double width=xmax-xmin;
      if(width<=0){
	cout<<"  boundary::enforce: Wrap: Negative width."<<endl;//debug
	return false;
      }
      double xt=fmod(x-xmin,width);
      if(xt<0)xt+=width;//fmod is stupid and finds the remainder after rounding *toward* zero.
      x=xmin+xt;//Note label range is open near xmax, closed to xmin
      //cout<<"  Wrap: OK."<<endl;//debug
      return true;
    } 
    //next check reflection
    if (lowertype==reflect&&uppertype==reflect){
      //double reflection is like wrapping, then folding
      double halfwidth=xmax-xmin;
      if(halfwidth<=0){
	return false;
	cout<<"boundary::enforce:  DoubleReflect: Negative width."<<endl;//debug
      }
      double width=2*halfwidth;
      double xt=fmod(x-xmin,width);
      if(xt<0)xt+=width;
      if(xt>=halfwidth)xt=halfwidth-xt;
      x=xmin+xt;
      cout<<"  DoubleReflect hasn't been tested...."<<endl;//debug
      return true;
    }
    if(lowertype==reflect&&x<xmin)x=xmin+(xmin-x);
    else if(uppertype==reflect&&x>xmax)x=xmax-(x-xmax);
    if(lowertype==limit&&x<xmin){
      //cout<<"  Lower limit failure."<<endl;//debug
      return false;
    }
    if(uppertype==limit&&x>xmax){
      //cout<<"  Upper limit failure."<<endl;//debug
      return false;
    };
    //cout<<"  All good!."<<endl;//debug
    return true;
  };

///Show structural info
string boundary::show()const{
  ostringstream s;
  if(lowertype==wrap)s<<"w["<<xmin<<","<<xmax<<")w";
  else {
    if(lowertype==reflect)s<<"R[";
    else if(lowertype==limit)s<<"[";
    else s<<"(";
    s<<xmin<<","<<xmax;
    if(uppertype==reflect)s<<"]R";
    else if(uppertype==limit)s<<"]";
    else s<<")";
  }
  return s.str();
};

/// State space class allows reference to overall limits and structure of the space.
/// Can provide support for boundary conditions, maybe special flows that can be used in proposals...
/// Should inner product be defined here?  probably...
/// Note: The 'limit' bound is currently problematical with state arithmetic.  If there is a 'limit' then
/// the state space is not complete wrt scalar multiplication and state addition, as the result may
/// exceed the limit. Resolutions could be to A) use 'limit' only if out-of-range values really don't even
/// make sense as differentials, etc.  Or B) ignore result of enforce in these operations.  The latter
/// is probably best, but will probably not be backward compatible. If such chages are to be made, then
/// we might want to move innerproduct (at minimum) to stateSpace and make some improvements with the
/// wrap dimensions (so that |A-B| = |B-A|, for eg).
bool stateSpace::enforce(valarray<double> &params)const{
    if(params.size()!=dim){
      cout<<"stateSpace::enforce:  Dimension error.  Expected "<<dim<<" params, but given "<<params.size()<<" for stateSpace="<<show()<<"."<<endl;
      exit(1);
    }
    for(uint i=0;i<dim;i++){
      //cout<<"stateSpace::enforce: testing parameter "<<i<<endl;//debug
      if(!bounds[i].enforce(params[i])){
	//cout<<"        FAILED."<<endl;
	return false;
      }
    }
    //cout<<"        PASSED."<<endl;
    return true;
};

///Show structural info
string stateSpace::show()const{
    ostringstream s;
    s<<"StateSpace:(dim="<<dim<<")\n";
    for(uint i=0;i<dim;i++){
      s<<"  "<<get_name(i)<<" in "<<bounds[i].show()<<"\n";
    }
    if(potentialSyms.size()>0){
      s<<"  Potential Symmetries:"<<endl;
      for(auto sym:potentialSyms){
	s<<"    "<<sym.get_label()<<endl;
      }
    }
    return s.str();
};

bool stateSpace::contains(const stateSpace &other){
  bool result=true;
  for(auto name: other.names)result=result and get_index(name)>=0;
  return result;
};

void stateSpace::attach(const stateSpace &other){
    if((!have_names&&dim!=0)||(!other.have_names&&other.dim!=0)){
      cout<<"stateSpace::attach: Warning attaching stateSpace without parameter names."<<endl;
      have_names=false;
      cout<<"have_names="<<have_names<<" dim="<<dim<<endl;
      cout<<"other:have_names="<<other.have_names<<" dim="<<other.dim<<endl;
    } else if(dim>0||other.dim>0)have_names=true;
    for(int i=0;i<other.dim;i++){
      bounds.push_back(other.bounds[i]);
      if(have_names){
	if(index.count(other.names[i])>0){
	  cout<<"stateSpace::attach: Attempted to attach a stateSpace with an identical name '"<<other.names[i]<<"'!"<<endl;
	  cout<<show()<<endl;
	  exit(1);
	}
	names.push_back(other.names[i]);
	index[other.names[i]]=dim;
      }
      dim+=1;
    }
    for(auto sym : other.potentialSyms)potentialSyms.push_back(sym);
  };

bool stateSpace::addSymmetry(stateSpaceInvolution &involution){
  if(contains(*involution.domainSpace)){
    potentialSyms.push_back(involution);
    return true;
  } else {
    cout<<"StateSpace::addSymmetry: Warning involution's domain is not a subspace of this space."<<endl;
    return false;
  }
};


      
void state::enforce(){
  if(!space)valid=false;
  if(!valid)return;
  valid=space->enforce(params);
  //cout<<"state::enforce:State was "<<(valid?"":"not ")<<"valid."<<endl;//debug
};

state::state(const stateSpace *space,int n):space(space){
  params.resize(n,0);
  valid=false;
  //cout<<"state::state():space="<<this->space<<endl;    
  if(space){
    valid=true;
    enforce();
  }
};
///build state from provided valarray params
state::state(const stateSpace *sp, const valarray<double>&array):space(sp),params(array){
  valid=false;
  if(space)valid=true;
  enforce();
};
//build state from provided vector params
state::state(const stateSpace *sp, const vector<double>&array):space(sp){
  int n=array.size();
  params.resize(n,0);
  for(int i=0;i<n;i++)params[i]=array[i];
  valid=false;
  if(space)valid=true;
  enforce();
};

//some algorithms rely on using the states as a vector space
state state::add(const state &other)const{
    //here we only require that the result is valid.
    state result(space,size());
    if(other.size()!=size()){
      cout<<"state::add: Sizes mismatch. ("<<size()<<"!="<<other.size()<<")\n";
      exit(1);
    }
    for(uint i=0;i<size();i++)result.params[i]=params[i]+other.params[i];
    result.enforce();
    return result;
  };

///Compute the squared distance between two states.  Should account for wrapped dimensions.
double state::dist2(const state &other)const{
  double result=0;
  //stateSpaces for the two states must match for a reasonable result, but we don't fully check this!
  if(other.size()!=size()){
    cout<<"state::diff: Sizes mismatch. ("<<size()<<"!="<<other.size()<<")\n";
    exit(1);
  }
  for(size_t i=0;i<size();i++){
    double xa=params[i],xb=other.params[i];
    double diff = fabs(xb-xa);
    boundary b=space->get_bound(i);
    if(b.isWrapped()){
      //We wind both points half the wrap distance and enforce the wrap to get the alternative distance
      double xmin,xmax,altdiff,halfwrap;
      b.getDomainLimits(xmin,xmax);
      halfwrap=(xmax-xmin)/2;
      xa-=halfwrap;
      xb-=halfwrap;
      b.enforce(xa);
      b.enforce(xb);
      altdiff=fabs(xb-xa);
      if(altdiff<diff)diff=altdiff;
    }
    result+=diff*diff;
  }
  return result;
}
      
      

state state::scalar_mult(double x)const{
    //we only require that the result is valid
    state result(space,size());
    for(uint i=0;i<size();i++)result.params[i]=params[i]*x;
    result.enforce();
    return result;
  };

///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
double state::innerprod(state other,bool constrained)const{
    if(constrained && !(valid&&other.valid))return NAN;
    double result=0;
    if(other.size()!=size()){
      cout<<"state::innerprod: sizes mismatch.\n";
      exit(1);
    }
    for(uint i=0;i<size();i++)result+=params[i]*other.params[i];
    return result;
};

string state::get_string(int prec)const{
    ostringstream s;
    if(prec>-1)s.precision(prec);
    int n=params.size();
    //cout<<"n="<<n<<endl;//debug
    for(int i=0;i<n-1;i++){
      //cout<<"i="<<i<<endl;
      s<<params[i]<<", ";
    }
    if(n>0)s<<params[n-1];
    else s<<"<empty>";
    if(!valid)s<<" [INVALID]";
    return s.str();
};

string state::show()const{
    ostringstream s;
    s<<"(\n";
    for(uint i=0;i<params.size();i++)s<<"  "<<(space?space->get_name(i):"[???]")<<" = "<<params[i]<<"\n";
    s<<")\n";
    return s.str();
};




