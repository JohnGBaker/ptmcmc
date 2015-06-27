///State and state space structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2014)

#include "states.hh"

bool boundary::enforce(double &x){
    //cout<<"boundary::enforce: testing value "<<x<<" in range "<<show()<<endl;//debug
    //check wrapping first
    if(lowertype==wrap^uppertype==wrap){//can't have just one wrap 
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
    if(uppertype==limit&&x<xmax){
      //cout<<"  Upper limit failure."<<endl;//debug
      return false;
    };
    //cout<<"  All good!."<<endl;//debug
    return true;
  };

///Show structural info
string boundary::show(){
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

bool stateSpace::enforce(valarray<double> &params){
    if(params.size()!=dim){
      cout<<"stateSpace::enforce:  Dimension error.  Expected "<<dim<<" params, but given "<<params.size()<<"."<<endl;
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
string stateSpace::show(){
    ostringstream s;
    s<<"StateSpace:(dim="<<dim<<")\n";
    for(uint i=0;i<dim;i++){
      s<<"  "<<get_name(i)<<" in "<<bounds[i].show()<<"\n";
    }
    return s.str();
};

      
void state::enforce(){
  if(!space)valid=false;
  if(!valid)return;
  valid=space->enforce(params);
  //cout<<"state::enforce:State was "<<(valid?"":"not ")<<"valid."<<endl;//debug
};

state::state(stateSpace *space,int n):space(space){
  params.resize(n,0);
  valid=false;
  //cout<<"state::state():space="<<this->space<<endl;    
  if(space){
    valid=true;
    enforce();
  }
};
state::state(stateSpace *sp, const valarray<double>&array):space(sp),params(array){
  //cout<<"state::state(stuff):space="<<this->space<<endl;
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

state state::scalar_mult(double x)const{
    //we only require that the result is valid
    state result(space,size());
    for(uint i=0;i<size();i++)result.params[i]=params[i]*x;
    result.enforce();
    return result;
  };

///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
double state::innerprod(state other)const{
    if(!(valid&&other.valid))return NAN;
    double result=0;
    if(other.size()!=size()){
      cout<<"state::innerprod: sizes mismatch.\n";
      exit(1);
    }
    for(uint i=0;i<size();i++)result+=params[i]*other.params[i];
    return result;
};

string state::get_string()const{
    ostringstream s;
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

string state::show(){
    ostringstream s;
    s<<"(\n";
    for(uint i=0;i<params.size();i++)s<<"  "<<(space?space->get_name(i):"[???]")<<" = "<<params[i]<<"\n";
    s<<")\n";
    return s.str();
};




