///State and state space structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2015)

#ifndef PTMCMC_STATES_HH
#define PTMCMC_STATES_HH
#include <map>
#include <valarray>
#include <vector>
#include <sstream>
//#include <cmath>
#include <iostream>
//#include <utility>
//#include <memory>
#include "restart.hh"

using namespace std;

typedef unsigned int uint;


//********* BASE CLASSES *************
/// A class for specifying 1-D boundary info. Used for stateSpace definition.
class boundary {
  int lowertype;
  int uppertype;
  double xmin;
  double xmax;
public:
  static const int open=0;
  static const int limit=1;
  static const int reflect=2;
  static const int wrap=3;
  boundary(int lowertype=open,int uppertype=open,double min=-INFINITY,double max=INFINITY):lowertype(lowertype),uppertype(uppertype),xmin(min),xmax(max){};
  ///enforce boundary condition. If consistent enforcement was achieved return true.
  bool enforce(double &x)const;
  ///Show structural info
  string show()const;
  void getDomainLimits(double &xmin_, double &xmax_)const{xmin_=xmin;xmax_=xmax;};
};

/// State space class allows reference to overall limits and structure of the space.
/// Can provide support for boundary conditions, maybe special flows that can be used in proposals...
/// Should inner product be defined here?  probably...
class stateSpace {
  int dim;
  vector<boundary> bounds;
  vector<string> names;
  map<string,int> index;
  bool have_names;
public:
  stateSpace(int dim=0):dim(dim){
    bounds.resize(dim,boundary());//default to default boundaries (ie allow all reals)
    have_names=false;    
  };
  int size()const{return dim;};
  void set_bound(int i, const boundary &b){
    if(i<dim)bounds[i]=b;
    else{
      cout<<"stateSpace::set_bound: Index out of range, "<<i<<">="<<dim<<"."<<endl;
      exit(1);
    }
  };      
  boundary get_bound(int i)const{
    if(i<0||i>=dim){
      cout<<"stateSpace::set_bound: Index out of range, "<<i<<">="<<dim<<"."<<endl;
      exit(1);
    }
    return bounds[i];
  };      
  void set_names(string stringnames[]){
    names.resize(dim,"");
    for(uint i=0;i<dim;i++){
      names[i]=stringnames[i];
      index[names[i]]=i;
    }
    have_names=true;
  };
  void set_names(vector<string> &stringnames){
    if(stringnames.size()<dim){
      cout<<"stateSpace::set_names: Vector of param names is too short. Quitting."<<endl;
      exit(-1);
    }
    names.resize(dim,"");
    for(uint i=0;i<dim;i++){
      names[i]=stringnames[i];
      index[names[i]]=i;
    }
    have_names=true;
  };
  string get_name(int i)const {
    if(have_names&&i<dim)return names[i];
    else return "[unnamed]";
  };      
  int get_index(const string &name)const{
    //cout<<"get_index: name="<<name<<endl;
    //cout<<"index.size="<<index.size()<<endl;
    if(have_names&&index.count(name)>0){
      //cout<<"index.size()="<<index.size()<<endl;
      //cout<<"index.at(name)="<<index.at(name)<<endl;
      return index.at(name);
    }
    else return -1;
  };      
  ///This is the same as get_index but fails with an error message if not found.
  int requireIndex(const string &name)const{
    //cout<<"requireIndex:name="<<name<<endl;
    int idx=get_index(name);
    //cout<<"requireIndex:idx="<<idx<<endl;
    if(idx<0){
      //cout<<"Oops:name="<<name<<", have_names="<<have_names<<endl;     
      cout<<"stateSpace::checkNames(): Name '"<<name<<"' not found in state:"<<show()<<endl;
      exit(1);
    }
    return idx;
  };     
  bool enforce(valarray<double> &params)const;
  ///Show structural info
  string show()const;
  ///replace a parameter in one space dimension
  void replaceParam(int i, const string &newname, const boundary &newbound){
    if(!have_names){
      cout<<"stateSpaceTransform::replaceParam: Cannot apply transform to a space for which names are not defined."<<endl;
      exit(1);
    }
    index.erase(names[i]);
    index[newname]=i;
    names[i]=newname;
    bounds[i]=newbound;
  };
  //join this space to another space
  void attach(const stateSpace &other){
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
  };
};
      
///Class for holding and manipulating bayesian parameter states
class state :restartable {
  bool valid;
  const stateSpace *space;
  valarray<double> params;
  void enforce();
public:
  //Need assignment operator since default valarray assignment is problematic
  const state& operator=(const state model){space=model.space,valid=model.valid,params.resize(model.size(),0);params=model.params;return *this;};
  state(const stateSpace *space=nullptr,int n=0);
  state(const stateSpace *sp, const valarray<double>&array);
  state(const stateSpace *sp, const vector<double>&array);
  void checkpoint(string path)override{//probably won't want to use this; woudl need a separate dir for each state...
    ostringstream ss;
    ss<<path<<"/state.cp";
    ofstream os;
    openWrite(os,ss.str());
    writeString(os,save_string());
  };
  void restart(string path)override{//probably won't want to use this; woudl need a separate dir for each state...
    ostringstream ss;
    ss<<path<<"/state.cp";
    ifstream os;
    openRead(os,ss.str());
    string s;
    readString(os,s);
    restore_string(s);
  };
  string save_string()const{
    //We assume *space is known independently
    ostringstream oss;
    writeInt(oss,valid);
    writeInt(oss,params.size());
    for(int i=0;i<params.size();i++)writeDouble(oss,params[i]);
    return oss.str();
  };
  void restore_string(const string s){
    //We assume *space is known independently
    istringstream iss(s);
    int n;
    readInt(iss,n);valid=n;
    readInt(iss,n);
    params.resize(n,0);
    for(int i=0;i<params.size();i++)readDouble(iss,params[i]);
  };
  int size()const{return params.size();}
  //some algorithms rely on using the states as a vector space
  virtual state add(const state &other)const;
  virtual state scalar_mult(double x)const;
  ///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
  virtual double innerprod(state other)const;
  virtual string get_string(int prec=-1)const;
  virtual void get_params_array(valarray<double> &outarray)const{
    //outarray.resize(params.size());
    //for(size_t i=0;i<size();i++)outarray[i]=params[i];
      outarray=std::move(params);
    return;
  }
  virtual valarray<double> get_params()const{return params;};
  virtual vector<double> get_params_vector()const{vector<double> v;v.assign(begin(params),end(params));return v;};
  double get_param(const int i)const{return params[i];};
  double get_param(const string name)const{if(space)return params[space->requireIndex(name)];else{cout<<"state::get_param(name):Need a stateSpace to get param by name."<<endl;exit(1);};};
  void set_param(const int i,const double v){params[i]=v;};
  const stateSpace * getSpace()const{return space;};
  ///Show param info
  string show()const;
  bool invalid()const{return !valid;};
};

///Interface class for objects using stateSpace to describe their content
class stateSpaceInterface {
  bool have_working_space;
public:
  stateSpaceInterface(){have_working_space=0;};
  virtual void defWorkingStateSpace(const stateSpace &sp)=0;
  virtual void haveWorkingStateSpace(){have_working_space=true;};
  //virtual stateSpace getObjectStateSpace()const{return stateSpace();};
  virtual const stateSpace* getObjectStateSpace()const=0;
  void checkWorkingStateSpace()const{
    if(!have_working_space){
      cout<<"stateSpaceTransform::checkWorkingSpace: [this="<<this<<"] Must define working space (defWorkingStateSpace) before applying it."<<endl;
      exit(1);
    }
  };
};

///Base class for stateSpace transformations
class stateSpaceTransform {
protected:
  bool have_working_space;
public:
  stateSpaceTransform(){have_working_space=false;};
  virtual stateSpace transform(const stateSpace &sp)=0;
  virtual stateSpace inverse_transform(const stateSpace &sp){cout<<"stateSpaceTransform:No inverse transform available"<<endl;exit(1);};
  virtual double jacobian(const stateSpace &sp){cout<<"stateSpaceTransform:No Jacobian available"<<endl;exit(1);};
  virtual state transformState(const state &s)const=0;
  virtual void defWorkingStateSpace(const stateSpace &sp)=0;
};

///Dynamically generated one-to-one generic state space transform class
///
///Perhaps a way to do this dynamically, rather than hand-coding each one as a separate derived class..
///This is just an idea, not yet developed for use....
class stateSpaceTransform1D : public stateSpaceTransform {
  string in;
  string out;
  double (*func)(double);
  boundary bound;
  bool have_bound;
  int idx_inout;
  public:
  stateSpaceTransform1D(string in="",string out="",double (*func)(double)=[](double a){return a;}):in(in),out(out),func(func){have_bound=false;};
  void set_bound(const boundary &b){
    have_bound=true;
    bound=b;
  }
  virtual stateSpace transform(const stateSpace &sp){
    
    stateSpace outsp=sp;
    int ind=sp.get_index(in);
    if(ind<0){
      cout<<"stateSpaceTransform1D::transform: Parameter name '"<<in<<"' not found."<<endl;
      exit(1);
    } else {
      boundary b;
      if(have_bound)b=bound;
      else b=sp.get_bound(ind);
      outsp.replaceParam(ind, out, b);
    }
    defWorkingStateSpace(sp);
    return outsp;    
  };
  virtual state transformState(const state &s)const{
    if(!have_working_space){
      cout<<"stateSpaceTransform1D::transformParams: Must call deWorkingStateSpace before transformParams."<<endl;
      exit(1);
    }
    state st=s;
    st.set_param(idx_inout,(*func)(s.get_param(idx_inout)));
    return st;
  };
  virtual void defWorkingStateSpace(const stateSpace &sp){
    int ind=sp.get_index(in);
    if(ind<0){
      cout<<"stateSpaceTransform1D::defWorkingStateSpace: Parameter name '"<<in<<"' not found."<<endl;
      exit(1);
    } else {
      idx_inout=ind;
      have_working_space=true;
    }
  };
};

///Dynamically generated n-to-n generic state space transform class
///
///Perhaps a way to do this dynamically, rather than hand-coding each one as a separate derived class..
///This is just an idea, not yet developed for use....
class stateSpaceTransformND : public stateSpaceTransform {
  int dim;
  vector<string> ins;
  vector<string> outs;
  vector<double> (*func)(vector<double>&a);
  vector<boundary> bounds;
  bool have_bounds;
  int idxs[2];
  public:
  stateSpaceTransformND(int dim=0,vector<string> ins={},vector<string> outs={},vector<double> (*func)(vector<double>&a)=[](vector<double> &a){return a;}):ins(ins),outs(outs),func(func),dim(dim){
    have_bounds=false;
    if(ins.size()!=dim||outs.size()!=dim){
      cout<<"stateSpaceTransformND::stateSpaceTransformND: Dimensions do not match."<<endl;
      exit(1);
    }
  };
  void set_bounds(const vector<boundary> &b){
    have_bounds=true;
    bounds=b;
  }
  virtual stateSpace transform(const stateSpace &sp){
    stateSpace outsp=sp;
    for(int i=0;i<dim;i++){
      int ind=sp.get_index(ins[i]);
      if(ind<0){
	cout<<"stateSpaceTransformND::transform: Parameter name '"<<ins[i]<<"' not found."<<endl;
	exit(1);
      } else {
	boundary b;
	if(have_bounds)b=bounds[i];
	else b=sp.get_bound(ind);
	outsp.replaceParam(ind, outs[i], b);
      }
    }
    defWorkingStateSpace(sp);
    return outsp;    
  };
  virtual state transformState(const state &s)const{
    if(!have_working_space){
      cout<<"stateSpaceTransformND::transformParams: Must call defWorkingStateSpace before transformParams."<<endl;
      exit(1);
    }
    state st=s;
    vector<double>inpars(2),outpars(2);
    for(int i=0;i<dim;i++)inpars[i]=s.get_param(idxs[i]);
    outpars=(*func)(inpars);
    for(int i=0;i<dim;i++)st.set_param(idxs[i],outpars[i]);
    return st;
  };
  virtual void defWorkingStateSpace(const stateSpace &sp){
    for(int i=0;i<dim;i++){
      int ind=sp.get_index(ins[i]);
      if(ind<0){
	cout<<"stateSpaceTransformND::defWorkingStateSpace: Parameter name '"<<ins[i]<<"' not found."<<endl;
	exit(1);
      } else {
	idxs[i]=ind;
      }
    }
    have_working_space=true;
  };
};

#endif

