//Code for parsing command-line options.
//Written by John Baker/NASA-GSFC (2010-2014)
#ifndef OPTIONS_HH
#define OPTIONS_HH

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <memory>
#include  <vector>
using namespace std;


class Options;
class Option {
  friend class Options;
 private:
  string name;
  string info;
  string value;
  bool have_default;
  bool is_set;
 public:
  Option():name(""),info(""),value(""),have_default(false),is_set(false){};
  //Option(const char *name, const char *info, const char* vdefault="<no default>"):name(name),info(info),value(vdefault){
  Option(const string & name, const string & info, const string& vdefault="<no default>"):name(name),info(info),value(vdefault){
    is_set= false;
    have_default=false;
    if(string(vdefault).compare("<no default>")!=0)have_default=true;
  };
  string describe()const{
    return string(name)+"('"+info+"')="+(have_default?string(value):"<no value>");
  };
};


class Options {
private:
  map<string,Option> flags;
  bool dash_dash;//if true expect two dashes to indicate flags
public:
  Options(bool dash_dash=true):dash_dash(dash_dash){};
  void add(const Option opt){  
    if(exists(opt.name)){
      if(opt.describe().compare(flags[opt.name].describe())!=0)
	cout<<"Options::add: Warning! Attempt to re-add an option with same name but non-identical information.\n  Retaining original option ("<<flags[opt.name].describe()<<")\n  Discarding new option ("<<opt.describe()<<") is discarded."<<endl;
      return;
    }
    flags[ opt.name ] = opt;
  };
  bool exists(const string &name)const{
    return flags.count(name)>0;
  };
  bool set(const string &name, string &return_value)const{
    const Option *opt =&flags.find(name)->second;
    if(flags.count(name)==0){
      cerr<<"Options: Error no option '"<<name<<"'."<<endl;
      return false;
    }
    //opt=
    if(opt->is_set||opt->have_default){
      return_value=opt->value;
      return true;
    }
    return false;
  };
  bool set(const string &name)const{
    string dummy;
    return set(name,dummy);
  };
  string value(const string &name)const{
    string s("");
    set(name,s);
    //cout<<" value for '"<<name<<"' is '"<<s<<"'"<<endl;
    return s;
  };
  string print_usage()const{
    ostringstream os("");
    os<<"Options:\n";
    map<string,Option>::const_iterator i;
    
    for(i=flags.begin();i!=flags.end();i++){
      // os.width(20);
      ostringstream flag("");
      if(dash_dash)flag<<"  --";
      else flag<<"  -";
      flag<<(*i).second.name;
      os<<setw(24)<<left<<flag.str()<<"\t"<<(*i).second.info<<"\n";
    }
    return os.str();
  };
  bool parse(int & argc, char* argv[],bool verbose=true){
    bool fail=false;
    int count=0;
    int i=1;
    //We record any arguments that we understand as flags and remove them from the argv array
    //otherwise we leave them in place, report if verbose=true, and ultimately return "fail"
    //In this way we work either with a partial list of flags, with others to be processed later
    //or will a presumed full list with errors to be called out.
    while(i<argc){
      if(argv[i][0]!='-'||(dash_dash&&(string(argv[i]).size()<=1||argv[i][1]!='-'))) i++;
      else {
	unsigned int iword=dash_dash?2:1;
	string flag( & argv[i][iword] );
	unsigned int pos=flag.find_first_of("=",iword);
	string name=flag.substr(0,pos); 
	//cout<<i<<" processsing flag '"<<name<<"'";
	if(flags.count(name)==0){
	  //cout<<"\t...not found"<<endl;
	  if(verbose)cerr<<"Option '"<<name<<"' not recognized."<<endl;
	  fail=true;
	  i++;
	} else {
	  //cout<<"\t...found"<<endl;
	  Option *opt=&flags[name];
	  opt->is_set=true;
	  if(pos!=(unsigned int)string::npos){
	    //cout<<"pos="<<pos<<endl;
	    opt->value=flag.substr(pos+1);
	    //cout<<"value='"<<opt->value<<"'"<<endl;
	  }
	  else
	    opt->value="true";
	  for(int ic=i;ic<argc-1;ic++)argv[ic]=argv[ic+1];
	  count++;
	  argc--;
	}
      }
    }
    //for(int i=0;i<argc;i++)argv[i+1]=argv[i+1+count];
    //cout<<"Counted "<<count<<" flags."<<endl;
    //cout<<"And "<<argc<<" arguments."<<endl;
    return fail;
  };
  ///This version supports more natural memory management
  bool parse(vector<string>argv ,bool verbose=true){
    bool fail=false;
    int count=0;
    int i=1;
    //We record any arguments that we understand as flags and remove them from the argv array
    //otherwise we leave them in place, report if verbose=true, and ultimately return "fail"
    //In this way we work either with a partial list of flags, with others to be processed later
    //or will a presumed full list with errors to be called out.
    while(i<argv.size()){
      if(argv[i][0]!='-'||(dash_dash&&((argv[i]).length()<=1||argv[i][1]!='-'))) i++;
      else {
	unsigned int iword=dash_dash?2:1;
	string flag( & argv[i][iword] );
	unsigned int pos=flag.find_first_of("=",iword);
	string name=flag.substr(0,pos); 
	//cout<<i<<" processsing flag '"<<name<<"'";
	if(flags.count(name)==0){
	  //cout<<"\t...not found"<<endl;
	  if(verbose)cerr<<"Option '"<<name<<"' not recognized."<<endl;
	  fail=true;
	  i++;
	} else {
	  //cout<<"\t...found"<<endl;
	  Option *opt=&flags[name];
	  opt->is_set=true;
	  if(pos!=(unsigned int)string::npos){
	    //cout<<"pos="<<pos<<endl;
	    opt->value=flag.substr(pos+1);
	    //cout<<"value='"<<opt->value<<"'"<<endl;
	  }
	  else
	    opt->value="true";
	  argv.erase(argv.begin() + i);
	  //for(int ic=i;ic<argc-1;ic++)argv[ic]=argv[ic+1];
	  count++;
	  //argc--;
	}
      }
    }
    //for(int i=0;i<argc;i++)argv[i+1]=argv[i+1+count];
    //cout<<"Counted "<<count<<" flags."<<endl;
    //cout<<"And "<<argc<<" arguments."<<endl;
    return fail;
  };
  string report(){
    ostringstream s;
    for (auto it:flags)
      s<< " " << it.first << ':' << (it.second.is_set?it.second.value:"(not set)") << '\n';
    return s.str();
  };

};


///This is an interface for classes (objects?) to provide options that can be realized (e.g.) on the command line.
class Optioned{
private:
  string prefix;
  Options *opt;
  bool have_options;
protected:
  void copyOptioned(Optioned &other){prefix=other.prefix;opt=other.opt;have_options=other.have_options;};
  //void addOption(const char * name, const char * info, const char * vdefault="<no default>"){
  void addOption(const string & name, const string & info, const string & vdefault="<no default>"){
    check_opt();
    opt->add(Option((prefix+name),info,vdefault));};
  void check_opt(){
    if(!this){
      cout<<"Optioned::check_opt: You've called this from a null pointer."<<endl;
      exit(1);
    }
    if(!have_options){
      cout<<"Optioned::check_opt: Must call Optioned::addOptions() before using options."<<endl;
      exit(1);
    }
  };
public:
  ///Add options to the program's option list.
  Optioned(){have_options=false;};
  virtual void addOptions(Options &opts,const string &prefix_=""){opt=&opts,prefix=prefix_;have_options=true;};
  //set the options for processing.
  void optGetValue(const string & name, int &val){
    check_opt();
    string s(opt->value(prefix+name));
    istringstream(s.c_str())>>val;
  };
  void optGetValue(const string & name, double &val){
    check_opt();
    string s(opt->value(prefix+name));
    istringstream(s.c_str())>>val;
  };
  void optGetValue(const string & name, string &val){
    check_opt();
    string s(opt->value(prefix+name));
    istringstream(s.c_str())>>val;
  };
  unique_ptr<istringstream> optValue(const string & name){
    check_opt();return unique_ptr<istringstream>(new istringstream(opt->value((prefix+name).c_str())));
  };
  bool optSet(const string & name){check_opt();return opt->set((prefix+name).c_str());};
  string reportOptions(){return opt->report();};
};
#endif
