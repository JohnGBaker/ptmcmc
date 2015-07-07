//Code for parsing command-line options.
//Written by John Baker/NASA-GSFC (2010-2014)
#ifndef OPTIONS_HH
#define OPTIONS_HH

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <memory>
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
  Option(const char *name, const char *info, const char* vdefault="<no default>"):name(name),info(info),value(vdefault){
    is_set= false;
    have_default=false;
    if(string(vdefault).compare("<no default>")!=0)have_default=true;
  };
};


class Options {
private:
  map<string,Option> flags;
public:
  Options(){};
  void add(Option opt){  flags[ opt.name ] = opt;};
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
      os<<"  -"<<(*i).second.name<<":\n";
      os<<"\t"<<(*i).second.info<<":\n";
    }
    return os.str();
  };
  bool parse(int & argc, char* argv[]){
    bool fail=false;
    int count=0;
    for(int i=1;i<argc;i++){
      if(argv[i][0]!='-')break;
      count++;
      string flag( & argv[i][1] );
      unsigned int pos=flag.find_first_of("=",1);
      string name=flag.substr(0,pos);
      if(flags.count(name)==0){
	cerr<<"Option '"<<name<<"' not recognized."<<endl;
	fail=true;
      } else {
	Option *opt=&flags[name];
	opt->is_set=true;
	if(pos!=(unsigned int)string::npos){
	  //cout<<"pos="<<pos<<endl;
	  opt->value=flag.substr(pos+1);
	  //cout<<"value='"<<opt->value<<"'"<<endl;
	}
	else
	  opt->value="true";
      }
    }
    argc-=count;
    for(int i=0;i<argc;i++)argv[i+1]=argv[i+1+count];
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
  void addOption(const char * name, const char * info, const char * vdefault="<no default>"){
    check_opt();opt->add(Option((prefix+name).c_str(),info,vdefault));};
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
  virtual void addOptions(Options &opts,const string &prefix_){opt=&opts,prefix=prefix_;have_options=true;};
  //set the options for processing.
  unique_ptr<istringstream> optValue(const string & name){
    check_opt();return unique_ptr<istringstream>(new istringstream(opt->value((prefix+name).c_str())));
  };
  bool optSet(const string & name){check_opt();return opt->set((prefix+name).c_str());};
};
#endif
