///Basic structure for handling checkpointing and restarting
///
///John G Baker - NASA-GSFC (2017)
#ifndef RESTART_HH
#define RESTART_HH
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace std;

///Checkpointing and restarting interface interface to be inherited and instantiated by child classes
class restartable {
  //ofstream os; (for some reason making these member seems to make swapping impossible)
  //ifstream is;
  
public:
  //restartable():os(NULL),is(NULL){};    
  ///Save whatever is needed for restart in path directory.
  virtual void checkpoint(string path){cout<<"Need to define checkpoint function for this object!"<<endl;exit(1);};
  ///Restart from data in path directory.
  virtual void restart(string path){cout<<"Need to define restart function for this object!"<<endl;exit(1);};
  ///This is for developmental testing it performs a a checkpont and restart of just this code component, without restarting the calling code.
  ///By default it does nothing.
  virtual void checkRestart(string path){
    checkpoint(path);
    restart(path);
  };
protected:
  //void openWrite(string path){
  void openWrite(ofstream &os, const string path, bool append=false)const{
    if(os.is_open()){
      cout<<"restartable::openwrite: Cannot reopen stream for checkpoint! (path='"<<path<<"')."<<endl;
      exit(1);
    }
    if(append)
      os.open(path.data(),ios::out | ios::binary| ios::app);
    else
      os.open(path.data(),ios::out | ios::binary);
    if(os.fail()){
      cout<<"restartable::openwrite: Could not open stream for checkpoint! (path='"<<path<<"')."<<endl;
    }      
  };
  //void openRead(string path){
  void openRead(ifstream &is,const string path)const{
    if(is.is_open()){
      cout<<"restartable::openread: Cannot reopen stream for restart! (path='"<<path<<"')."<<endl;
      exit(1);
    }
    is.open(path.data(),ios::in | ios::binary);
    if(is.fail()){
      cout<<"restartable::openread: Could not open stream for restart! (path='"<<path<<"')."<<endl;
      exit(1);
    }      
  };
  /*
  void close(){
    if(os.is_open()){
      os.close();
    }
    if(is.is_open()){
      is.close();
    }
    };*/
  //void writeInt(int val){
  //void writeInt(int val){
  void writeInt(ostream & os, int val)const{
    os.write(reinterpret_cast<char *>(&val),sizeof(int));
  };
  //void readInt(int &val){
  void readInt(istream & is, int &val)const{
    is.read(reinterpret_cast<char *>(&val),sizeof(int));
  };
  //void writeDouble(double val){
  void writeDouble(ostream & os, double val)const{
    os.write(reinterpret_cast<char *>(&val),sizeof(double));
  };
  //void readDouble(double &val){
  void readDouble(istream & is, double &val)const{
    is.read(reinterpret_cast<char *>(&val),sizeof(double));
  };
  void writeString(ostream & os, const string &s)const{
    size_t len=s.length();
    os.write(reinterpret_cast<char *>(&len),sizeof(size_t));
    os.write(&s[0],len);
  };
  void readString(istream & is, string &s)const{
    size_t len;
    is.read(reinterpret_cast<char *>(&len),sizeof(size_t));
    s.resize(len);
    is.read(&s[0],len);
  };
  void writeDoubleVector(ostream & os, const vector<double> &vec)const{
    size_t len=vec.size();
    os.write(reinterpret_cast<char *>(&len),sizeof(size_t));
    os.write(reinterpret_cast<const char *>(&vec[0]),len*sizeof(double));
  };
  void readDoubleVector(istream & is, vector<double> &vec)const{
    size_t len;
    is.read(reinterpret_cast<char *>(&len),sizeof(size_t));
    vec.resize(len);
    is.read(reinterpret_cast<char *>(&vec[0]),len*sizeof(double));
  };
  void writeIntVector(ostream & os, const vector<int> &vec)const{
    size_t len=vec.size();
    os.write(reinterpret_cast<char *>(&len),sizeof(size_t));
    os.write(reinterpret_cast<const char *>(&vec[0]),len*sizeof(int));
  };
  void readIntVector(istream & is, vector<int> &vec)const{
    size_t len;
    is.read(reinterpret_cast<char *>(&len),sizeof(size_t));
    vec.resize(len);
    is.read(reinterpret_cast<char *>(&vec[0]),len*sizeof(int));
  };
  
};

#endif
