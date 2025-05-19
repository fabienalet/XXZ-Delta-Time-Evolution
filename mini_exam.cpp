#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <numeric>
#include <map>
using namespace std;

int main (int argc, char** argv)

{
   if ((argc<7) || (argc>10))
    { std::cout << "./mini-exam seed ExtremaFile.dat EPositionFile.dat CorrFile.dat ScarsfromSigma.dat ScarsfromKL.dat (C*) (KL*) (Sigma*) ";
    }

 else {

    int Lover2=9;

  long int seed =atoi(argv[1]);
  ifstream fe(argv[2]);
  if (!fe.good()) cerr<<"\nError: I cannot open Extrema file.\n",exit(0);
  ifstream fep(argv[3]);
  if (!fep.good()) cerr<<"\nError: I cannot open Eposition file.\n",exit(0);
  ifstream fc(argv[4]);
  if (!fc.good()) cerr<<"\nError: I cannot open Corr file.\n",exit(0);
  ifstream fs(argv[5]);
  if (!fs.good()) cerr<<"\nError: I cannot open Sigma file.\n",exit(0);
  ifstream fk(argv[6]);
  if (!fk.good()) cerr<<"\nError: I cannot open KL file.\n",exit(0);
  double Cstar=0.1;
  if (argc==8)  { Cstar=atof(argv[7]); }
  double KLmax=2.20;
  if (argc==9)  { KLmax=atof(argv[8]); }
  double Sigmamin=0.08;
  if (argc==10)  { Sigmamin=atof(argv[9]); }
    cout.precision(16);
  double s,emin,emax,de;
  while (fe >> s >> emin >> emax >> de) {
   }
  std::map<double,int > energies_position; energies_position.clear();
  double e; int p;
  while (fep >> s >> e >> p) {
  energies_position[e]=p;
  }
  std::vector<double> energies_correl(0); energies_correl.resize(0);
  std::vector<double> correl; correl.resize(0);
  std::vector<int> x1_correl;x1_correl.resize(0);
  std::vector<int> x2_correl; x2_correl.resize(0);
  int x1; int x2; double C;
  while (fc >> x1 >> x2 >> C >> e) {
  if  ( (fabs(C)>Cstar) && ( fabs(x1-x2)==Lover2) ){
  energies_correl.push_back(e); x1_correl.push_back(x1); x2_correl.push_back(x2); correl.push_back(C);
  }
  }
  std::vector<double> energies_sigma1; energies_sigma1.resize(0); std::vector<double> energies_sigma2; energies_sigma2.resize(0);
  std::vector<double> sigma; sigma.resize(0);
  std::vector<int> x_sigma(0); x_sigma.resize(0);
  int x; double S; double ep; string aa;
  while (fs >> aa >> x1 >> S >> e >> ep) {  
    if (fabs(S)>Sigmamin) {
  energies_sigma1.push_back(e); energies_sigma2.push_back(ep);  x_sigma.push_back(x1); sigma.push_back(S);
    }
  }

  std::vector<double> energies_KL1; std::vector<double> energies_KL2; energies_KL1.resize(0); energies_KL2.resize(0);
  std::vector<double> KL_12; std::vector<double> KL_21;  KL_12.resize(0); KL_21.resize(0);
  double K1,K2; 
  while (fk >> K1 >> K2 >> e >> ep) {
    if ( (K1<KLmax) || (K2<KLmax) ) {
  energies_KL1.push_back(e); energies_KL2.push_back(ep);  KL_12.push_back(K1); KL_21.push_back(K2); 
    }
  }
  //  cerr << "Treating seed " << seed << " Corr > " << Cstar << " _size = " <<  energies_correl.size() 
  //  << " Sigma > " << Sigmamin << " _size= " << energies_sigma1.size() << " KL < " << KLmax << " _size=" <<  energies_KL1.size() << endl;
    bool careful=0;
    if ((2*energies_KL1.size())!=(energies_correl.size())) { cerr << "**** Careful with seed=" << seed << endl; careful=1;}
    int nk=KL_12.size();
    std::vector < std::vector<int> > possible_sites; possible_sites.resize(nk);
    std::vector < std::vector<int> > sigma_indices; sigma_indices.resize(nk);
    int nb_cats=0;
    for (int kk=0;kk<nk;++kk) {
        double e1=energies_KL1[kk];
        double e2=energies_KL2[kk];
        std::vector<double> it;
        for (int ss=0;ss<energies_sigma1.size();++ss) {
        if ( ( (e1==energies_sigma1[ss]) && (e2==energies_sigma2[ss]) ) || ( (e1==energies_sigma2[ss]) && (e2==energies_sigma1[ss]) ) )
            {possible_sites[kk].push_back(x_sigma[ss]); sigma_indices[kk].push_back(ss); }
        }
      //  cout << "Psites " << kk << " size= " << possible_sites[kk].size() << " " << possible_sites[kk][0] << " " << possible_sites[kk][1] << endl;
        double previous_C; double previous_E; int previous_x1;
        for (int cc=0;cc<energies_correl.size();++cc) {
            if ((e1==energies_correl[cc]) || (e2==energies_correl[cc])) {
        //        cout << "Energy ok fpr cc=" << cc << endl; 
            if ( std::find(possible_sites[kk].begin(), possible_sites[kk].end(), x1_correl[cc]) != possible_sites[kk].end() ) 
                { if ( std::find(possible_sites[kk].begin(), possible_sites[kk].end(), x2_correl[cc]) != possible_sites[kk].end() ) 
                        { // those are cat states
                            bool first=1;
                            if (e2==energies_correl[cc]) { first=0;}
                            if (first) { previous_C=correl[cc]; previous_E=e1; previous_x1=x1_correl[cc]; }

                            double fgap=fabs(e2-e1);
                            double scaled_fgap=fgap/de;
                            // find indices
                            double fdistance=fabs(energies_position[e1]-energies_position[e2]);
                            if (!(first)) {
                                if ( (previous_E==e1) && (previous_x1==x1_correl[cc])) {
                                    nb_cats++;
                            if (careful) { cout << "CARCATS ";} else { cout << "CATS ";}
                            cout << seed << " " << e1 << " " << e2 << " " << fgap << " " << scaled_fgap << " " << fdistance 
                                << " " << KL_12[kk] << " " << KL_21[kk] 
                                << " " << sigma[sigma_indices[kk][0]] << " " << sigma[sigma_indices[kk][1]] 
                                << " " << previous_C << " " << correl[cc] << " " << endl;
                                }
                            }
                        }
                }
            }
        }

    }
 
    //cerr << "seed " << seed << " --> Resulting in " << nb_cats << " cats\n";
 
 }
 return 0;
}






