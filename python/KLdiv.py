#Code for estimating Kullback-Leibler divergence among two set of chain data
#John Baker (2018-19)
#Find at github.com/johngbaker/ptmcmc
#
import numpy as np
import argparse
import sys
import ptmcmc_analysis

################# MAIN ################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='KL divergence among two chain files.')
    parser.add_argument('fnameQ', metavar='chain_file', type=str, help='chain file path')
    parser.add_argument('fnameP', metavar='chain_file', type=str, help='chain file path')
    #parser.add_argument('fname', metavar='chain_file', nargs='+', type=str, help='chain file path')
    #parser.add_argument('-uselike',action='store_true',help='Include the likelihood')
    parser.add_argument('-noPost',action='store_true',help='Data has no Posterior or Likelihood in first columns ')
    parser.add_argument('-upsample',help='Factor by which to upsample the resampled posterior. (Default=1)',default='1')
    parser.add_argument('-esslimit',help='Assume ESS is less than this value.',default='10000')
    args = parser.parse_args()
    print(args)

    ptmcmc_analysis.noPostDefault=args.noPost

    args = parser.parse_args()

    chainQ=ptmcmc_analysis.chainData(args.fnameQ)
    chainP=ptmcmc_analysis.chainData(args.fnameP)
    upsample_fac=float(args.upsample)
    ntrials=100
    if False:
        div=chainP.KLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit))
        print("KL divergence is ",div)
        print("retry:",chainP.KLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit)))
        print("retry:",chainP.KLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit)))
        print("retry:",chainP.KLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit)))
    if True:
        divs=[]
        for trial in range(ntrials):            
            div=chainP.fakeKLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit))
            divs.append(div)
        print("fake KL divergence is ",np.mean(divs),"+/-",np.std(divs))
        divs=[]
        for trial in range(ntrials):            
            div=chainQ.fakeKLdivergence(chainQ,upsample_fac,esslimit=float(args.esslimit))
            divs.append(div)
        print("QQ fake KL divergence is ",np.mean(divs),"+/-",np.std(divs))
        divs=[]
        for trial in range(ntrials):            
            div=chainP.fakeKLdivergence(chainP,upsample_fac,esslimit=float(args.esslimit))
            divs.append(div)
        print("PP fake KL divergence is ",np.mean(divs),"+/-",np.std(divs))
    
