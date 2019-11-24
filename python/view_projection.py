#Interactively view 2-D projections of chain data
#This is now a front end for ptmcmc_analysis (with -selectX=True)
import argparse
import ptmcmc_analysis

parser = argparse.ArgumentParser(description='Provide snapshot of chain state.')
parser.add_argument('fname', metavar='chain_file', nargs='+', type=str, 
                    help='chain file path')
parser.add_argument('-uselike',action='store_true',help='Include the likelihood')
parser.add_argument('-noPost',action='store_true',help='Data has no Posterior or Likelihood in first columns ')
args = parser.parse_args()
print(args)


ptmcmc_analysis.useLikeDefault=args.uselike
ptmcmc_analysis.noPostDefault=args.noPost
ptmcmc_analysis.viewer(args.fname,selectX=True)




