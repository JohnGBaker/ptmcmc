all:  exampleLISA cp-test


exampleLISA:
	(export OMP_NUM_THREADS=4 ; time ../../exampleLISA --seed=0.012556 --save_every=40 --prop=7 --pt=20 --pt_evolve_rate=0.01 --pt_stop_evid_err=0.05 --chain_ess_stop=2000 --checkp_at_step=10000 --outname=exampleLISA_test_1 ) > exampleLISA_test_1.out
	@diff exampleLISA_test_[01]_t0.dat ; if [ $$? -ne 0 ] ; then  echo $@ FAIL ; else echo $@ PASS ;fi

cp-test:
	(export OMP_NUM_THREADS=4 ; time ../../exampleLISA --seed=0.012556 --save_every=40 --prop=7 --pt=20 --pt_evolve_rate=0.01 --pt_stop_evid_err=0.05 --chain_ess_stop=2000 --checkp_at_step=2000 --outname=exampleLISA_test_2 ) > exampleLISA_test_2.p1.out
	(export OMP_NUM_THREADS=4 ; time ../../exampleLISA --seed=0.012556 --save_every=40 --prop=7 --pt=20 --pt_evolve_rate=0.01 --pt_stop_evid_err=0.05 --chain_ess_stop=2000 --checkp_at_step=10000 --restart_dir=step_2000-cp --outname=exampleLISA_test_2 ) > exampleLISA_test_2.p2.out
	@diff exampleLISA_test_[12]_t0.dat ; if [ $$? -ne 0 ] ; then  echo $@ FAIL ; else echo $@ PASS ;fi

