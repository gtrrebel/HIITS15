from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.MAXLOAD = 1
print runner.check_slots()
runner.add_jobs([('github/pol2rna/matlab/run_ode_savesamples.sh %(index)s:%(count)s:100000', 1)])