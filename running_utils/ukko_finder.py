from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.remove_bad_hosts()
print runner.machines[0][0]

