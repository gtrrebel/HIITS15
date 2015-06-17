from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.MAXLOAD = 1

print runner.check_slots()
runner.add_jobs([('Windows/Desktop/hiit/HIITS15/scripts/test_runner.sh Windows/Desktop/hiit/HIITS15/colvb-master/examples/MOG_demo2.py', 1)])
print runner.jobs
runner.start_batches()

