import urllib2, re, os, math, time

MACHINERE = re.compile(r'ukko[0-9]{3}\.hpc\.cs')

class runner:
    def __init__(self):
        self.update_machine_list()
        self.MAXLOAD = 8
        self.JOBLOAD = 1
        self.jobs = []
        self.skipfirst = 0

    def update_machine_list(self, mymachines = None):
        f = urllib2.urlopen('http://www.cs.helsinki.fi/ukko/hpc-report.txt')
        machines = []
        for l in f:
            l = l.strip()
            if MACHINERE.match(l):
                t = l.split()
                if (mymachines is None or t[0] in mymachines) and (t[7].startswith('ok') or t[7].startswith('cs')) and t[2] != 'no' and t[3] != 'no':  # and t[7] != 'N/A':
                    try:
                        machines.append((t[0], float(t[5])))
                    except:
                        pass
        f.close()
        machines.sort(key=lambda x: x[1])
        self.machines = machines

    def remove_bad_hosts(self):
        with open(os.path.expanduser('~/bad_hosts.txt')) as f: 
            h = [x.strip() for x in f.readlines()]
        self.machines = [x for x in self.machines if x[0][0:7] not in h]

    def start_job(self, job, machine):
        os.system('ssh %s "nohup %s </dev/null >/dev/null 2>&1 &"' % (machine, job))

    def check_slots(self):
        n=0;
        for k in self.machines:
            if math.floor(k[1]) < self.MAXLOAD:
                n += self.MAXLOAD - math.floor(k[1])
        return n

    def start_batch(self, job, count):
        if count > self.check_slots():
            raise("Batch too big to fit to available pool of machines")
        mymachine = self.machines.pop(0)
        curmachine = mymachine[0]
        curcount = math.floor(mymachine[1])
        for k in range(count):
            myjob = job % {'count': str(count), 'index': str(k+1)}
            #print myjob
            if curcount >= self.MAXLOAD:
                mymachine = self.machines.pop(0)
                curmachine = mymachine[0]
                curcount = math.floor(mymachine[1]) + 1
            else:
                curcount += 1
            print curmachine, myjob
            self.start_job(myjob, curmachine)
            time.sleep(0.5)

    def start_batches(self):
        if self.JOBLOAD*(sum([x[1] for x in self.jobs])-self.skipfirst) > self.check_slots():
            raise RuntimeError("Batch too big to fit to available pool of machines")
        mymachine = self.machines.pop(0)
        curmachine = mymachine[0]
        curcount = math.floor(mymachine[1])
        processed = 0
        for job, count in self.jobs:
            for k in range(count):
                myjob = job % {'count': str(count), 'index': str(k+1)}
                #print myjob
                if processed < self.skipfirst:
                    processed += 1
                    continue
                else:
                    processed += 1
                if curcount >= self.MAXLOAD:
                    mymachine = self.machines.pop(0)
                    curmachine = mymachine[0]
                    curcount = math.floor(mymachine[1]) + self.JOBLOAD
                else:
                    curcount += self.JOBLOAD
                print curmachine, myjob
                self.start_job(myjob, curmachine)
                time.sleep(0.5)
        self.jobs = []
        self.skipfirst = 0

    def add_jobs(self, jobs, numcpus=1):
        self.jobs += jobs
        self.JOBLOAD = max(self.JOBLOAD, numcpus)
        if self.JOBLOAD > numcpus:
            print "Warning: mixing different numbers of CPUs not properly supported yet, using max(numcpus)"

    def skip_first(self, skipfirst=0):
        self.skipfirst = skipfirst
