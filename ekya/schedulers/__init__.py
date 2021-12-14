from ekya.schedulers.fair_scheduler import FairScheduler
from ekya.schedulers.no_retraining import NoRetrainingScheduler
from ekya.schedulers.profiling_scheduler import ProfilingScheduler
from ekya.schedulers.thief_scheduler import ThiefScheduler
from ekya.schedulers.utilitysim_scheduler import UtilitySimScheduler


def get_scheduler(scheduler_name):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'fair':
        return FairScheduler
    if scheduler_name == 'noretrain':
        return NoRetrainingScheduler
    if scheduler_name == 'utilitysim':
        return UtilitySimScheduler
    if scheduler_name == 'profiling':
        return ProfilingScheduler
    if scheduler_name == 'thief':
        return ThiefScheduler
    raise NotImplementedError("Cannot find scheduler {}".format(scheduler_name))