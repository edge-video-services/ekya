from typing import List


class BaseMicroprofiler(object):
    def __int__(self):
        pass

    def run_microprofiling(self,
                           candidate_hyperparams: List[dict],
                           dataloaders: dict) -> dict:
        pass