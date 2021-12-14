from ekya.schedulers.scheduler import fair_reallocation


def test_noinferencejobs():
    inference_resource_weight = {}
    training_resource_weight = {'trgjob': 1}
    ifwt, twt = fair_reallocation('trgjob', inference_resource_weight,
                                  training_resource_weight)
    assert ifwt == {}
    assert twt['trgjob'] == 0


def test_inferencejob():
    inference_resource_weight = {'infjob': 0.1}
    training_resource_weight = {'trgjob': 0.9}
    ifwt, twt = fair_reallocation('trgjob', inference_resource_weight, training_resource_weight)
    assert ifwt['infjob'] == 1
    assert twt['trgjob'] == 0

if __name__ == '__main__':
    test_noinferencejobs()
    test_inferencejob()
    print("tests done.")
