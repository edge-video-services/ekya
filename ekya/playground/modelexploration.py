from inclearn.models.icarl import ICarl

args = {
    'convnet': 'resnet50'
}
ic = ICarl(args)

def get_params():
    itr = filter(lambda p: p.requires_grad, ic._network.parameters())
    num_params=0
    for p in itr:
        num_params += p.numel()
    print(num_params)