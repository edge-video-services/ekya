import ray

from ekya.models.resnet_inference import ResnetInference

if __name__ == '__main__':
    ray.init()
    model = ray.remote(ResnetInference).remote()
    result = model.infer_images.remote(["/home/romilb/image.jpg"]*1000)
    print(ray.get(result))