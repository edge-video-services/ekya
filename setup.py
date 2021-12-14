from setuptools import setup, find_packages

setup(
    name='ekya',
    version='0.0.1',
    url='https://github.com/ekya-project/ekya.git',
    author='Romil Bhardwaj, Zhengxu Xia',
    author_email='romilb@eecs.berkeley.edu',
    description='Ekya - A system for online training',
    packages=find_packages(),
    install_requires=['ray', 'tensorflow', 'waymo-open-dataset-tf-2-2-0',
                      'opencv-contrib-python', 'tensorflow==2.2.0',
                      'matplotlib', 'pandas'],
)
