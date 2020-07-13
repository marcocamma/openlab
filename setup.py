from setuptools import setup,find_packages

setup(name='openlab',
      version='0.0.1',
      description='Control your instruments with python',
      url='https://github.com/marcocamma/openlab',
      author='marco cammarata',
      author_email='marcocammarata@gmail.com',
      license='MIT',
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'h5py',
          'tqdm',
      ],
      zip_safe=False)
