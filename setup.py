from io import open
from setuptools import find_packages, setup

setup(
    name="pytorch_quantization_tool",
    version="0.1",
    author="Liangang, Zhang",
    author_email="liangang.zhang@intel.com",
    description="Repository of pytorch imperatie quantization tool",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='quantization post-training fallback auto-tuning',
    license='',
    url="https://github.com/liangan1/pytorch_imperative_quantization_tool",
    packages = find_packages('src'), 
    package_dir = {'':'src'},         

    install_requires=['torch>=1.3.0',
                      'numpy',
                      'tqdm',
                      'regex'],
    entry_points={
      'console_scripts':  [""]
    },
    # python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
