import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name='MultiobjectiveLamaCoDeepNEAT',
    version='0.1',
    author='Renata Pivodova',
    author_email='renata.pivodova@gmail.com',
    url="https://github.com/PaulPauls/Tensorflow-Neuroevolution",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow >= 2.0.0",
        "ray",
        "graphviz",
        "matplotlib",
        "PyQt5",
        "pydot",
    ],
    python_requires='>= 3.7',
)
