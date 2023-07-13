# Master Thesis
Title: Multi-objective Neural Architecture Search

Author: Renáta Pivodová

Department: Deparment of Theoretical Computer Science and Mathematical Logic

Supervisor: Mgr. Martin Pilát, Ph.D.

Abstract: Neural architecture search is a promising approach to automatic neural network architecture design, which can save a designer's work. The real world contains a lot of problems, which are time-consuming to solve even by neural architecture search techniques. A lot of these problems require architectures optimized according to different criteria such as quality, time of search, etc. In this work, we present two methods extending the CoDeepNEAT, a state-of-the-art neural architecture search algorithm. The Lamarckian CoDeepNEAT is the CoDeepNEAT enriched with weight inheritance implementation inspired by the Lamarckian theory of evolution. The Multi-objective CoDeepNEAT performs a multi-objective minimization of two chosen neural network objectives - the error rate and the number of floating point operations. Thanks to the base NSGA-II algorithm, the Multi-objective CoDeepNEAT searches for well-performing and fast networks. The methods are evaluated on the MNIST and CIFAR-10 datasets. 

Full text: https://dspace.cuni.cz/handle/20.500.11956/181945

# Run experiments
All experiments are in `examples` directory.

An example of running LamaCoDeepNEAT MNIST experiment:

```python
cd ./examples/lamacodeepneat/lamacodeepneat_mnist_example/
python lamacodeepneat_mnist_example.py
```
 Do not forget to define a config file with experiment parameters. 
 
 # Content
 ```
 - LamaCoDeepNEAT
 |  - examples  - CDN, LamaCDN experiment scripts
 |  - tests     - CDN, LamaCDN implementation test scripts
 |  - tfne      - CDN and LamaCDN implementation
 - MOLamaCoDeepNEAT
 |  - examples  - MOLamaCDN experiment scripts
 |  - tests     - MOLamaCDN implementation test scripts
 |  - tfne      - MOLamaCDN implementation
 ```
