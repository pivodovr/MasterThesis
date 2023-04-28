from .lamacodeepneat_module_densedropout import LamaCoDeepNEATModuleDenseDropout
from .lamacodeepneat_module_conv2dmaxpool2ddropout import LamaCoDeepNEATModuleConv2DMaxPool2DDropout

# Dict associating the string name of the module when referenced in LamaCoDeepNEAT config with the concrete instance of
# the respective module
LAMAMODULES = {
    'DenseDropout': LamaCoDeepNEATModuleDenseDropout,
    'Conv2DMaxPool2DDropout': LamaCoDeepNEATModuleConv2DMaxPool2DDropout
}