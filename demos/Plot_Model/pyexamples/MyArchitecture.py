import sys
import os
import subprocess
sys.path.append('../')
from demos.Plot_Model.pycore.tikzeng import *


class Plot_Model():
    # defined your arch
    layer_list=[]
    layer_list=[]
    def __init__(self,layer_name,args_items):
        self.args_items=args_items
        self.layer_name=layer_name




    def Plot_Layer(layer_names,args_items):
            arch = [
                to_head('..'),
                to_cor(),
                to_begin()
            ]
            #[ 'Add' 'Concatenate'  'Dropout' 'Flatten'  'Reshape' 'Shuffle'
            for layer_name,layer_args in zip(layer_names,args_items):
                #layer_args:dict
                if layer_name == 'Conv2D' or layer_name == 'DepthwiseConv2D' or layer_name == 'PointwiseConv2D' or layer_name=='GroupedConv2D' or layer_name=='GroupedPointwiseConv2D' or layer_name=='SameConv2D' or layer_name=='SeparableConv2D':
                    for key, value in layer_args.items():
                        if key.name == "KERNEL_SIZE":
                            s_filer=value
                        if key.name =="FILTERS":
                            n_filer=value
                    arch.append(to_Conv(layer_name,s_filer,n_filer,offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "))

                if layer_name == 'MaxPool3D' or layer_name =='MaxPool1D' or layer_name =='MaxPool2D' or layer_name =='SameMaxPool2D' or layer_name =='GlobalAveragePool2D':
                    for key,value in layer_args.items():
                        if key.name =="POOL_SIZE":
                            pool_size=value
                    arch.append(to_Pool(layer_name, pool_size, offset="(0,0,0)", to="(cr1-east)",
                            width=1, height=35, depth=35, opacity=0.5))


                if layer_name == 'HiddenDense':
                    for key, value in layer_args.items():
                        if key.name == "UNITS":
                            units = value
                    arch.append(to_FullyConnected(layer_name,units, offset="(1.25,0,0)",
                                              to="(fl-east)", width=1, height=1, depth=40,caption="fc1\ndr"))

                if layer_name == 'OutputDense':
                    for key, value in layer_args.items():
                        if key.name == "UNITS":
                            units = value
                    arch.append(to_SoftMax(layer_name,units, offset="(1.25,0,0)", to="(fc3-east)",
                               width=1, height=1, depth=10,
                               caption="SIGMOID", opacity=1.0))
                    arch.append(to_end())

            namefile = str(os.path.splitext(os.path.basename(__file__))[0])
            #print(os.path.splitext("/path/to/some/file.txt")[0])
            to_generate(arch, namefile + '.tex')
            os.system("sh demos/Plot_Model/tikzmake.sh demos/Plot_Model/pyexamples/MyArchitecture demos/MyArchitecture")
            #os.system("mv MyArchitecture ../../../../demos/MyArchitecture ")
# sh ../tikzmake.sh pyexamples/MyArchitecture ../../DemoClassification






