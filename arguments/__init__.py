#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    """
    这段代码定义了一个ParamGroup类，用于通过ArgumentParser解析命令行参数，并根据传入的对象实例中的属性动态的添加参数选项。
    """
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        # 使用parser创建一个新的参数组，组的名称由name参数决定。这个参数组包含所有从当前实例self中提取的参数。
        group = parser.add_argument_group(name)
        # 遍历当前实例的所有属性，vars(self)返回一个字典，包含了当前实例的所有属性和对应的值, item()方法返回一个元组列表，元组的第一个元素是属性名，第二个元素是属性值。
        for key, value in vars(self).items():
            # 初始化一个标记 shorthand，用于判断参数名称是否以 _ 开头
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    # 添加参数 --key 和 -k（简写），其中 key[0:1] 获取属性名的首字母。default=value 指定默认值，action="store_true" 表示如果命令行中出现了 --key 或 -k，则将 key 的值设为 True。
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

# 定义extract方法，用于提取和存储函数参数。
    def extract(self, args):
        # 创建GroupParams对象，用于存储解析后的参数。
        group = GroupParams()
        # 遍历传入的args中所有参数和对应值。
        for arg in vars(args).items():
            #  检查args中的参数是否在当前实例的属性中，如果在，则将参数值赋给GroupParams对象。
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """
    这个函数将命令行参数和配置文件中的参数合并。它的作用是解析命令行参数，然后读取配置文件中的参数，将两者合并，返回一个新的Namespace对象。
    """
    # 从系统的命令行参数中获取除第一个参数（通常是脚本名之外的所有参数，并将他们存储在cmdlne_string）中。
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    # 使用parser解析命令行参数，将结果存储在args_cmdline中, parse_args会根据add_argument()方法添加的参数选项解析命令行参数。
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        # 假设args_cmdline中存在model_path参数，将其与“cfg_args”,拼接生成配置文件的路径cfgfilepath。
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        # 尝试打开配置文件，如果成功，则将配置文件的内容读取到cfgfile_string中。
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            # 读取配置文件的内容并存储在cfgfile_string中。
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    # 使用eval将cfgfile_string转换为一个Namespace对象,从而接卸配置文件中的参数。
    args_cfgfile = eval(cfgfile_string)
    # 将args_cfgfile中的参数转换为自定并复制到merged_dict中。
    merged_dict = vars(args_cfgfile).copy()
    # 检查value是否为空，如果不为空则将其添加到merged_dict中。
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

    # Namespace 是一个简单的类，可以理解为一个 "对象化的字典"。通过 ArgumentParser 的 parse_args() 方法将命令行参数解析并存储到 Namespace 对象中，这样可以像访问对象属性一样访问这些参数值。
