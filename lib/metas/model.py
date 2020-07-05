

class MetaModel(type):
    """ 所有模型的元类，用于配置部分特性
    """
    def __init__(self, cls_name, cls_base, cls_dict):
        super().__init__(cls_name, cls_base, cls_dict)
    
    def __new__(cls, cls_name, cls_base, cls_dict):
        return super().__new__(cls, cls_name, cls_base, cls_dict)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
