class MetaPlotable(type):
    """ 可绘制对象的元类
    """
    def __call__(cls, *args, **kwargs):
        if "index" in kwargs:
            cls.index = kwargs.pop("index")
        return super().__call__(*args, **kwargs)
