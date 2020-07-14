class MetaModel(type):
    """ 所有模型的元类，用于配置部分特性
    """
    def __init__(self, cls_name, cls_base, cls_dict):
        super().__init__(cls_name, cls_base, cls_dict)

    def __new__(cls, cls_name, cls_base, cls_dict):
        use_ga = cls_dict.get("use_ga", False)

        def call(instance, *args, **kwargs):
            instance.setup()
            if use_ga:
                assert "ga_parameter" in kwargs, \
                    "使用GA优化的模型必须指定ga_parameter"
                ga = cls_dict["ga"]
                for i in ga(**kwargs.pop("ga_parameter")):
                    parameter = ga.best_individual.parameter
                instance.set_parameter(parameter)
            instance.predict(*args, **kwargs)

        cls_dict["__call__"] = call
        return super().__new__(cls, cls_name, cls_base, cls_dict)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
