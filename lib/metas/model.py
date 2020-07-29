from progress.bar import Bar


class MetaModel(type):
    """ 所有模型的元类，用于配置部分特性
    """
    def __init__(self, cls_name, cls_base, cls_dict):
        super().__init__(cls_name, cls_base, cls_dict)

    def __new__(cls, cls_name, cls_base, cls_dict):
        use_ga = cls_dict.get("use_ga", False)
        if use_ga:
            origin_fit_func = cls_dict.get("fit", cls_base[0].fit)

            def fit(instance, *args, **kwargs):
                instance.input_data = kwargs.get("input_data", args[0])
                instance.target_data = kwargs.get("target_data", args[1])
                ga = instance.ga
                bar = Bar(
                   'Ga Runing...',
                   max=instance.ga_parameter.get("max_iter_count"))
                # 在执行计算前，优先进行ga优化
                for _ in ga(**instance.ga_parameter):
                    bar.next()
                bar.finish()
                instance.set_parameter(ga.best_individual.parameters)
                return origin_fit_func(instance, *args, **kwargs)

            cls_dict["fit"] = fit
        return super().__new__(cls, cls_name, cls_base, cls_dict)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
