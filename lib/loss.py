from lib.metas import MetaLoss


class LossLog:
    """Loss 记录表
    """
    loss_map = []

    def log(self, value, meta={}):
        self.loss_map.append( {"meta": meta, "data": value}
         )

    def clear(self):
        self.loss_map = []


class BaseLoss(metaclass=MetaLoss):
    loss = None  # 残差实际值
    true_value = None  # 目前绑定的真实值
    preb_value = None  # 目前绑定的预测值
    loss_log = LossLog()
    meta = None  # 元数据
    def calc_loss(
        self, 
        preb_value, 
        true_value
        ):
        """计算残差值
        """
        raise NotImplementedError

    def snap_point(self):
        """执行残差记录
        """
        self.loss_log.log(
            self.calc_loss(
                self.preb_value, 
                self.true_value
            ),
            self.meta
        )

    def __call__(self, preb_value, true_value):
        self.preb_value = preb_value
        self.true_value = true_value
        return self
