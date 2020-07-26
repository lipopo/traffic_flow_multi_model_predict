# 预测模型包
# 所有构建的预测模型 都会记录在此处
from model.bp import BP, GaBP
from model.svr import Svr, GaSvr
from model.lssvr import Lssvr, GaLssvr
from model.ga_knn_lssvr import GaKnnLssvr


__all__ = (
    "BP", "GaBP", "Svr", "GaSvr",
    "Lssvr", "GaLssvr", "GaKnnLssvr"
)
