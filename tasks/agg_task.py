import os

from data import XLSXReader

from tasks.base_task import BaseTask


class AggTask(BaseTask):
    task_name = "agg_data"
    __doc__ = """合并已有的数据"""

    def run(self):
        file_list = filter(
                lambda fname: fname.startswith("2011"),
                os.listdir("./asset"))
        data = None
        for fname in file_list:
            _d = XLSXReader(
                f"./asset/{fname}",
                usecols=["交易时间", "WEIGHT_FACT_CASH"],
                index_col=0).convert_index_to_datetime()
            if not data:
                data = _d
            else:
                data += _d
        data.save("./asset/agg.xlsx")
