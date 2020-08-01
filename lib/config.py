import json


class ConfigLoader:
    """配置加载器，用于生成配置对象
    """
    def __init__(self, config_name):
        with open(config_name, "r") as f:
            self._config = json.load(f)

    def load(self, section_name):
        return self._config.get(section_name, {})
