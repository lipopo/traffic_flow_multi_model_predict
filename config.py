from lib import ConfigLoader

config = ConfigLoader("config.json")

model_config = config.load("model")
time_range = config.load("time_range")
