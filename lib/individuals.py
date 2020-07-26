import random

import numpy as np

from lib.ga import Individual


class ParameterIndividual(Individual):
    @staticmethod
    def rand_feature(parameter_size, loss_function):
        parameters = np.random.random(parameter_size)
        return {
            "parameters": parameters,
            "loss_func": loss_function
        }

    @property
    def loss_function(self):
        return self.feature.get("loss_func")

    @property
    def parameters(self):
        return self.feature.get("parameters")

    def calc_fitness(self):
        """计算适应度
        """
        loss_values = self.loss_function(self.parameters)
        # indivdual handler their own fitness and the model
        # handler the loss calc method and others
        return -loss_values[0]

    def crossover(self, other):
        """交叉过程
        """
        parameters = self.feature.get("parameters")
        other_parameters = other.feature.get("parameters")
        for p_idx in range(len(parameters)):
            if random.random() < self.crossover_value:
                parameters[p_idx], other_parameters[p_idx] = \
                        other_parameters[p_idx], parameters[p_idx]
        self.feature["parameters"] = parameters
        other.feature["parameters"] = other_parameters

    def mutation(self, mutation_value):
        """变异过程
        """
        parameters = self.feature.get("parameters")
        for p_idx in range(len(parameters)):
            if random.random() > mutation_value:
                parameters[p_idx] = random.random()
        self.feature["parameters"] = parameters
