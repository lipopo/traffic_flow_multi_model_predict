# GA算法流程
from copy import deepcopy
import random
from typing import Any, List, Dict


class Individual:
    """ 个体抽象
    """
    _fitness = None  # 适应度
    crossover_value = None

    def __init__(self, feature: Dict[str, Any]):
        """ 基因序列
        @parameter feature Dict[str, Any] 个体特征
        """
        self.feature = feature

    def __mul__(self, right_value):
        """ 交叉变异过程
        """
        # 重设适应度数值
        self._fitness = None
        if type(self) == type(right_value):  # 同类个体才可以交叉
            # 交叉过程
            self.crossover(right_value)
        elif isinstance(right_value, float):
            # 变异过程
            self.mutation(right_value)
        return self

    def __pow__(self, right_value):
        """ 设置交叉率
        """
        if not isinstance(right_value, float):
            raise Exception("交叉率必须设定为浮点类型")
        self.crossover_value = right_value
        return self

    def __gt__(self, right_value):
        """ 对比个体适应度
        """
        return self.fitness > right_value.fitness

    @property
    def fitness(self):
        """ 个体适应度
        """
        if not self._fitness:
            # 计算适应度
            self._fitness = self.calc_fitness()
        return self._fitness

    @classmethod
    def rand_individual(cls):
        """生成随机个体
        """
        return cls(cls.rand_feature())

    @staticmethod
    def rand_feature():
        """生成随机特性
        """
        raise NotImplementedError("生成随机的特性")

    def calc_fitness(self):
        """ 计算适应度
        """
        raise NotImplementedError("定义个体需要定义calc_fitness函数")

    def crossover(self, other):
        """ 交叉过程
        """
        raise NotImplementedError("定义个体间交叉过程")

    def mutation(self, mutation_value):
        """ 变异过程
        """
        raise NotImplementedError("定义个体变异过程")


class Population:
    """ 种群，个体的集合
    """

    def __init__(self, individual_list: List[Individual] = []):
        self.individual_list = individual_list

    def __add__(self, right_value):
        """ 添加指定个体
        """
        if isinstance(right_value, Individual):
            self.individual_list.append(right_value)
            return self
        else:
            raise Exception("只有Individual类型可以被添加到种群中")

    def __sub__(self, right_value):
        """ 将个别个体移出种群
        """
        if not isinstance(right_value, Individual):
            raise Exception("只有Individual类型的个体可以被移除种群")
        else:
            try:
                self.individual_list.remove(right_value)
            except ValueError:
                raise ValueError("个体不在种群中")
        return self

    def __iter__(self):
        for i in self.individual_list:
            yield i

    @classmethod
    def generate_population(cls, individual: type, individual_count: int):
        """种群生成
        @parameter individual_cls 个体实现类别
        @parameter individual_count 个体数量
        """
        individual_list = [
            individual.rand_individual()
            for i in range(individual_count)
        ]
        return cls(individual_list)

    @property
    def weights(self):
        """ 个体适应度数值
        """
        return [
            individual.fitness
            for individual in self.individual_list
        ]

    @property
    def individuals(self) -> List[Individual]:
        """ 个体列表
        """
        return self.individual_list

    @property
    def size(self) -> int:
        """ 返回种群数量
        """
        return len(self.individual_list)

    def update(self, individual_list):
        """ 更新个体列表
        """
        self.individual_list = [
                deepcopy(individual)
                for individual in individual_list
            ]

    def remove(self, individual_list):
        """ 删除部分个体
        """
        for individual in individual_list:
            self -= individual

    def add(self, individual_list):
        """ 增加部分个体
        """
        for individual in individual_list:
            self += individual


class GA:
    max_iter_count = 0  # 最大迭代次数
    max_fitness = 0  # 最大适应度
    iter_count = 0  # 当前迭代次数

    population = None  # 当前种群
    crossover_pair_count = 1  # 交叉组数
    mutation_value = 0.1  # 变异率
    crossover_value = 0.1  # 交叉率

    def __init__(self, population: Population = None):
        """ 初始化种群编码
        @parameter population List[Individual] 初始种群
        """
        self.population = population

    def __iter__(self):
        """ 优化迭代
        """
        return self

    def __next__(self):
        """ 调用种群优化流程
        """
        if (self.max_iter_count is not None and
                self.iter_count >= self.max_iter_count):
            raise StopIteration
        self.iter_count += 1
        self.selection()  # 选择
        self.crossover()  # 交叉
        self.mutation()  # 变异
        return self

    def __call__(
        self,
        max_iter_count: int = None,
        # max_fitness: float = None,
        crossover_pair_count: int = 1,
        crossover_value: float = 0.1,
        mutation_value: float = 0.1
    ):
        """ 定义优化参数
        @parameter max_iter_count int 最大迭代次数
        @parameter [DEPRECT] max_fitness float 最大适应度
        @parameter crossover_pair_count int 交叉变异组数
        @parameter crossover_value float 交叉率
        @parameter mutation_value float 变异率
        """
        self.max_iter_count += max_iter_count
        # self.max_fitness = max_fitness
        self.crossover_pair_count = crossover_pair_count
        self.crossover_value = crossover_value
        self.mutation_value = mutation_value
        return self

    @property
    def best_individual(self):
        """ 最有的个体
        """
        return max(self.population)

    @property
    def mean_fitness(self):
        """平均适应度
        """
        return sum(self.population.weights) / self.population.size

    def selection(self):
        """ 选择
        @description 根据轮盘赌法选取种群中的部分适应度的个体，组成新的种群
        """
        new_individual_list = random.choices(
            self.population.individual_list,
            self.population.weights,
            k=self.population.size
        )
        self.population.update(new_individual_list)

    def crossover(self):
        """ 交叉
        """
        # 随机选择个体 进行交叉
        sample_individuals = random.sample(
            list(self.population), self.crossover_pair_count * 2)
        # 相邻个体键进行交叉
        for individual_father, individual_mother in zip(
            sample_individuals[:self.crossover_pair_count],
                sample_individuals[self.crossover_pair_count:]):
            (individual_father ** self.crossover_value) \
                    * individual_mother  # 交叉

    def mutation(self):
        """ 变异
        """
        mutation_individual = random.choice(list(self.population))
        # 引发个体变异 指定变异率
        mutation_individual *= self.mutation_value


if __name__ == "__main__":
    pass
