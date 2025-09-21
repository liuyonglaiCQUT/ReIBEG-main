import torch

class EMAModel:
    """
    简单的指数移动平均(EMA)模型
    用于跟踪模型参数的平滑版本
    """
    def __init__(self, parameters, power=0.999):
        self.params = list(parameters)
        self.power = power
        # 创建 shadow 副本，用于存储 EMA 权重
        self.shadow = [p.clone().detach() for p in self.params]

    def update(self):
        """
        更新 EMA 参数
        """
        for s, p in zip(self.shadow, self.params):
            s.data = self.power * s.data + (1 - self.power) * p.data

    def apply_shadow(self):
        """
        将 EMA 参数应用到模型中
        """
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.data)

    def restore(self, backup_params):
        """
        将模型参数恢复为备份参数（可选）
        """
        for p, b in zip(self.params, backup_params):
            p.data.copy_(b.data)