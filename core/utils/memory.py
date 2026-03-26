import math
import torch


class MemoryItem:
    def __init__(self, data=None, prob=None, uncertainty=0, age=0):
        self.data = data
        self.prob = prob          # shape: [num_class]
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.prob, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_balance=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.lambda_balance = lambda_balance

        self.data = []   # 클래스별 분리 제거
        self.prob_sum = torch.zeros(num_class, dtype=torch.float32)

    def get_occupancy(self):
        return len(self.data)

    def add_age(self):
        for item in self.data:
            item.increase_age()

    def heuristic_score(self, age, uncertainty):
        return (
            self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))
            + self.lambda_u * uncertainty / math.log(self.num_class)
        )

    def uniformity_loss_from_sum(self, prob_sum, n):
        if n == 0:
            return 0.0
        mean_prob = prob_sum / n
        target = torch.ones(self.num_class, dtype=mean_prob.dtype, device=mean_prob.device) / self.num_class
        return torch.sum((mean_prob - target) ** 2).item()

    def add_instance(self, instance):
        assert len(instance) == 4
        x, prediction, uncertainty, prob = instance

        if not isinstance(prob, torch.Tensor):
            prob = torch.tensor(prob, dtype=torch.float32)
        prob = prob.detach().cpu()

        new_item = MemoryItem(data=x, prob=prob, uncertainty=uncertainty, age=0)

        # 1) 아직 안 찼으면 바로 추가
        if self.get_occupancy() < self.capacity:
            self.data.append(new_item)
            self.prob_sum += prob
            self.add_age()
            return True

        # 2) 꽉 찼으면 "누구를 뺄지" 최적 선택
        best_idx = None
        best_obj = None

        for idx, old_item in enumerate(self.data):
            candidate_sum = self.prob_sum - old_item.prob + prob
            balance_loss = self.uniformity_loss_from_sum(candidate_sum, self.capacity)

            # 기존 heuristic도 같이 반영 가능
            old_remove_score = self.heuristic_score(old_item.age, old_item.uncertainty)
            new_keep_score = self.heuristic_score(0, uncertainty)

            # 목적함수:
            # balance_loss는 작을수록 좋음
            # old_remove_score가 클수록 오래되고 불확실한 샘플 제거에 유리
            # new_keep_score가 클수록 새 샘플 유지에 유리
            obj = self.lambda_balance * balance_loss - old_remove_score + new_keep_score

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_idx = idx

        # 필요하면 "교체가 실제로 이득일 때만" 교체
        current_balance_loss = self.uniformity_loss_from_sum(self.prob_sum, self.capacity)
        current_obj = self.lambda_balance * current_balance_loss

        if best_obj is not None and best_obj < current_obj:
            removed = self.data.pop(best_idx)
            self.prob_sum -= removed.prob
            self.data.append(new_item)
            self.prob_sum += prob

        self.add_age()
        return True

    def get_memory(self):
        tmp_data = []
        tmp_age = []
        tmp_prob = []

        for item in self.data:
            tmp_data.append(item.data)
            tmp_age.append(item.age / self.capacity)
            tmp_prob.append(item.prob)

        return tmp_data, tmp_age #, tmp_prob

    def get_mean_prob(self):
        n = self.get_occupancy()
        if n == 0:
            return torch.zeros(self.num_class)
        return self.prob_sum / n