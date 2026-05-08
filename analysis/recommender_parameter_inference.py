from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class StoryObservation:
    """单条内容的观测数据。"""

    story_id: str
    repost_count: float
    view_count: float


class BayesianAgent:
    """用于模拟信息扩散的个体。"""

    def __init__(self, agent_id: int, p_online: float = 0.1):
        self.id = agent_id
        self.p_online = p_online
        self.state = 0
        self.init_belief = random.random()
        self.belief = self.init_belief
        self.sigma = 0.1
        self.is_verified = False

    def reset_state(self) -> None:
        self.state = 0
        self.belief = self.init_belief

    def decide_probability_check(self, neighbor_belief: float, p_reshare: float, p_reject: float) -> int:
        distance = abs(self.belief - neighbor_belief)
        affinity = 1.0 - distance
        effective_reshare = p_reshare * affinity
        effective_reject = p_reject * (1 - affinity)
        value = random.random()
        if value < effective_reshare:
            return 1
        if value < effective_reshare + effective_reject:
            return -1
        return 0

    def update_belief(self, neighbor_belief: float, mu_tweet: float = 0.5, sigma_tweet: float = 0.5) -> None:
        sigma_self = self.sigma
        evidence_self = self._gaussian_pdf(neighbor_belief, self.belief, sigma_self)
        evidence_tweet = self._gaussian_pdf(neighbor_belief, mu_tweet, sigma_tweet)
        new_belief = self.belief * (evidence_self / max(evidence_tweet, 1e-10))
        self.belief = float(np.clip(new_belief, 0.0, 1.0))

    def calculate_recommendation_score(
        self,
        source_agent: 'BayesianAgent',
        tweet_stats: Dict[str, float],
        weights: Dict[str, float],
        current_step: int,
    ) -> float:
        chrono_weight = weights['chrono']
        belief_weight = weights['belief']
        pop_weight = weights['pop']
        rand_weight = weights['rand']
        decay_lambda = weights.get('decay_lambda', 0.1)

        score_chrono = math.exp(-decay_lambda * current_step)
        score_belief = 1.0 - abs(self.belief - source_agent.belief)
        max_pop_reposts = 50.0
        score_pop = math.log1p(tweet_stats.get('current_retweets', 0.0)) / math.log1p(max_pop_reposts)
        score_pop = min(score_pop, 1.0)
        if source_agent.is_verified:
            score_pop = min(max(score_pop, 0.4) * 1.2, 1.0)
        score_rand = random.random()

        return (
            chrono_weight * score_chrono
            + belief_weight * score_belief
            + pop_weight * score_pop
            + rand_weight * score_rand
        )

    @staticmethod
    def _gaussian_pdf(value: float, mean: float, sigma: float) -> float:
        denominator = sigma * math.sqrt(2 * math.pi)
        exponent = -((value - mean) ** 2) / (2 * sigma**2)
        return (1.0 / max(denominator, 1e-10)) * math.exp(exponent)


class SocialABM:
    """轻量化社交扩散模拟器。"""

    def __init__(
        self,
        num_agents: int = 1500,
        avg_degree: int = 20,
        verified_ratio: float = 0.01,
        p_online: float = 0.1,
        visibility_temperature: float = 0.12,
    ):
        self.num_agents = num_agents
        self.avg_degree = avg_degree
        self.p_online = p_online
        self.visibility_temperature = max(visibility_temperature, 1e-3)
        self.agents = [BayesianAgent(i, p_online=p_online) for i in range(num_agents)]
        self.adj_list = self._build_preferential_attachment_graph(num_agents, avg_degree)
        self._assign_beliefs_based_on_community()
        self._assign_verified_status(verified_ratio)

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset_state()

    def run_simulation(
        self,
        p_re: float,
        p_rj: float,
        weights: Optional[Dict[str, float]] = None,
        duration: int = 24,
        fixed_seed: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed % (2**32 - 1))

        try:
            self.reset()
            seed = fixed_seed if fixed_seed is not None else random.randrange(self.num_agents)
            self.agents[seed].state = 1
            susceptible_ids = [idx for idx in range(self.num_agents) if idx != seed]
            history: List[int] = []
            current_count = 1
            view_count = 0
            use_recommendation = bool(weights) and {'chrono', 'belief', 'pop', 'rand'}.issubset(weights)

            for step in range(duration):
                new_infections = 0
                tweet_stats = {'current_retweets': max(current_count - 1, 0)}
                next_susceptible: List[int] = []

                for agent_id in susceptible_ids:
                    agent = self.agents[agent_id]
                    if random.random() > agent.p_online:
                        next_susceptible.append(agent_id)
                        continue

                    spreaders = [neighbor for neighbor in self.adj_list[agent_id] if self.agents[neighbor].state == 1]
                    if not spreaders:
                        next_susceptible.append(agent_id)
                        continue

                    is_visible = False
                    source_agent: Optional[BayesianAgent] = None
                    if use_recommendation and weights is not None:
                        best_score = -1.0
                        for spreader_id in spreaders:
                            candidate = self.agents[spreader_id]
                            score = agent.calculate_recommendation_score(candidate, tweet_stats, weights, step)
                            if score > best_score:
                                best_score = score
                                source_agent = candidate
                        threshold = random.gauss(0.55, 0.15)
                        score_margin = best_score - threshold
                        visible_probability = 1.0 / (
                            1.0 + math.exp(-score_margin / self.visibility_temperature)
                        )
                        is_visible = random.random() < visible_probability
                    else:
                        source_agent = self.agents[random.choice(spreaders)]
                        is_visible = True

                    if not is_visible or source_agent is None:
                        next_susceptible.append(agent_id)
                        continue

                    decision = agent.decide_probability_check(source_agent.belief, p_re, p_rj)
                    view_count += 1
                    if decision == 1:
                        agent.state = 1
                        agent.update_belief(source_agent.belief)
                        new_infections += 1
                    elif decision == -1:
                        agent.state = -1
                    else:
                        next_susceptible.append(agent_id)

                susceptible_ids = next_susceptible
                current_count += new_infections
                history.append(max(current_count - 1, 0))
                if not susceptible_ids:
                    history.extend([history[-1]] * (duration - len(history)))
                    break

            if not history:
                history = [max(current_count - 1, 0)] * duration
            if len(history) < duration:
                history.extend([history[-1]] * (duration - len(history)))
            return np.array(history, dtype=float), view_count
        finally:
            if rng_seed is not None:
                random.setstate(python_random_state)
                np.random.set_state(numpy_random_state)

    def _assign_beliefs_based_on_community(self) -> None:
        community_count = max(5, self.num_agents // 50)
        communities: List[List[int]] = [[] for _ in range(community_count)]
        for node_id in range(self.num_agents):
            communities[node_id % community_count].append(node_id)

        for community in communities:
            community_mu = random.random()
            for node_id in community:
                belief = float(np.clip(random.gauss(community_mu, 0.2), 0.001, 0.999))
                self.agents[node_id].init_belief = belief
                self.agents[node_id].belief = belief

    def _assign_verified_status(self, ratio: float) -> None:
        degrees = [(idx, len(neighbors)) for idx, neighbors in enumerate(self.adj_list)]
        degrees.sort(key=lambda item: item[1], reverse=True)
        verified_count = int(self.num_agents * ratio)
        for node_id, _ in degrees[:verified_count]:
            self.agents[node_id].is_verified = True

    @staticmethod
    def _build_preferential_attachment_graph(num_agents: int, avg_degree: int) -> List[List[int]]:
        if num_agents <= 1:
            return [[]]

        attach_size = max(1, min(avg_degree, max(1, num_agents // 10)))
        adjacency = [set() for _ in range(num_agents)]

        for left in range(attach_size):
            for right in range(left + 1, attach_size):
                adjacency[left].add(right)
                adjacency[right].add(left)

        targets: List[int] = []
        for node_id in range(attach_size):
            targets.extend([node_id] * max(1, len(adjacency[node_id])))

        for new_node in range(attach_size, num_agents):
            if not targets:
                selected = random.sample(range(new_node), k=min(attach_size, new_node))
            else:
                selected_set = set()
                while len(selected_set) < min(attach_size, new_node):
                    selected_set.add(random.choice(targets))
                selected = list(selected_set)

            for old_node in selected:
                adjacency[new_node].add(old_node)
                adjacency[old_node].add(new_node)

            targets.extend(selected)
            targets.extend([new_node] * len(selected))

        return [sorted(list(neighbors)) for neighbors in adjacency]


class RecommendationParameterInferer:
    """用于反推推荐系统参数的 EM 风格求解器。"""

    def __init__(
        self,
        num_agents: int = 1500,
        avg_degree: int = 20,
        verified_ratio: float = 0.01,
        min_scaled_target: int = 3,
        n_trials_per_story: int = 40,
        n_trials_per_weight: int = 100,
        n_simulations_per_trial: int = 5,
    ):
        self.duration = 24
        self.p_online = 0.1
        self.num_agents = num_agents
        self.min_scaled_target = min_scaled_target
        self.n_trials_per_story = n_trials_per_story
        self.n_trials_per_weight = n_trials_per_weight
        self.n_simulations_per_trial = n_simulations_per_trial
        self.objective_weights = self._normalize_objective_weights(
            {
                'view': 0.4,
                'repost': 0.4,
                'rate': 0.2,
            }
        )
        self.abm = SocialABM(
            num_agents=num_agents,
            avg_degree=avg_degree,
            verified_ratio=verified_ratio,
            p_online=self.p_online,
        )
        self.representative_stories: Dict[str, Dict[str, float]] = {}
        self.calibrated_probs: Dict[str, Dict[str, float]] = {}
        self.weight_fit_diagnostics: Dict[str, Dict[str, float]] = {}

    def select_representative_stories(
        self,
        observations: Sequence[StoryObservation],
        anchor_percentile: float = 0.8,
    ) -> Dict[str, Dict[str, float]]:
        """根据观测数据筛选代表性内容，并转换成 ABM 规模下的目标值。"""
        valid = [item for item in observations if item.repost_count > 0 and item.view_count > 100]
        if not valid:
            self.representative_stories = {}
            return {}

        view_values = np.array([item.view_count for item in valid], dtype=float)
        target_view_anchor = float(np.quantile(view_values, anchor_percentile))
        if target_view_anchor <= 100:
            target_view_anchor = float(np.max(view_values))
        global_scale_ratio = self.num_agents / max(target_view_anchor, 1.0)

        scaled_items: List[Tuple[StoryObservation, Dict[str, float]]] = []
        for item in valid:
            scaled_repost_target = min(
                float(round(item.repost_count * global_scale_ratio)),
                float(self.num_agents),
            )
            if scaled_repost_target < self.min_scaled_target:
                continue
            scaled_items.append(
                (
                    item,
                    {
                        'real_repost': float(item.repost_count),
                        'view_count': float(item.view_count),
                        'real_rate': float(item.repost_count / max(item.view_count, 1.0)),
                        'scaled_target': scaled_repost_target,
                        'scaled_repost_target': scaled_repost_target,
                        'scaled_view_target': float(item.view_count * global_scale_ratio),
                    },
                )
            )

        if not scaled_items:
            self.representative_stories = {}
            return {}

        scaled_values = np.array(
            [story_info['scaled_repost_target'] for _, story_info in scaled_items],
            dtype=float,
        )
        rate_values = np.array(
            [story_info['real_rate'] for _, story_info in scaled_items],
            dtype=float,
        )
        selected: Dict[str, Dict[str, float]] = {}
        repost_percentiles = np.percentile(scaled_values, np.arange(10, 101, 10))
        rate_percentiles = np.percentile(rate_values, np.arange(10, 101, 10))

        for threshold in repost_percentiles:
            candidate = min(
                scaled_items,
                key=lambda pair: abs(pair[1]['scaled_repost_target'] - threshold),
            )
            selected[candidate[0].story_id] = dict(candidate[1])

        for threshold in rate_percentiles:
            candidate = min(
                scaled_items,
                key=lambda pair: abs(pair[1]['real_rate'] - threshold),
            )
            selected[candidate[0].story_id] = dict(candidate[1])

        self.representative_stories = selected
        return selected

    def calibrate_probabilities(self, current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """固定权重，校准单条内容的转发与拒绝概率。"""
        calibrated: Dict[str, Dict[str, float]] = {}
        for story_id, info in self.representative_stories.items():
            best_loss = float('inf')
            best_result = {
                'p_re': 0.1,
                'p_rj': 0.1,
                'mean_scaled_repost': 0.0,
                'mean_scaled_view': 0.0,
                'mean_rate': 0.0,
                'view_loss': float('inf'),
                'repost_loss': float('inf'),
                'rate_loss': float('inf'),
            }

            for _ in range(self.n_trials_per_story):
                p_re = random.uniform(0.001, 0.99)
                p_rj = random.uniform(0.0, max(0.0, 1.0 - p_re - 0.001))
                mean_scaled_repost, mean_scaled_view = self._simulate_story_statistics(
                    story_id=story_id,
                    p_re=p_re,
                    p_rj=p_rj,
                    weights=current_weights,
                )
                story_loss = self._calculate_story_loss(
                    story_info=info,
                    mean_scaled_repost=mean_scaled_repost,
                    mean_scaled_view=mean_scaled_view,
                )
                if story_loss['total_loss'] < best_loss:
                    best_loss = story_loss['total_loss']
                    best_result = {
                        'p_re': p_re,
                        'p_rj': p_rj,
                        'mean_scaled_repost': mean_scaled_repost,
                        'mean_scaled_view': mean_scaled_view,
                        'mean_rate': story_loss['mean_rate'],
                        'view_loss': story_loss['view_loss'],
                        'repost_loss': story_loss['repost_loss'],
                        'rate_loss': story_loss['rate_loss'],
                    }

            calibrated[story_id] = {
                **best_result,
                'real_repost': info['real_repost'],
                'view_count': info['view_count'],
                'real_rate': info['real_rate'],
                'scaled_target': info['scaled_target'],
                'scaled_repost_target': info['scaled_repost_target'],
                'scaled_view_target': info['scaled_view_target'],
                'best_loss': best_loss,
            }

        self.calibrated_probs = calibrated
        return calibrated

    def optimize_recommendation_weights(
        self,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """固定概率，搜索推荐系统权重。"""
        if not self.calibrated_probs:
            raise RuntimeError('请先完成概率校准')

        best_loss = float('inf')
        best_repost_rmse = float('inf')
        best_weights = self._normalize_weights(current_weights or {
            'chrono': 0.15,
            'belief': 0.45,
            'pop': 0.30,
            'rand': 0.10,
            'decay_lambda': 0.2,
        })
        best_diagnostics: Dict[str, Dict[str, float]] = {}

        for _ in range(self.n_trials_per_weight):
            raw_weights = {
                'chrono': random.random(),
                'belief': random.random(),
                'pop': random.random(),
                'rand': random.uniform(0.0, 0.1),
                'decay_lambda': 10 ** random.uniform(-2, 0),
            }
            weights = self._normalize_weights(raw_weights)
            losses = []
            repost_errors = []
            diagnostics: Dict[str, Dict[str, float]] = {}
            for story_id, story_info in self.calibrated_probs.items():
                mean_scaled_repost, mean_scaled_view = self._simulate_story_statistics(
                    story_id=story_id,
                    p_re=story_info['p_re'],
                    p_rj=story_info['p_rj'],
                    weights=weights,
                )
                story_loss = self._calculate_story_loss(
                    story_info=story_info,
                    mean_scaled_repost=mean_scaled_repost,
                    mean_scaled_view=mean_scaled_view,
                )
                losses.append(story_loss['total_loss'])
                repost_errors.append(
                    (mean_scaled_repost - story_info['scaled_repost_target']) ** 2
                )
                diagnostics[story_id] = {
                    'mean_scaled_repost': mean_scaled_repost,
                    'mean_scaled_view': mean_scaled_view,
                    'mean_rate': story_loss['mean_rate'],
                    'view_loss': story_loss['view_loss'],
                    'repost_loss': story_loss['repost_loss'],
                    'rate_loss': story_loss['rate_loss'],
                    'total_loss': story_loss['total_loss'],
                }

            objective_loss = float(np.mean(losses)) if losses else float('inf')
            repost_rmse = math.sqrt(sum(repost_errors) / max(len(repost_errors), 1))
            if objective_loss < best_loss:
                best_loss = objective_loss
                best_repost_rmse = repost_rmse
                best_weights = weights
                best_diagnostics = diagnostics

        self.weight_fit_diagnostics = best_diagnostics

        return {
            **best_weights,
            'best_loss': best_loss,
            'best_repost_rmse': best_repost_rmse,
            'duration': float(self.duration),
            'p_online': self.p_online,
            'objective_weight_view': self.objective_weights['view'],
            'objective_weight_repost': self.objective_weights['repost'],
            'objective_weight_rate': self.objective_weights['rate'],
        }

    def run_em_calibration_loop(self, max_iterations: int = 3) -> Dict[str, float]:
        """执行 EM 风格的交替校准。"""
        current_weights = {
            'chrono': 0.15,
            'belief': 0.45,
            'pop': 0.30,
            'rand': 0.10,
            'decay_lambda': 0.2,
        }

        for _ in range(max_iterations):
            normalized = self._normalize_weights(current_weights)
            self.calibrate_probabilities(current_weights=normalized)
            current_weights = self.optimize_recommendation_weights(current_weights=normalized)

        return current_weights

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        rand_weight = min(max(weights.get('rand', 0.1), 0.0), 0.3)
        budget = 1.0 - rand_weight
        chrono = max(weights.get('chrono', 0.0), 0.0)
        belief = max(weights.get('belief', 0.0), 0.0)
        pop = max(weights.get('pop', 0.0), 0.0)
        total = chrono + belief + pop
        if total <= 0:
            chrono = belief = pop = 1.0 / 3.0
            total = 1.0
        return {
            'chrono': chrono / total * budget,
            'belief': belief / total * budget,
            'pop': pop / total * budget,
            'rand': rand_weight,
            'decay_lambda': max(float(weights.get('decay_lambda', 0.2)), 1e-4),
        }

    @staticmethod
    def _normalize_objective_weights(weights: Dict[str, float]) -> Dict[str, float]:
        view_weight = max(float(weights.get('view', 0.4)), 0.0)
        repost_weight = max(float(weights.get('repost', 0.4)), 0.0)
        rate_weight = max(float(weights.get('rate', 0.2)), 0.0)
        total = view_weight + repost_weight + rate_weight
        if total <= 0:
            view_weight = repost_weight = 0.4
            rate_weight = 0.2
            total = 1.0
        return {
            'view': view_weight / total,
            'repost': repost_weight / total,
            'rate': rate_weight / total,
        }

    def _simulate_story_statistics(
        self,
        story_id: str,
        p_re: float,
        p_rj: float,
        weights: Optional[Dict[str, float]],
    ) -> Tuple[float, float]:
        scaled_reposts: List[float] = []
        scaled_views: List[float] = []
        for simulation_index in range(self.n_simulations_per_trial):
            rng_seed = self._build_simulation_seed(story_id, simulation_index)
            history, view_count = self.abm.run_simulation(
                p_re=p_re,
                p_rj=p_rj,
                weights=weights,
                duration=self.duration,
                fixed_seed=rng_seed % self.num_agents,
                rng_seed=rng_seed,
            )
            scaled_reposts.append(float(history[-1]))
            scaled_views.append(float(view_count))
        return float(np.mean(scaled_reposts)), float(np.mean(scaled_views))

    def _calculate_story_loss(
        self,
        story_info: Dict[str, float],
        mean_scaled_repost: float,
        mean_scaled_view: float,
    ) -> Dict[str, float]:
        target_view = max(story_info['scaled_view_target'], 1.0)
        target_repost = max(story_info['scaled_repost_target'], 1.0)
        real_rate = story_info['real_rate']
        mean_rate = mean_scaled_repost / max(mean_scaled_view, 1.0)

        view_loss = abs(math.log1p(mean_scaled_view) - math.log1p(target_view))
        repost_loss = abs(math.log1p(mean_scaled_repost) - math.log1p(target_repost))
        rate_loss = abs(mean_rate - real_rate)
        total_loss = (
            self.objective_weights['view'] * view_loss
            + self.objective_weights['repost'] * repost_loss
            + self.objective_weights['rate'] * rate_loss
        )
        return {
            'view_loss': view_loss,
            'repost_loss': repost_loss,
            'rate_loss': rate_loss,
            'mean_rate': mean_rate,
            'total_loss': total_loss,
        }

    @staticmethod
    def _build_simulation_seed(story_id: str, simulation_index: int) -> int:
        text = f'{story_id}:{simulation_index}'
        accumulator = 0
        for index, char in enumerate(text):
            accumulator += (index + 1) * ord(char)
        return accumulator


def build_story_observations(records: Iterable[Dict[str, float]]) -> List[StoryObservation]:
    """把外部表格或数据库记录转换成统一观测结构。"""
    observations: List[StoryObservation] = []
    for index, record in enumerate(records):
        story_id = str(record.get('story_id', index))
        repost_count = float(record.get('repost_count', 0.0))
        view_count = float(record.get('view_count', 0.0))
        observations.append(
            StoryObservation(
                story_id=story_id,
                repost_count=repost_count,
                view_count=view_count,
            )
        )
    return observations
