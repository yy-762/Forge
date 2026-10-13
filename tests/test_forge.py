import inspect
import subprocess
import sys
import unittest
import numpy as np
import torch
from forge.policy import Actor, Critic
from forge.semantic import Memory, Record, guidance


class ForgeTests(unittest.TestCase):
    def test_actor_interface_and_feasibility(self):
        self.assertEqual(list(inspect.signature(Actor.forward).parameters), ['self', 'observation'])
        self.assertEqual(list(inspect.signature(Critic.forward).parameters), ['self', 'state'])
        actor = Actor()
        self.assertFalse(any('mem' in name for name, _ in actor.named_parameters()))
        for deterministic in (False, True):
            packets, power, logp = actor.act(torch.randn(64, 9), deterministic)
            self.assertTrue(torch.all(packets.sum(-1) == 10))
            self.assertTrue(torch.all(packets >= 0))
            self.assertTrue(torch.allclose(power.sum(-1), torch.ones(64), atol=1e-6))
            self.assertTrue(torch.isfinite(logp).all())
            p, d = actor.distributions(torch.zeros(8, 9))
            counts, watts, old = actor.act(torch.zeros(8, 9))
            self.assertTrue(torch.allclose(old, p.log_prob(counts) + d.log_prob(watts)))

    def test_invalid_and_low_confidence_records(self):
        with self.assertRaises(ValueError):
            Record.validate({}, 0)
        record = Record('low quality', 'concentrated packets', 'violation', 'split traffic',
                        ['deadline'], 1, 'split', 'split', .4, 0)
        bank = Memory(); bank.add(record)
        retrieved = bank.retrieve('low quality', ['deadline'], 0)
        self.assertEqual(retrieved, [])
        bonus, _ = guidance([4, 3, 3], [1/3]*3, np.zeros(9), retrieved)
        self.assertEqual(bonus, 0)

    def test_bonus_bound_and_semantic_direction(self):
        record = Record('high traffic', 'concentrate', 'success', 'use best quality link',
                        ['routing'], 1, 'concentrate', 'concentrate', 1., 0)
        obs = np.array([0, .5, .9, 0, .5, .4, 0, .5, .1])
        aligned, _ = guidance([8, 1, 1], [.8, .1, .1], obs, [(record, 1.)])
        mismatched, _ = guidance([1, 1, 8], [.1, .1, .8], obs, [(record, 1.)])
        self.assertTrue(0 <= mismatched < aligned <= .5)
        self.assertEqual(guidance([8, 1, 1], [.8, .1, .1], obs, [(record, 1.)], 0)[0], 0.)

    def test_deployment_imports_no_semantic_module(self):
        code = 'import evaluate,sys; assert "forge.semantic" not in sys.modules; assert "forge.train" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_risk_targets_and_terminal_boundary(self):
        from forge.train import discounted_costs, gae
        targets = discounted_costs([1., 2.], [0., 1.], .9)
        np.testing.assert_allclose(targets, [2.8, 2.])
        isolated = discounted_costs([1., 2.], [1., 1.], .9)
        np.testing.assert_allclose(isolated, [1., 2.])
        returns, advantage = gae([1., 2.], [0., 0.], [0., 1.], .9, 1.)
        np.testing.assert_allclose(returns, targets)
        np.testing.assert_allclose(advantage, targets)


if __name__ == '__main__':
    unittest.main()
