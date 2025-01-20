import unittest
from src.policy import PolicyNetwork

class TestPolicyNetwork(unittest.TestCase):
    def test_policy_output(self):
        policy = PolicyNetwork(24, 5)
        input_data = torch.randn(1, 24)
        output = policy(input_data)
        self.assertEqual(output.shape, (1, 5))
