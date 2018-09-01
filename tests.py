import unittest
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tempfile
import os
import numpy as np
import ipdb
import itertools

from policy import Policy

# tf.enable_eager_execution()


class TestCheckpoints(unittest.TestCase):

    def test_save_restore(self):
        pol = Policy()
        episode = [(0, 0), (1, 0), (2, 3)]
        expected = pol(episode).numpy()
        with tempfile.TemporaryDirectory() as tdir:
            path = os.path.join(tdir, "checkpt")

            saver = tfe.Saver(pol.named_variables)
            saver.save(path)

            pol2 = Policy()
            def diff():
                actual = pol2(episode).numpy()
                return np.linalg.norm(actual-expected)

            self.assertGreater(diff(), 0.0001)
            saver = tfe.Saver(pol2.named_variables)
            saver.restore(path)
            self.assertGreaterEqual(0.00001, diff())


if __name__ == "__main__":
    tf.set_random_seed(230)
    assert(tf.executing_eagerly())
    unittest.main(module='tests')
    # need to specify 'tests' as the module
    # since there's a competing 'main' method somewhere

