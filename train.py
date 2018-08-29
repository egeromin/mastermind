import tensorflow.contrib.eager as tfe
import tensorflow as tf
import random

from episode import Episode
from policy import Policy
import config


tf.enable_eager_execution()



def train(num_episodes=1000):
    p = Policy()
    for _ in range(num_episodes):
        random_secret = random.randint(0, config.max_guesses - 1)
        e = Episode(p, random_secret)
        history = e.generate()

        print("Episode length: {}".format(len(history)))

        G = -1 

        optimizer = \
            tf.train.GradientDescentOptimizer(
                learning_rate=config.reinforce_alpha*G)

        for i in reversed(range(1, len(history))):
            history_so_far = history[:i]
            next_action, _ = history[i]
            with tfe.GradientTape() as tape:
                action_logits = p(history_so_far, with_softmax=False)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.one_hot(
                        tf.convert_to_tensor([next_action]),
                        config.max_guesses),
                    logits=action_logits
                )

            grads = tape.gradient(loss, p.variables)
            optimizer.apply_gradients(zip(grads, p.variables))

            G -= 1
            optimizer._learning_rate = G * config.reinforce_alpha
            optimizer._learning_rate_tensor = None
            # hack. Should be able to pass a callable as learning_rate, see
            # https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer#args
            # can I perhaps submit a PR to fix this bug?


if __name__ == "__main__":
    train()

