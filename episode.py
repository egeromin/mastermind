import config
import numpy as np
import ipdb

class Episode:

    start_guess = config.max_guesses
    start_feedback = config.max_feedback
    triangle_numbers = [0, 1, 3, 6, 10]
    max_episode_length = 30

    def __init__(self, policy, secret):
        self.policy = policy
        self.secret = secret
        self.secret_sorted = sorted(secret)

    @staticmethod
    def _index_from_number(number):
        """
        Convert a 4-digit guess to an index between 0 and 6**4 
        """
        assert(len(number) <= 4)
        assert(set(number) <= set(map(str, range(5))))
        return int(number, base=6)

    @staticmethod
    def _number_from_index(index):
        assert(0 <= index < config.max_guesses)
        digits = []
        while index > 0:
            digits.append(str(index % 6))
            index = index // 6
        return "".join(reversed(digits)).zfill(4)

    @classmethod
    def _index_from_feedback(cls, feedback):
        """
        Convert a feedback dict to an index

        Use triangle numbers to do this. E.g.

        ++++ -> 14
        / -> 0
        +-- -> 7
        """
        assert(list(sorted(feedback.keys())) == ['+', '-'])
        return cls.triangle_numbers[sum(feedback.values())] + feedback['+']
        
    def _feedback_from_guess(self, guess):
        def negative_hamming(a, b):
            """'opposite' of the Hamming distance

            Counts the number of symbols that are equal
            """
            return sum([x == y for x, y in zip(a, b)])

        compare = negative_hamming(self.secret, guess)
        compare_sorted = negative_hamming(self.secret_sorted, guess)
        return {
            '+': compare,
            '-': compare_sorted - compare
        }

    @staticmethod
    def _select_next_action(action_distribution):
        """Sample the next action based on the action distribution"""
        action_dist = action_distribution.numpy().reshape(-1)
        # `action_distribution` is a tensor
        return np.random.choice(action_dist.shape[0],
                                p=action_dist)

    def generate(self):
        episode = [(self.start_guess, self.start_feedback)]

        for _ in range(self.max_episode_length):
            guess = self._select_next_action(self.policy(episode))
            feedback = self._index_from_feedback(
                self._feedback_from_guess(
                    self._number_from_index(guess)))
            if feedback == 14:
                break  # this corresponds to a correct guess
            episode.append((guess, feedback))

        return episode

