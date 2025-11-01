import numpy as np
import tensorflow.compat.v1 as tf
import random
import logging
from collections import deque, namedtuple

logger = logging.getLogger(__name__)

# Define the structure for transitions stored in the replay buffer
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """
    A fixed-size replay buffer using collections.deque for efficient
    O(1) appends and pops.
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        logger.info(f"ReplayBuffer initialized with size {buffer_size}")

    def add(self, state, action, reward, next_state, done):
        """Adds a new transition to the buffer."""
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)
        
    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer.
        
        Args:
            batch_size (int): The number of transitions to sample.
            
        Returns:
            A tuple of (states, actions, rewards, next_states, dones)
        """
        if batch_size > len(self.memory):
            logger.warning(f"Requested batch size {batch_size} but buffer only has {len(self.memory)} elements.")
            batch_size = len(self.memory)
            
        samples = random.sample(self.memory, batch_size)
        
        # Unzip batch of transitions into separate numpy arrays
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class ModelParametersCopier:
    """
    Helper class to copy model parameters from one estimator (Q-Network)
    to another (Target-Network).
    """
    def __init__(self, estimator_from, estimator_to):
        """
        Defines the copy operation graph.
        """
        try:
            e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_from.scope)]
            e1_params = sorted(e1_params, key=lambda v: v.name)
            
            e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_to.scope)]
            e2_params = sorted(e2_params, key=lambda v: v.name)

            self.update_ops = []
            if len(e1_params) != len(e2_params):
                logger.error(f"Param mismatch: {estimator_from.scope} has {len(e1_params)} params, "
                             f"{estimator_to.scope} has {len(e2_params)} params.")
                raise ValueError("Parameter count mismatch between estimators.")

            for e1_v, e2_v in zip(e1_params, e2_params):
                op = e2_v.assign(e1_v)
                self.update_ops.append(op)
            logger.info(f"ModelParameterCopier created for {estimator_from.scope} -> {estimator_to.scope}")

        except Exception as e:
            logger.error(f"Failed to create ModelParametersCopier: {e}")
            raise

    def make_copy(self, sess):
        """
        Executes the copy operation in the given session.
        """
        sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy function.
    
    Args:
        estimator: The Q-Network (Estimator object) to use for predictions.
        nA: Number of valid actions.
        
    Returns:
        A function policy_fn(sess, observation, epsilon)
    """
    def policy_fn(sess, observation, epsilon):
        """
        Takes an observation and epsilon, returns action probabilities.
        """
        A = np.ones(nA, dtype=float) * epsilon / nA
        try:
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
        except Exception as e:
            logger.warning(f"Error during policy prediction: {e}. Returning random policy.")
            # Fallback to random action if prediction fails
            A = np.ones(nA, dtype=float) / nA
        return A
    return policy_fn
