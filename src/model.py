import tensorflow.compat.v1 as tf
import os
import logging
from .utils import VALID_ACTIONS

tf.disable_v2_behavior()
logger = logging.getLogger(__name__)

class Estimator:
    """
    Q-Value Estimator Network (DQN).
    This network is used for both the Q-Network and the Target Network.
    """
    
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        
        with tf.variable_scope(scope):
            # Step counter
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # Build the graph
            try:
                self._build_model()
                logger.info(f"Model graph built for scope: {scope}")
            except Exception as e:
                logger.error(f"Failed to build model for scope '{scope}': {e}")
                raise
            
            # TensorBoard summaries
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, f"summaries_{scope}")
                os.makedirs(summary_dir, exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
                logger.info(f"Summaries will be written to: {summary_dir}")

    def _build_model(self):
        """
        Builds the Tensorflow graph using the architecture from the
        "Playing Atari with Deep Reinforcement Learning" paper.
        
        Uses tf.compat.v1.layers for consistency.
        """
        
        # --- Placeholders ---
        # Input: [None, 84, 84, 4] (batch_size, height, width, frame_stack)
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # Target Q-values
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Actions taken
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # --- Preprocessing ---
        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        X_float = tf.cast(self.X_pl, tf.float32) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # --- Convolutional Layers ---
        # Conv 1: 32 filters, 8x8, stride 4
        conv1 = tf.layers.conv2d(
            X_float, 32, 8, 4, activation=tf.nn.relu, name="conv1"
        )
        # Conv 2: 64 filters, 4x4, stride 2
        conv2 = tf.layers.conv2d(
            conv1, 64, 4, 2, activation=tf.nn.relu, name="conv2"
        )
        # Conv 3: 64 filters, 3x3, stride 1
        conv3 = tf.layers.conv2d(
            conv2, 64, 3, 1, activation=tf.nn.relu, name="conv3"
        )
        
        # --- Dense (Fully Connected) Layers ---
        flattened = tf.layers.flatten(conv3)
        
        # Dense 1: 512 units
        fc1 = tf.layers.dense(
            flattened, 512, activation=tf.nn.relu, name="fc1"
        )
        
        # Output Layer: NUM_ACTIONS units
        self.predictions = tf.layers.dense(
            fc1, len(VALID_ACTIONS), name="predictions"
        )

        # --- Loss and Optimization ---
        
        # Get Q-values for the specific actions taken
        # 1. Create indices for action_pl
        action_indices = tf.stack([tf.range(batch_size), self.actions_pl], axis=1)
        # 2. Gather the Q-values
        self.action_predictions = tf.gather_nd(self.predictions, action_indices)

        # Loss (Huber Loss is often used, but squared diff is from original paper)
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer (RMSProp as in the paper)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # --- Summaries for TensorBoard ---
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values (Q-values) for a given state.
        
        Args:
            sess: Tensorflow session.
            s: State input of shape [batch_size, 84, 84, 4].
        
        Returns:
            Tensor of shape [batch_size, NUM_VALID_ACTIONS]
        """
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Performs one update step (gradient descent) on the network.
        
        Args:
            sess: Tensorflow session.
            s: State batch [batch_size, 84, 84, 4].
            a: Action batch [batch_size].
            y: Target Q-value batch [batch_size].
            
        Returns:
            The calculated loss on the batch.
        """
        try:
            feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
            
            summaries, global_step, _, loss = sess.run(
                [self.summaries, self.global_step, self.train_op, self.loss],
                feed_dict
            )
            
            if self.summary_writer:
                self.summary_writer.add_summary(summaries, global_step)
                
            return loss
        except Exception as e:
            logger.error(f"Error during model update: {e}")
            logger.debug(f"Shapes - s: {s.shape}, a: {a.shape}, y: {y.shape}")
            raise
