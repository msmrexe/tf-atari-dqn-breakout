import tensorflow.compat.v1 as tf
import logging

logger = logging.getLogger(__name__)

class StateProcessor:
    """
    Processes a raw Atari frame (210x160x3) and converts it
    to a processed (84x84x1) grayscale frame.
    
    This uses a TF graph for efficient preprocessing.
    """
    def __init__(self):
        try:
            with tf.variable_scope("state_processor"):
                # Input placeholder for a single raw frame
                self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
                
                # Convert to grayscale
                grayscale = tf.image.rgb_to_grayscale(self.input_state)
                
                # Crop the relevant playing area (removes score, etc.)
                cropped = tf.image.crop_to_bounding_box(grayscale, 34, 0, 160, 160)
                
                # Resize to 84x84
                resized = tf.image.resize_images(
                    cropped, 
                    [84, 84], 
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
                
                # Squeeze to [84, 84]
                self.output = tf.squeeze(resized)
                logger.debug("StateProcessor graph built successfully.")
        except Exception as e:
            logger.error(f"Failed to build StateProcessor graph: {e}")
            raise

    def process(self, sess, state):
        """
        Processes a raw frame using the initialized TF graph.
        
        Args:
            sess: A Tensorflow session.
            state: A [210, 160, 3] Atari RGB State.

        Returns:
            A processed [84, 84] grayscale state.
        """
        if state is None:
            logger.warning("Received None state in StateProcessor.process")
            return None
        try:
            return sess.run(self.output, {self.input_state: state})
        except Exception as e:
            logger.error(f"Error during state processing: {e}")
            return None
