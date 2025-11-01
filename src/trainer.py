import numpy as np
import tensorflow.compat.v1 as tf
import os
import psutil
import itertools
import logging
from gymnasium.wrappers.record_video import RecordVideo

from .agent import ReplayBuffer, ModelParametersCopier, make_epsilon_greedy_policy
from .utils import VALID_ACTIONS, EpisodeStats

logger = logging.getLogger(__name__)

def populate_replay_buffer(sess, env, state_processor, replay_buffer, init_size):
    """
    Populates the replay buffer with initial random experiences.
    """
    logger.info(f"Populating replay memory with {init_size} transitions...")
    
    state, _ = env.reset()
    state = state_processor.process(sess, state)
    # Stack 4 initial identical frames
    state = np.stack([state] * 4, axis=2)
    
    for i in range(init_size):
        if i % 1000 == 0:
            logger.debug(f"Populating buffer... {i}/{init_size}")
            
        action = random.choice(VALID_ACTIONS)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        
        next_observation_p = state_processor.process(sess, next_observation)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_observation_p, 2), axis=2)
        
        done = terminated or truncated
        
        replay_buffer.add(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
    logger.info("Replay memory populated.")


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50,
                    max_steps_per_episode=10000):
    """
    The main Deep Q-Learning training loop.
    
    Args:
        sess: Tensorflow Session object.
        env: The gymnasium environment.
        q_estimator: The Q-Network.
        target_estimator: The Target Network.
        state_processor: The StateProcessor instance.
        num_episodes: Number of episodes to run.
        experiment_dir: Directory to save checkpoints and summaries.
        ... (other hyperparameters)
        
    Yields:
        (total_t, episode_stats, loss) for each completed episode.
    """

    # --- Setup ---
    replay_buffer = ReplayBuffer(replay_memory_size)
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    current_process = psutil.Process()
    
    # --- Checkpoints and Video ---
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    monitor_path = os.path.join(experiment_dir, "monitor")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(monitor_path, exist_ok=True)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        logger.info(f"Loading model checkpoint {latest_checkpoint}...")
        saver.restore(sess, latest_checkpoint)
    
    total_t = sess.run(tf.train.get_global_step())

    # Epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Epsilon-greedy policy
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # --- Populate Replay Buffer ---
    populate_replay_buffer(sess, env, state_processor, replay_buffer, replay_memory_init_size)

    # --- Video Recording Wrapper ---
    try:
        env = RecordVideo(
            env, 
            video_folder=monitor_path, 
            episode_trigger=lambda ep_id: ep_id % record_video_every == 0
        )
    except Exception as e:
        logger.warning(f"Could not attach RecordVideo wrapper: {e}. Continuing without video.")

    # --- Main Training Loop ---
    for i_episode in range(num_episodes):
        
        # Save checkpoint
        saver.save(sess, checkpoint_path)

        # Reset environment
        state, _ = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        for t in itertools.count():
            
            # 1. Select Epsilon
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            
            # 2. Update Target Network
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make_copy(sess)
                logger.info(f"\nStep {total_t}: Copied model parameters to target network.")

            # 3. Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            
            next_observation_p = state_processor.process(sess, next_observation)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_observation_p, 2), axis=2)
            
            done = terminated or truncated
            
            # Stop if agent is stuck
            if t >= max_steps_per_episode:
                logger.warning(f"Episode {i_episode+1} reached max steps. Truncating.")
                done = True

            # 4. Save to Replay Buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Update stats
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # 5. Sample minibatch
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = \
                replay_buffer.sample(batch_size)

            # 6. Calculate targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                            discount_factor * np.amax(q_values_next, axis=1)

            # 7. Perform gradient descent
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break
                
            state = next_state
            total_t += 1

        # --- End of Episode ---
        
        # Add episode summaries to TensorBoard
        try:
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
            episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
            episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
            episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
            episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memory_usage_percent")
            q_estimator.summary_writer.add_summary(episode_summary, i_episode)
            q_estimator.summary_writer.flush()
        except Exception as e:
            logger.warning(f"Failed to write episode summary: {e}")

        # Yield stats for logging
        yield total_t, EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1]), loss

    logger.info("Training complete.")
