import random
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
from typing import Optional


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def build_network(state_size: int, action_size: int, hidden_units: list) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(state_size,))
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(action_size, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_units: Optional[list] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._step_count = 0

        if hidden_units is None:
            hidden_units = [256, 256, 128]

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.policy_net = build_network(state_size, action_size, hidden_units)
        self.target_net = build_network(state_size, action_size, hidden_units)
        self.target_net.set_weights(self.policy_net.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.policy_net(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            next_actions = tf.argmax(self.policy_net(next_states, training=False), axis=1)
            next_q_values = self.target_net(next_states, training=False)
            next_q = tf.gather(next_q_values, next_actions, batch_dims=1)
            targets_full = self.policy_net(states, training=True)
            targets = targets_full.numpy()

            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * next_q[i].numpy()

            targets_tensor = tf.constant(targets, dtype=tf.float32)
            q_values = self.policy_net(states, training=True)
            loss = self.loss_fn(targets_tensor, q_values)

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss)

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.policy_net.save(p / "policy_net.keras")
        self.target_net.save(p / "target_net.keras")

    def load(self, path: str):
        p = Path(path)
        self.policy_net = tf.keras.models.load_model(p / "policy_net.keras")
        self.target_net = tf.keras.models.load_model(p / "target_net.keras")
        self.epsilon = self.epsilon_min
