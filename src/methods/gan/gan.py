import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pm4py.objects.log.obj import EventLog
import pm4py.util.xes_constants as xes
from pm4py.statistics.attributes.log import get as attributes_get
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict
import os


def set_seeds(seed=42):
    """Set seeds for reproducibility across libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    # MODIFICATION: Added TF function for determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1' 
    tf.config.experimental.enable_op_determinism() # Enforce determinism
    tf.random.set_seed(seed)
    np.random.seed(seed)

def preprocess_log_for_gan(log: EventLog, activity_key: str = xes.DEFAULT_NAME_KEY) -> Tuple[np.ndarray, Dict, int, int]:
    """
    Converts an event log into a feature matrix for the GAN, including
    control-flow (one-hot) and time features (time since last event).
    """
    print("Preprocessing log: Extracting features...")
    all_activities = attributes_get.get_attribute_values(log, activity_key)
    unique_activities = sorted(list(all_activities.keys()))
    activity_to_int = {act: i + 1 for i, act in enumerate(unique_activities)}
    vocab_size = len(unique_activities) + 1

    integer_traces, transition_times = [], []
    for trace in log:
        integer_traces.append([activity_to_int[event[activity_key]] for event in trace])
        timestamps = [event[xes.DEFAULT_TIMESTAMP_KEY] for event in trace]
        times = [0.0] + [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        transition_times.append(times)

    max_length = max(len(seq) for seq in integer_traces)
    padded_int_traces = pad_sequences(integer_traces, maxlen=max_length, padding='post')
    padded_times = pad_sequences(transition_times, maxlen=max_length, padding='post', dtype='float32')

    scaler = MinMaxScaler()
    normalized_times = scaler.fit_transform(padded_times.reshape(-1, 1)).reshape(padded_times.shape[0], max_length, 1)

    control_flow_matrix = np.eye(vocab_size)[padded_int_traces]
    final_features = np.concatenate([control_flow_matrix, normalized_times], axis=2).astype('float32')
    
    print(f"Preprocessing complete. Feature matrix shape: {final_features.shape}")
    return final_features, activity_to_int, max_length, vocab_size


class StableDriftDetector:
    """
    MODIFICATION: Switched to a GAN with Zero-Centered Gradient Penalty (0-GP)
    based on the provided research paper for improved stability and generalization.
    """
    def __init__(self, vocab_size, num_time_features, max_seq_length, latent_dim=64, seed=42):
        self.vocab_size = vocab_size
        self.num_time_features = num_time_features
        self.num_features = vocab_size + num_time_features
        self.max_seq_length = max_seq_length
        self.latent_dim = latent_dim
        self.seed = seed
        
        set_seeds(self.seed)
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.7)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.7)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def _build_generator(self):
        # This architecture is fine, no major changes needed
        initializer = tf.keras.initializers.GlorotNormal(seed=self.seed)
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        
        x = tf.keras.layers.Dense(256, kernel_initializer=initializer)(noise)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.max_seq_length * 256, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Reshape((self.max_seq_length, 256))(x)
        
        output_control_flow = tf.keras.layers.Dense(self.vocab_size, activation='softmax', name='control_flow_output')(x)
        output_time = tf.keras.layers.Dense(self.num_time_features, activation='sigmoid', name='time_output')(x)
        combined_output = tf.keras.layers.Concatenate(name='combined_output')([output_control_flow, output_time])
        
        return tf.keras.Model(noise, combined_output, name='generator')

    def _build_discriminator(self):
        """
        Build the Discriminator. The final activation is sigmoid for the original GAN loss.
        """
        initializer = tf.keras.initializers.GlorotNormal(seed=self.seed)
        sequence_input = tf.keras.layers.Input(shape=(self.max_seq_length, self.num_features))
        
        x = tf.keras.layers.Conv1D(128, 3, padding='same', kernel_initializer=initializer)(sequence_input)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1, seed=self.seed)(x)
        x = tf.keras.layers.Conv1D(256, 3, padding='same', kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.Dropout(0.1, seed=self.seed)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1, seed=self.seed)(x)
        # Using sigmoid for the original GAN loss objective L
        output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(x)
        
        return tf.keras.Model(sequence_input, output, name='discriminator')

    def _discriminator_loss(self, real_output, fake_output):
        """ The original GAN loss function, which the discriminator seeks to maximize """
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

    def _generator_loss(self, fake_output):
        """ Non-saturating generator loss """
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

    def _gradient_penalty(self, real_traces, fake_traces):
        """ 
        MODIFICATION: Calculates the Zero-Centered Gradient Penalty (0-GP).
        This replaces the 1-GP from WGAN-GP. The goal is to push the norm of the 
        gradient towards 0 on points interpolated between real and fake data.
        """
        alpha = tf.random.uniform([real_traces.shape[0], 1, 1], 0., 1.)
        # Interpolate points between real and fake samples [cite: 259]
        interpolated = (alpha * real_traces) + ((1 - alpha) * fake_traces)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        
        # Calculate the squared L2 norm of the gradients
        squared_norm = tf.reduce_sum(tf.square(grads), axis=np.arange(1, len(grads.shape)))
        
        # The penalty is the mean of these squared norms 
        gp = tf.reduce_mean(squared_norm)
        return gp

    @tf.function
    def _train_step(self, real_traces, gp_weight=10.0, n_discriminator=1):
        """ MODIFICATION: A full 0-GP GAN training step. """
        noise = tf.random.normal([real_traces.shape[0], self.latent_dim])

        # Train Discriminator
        for _ in range(n_discriminator):
            with tf.GradientTape() as tape:
                fake_traces = self.generator(noise, training=True)
                
                real_output = self.discriminator(real_traces, training=True)
                fake_output = self.discriminator(fake_traces, training=True)

                d_loss = self._discriminator_loss(real_output, fake_output)
                
                # Apply the Zero-Centered Gradient Penalty from the paper
                gp = self._gradient_penalty(real_traces, fake_traces)
                total_d_loss = d_loss + gp * gp_weight

            d_grads = tape.gradient(total_d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape:
            fake_traces = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_traces, training=True)
            g_loss = self._generator_loss(fake_output)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return total_d_loss, g_loss

    def train(self, training_data, epochs=100, batch_size=32, verbose=True):
        set_seeds(self.seed)
        num_batches = training_data.shape[0] // batch_size
        if num_batches == 0:
          num_batches = 1
        
        
        for epoch in range(epochs):
            indices = np.arange(training_data.shape[0])
            np.random.shuffle(indices)
            epoch_d_loss, epoch_g_loss = 0, 0

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]
                real_batch = tf.constant(training_data[batch_indices])
                
                d_loss, g_loss = self._train_step(real_batch)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}")

    def detect_drift(self, feature_matrix, window_size=30, threshold_factor=3.0):
        """
        The discriminator's output is a probability. Lower probabilities suggest
        a sample is more likely "fake" or anomalous (a potential drift).
        """
        set_seeds(self.seed)
        scores = self.discriminator.predict(feature_matrix, batch_size=64, verbose=0).flatten()
        
        # We invert the scores because a lower score (more fake) indicates drift
        anomaly_scores = 1.0 - scores
        
        window_scores = np.convolve(anomaly_scores, np.ones(window_size), 'valid') / window_size
        
        baseline = window_scores[:200]
        # A drift is detected if the anomaly score rises significantly ABOVE the baseline
        threshold = np.mean(baseline) + threshold_factor * np.std(baseline)
        
        drift_indices = np.where(window_scores > threshold)[0]
        
        if len(drift_indices) == 0:
            return window_scores, [], threshold
            
        consolidated = [drift_indices[0]]
        for idx in drift_indices[1:]:
            if idx - consolidated[-1] >= window_size // 2:
                consolidated.append(idx)
        
        drift_points = [int(i + window_size) for i in consolidated]
        return window_scores, drift_points, threshold

# The AdaptiveDriftDetector will automatically inherit the new 0-GP logic
# without any changes to its own code.

class AdaptiveDriftDetector(StableDriftDetector):
    """
    Extends the stable detector with continual learning capabilities. It automatically
    uses the 0-GP logic from its parent class.
    """
    def __init__(self, vocab_size, num_time_features, max_seq_length, latent_dim=64, seed=42, 
                 buffer_capacity=500, min_finetune_size=5):
        """
        Initializes the adaptive detector with a buffer for storing normal data.
        
        Args:
            buffer_capacity (int): The maximum number of traces to store in the normal buffer.
            min_finetune_size (int): The minimum number of new traces required to trigger fine-tuning.
        """
        super().__init__(vocab_size, num_time_features, max_seq_length, latent_dim, seed)
        self.buffer_capacity = buffer_capacity
        self.min_finetune_size = min_finetune_size
        # Initialize the buffer with the initial training data
        self.normal_buffer = None

    def train(self, training_data, epochs=100, batch_size=32, verbose=True):
        """
        Initial training for the model. Also populates the initial normal buffer.
        """
        # Call the parent training method
        super().train(training_data, epochs, batch_size, verbose)
        # Populate the buffer with the initial "normal" data
        self._update_buffer(training_data)
        print(f"Initial training complete. Normal buffer populated with {len(self.normal_buffer)} traces.")

    def _update_buffer(self, new_normal_data):
        """Helper to update the buffer with new data, ensuring it doesn't exceed capacity."""
        if self.normal_buffer is None:
            self.normal_buffer = new_normal_data
        else:
            # Append new data and keep the most recent traces up to the capacity
            self.normal_buffer = np.concatenate([self.normal_buffer, new_normal_data], axis=0)
        
        if len(self.normal_buffer) > self.buffer_capacity:
            self.normal_buffer = self.normal_buffer[-self.buffer_capacity:]

    def adapt(self, new_normal_data, finetune_epochs=20, batch_size=32, replay_ratio=0.1):
        """
        Fine-tunes the model on new data to adapt to a concept drift.

        Args:
            new_normal_data (np.ndarray): The traces representing the new normal process.
            finetune_epochs (int): Number of epochs for fine-tuning.
            replay_ratio (float): The fraction of old data from the buffer to mix in
                                  to prevent catastrophic forgetting.
        """
        print(f"\n---  Adapting to new concept drift with {len(new_normal_data)} traces... ---")
        
        # 1. Get a small sample of old data from the buffer for replay
        replay_size = int(len(self.normal_buffer) * replay_ratio)
        replay_indices = np.random.choice(len(self.normal_buffer), size=replay_size, replace=False)
        replay_data = self.normal_buffer[replay_indices]

        # 2. Combine new data with the replayed old data
        finetuning_data = np.concatenate([new_normal_data, replay_data], axis=0)
        
        # 3. Fine-tune the model using the parent's training logic
        print(f"Fine-tuning on a mixed dataset of {len(finetuning_data)} traces ({len(new_normal_data)} new, {len(replay_data)} replayed).")
        super().train(finetuning_data, epochs=finetune_epochs, batch_size=batch_size, verbose=True)

        # 4. Update the buffer to contain only the new normal process
        self._update_buffer(new_normal_data)
        print(f"--- Adaptation complete. Buffer updated. Ready for continued monitoring. ---\n")
