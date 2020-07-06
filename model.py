import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl
import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import backend as K
import tensorflow as tf
import joblib

import numpy as np

class MVAE(tfk.Model):
    def __init__(self, x_depth, enc_rnn_dim, enc_dropout, dec_rnn_dim, dec_dropout, cont_dim, cat_dim, mu_force, t_gumbel, 
                       style_embed_dim, beta_anneal_steps, kl_reg):
        super(MVAE, self).__init__()
        
        self.summaries = []
        self.features = ['pitch', 'dt', 'duration']

        self.x_depth = x_depth
        self.x_dim = sum(x_depth)
        self.content_embed_dim = 32
        self.t_gumbel = float(t_gumbel)
        self.cont_dim = int(cont_dim)
        self.cat_dim = int(cat_dim)
        self.mu_force = float(mu_force)
        self.beta_anneal_steps = int(beta_anneal_steps)
        self.kl_reg = float(kl_reg)

        enc_rnn_dim = int(enc_rnn_dim)
        enc_dropout = float(enc_dropout)
        dec_rnn_dim = int(dec_rnn_dim)
        dec_dropout = float(dec_dropout)
        style_embed_dim = int(style_embed_dim)
        
        self.ohc = tfp.distributions.OneHotCategorical
        self.relaxed_ohc = tfp.distributions.RelaxedOneHotCategorical

        # music content variables
        self.pitch_embedding = tfkl.Embedding(x_depth[0], self.content_embed_dim, name='pitch_embedding')
        self.lstm_encoder = tfkl.Bidirectional(tfkl.LSTM(enc_rnn_dim, dropout=enc_dropout, return_state=True), name='lstm_encoder')
        self.content_head = tfkl.Dense(512, activation='relu', name='content_head')
        self.z_params = tfkl.Dense(self.cont_dim + self.cont_dim, name='z_params')
        
        self.decoder_init_state = tfkl.Dense(dec_rnn_dim * 2, name='decoder_init', activation='tanh')
        self.lstm_decoder = tfkl.LSTM(dec_rnn_dim, return_sequences=True, return_state=True, dropout=dec_dropout, name='lstm_decoder')
        self.logit_layer = tfkl.Dense(self.x_dim, name='content_logits')
        
        self.anneal_step = tf.Variable(0.0, trainable=False)
        
        # music style variables
        self.cat_head = tfkl.Dense(512, activation='relu', name='style_head')
        self.z_cat_logit = tfkl.Dense(self.cat_dim, name='z_cat_logit')
        self.style_embedding = tf.Variable(tf.random.uniform([self.cat_dim, style_embed_dim], -1.0, 1.0), trainable=True, name="style_embedding")
        
        # train metric trackers
        self.recon_loss_tracker = tfk.metrics.Mean(name="recon_loss_tracker")
        self.kl_loss_tracker = tfk.metrics.Mean(name="kl_loss_tracker")
        self.style_loss_tracker = tfk.metrics.Mean(name="style_loss_tracker")
        self.loss_tracker = tfk.metrics.Mean(name="loss_tracker")
        
        self.p_acc_tracker = tfk.metrics.Mean(name="p_accuracy_tracker")
        self.dt_acc_tracker = tfk.metrics.Mean(name="dt_accuracy_tracker")
        self.d_acc_tracker = tfk.metrics.Mean(name="d_accuracy_tracker")
        
        # test metric trackers
        self.val_recon_loss_tracker = tfk.metrics.Mean(name="val_recon_loss_tracker")
        self.val_kl_loss_tracker = tfk.metrics.Mean(name="val_kl_loss_tracker")
        self.val_style_loss_tracker = tfk.metrics.Mean(name="val_style_loss_tracker")
        self.val_loss_tracker = tfk.metrics.Mean(name="val_loss_tracker")
        
        self.val_p_acc_tracker = tfk.metrics.Mean(name="val_p_accuracy_tracker")
        self.val_dt_acc_tracker = tfk.metrics.Mean(name="val_dt_accuracy_tracker")
        self.val_d_acc_tracker = tfk.metrics.Mean(name="val_d_accuracy_tracker")
        
        # generic accuracy used for computing raw accuracies
        self.accuracy_tracker = tfk.metrics.Accuracy(name='accuracy_tracker')
        
    def reset_trackers(self):
        self.recon_loss_tracker.reset_states()
        self.kl_loss_tracker.reset_states()
        self.style_loss_tracker.reset_states()
        self.loss_tracker.reset_states()
        
        self.p_acc_tracker.reset_states()
        self.dt_acc_tracker.reset_states()
        self.d_acc_tracker.reset_states()
        
        self.val_recon_loss_tracker.reset_states()
        self.val_kl_loss_tracker.reset_states()
        self.val_style_loss_tracker.reset_states()
        self.val_loss_tracker.reset_states()
        
        self.val_p_acc_tracker.reset_states()
        self.val_dt_acc_tracker.reset_states()
        self.val_d_acc_tracker.reset_states()
        
      
    def embed_input(self, X):
        p, dt, d = tf.split(X, self.x_depth, axis=-1)
        
        e_p = tf.argmax(p, axis=-1)
        e_p = self.pitch_embedding(e_p)

        vae_in = tfkl.Concatenate(axis=-1)([e_p, dt, d])
        return vae_in
    
    def create_start_token(self, batch_size):
        return tf.zeros(shape=(batch_size, 1, self.x_dim), dtype=tf.float32)
        
    def encode(self, X_embed, S=None, training=True):
        
        # create mask for padded inputs. Ignore <end> token. TODO: should stop gradient?
        if S != None:  # S is None only when generation graph model => to get the model summary
            mask = tf.sequence_mask(S - 1, tf.math.reduce_max(S), dtype=tf.bool)
            output, h_fw, c_fw, h_bw, c_bw = self.lstm_encoder(X_embed, mask=mask, training=training)
        else:
            output, h_fw, c_fw, h_bw, c_bw = self.lstm_encoder(X_embed, training=training)
        
        # use only cell states since they contain all the intuition from the previous units
        states = tf.concat([c_fw, c_bw], axis=-1)
        
        cont_head = self.content_head(states)
        z_params = self.z_params(cont_head)
        z_mean, z_log_var = tf.split(z_params, num_or_size_splits=2, axis=1)
        
        cat_head = self.cat_head(states)
        z_cat_logit = self.z_cat_logit(cat_head)
        
        return z_mean, z_log_var, z_cat_logit

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def reparametrize_cat(self, logit, training=True):
        if training:
            return self.relaxed_ohc(temperature=self.t_gumbel, logits=logit).sample()
        else:
            return self.ohc(logits=logit, dtype=tf.float32).sample()

    def decode(self, Z, X_embed, timesteps, S=None, training=True):
        # timesteps = tf.max(S)
        init_state = tf.split(self.decoder_init_state(Z), 2, axis=-1)
        
        start_token = self.create_start_token(tf.shape(Z)[0])
        start_token = self.embed_input(start_token)
        
        # add start_token to the beginning of the sequence and discard end token from the sequence
        o_p = tf.concat([start_token, X_embed[:, :-1, :]], axis=1)
                
        if S != None:  # S is None only when generation graph model => to get the model summary
            mask = tf.sequence_mask(S, dtype=tf.bool)
            o_p, _, _ = self.lstm_decoder(o_p, training=training, mask=mask, initial_state=init_state)
        else:
            o_p, _, _ = self.lstm_decoder(o_p, training=training, initial_state=init_state)
            
        # we require that output equal to network input X (with the final <END> token) as a seq2seq.
        logits = self.logit_layer(o_p)
        return logits
    
    # generate batch_size random pieces of the given style
    # style should be an integer
    def sample(self, genre, batch_size=16, eps=None, max_iterations=512):
        
        def sample_single_batch():
            def sample_fn(logits):    
                logits = tf.split(logits, self.x_depth, axis=-1)

                samples = [self.ohc(logits=logit, dtype=tf.float32).sample() for logit in logits]
                samples = tf.concat(samples, axis=-1)

                return samples
            
            def is_finished(samples):
                p, _, _ = tf.split(samples, self.x_depth, axis=-1)
                p = tf.argmax(tf.squeeze(p))
                return p == 88
                   
            x = self.create_start_token(1)
            
            z_cont = tf.random.normal(shape=(1, self.cont_dim))
            z_cat = np.zeros((1, self.cat_dim), np.float32)
            z_cat[:, genre] = 1
            z_cat = tf.linalg.matmul(z_cat, self.style_embedding)
            
            z = tf.concat([z_cont, z_cat], axis=-1)
            
            init = self.decoder_init_state(z)
            
            state = tf.split(init, 2, axis=-1)
            
            
            for _ in range(max_iterations):
                x_embed = self.embed_input(x)
                o, s_h, s_c = self.lstm_decoder(x_embed, training=False, initial_state=state)
                
                # only keep the final output
                o = o[:, -1, :]
                o = tf.reshape(o, shape=(-1, 1, 512))
                
                # compute logits
                logits = self.logit_layer(o)
                samples = sample_fn(logits)
                
                # combine 'sequence until now' with the generated output
                x = tf.concat([x, samples], axis=1)
                
                if is_finished(samples):
                    break
            
            # discard first timestep, the start token
            return x[0, 1:, :]
        
        return [sample_single_batch() for _ in range(batch_size)]
    
    def compute_reconstruction_accuracy(self, logits, X, seq_len):
        # logits = model(X)
        logits = tf.split(logits, self.x_depth, axis=-1)
        X = tf.split(X, self.x_depth, axis=-1)
        mask = tf.sequence_mask(seq_len, dtype=tf.float32)
                
        accuracy = []

        for logit, x, feature in zip(logits, X, self.features):
            self.accuracy_tracker.reset_states()
            
            tmp_x = tf.argmax(x, axis=-1, output_type=tf.int32)
            tmp_logit = tf.argmax(logit, axis=-1, output_type=tf.int32)
            
            self.accuracy_tracker.update_state(tmp_x, tmp_logit, sample_weight=mask)
            
            accuracy.append(self.accuracy_tracker.result())
            
        return accuracy

    def compute_loss(self, logits, X, seq_len):
        # logits = model(X)
        logits = tf.split(logits, self.x_depth, axis=-1)
        X = tf.split(X, self.x_depth, axis=-1)
        mask = tf.stop_gradient(tf.sequence_mask(seq_len, dtype=tf.float32))
        
        loss = []

        for logit, x, feature in zip(logits, X, self.features):
            tmp_loss = tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=logit)
            tmp_loss = tf.compat.v1.losses.compute_weighted_loss(tmp_loss, weights=mask)
            
            if feature == 'pitch':
                tmp_loss = tmp_loss * 3
                
            loss.append(tmp_loss)
            
        return tf.reduce_sum(loss)
    
    def compute_kl_loss(self, enc_out):
        z_mean, z_log_var = enc_out
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return kl_loss
    
    def compute_mu_regularization(self, z_mean):
        z_mean_mu = tf.reduce_mean(z_mean, 0)
        mu_losses = tfk.losses.MSE(z_mean_mu, z_mean)
        mu_loss = tf.nn.relu(self.mu_force - K.mean(mu_losses))
        return mu_loss
    
    def compute_label_loss(self, logits, labels):
        cat_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(cat_loss)

    def test_step(self, data):
        X, S, labels = data
        labels = tf.one_hot(labels, self.cat_dim)
        
        X_embed = self.embed_input(X)
            
        mean, logvar, cat_logit = self.encode(X_embed, S, training=False)
        z_cont = self.reparametrize(mean, logvar)
        z_cat = self.reparametrize_cat(cat_logit, training=False)
        z_cat = tf.linalg.matmul(z_cat, self.style_embedding)
        
        z = tf.concat([z_cont, z_cat], axis=-1)
        
        logits = self.decode(z, X_embed, X.shape[1], S, training=False)

        recon_loss = self.compute_loss(logits, X, S)
        kl_loss = tf.reduce_mean(self.compute_kl_loss([mean, logvar]) / self.cont_dim)
        mu_loss = self.compute_mu_regularization(mean)
        style_loss = self.compute_label_loss(cat_logit, labels)

        beta = 1.0 - self.beta_anneal_steps / (self.beta_anneal_steps + tf.exp(self.anneal_step / self.beta_anneal_steps))
        beta = tf.cast(beta, tf.float32)     

        loss = recon_loss + self.kl_reg * beta * kl_loss + mu_loss + style_loss
        
        pitch_acc, dt_acc, dur_acc = self.compute_reconstruction_accuracy(logits, X, S)
        
        self.val_recon_loss_tracker.update_state(recon_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_style_loss_tracker.update_state(style_loss)
        self.val_loss_tracker.update_state(loss)
        
        self.val_p_acc_tracker.update_state(pitch_acc)
        self.val_dt_acc_tracker.update_state(dt_acc)
        self.val_d_acc_tracker.update_state(dur_acc)
        
        return { "loss": self.val_loss_tracker.result(), "recon_loss": self.val_recon_loss_tracker.result(), "kl_loss": self.val_kl_loss_tracker.result(),
                 "style_loss": self.val_style_loss_tracker.result(),
                 "p_acc": self.val_p_acc_tracker.result(), "dt_acc": self.val_dt_acc_tracker.result(), "dur_acc": self.val_d_acc_tracker.result() }
        
    def train_step(self, data):
        X, S, labels = data

        with tf.GradientTape() as tape:
            labels = tf.one_hot(labels, self.cat_dim)
            self.anneal_step.assign_add(1.0)
            self.learning_rate = tf.maximum(5e-4 * 0.95 ** ((self.anneal_step - 10000) / 5000), 1e-4)
            
            X_embed = self.embed_input(X)
            
            mean, logvar, cat_logit = self.encode(X_embed, S, training=True)
            
            z_cont = self.reparametrize(mean, logvar)
            z_cat = self.reparametrize_cat(cat_logit, training=True)
            z_cat = tf.linalg.matmul(z_cat, self.style_embedding)
            
            z = tf.concat([z_cont, z_cat], axis=-1)
            
            logits = self.decode(z, X_embed, X.shape[1], S, training=True)
            
            recon_loss = self.compute_loss(logits, X, S)
            kl_loss = tf.reduce_mean(self.compute_kl_loss([mean, logvar]) / self.cont_dim)
            mu_loss = self.compute_mu_regularization(mean)
            style_loss = self.compute_label_loss(cat_logit, labels)
            
            beta = 1.0 - self.beta_anneal_steps / (self.beta_anneal_steps + tf.exp(self.anneal_step / self.beta_anneal_steps))
            beta = tf.cast(beta, tf.float32)     
            
            loss = recon_loss + self.kl_reg * beta * kl_loss + mu_loss + style_loss
            
        trainable_vars = self.trainable_variables
        
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.lr.assign(self.learning_rate)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        pitch_acc, dt_acc, dur_acc = self.compute_reconstruction_accuracy(logits, X, S)
        
        # gather metrics (mean is taken over the entire epoch)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.style_loss_tracker.update_state(style_loss)
        self.loss_tracker.update_state(loss)
        
        self.p_acc_tracker.update_state(pitch_acc)
        self.dt_acc_tracker.update_state(dt_acc)
        self.d_acc_tracker.update_state(dur_acc)
        
        return { "loss": self.loss_tracker.result(), "recon_loss": self.recon_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result(),
                 "style_loss": self.style_loss_tracker.result(), "l_r": self.learning_rate,
                 "p_acc": self.p_acc_tracker.result(), "dt_acc": self.dt_acc_tracker.result(), "dur_acc": self.d_acc_tracker.result() }
    
    # should be used only for model summary -- computation is wrong -- dimensionality is right (we hope)
    def model(self):
        X = tfk.Input(shape=(None, self.x_dim), name='network_input')
        X_embed = self.embed_input(X)
        
        mean, logvar, z_cat = self.encode(X_embed)
        z_cat = tf.linalg.matmul(z_cat, self.style_embedding)

        z = tf.concat([mean, z_cat], axis=-1)

        output = self.decode(z, X_embed, 1)
        
        return tfk.Model(inputs=X, outputs=output, name='MVAE')

    # returns a MVAE model from a configuration dictionary
    @staticmethod
    def build_model_from_config():
        pass
        