import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization


def build_lstm_model(seq_length=30, lstm_units=64, dropout_rate=0.2):
    """
    LSTM for Indian daily power consumption forecasting.
    seq_length=30: looks back 30 days to predict day 31.

    Key fixes vs original:
    - tanh activation (not relu) — relu causes gradient explosions in LSTM
    - BatchNormalization — handles different MW scales per state
    - Gradient clipping in compile_model — stops loss from going to billions
    """
    model = Sequential([
        LSTM(lstm_units,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=True,
             input_shape=(seq_length, 1),
             name='LSTM_Layer_1'),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(lstm_units // 2,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=False,
             name='LSTM_Layer_2'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(lstm_units // 4, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(1, name='Output_Layer')
    ])
    return model


def compile_model(model, learning_rate=0.0005):
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0        # prevents gradient explosions
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    model = build_lstm_model(seq_length=30)
    model = compile_model(model)
    model.summary()