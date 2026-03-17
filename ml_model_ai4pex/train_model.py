#!/usr/bin/env python3

'''
    Description: 
'''

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import xarray as xr
import xbatcher as xb
import keras
import tensorflow as tf
import io

from ml_model_ai4pex.cnn import CNN
from ml_model_ai4pex.unet import UNet

@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 30
    # optimiser
    learning_rate: float = 0.001
    use_ema: bool = False
    ema_momentum: float = 0.99
    # regularisation
    dropout_rate: float = 0.2
    # learning rate scheduler
    use_learning_rate_scheduler: bool = False
    learning_rate_decay_steps_multiplier: int = 10
    learning_rate_decay_rate: float = 0.05
    # early stopping
    use_early_stopping: bool = False
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_start_from_epoch: int = 30
    # attention
    use_attention: bool = False
    # training control
    pickup: bool = False
    # model selection
    model_type: str = "cnn"
    base_filters: int = 64
    depth: int = 2
    # paths
    model_dir: str = "./"
    model_filename: Optional[str] = None
    tensboard_log_dir: Optional[str] = None
    # logging
    verbose: bool = False
    current_period: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # runs any validation checks
    def __post_init__(self): 
        if self.pickup and self.model_filename is None:
            raise ValueError("pickup=True requires model_filename to be set.")
        #TODO add more checks

def _keras_data_generator(bgen, sc):
    """
    A wrapper generator that takes xbatcher generator (bgen)
    and yields Keras-compatible (inputs, targets) tuples.
    """
    while True: # infinite generator for multiple epochs
        for batch in bgen:
            batch_input = [batch[x] for x in sc.input_var]
            batch_target = [batch[x] for x in sc.target]
            
            batch_input = xr.merge(batch_input).\
                to_array('var').transpose('sample', 'y_c', 'x_c','var')
            batch_target = xr.merge(batch_target).\
                to_array('var').transpose('sample', 'y_c', 'x_c','var')
    
            # Yield the final NumPy arrays
            yield (batch_input.to_numpy(), batch_target.to_numpy())

def train_model(scenario, ds_training, ds_validation,
                training_config: TrainingConfig, 
                mask=None, logger=None):
    
    """Train model with the given scenario, data, and config."""

    if logger and training_config.verbose:
        logger.info(
            f"Starting training with scenario: {scenario}, \
             training_config: {training_config}"
        )

    # Get y_c and x_c dimension sizes from input data
    y_c_size = ds_training.sizes['y_c']
    x_c_size = ds_training.sizes['x_c']

    # Set up batch generators
    bgen_training = xb.BatchGenerator(
        ds_training,
        input_dims={'y_c': y_c_size, 'x_c': x_c_size},
        batch_dims={'sample': training_config.batch_size},
    )
    bgen_validation = xb.BatchGenerator(
        ds_validation,
        input_dims={'y_c': y_c_size, 'x_c': x_c_size},
        batch_dims={'sample': training_config.batch_size},
    )

    train_generator = _keras_data_generator(bgen_training, scenario)
    val_generator = _keras_data_generator(bgen_validation, scenario)

    # Inspect the first batch
    batch_inputs, batch_targets = next(train_generator)

    validation_steps = len(ds_validation.sample) // training_config.batch_size
    steps_per_epoch = len(ds_training.sample) // training_config.batch_size

    # set learning rate
    if training_config.use_learning_rate_scheduler:
        decay_steps = training_config.learning_rate_decay_steps_multiplier \
                        * steps_per_epoch
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=training_config.learning_rate,
            decay_steps=decay_steps,
            decay_rate=training_config.learning_rate_decay_rate,
            staircase=False
        )
    else:
        lr_schedule = training_config.learning_rate

    #TODO add pickup option

    # Build the model
    input_shape = (y_c_size, x_c_size, len(scenario.input_var))

    if training_config.model_type == "unet":
        model = UNet(
            scenario,
            input_shape,
            dropout_rate=training_config.dropout_rate,
            use_attention=training_config.use_attention,
        )
    else:
        model = CNN(scenario, 
                    input_shape, 
                    dropout_rate=training_config.dropout_rate,
                    use_attention=training_config.use_attention,
        )
    model(tf.keras.Input(shape=input_shape))

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))

    logger.info('Model summary is:\n%s', stream.getvalue())

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr_schedule,
            use_ema=training_config.use_ema,
            ema_momentum=training_config.ema_momentum,
        ),
        # loss=cnn.MaskedMSELoss(mask=mask),
        loss=tf.keras.losses.MeanAbsoluteError(
            reduction='sum_over_batch_size',
            name='mean_absolute_error'
        ),
        run_eagerly=False,
    )

    # Set up callbacks
    callbacks = []

    log_dir = training_config.tensboard_log_dir \
        or f'logs/fit/{training_config.current_period}'
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1,
            write_images=True, update_freq='epoch',
        )
    ]

    if training_config.use_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=training_config.early_stopping_monitor,
                patience=training_config.early_stopping_patience,
                min_delta=training_config.early_stopping_min_delta,
                start_from_epoch=training_config.early_stopping_start_from_epoch,
                restore_best_weights=True,
            )
        )

    # Train the model
    logger.info("Starting model training...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=training_config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    if logger and training_config.verbose:
        logger.info(
            f"Training complete. Final training loss: {history.history['loss'][-1]}, \
             final validation loss: {history.history['val_loss'][-1]}"
        )

    return model