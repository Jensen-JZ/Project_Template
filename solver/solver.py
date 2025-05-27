import datetime
import os
import time
import functools

import torch
from torch import nn
from munch import Munch

from data.fetcher import Fetcher
# from metrics.eval import calculate_total_fid # Removed
# from metrics.fid import calculate_fid_given_paths # Removed
from models.build import build_model
from solver.loss import compute_loss # Updated to generic compute_loss
# from solver.misc import translate_using_latent, generate_samples # Removed, misc.py is now minimal
from solver.utils import he_init # moving_average removed from direct use here
from utils.checkpoint import CheckpointIO
from utils.file import delete_dir, write_record, delete_model, delete_sample # delete_model/sample might be simplified later
from utils.misc import get_datetime, send_message
from utils.model import count_parameters


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        # build_model now returns nets (with .model) and optionally nets_ema
        # We will primarily use nets.model and ignore nets_ema for simplification
        self.nets, nets_ema_from_build = build_model(args) # nets_ema_from_build might be None or a simple model
        
        # Count parameters for the main model
        if hasattr(self.nets, 'model') and self.nets.model is not None:
            count_parameters(self.nets.model, 'model')
            if args.multi_gpu:
                self.nets.model = nn.DataParallel(self.nets.model)
            self.nets.model.to(self.device)
        else:
            # Fallback or error if nets.model is not what's expected
            # This could happen if build_model is not returning the expected structure.
            # For now, we assume nets.model exists.
            print("Warning: self.nets.model is not properly defined by build_model.")
            # As a minimal fallback, create a dummy model if none exists to prevent crashes,
            # though this indicates a deeper issue with model definition.
            if not hasattr(self.nets, 'model'):
                 self.nets.model = nn.Linear(1,1).to(self.device) # Dummy model

        # EMA models are being simplified out of the main solver logic
        # If nets_ema_from_build.model_ema exists, move to device but don't create optimizer or checkpoint for it.
        if hasattr(nets_ema_from_build, 'model_ema') and nets_ema_from_build.model_ema is not None:
            nets_ema_from_build.model_ema.to(self.device)
            self.nets_ema = nets_ema_from_build # Keep it if provided, but won't be actively used in training loop
        else:
            self.nets_ema = Munch(model_ema=None) # Ensure nets_ema exists but is empty

        if args.mode == 'train':
            # Setup a single optimizer for self.nets.model
            self.optims = Munch()
            if hasattr(self.nets, 'model') and self.nets.model is not None:
                 # Check if model is in pretrained_models (assuming 'model' is the key if it was)
                if 'model' not in args.pretrained_models: # Adjust if model name in nets can vary
                    self.optims.model = torch.optim.Adam(
                        params=self.nets.model.parameters(),
                        lr=args.lr, # Use generic learning rate
                        betas=(args.beta1, args.beta2), # These args were kept in config
                        weight_decay=args.weight_decay) # This arg was kept in config
                else:
                    print("Model is specified in pretrained_models, not creating optimizer for it.")
            else:
                print("Error: self.nets.model is None, cannot create optimizer.")


            # Checkpoint for nets.model and its optimizer
            # Ensure self.nets.model exists before creating CheckpointIO
            if hasattr(self.nets, 'model') and self.nets.model is not None:
                ckpt_objects = {'model': self.nets.model}
                if hasattr(self.optims, 'model'): # only if optimizer was created
                     ckpt_objects_optims = {'model_optim': self.optims.model}
                     self.ckptios = [
                         CheckpointIO(args.model_dir + '/{:06d}_model.ckpt', multi_gpu=args.multi_gpu, **ckpt_objects),
                         CheckpointIO(args.model_dir + '/{:06d}_optims.ckpt', **ckpt_objects_optims)
                     ]
                else: # No optimizer (e.g. model is pretrained)
                    self.ckptios = [
                        CheckpointIO(args.model_dir + '/{:06d}_model.ckpt', multi_gpu=args.multi_gpu, **ckpt_objects)
                    ]
            else: # Fallback if model is not defined
                self.ckptios = []
                print("Warning: self.nets.model not found, no checkpoints will be saved for the model.")

        else: # eval or sample mode
            if hasattr(self.nets, 'model') and self.nets.model is not None:
                 # Load the main model for eval/sample. EMA model is not used.
                 self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_model.ckpt', multi_gpu=args.multi_gpu, model=self.nets.model)]
            elif hasattr(self.nets_ema, 'model_ema') and self.nets_ema.model_ema is not None:
                # Fallback to EMA model if main model not found and EMA exists (legacy case)
                print("Warning: Main model not found for eval, attempting to load EMA model.")
                self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_nets_ema.ckpt', model_ema=self.nets_ema.model_ema)] # Old naming
            else:
                self.ckptios = []
                print("Warning: No model found for eval/sample mode checkpointing.")


        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from utils.logger import Logger
            self.logger = Logger(args.log_dir)
        self.record = functools.partial(write_record, file_path=args.record_file)
        # self.record(f"Please notice eval_use_ema is set to {args.eval_use_ema}.") # eval_use_ema removed

    def init_weights(self):
        if self.args.init_weights == 'he':
            if hasattr(self.nets, 'model') and self.nets.model is not None:
                # Assuming 'model' is the key for the main model to initialize
                # And it's not in pretrained_models
                if 'model' not in self.args.pretrained_models:
                    print('Initializing model with he_init...')
                    self.nets.model.apply(he_init)
                    print('Done.')
            else:
                print("Warning: self.nets.model not found for weight initialization.")

        elif self.args.init_weights == 'default':
            # Do nothing because the weights has been initialized in this manner.
            pass

    def train_mode(self, training=True):
        if hasattr(self.nets, 'model') and self.nets.model is not None:
            # Set only the main model to train/eval mode
            # Assuming 'model' is not in pretrained_models if it needs mode switching
            if 'model' not in self.args.pretrained_models:
                self.nets.model.train(mode=training)
        # self.nets_ema is no longer actively managed for mode switching

    def eval_mode(self):
        self.train_mode(training=False)

    def save_model(self, step):
        # self.ckptios is already simplified to handle only main model and its optimizer
        for ckptio in self.ckptios:
            ckptio.save(step)

    def load_model(self, step):
        # self.ckptios is already simplified
        for ckptio in self.ckptios:
            ckptio.load(step)

    def load_model_from_path(self, path):
        # self.ckptios is already simplified
        for ckptio in self.ckptios:
            ckptio.load_from_path(path)

    def zero_grad(self):
        # self.optims is simplified to self.optims.model
        if hasattr(self.optims, 'model') and self.optims.model is not None:
            self.optims.model.zero_grad()
        # else: print an error or handle cases where no optimizer is present (e.g. eval mode)

    def train(self, loaders):
        args = self.args
        nets = self.nets # This now primarily means nets.model
        optims = self.optims # This now primarily means optims.model

        # Use the Fetcher for iterating over train data
        train_fetcher = Fetcher(loaders.train, args)
        # For evaluation during training, we'll iterate test_loader directly
        test_loader = loaders.test

        # Load or initialize the model parameters.
        if args.start_iter > 0:
            self.load_model(args.start_iter)
        else:
            self.init_weights()

        # Placeholder for best model tracking based on a generic metric (e.g., loss)
        best_metric = float('inf') # Assuming lower is better for the metric
        best_step = 0
        
        print('Start training...')
        start_time = time.time()
        for step in range(args.start_iter + 1, args.end_iter + 1):
            self.train_mode() # Set model to training mode

            try:
                # Fetch a batch of data
                # Fetcher returns a Munch with x (data) and possibly y (target)
                batch = next(train_fetcher)
                inputs = batch.x
                # Ensure ground_truth is available. Fetcher handles if y is not present.
                ground_truth = batch.y if hasattr(batch, 'y') else None 
                
                if ground_truth is None and args.mode == 'train':
                    # This is a fallback, in many supervised tasks, ground_truth is required.
                    # For unsupervised tasks, loss computation must not expect ground_truth.
                    # The placeholder loss.py handles ground_truth being None by using a dummy target.
                    pass # print("Warning: ground_truth is None during training.")

            except Exception as e:
                print(f"Error fetching training data: {e}")
                continue # Skip this iteration or handle error appropriately

            # Forward pass
            if not hasattr(nets, 'model') or nets.model is None:
                print("Error: nets.model is not defined. Skipping training step.")
                continue
            model_output = nets.model(inputs)

            # Compute loss
            # compute_loss is imported from solver.loss
            # It should handle cases where ground_truth might be None (e.g. for unsupervised tasks)
            loss, loss_items = compute_loss(model_output, ground_truth, args)
            
            # Backward and optimize
            if hasattr(optims, 'model') and optims.model is not None:
                self.zero_grad()
                loss.backward()
                optims.model.step()
            else: # Should not happen if model is trainable
                print("Warning: Optimizer for model not found. Skipping optimization step.")


            # EMA update is removed for simplification
            # moving_average(nets.generator, nets_ema.generator, beta=args.ema_beta)
            # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=args.ema_beta)

            self.eval_mode() # Set model to evaluation mode for logging/sampling/eval

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = f"[{elapsed}]-[{step}/{args.end_iter}]: "
                
                # loss_items is a Munch from compute_loss
                log += ' '.join([f'{key}: [{value:.4f}]' for key, value in loss_items.items()])
                print(log)
                
                if args.save_loss: # Generic loss saving
                    if step == args.log_every: # First log step, write header
                        header = ','.join(['iter'] + list(loss_items.keys()))
                        write_record(header, args.loss_file, append=False)
                    log_csv = ','.join([str(step)] + [f'{value:.4f}' for value in loss_items.values()])
                    write_record(log_csv, args.loss_file, append=True)

                if self.use_tensorboard:
                    for tag, value in loss_items.items():
                        self.logger.scalar_summary(f"train_loss/{tag}", value, step)
            
            # Visualization (simplified)
            if step % args.visualize_every == 0: # visualize_every was sample_every
                print(f"Step {step}: Visualizing model performance (placeholder)...")
                # Placeholder: run model on a fixed batch from test_loader and log loss
                try:
                    # Get a fixed batch from test_loader if not already available
                    # This might re-iterate test_loader if not handled carefully.
                    # For simplicity, let's just take one batch.
                    vis_batch = next(iter(test_loader)) # Fetcher not used here for simplicity
                    vis_inputs = vis_batch.x.to(self.device) if hasattr(vis_batch, 'x') else vis_batch[0].to(self.device)
                    vis_gt = vis_batch.y.to(self.device) if hasattr(vis_batch, 'y') else (vis_batch[1].to(self.device) if len(vis_batch)>1 else None)

                    with torch.no_grad():
                        vis_output = nets.model(vis_inputs)
                        vis_loss, vis_loss_items = compute_loss(vis_output, vis_gt, args)
                    
                    log_msg = f"Visualization at step {step}: Test Batch Loss: "
                    log_msg += ' '.join([f'{key}: [{value:.4f}]' for key, value in vis_loss_items.items()])
                    print(log_msg)
                    if self.use_tensorboard:
                        for tag, value in vis_loss_items.items():
                            self.logger.scalar_summary(f"visualize_loss/{tag}", value, step)
                    # User can add custom image saving or other visualization here.
                except Exception as e:
                    print(f"Error during visualization: {e}")


            # Model Saving
            if step % args.save_every == 0:
                self.save_model(step)
                # Simplified: remove deletion of non-best models for now, user can manage manually
                # last_step = step - args.save_every
                # if last_step != best_step and not args.keep_all_models:
                #     delete_model(args.model_dir, last_step)

            # Evaluation (simplified)
            if step % args.eval_every == 0:
                print(f"Evaluating at step {step}...")
                total_eval_loss = 0
                num_eval_batches = 0
                all_eval_loss_items = Munch()

                with torch.no_grad():
                    for eval_batch in test_loader: # Iterate directly
                        eval_inputs = eval_batch.x.to(self.device) if hasattr(eval_batch, 'x') else eval_batch[0].to(self.device)
                        eval_gt = eval_batch.y.to(self.device) if hasattr(eval_batch, 'y') else (eval_batch[1].to(self.device) if len(eval_batch)>1 else None)

                        eval_output = nets.model(eval_inputs)
                        eval_loss, eval_loss_items = compute_loss(eval_output, eval_gt, args)
                        
                        # Accumulate primary loss (e.g. first item in loss_items)
                        # This assumes the first loss in loss_items is the primary one for comparison
                        primary_loss_key = list(eval_loss_items.keys())[0]
                        total_eval_loss += eval_loss_items[primary_loss_key]

                        for key, value in eval_loss_items.items():
                            all_eval_loss_items[key] = all_eval_loss_items.get(key, 0) + value
                        num_eval_batches += 1
                
                if num_eval_batches > 0:
                    avg_eval_loss_items = Munch({k: v / num_eval_batches for k, v in all_eval_loss_items.items()})
                    avg_primary_loss = total_eval_loss / num_eval_batches
                    
                    log_msg = f"Evaluation at step {step}: Average Test Loss: "
                    log_msg += ' '.join([f'{key}: [{value:.4f}]' for key, value in avg_eval_loss_items.items()])
                    print(log_msg)
                    self.record(log_msg) # Save to records.txt

                    if self.use_tensorboard:
                        for tag, value in avg_eval_loss_items.items():
                            self.logger.scalar_summary(f"eval_loss/{tag}", value, step)
                    
                    # Simplified best model tracking based on the first reported loss value
                    if avg_primary_loss < best_metric:
                        best_metric = avg_primary_loss
                        # To save the "best" model, we can save it with a specific name or overwrite a "best" file
                        # For simplicity, we'll just note it. User can implement saving logic.
                        # self.save_model("best") # Example: could save to "best_model.ckpt"
                        best_step = step
                        self.record(f"New best model found at step {step} with metric: {best_metric:.4f}")
                else:
                    print("No data in test_loader for evaluation.")
                
                # Removed FID and complex best model tracking/deletion
                # send_message(info, args.exp_id) # Removed send_message for simplicity

        self.record(f"Model training completed. Best validation metric ({list(all_eval_loss_items.keys())[0] if all_eval_loss_items else 'N/A'}): {best_metric:.4f} at step {best_step}")
        # send_message("Model training completed.", args.exp_id) # Removed

    @torch.no_grad()
    def sample(self, loaders):
        # Re-purposed for generic inference/prediction
        args = self.args
        nets = self.nets

        if not hasattr(nets, 'model') or nets.model is None:
            print("Error: Model not found (nets.model is None). Cannot perform sampling.")
            return

        # Load model (user should specify which iteration, or use 'best' if implemented)
        # For simplicity, assume start_iter can be used to specify a model, or a fixed name like 'best_model.ckpt'
        if args.start_iter > 0: # Assuming start_iter specifies the model to load for sampling
            print(f"Loading model from iteration {args.start_iter} for sampling...")
            self.load_model(args.start_iter)
        else:
            # Try to load a 'best' model if available, or latest. This part needs user convention.
            # For now, we'll just use the current state or require start_iter.
            print("Warning: `start_iter` not specified for sampling. Using current model state or a pre-loaded one if available.")
            # Or, implement logic to find the latest/best checkpoint.
        
        self.eval_mode() # Set model to evaluation mode

        sample_loader = loaders.test # Use test_loader, or user can specify a different one via args
        if args.sample_dir is None:
            args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "generic_samples")
        os.makedirs(args.sample_dir, exist_ok=True)

        print(f"Running inference on data from: {args.test_path}")
        print(f"Saving outputs to: {args.sample_dir}")

        for i, batch in enumerate(sample_loader):
            inputs = batch.x.to(self.device) if hasattr(batch, 'x') else batch[0].to(self.device)
            
            model_output = nets.model(inputs)
            
            # Process and save output - this is highly dependent on the task
            # Example for image classification: save predicted class labels
            # Example for image generation: save images
            # For a generic template, we can save raw output or a simple interpretation
            for j, output_item in enumerate(model_output):
                output_filename = os.path.join(args.sample_dir, f"sample_{i * args.batch_size + j}_output.pt")
                torch.save(output_item.cpu(), output_filename)
            if i < 5: # Limit number of console printouts
                 print(f"Processed batch {i}, saved {model_output.size(0)} outputs. Example output head: {model_output[0][:10]}...")
            if i*args.batch_size > 1000: # Safety break for very large test sets
                print("Stopping sampling after 1000 samples to avoid excessive output.")
                break
        print(f"Inference complete. Outputs saved in {args.sample_dir}")
        return args.sample_dir


    @torch.no_grad()
    def evaluate(self, loaders):
        # Re-purposed for generic evaluation (e.g., calculate average loss on test set)
        args = self.args
        nets = self.nets

        if not hasattr(nets, 'model') or nets.model is None:
            print("Error: Model not found (nets.model is None). Cannot perform evaluation.")
            return

        if args.start_iter > 0: # Assuming start_iter specifies the model to load
            print(f"Loading model from iteration {args.start_iter} for evaluation...")
            self.load_model(args.start_iter)
        else:
            print("Warning: `start_iter` not specified for evaluation. Using current model state or a pre-loaded one if available.")

        self.eval_mode() # Set model to evaluation mode

        eval_loader = loaders.test # Use test_loader from args.test_path
        
        print(f"Evaluating model on data from: {args.test_path}")
        
        total_loss = 0
        num_batches = 0
        all_loss_items_eval = Munch()

        for batch in eval_loader:
            inputs = batch.x.to(self.device) if hasattr(batch, 'x') else batch[0].to(self.device)
            ground_truth = batch.y.to(self.device) if hasattr(batch, 'y') else (batch[1].to(self.device) if len(batch)>1 else None)

            model_output = nets.model(inputs)
            loss, loss_items = compute_loss(model_output, ground_truth, args) # Using the generic compute_loss
            
            # Accumulate all loss items
            for key, value in loss_items.items():
                all_loss_items_eval[key] = all_loss_items_eval.get(key, 0) + value
            num_batches += 1

        if num_batches > 0:
            avg_loss_items = Munch({k: v / num_batches for k, v in all_loss_items_eval.items()})
            log_msg = f"Evaluation Results on {args.test_path}: "
            log_msg += ' '.join([f'Avg {key}: [{value:.4f}]' for key, value in avg_loss_items.items()])
            print(log_msg)
            self.record(log_msg) # Save to records.txt
            # User can add more metrics here
        else:
            print("No data in eval_loader for evaluation.")
            self.record("Evaluation: No data found in test_path.")
        
        # Removed FID calculation and related file deletions
