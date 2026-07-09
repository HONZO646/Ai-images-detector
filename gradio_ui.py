"""
Gradio Web UI for NPR AI/Real Image Detector
"""
import os
import sys
import json
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

from config import Config
from models.detector import create_detector
from utils import load_checkpoint, get_device


# Global state
_state = {
    "model": None,
    "config": None,
    "trainer": None,
    "stop_event": None,
    "training_thread": None,
    "log_queue": queue.Queue(),
    "training_complete": False,
    "test_metrics": None,
}
_state_lock = threading.Lock()


class QueueLogHandler(logging.Handler):
    """Log handler that puts records into a queue."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


def setup_logging(log_queue):
    """Setup logging with queue handler."""
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(queue_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger


def create_config_from_ui(learning_rate, epochs, batch_size, early_stop_patience, optimizer, scheduler_type, use_amp):
    """Create Config from UI parameters."""
    config = Config()
    config.training.learning_rate = learning_rate
    config.training.epochs = epochs
    config.training.batch_size = batch_size
    config.training.early_stop_patience = early_stop_patience
    config.training.optimizer = optimizer
    config.training.scheduler_type = scheduler_type
    config.training.use_amp = use_amp
    return config


def preprocess_image(image, image_size=640):
    """Preprocess image for inference."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_rgb = image.resize((image_size, image_size), Image.LANCZOS)

    rgb_array = np.array(img_rgb, dtype=np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)

    img_gray = img_rgb.convert('L')
    gray_array = np.array(img_gray, dtype=np.float32) / 255.0
    gray_tensor = torch.from_numpy(gray_array).unsqueeze(0).unsqueeze(0)

    return gray_tensor, rgb_tensor


def run_training(config, log_queue, stop_event):
    """Run training in background thread."""
    try:
        logger = setup_logging(log_queue)
        logger.info("Starting training...")

        from train import Trainer
        trainer = Trainer(config)

        with _state_lock:
            _state["trainer"] = trainer

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)

        trainer.val_metrics = []
        start_time = time.time()

        for epoch in range(config.training.epochs):
            if stop_event.is_set():
                logger.info("Training stopped by user")
                break

            trainer.current_epoch = epoch

            train_metrics = trainer.train_epoch()
            trainer.train_losses.append(train_metrics['loss'])

            val_metrics = trainer.validate()
            trainer.val_losses.append(val_metrics['loss'])
            trainer.val_aucs.append(val_metrics['auc'])
            trainer.val_metrics.append(val_metrics)

            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{config.training.epochs} [{elapsed:.0f}s] Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

            if trainer.scheduler:
                if hasattr(trainer.scheduler, 'step'):
                    if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.scheduler.step(val_metrics['auc'])
                    else:
                        trainer.scheduler.step()

            if val_metrics['auc'] > trainer.best_val_auc:
                trainer.best_val_auc = val_metrics['auc']
                trainer.epochs_without_improvement = 0
                trainer.save_checkpoint(is_best=True)
            else:
                trainer.epochs_without_improvement += 1

            if (epoch + 1) % config.training.save_interval == 0:
                trainer.save_checkpoint(is_best=False)

            if trainer.epochs_without_improvement >= config.training.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)

        if os.path.exists(config.training.best_model_path):
            logger.info(f"Loading best model from: {config.training.best_model_path}")
            load_checkpoint(config.training.best_model_path, trainer.model, trainer.device)

        test_metrics = trainer.evaluate_on_test()

        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")

        with _state_lock:
            _state["test_metrics"] = test_metrics
            _state["training_complete"] = True

        log_queue.put("Training completed!")
        return test_metrics

    except Exception as e:
        log_queue.put(f"Error during training: {str(e)}")
        raise


def build_training_tab():
    """Build the Training Control tab."""
    with gr.Tab("Training Control") as tab:
        gr.Markdown("## Training Configuration")

        with gr.Row():
            with gr.Column():
                learning_rate = gr.Number(label="Learning Rate", value=3e-4)
                epochs = gr.Number(label="Epochs", value=50, precision=0)
                batch_size = gr.Number(label="Batch Size", value=512, precision=0)
                early_stop_patience = gr.Number(label="Early Stop Patience", value=8, precision=0)

            with gr.Column():
                optimizer = gr.Dropdown(label="Optimizer", choices=["adamw", "adam", "sgd"], value="adamw")
                scheduler_type = gr.Dropdown(label="Scheduler Type", choices=["cosine", "plateau"], value="cosine")
                use_amp = gr.Checkbox(label="Use Mixed Precision (AMP)", value=False)

        with gr.Row():
            start_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Training", variant="stop")

        training_log = gr.Textbox(lines=15, label="Training Log", interactive=False)

        with gr.Row():
            loss_plot = gr.Plot(label="Live Loss Curve")
            auc_plot = gr.Plot(label="Live AUC Curve")

        final_metrics = gr.JSON(label="Final Test Metrics")

    def start_training(learning_rate, epochs, batch_size, early_stop_patience, optimizer, scheduler_type, use_amp):
        """Start training in background thread."""
        with _state_lock:
            if _state["training_thread"] and _state["training_thread"].is_alive():
                return "Training already in progress!", {}

            config = create_config_from_ui(
                float(learning_rate), int(epochs), int(batch_size),
                int(early_stop_patience), optimizer, scheduler_type, use_amp
            )

            stop_event = threading.Event()

            while not _state["log_queue"].empty():
                try:
                    _state["log_queue"].get_nowait()
                except queue.Empty:
                    break

            _state["stop_event"] = stop_event
            _state["training_complete"] = False
            _state["test_metrics"] = None

            training_thread = threading.Thread(
                target=run_training,
                args=(config, _state["log_queue"], stop_event),
                daemon=True
            )
            training_thread.start()
            _state["training_thread"] = training_thread

        return "Training started!", {}

    def stop_training():
        """Signal training to stop."""
        with _state_lock:
            if _state["stop_event"]:
                _state["stop_event"].set()
                return "Stopping training..."
            return "No training in progress"

    def update_log():
        """Poll log queue."""
        logs = []
        try:
            while True:
                msg = _state["log_queue"].get_nowait()
                logs.append(msg)
        except queue.Empty:
            pass
        return "\n".join(logs) if logs else ""

    def update_plots():
        """Update loss and AUC plots."""
        with _state_lock:
            trainer = _state.get("trainer")

        if trainer is None or not hasattr(trainer, 'train_losses') or len(trainer.train_losses) == 0:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            return fig1, fig2

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(trainer.train_losses) + 1)
        ax1.plot(epochs_range, trainer.train_losses, 'b-', label='Train Loss')
        if hasattr(trainer, 'val_losses') and trainer.val_losses:
            ax1.plot(epochs_range[:len(trainer.val_losses)], trainer.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if hasattr(trainer, 'val_aucs') and trainer.val_aucs:
            ax2.plot(epochs_range[:len(trainer.val_aucs)], trainer.val_aucs, 'g-', label='Val AUC')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('AUC')
            ax2.set_title('Validation AUC')
            ax2.legend()
            ax2.grid(True)
            ax2.set_ylim([0, 1])

        return fig1, fig2

    def get_final_metrics():
        """Get final test metrics."""
        with _state_lock:
            if _state["training_complete"] and _state["test_metrics"]:
                return _state["test_metrics"]
        return {}

    start_btn.click(
        fn=start_training,
        inputs=[learning_rate, epochs, batch_size, early_stop_patience, optimizer, scheduler_type, use_amp],
        outputs=[training_log, final_metrics]
    )

    stop_btn.click(fn=stop_training, outputs=[training_log])

    timer = gr.Timer(value=2)
    timer.tick(fn=update_log, outputs=[training_log])
    timer.tick(fn=update_plots, outputs=[loss_plot, auc_plot])
    timer.tick(fn=get_final_metrics, outputs=[final_metrics])


def build_inference_tab():
    """Build the Interactive Inference tab."""
    with gr.Tab("Interactive Inference") as tab:
        gr.Markdown("## Single Image Inference")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Detection Threshold")
                classify_btn = gr.Button("Classify", variant="primary")
                checkpoint_path = gr.Textbox(label="Checkpoint Path", value="checkpoints/best_model.pt")
                load_model_btn = gr.Button("Load Model")

            with gr.Column():
                prediction_label = gr.Label(label="Prediction")
                raw_probability = gr.Number(label="Raw Probability")
                gate_weights_plot = gr.Plot(label="Gate Weights")
                status_text = gr.Textbox(label="Status", interactive=False)

    def load_model(checkpoint_path):
        """Load model from checkpoint."""
        try:
            config = Config()
            device = get_device(config.training.device)
            model = create_detector(config.model, device=device.type)
            model = model.to(device)

            if os.path.exists(checkpoint_path):
                load_checkpoint(checkpoint_path, model, device)
                model.eval()
                with _state_lock:
                    _state["model"] = model
                    _state["config"] = config
                return f"Model loaded from {checkpoint_path}"
            else:
                return f"Checkpoint not found: {checkpoint_path}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def classify_image(image, threshold, checkpoint_path):
        """Classify uploaded image."""
        try:
            with _state_lock:
                model = _state["model"]
                config = _state["config"]

            if model is None:
                status = load_model(checkpoint_path)
                if "loaded" not in status.lower():
                    return {}, 0.0, plt.figure(), status
                with _state_lock:
                    model = _state["model"]
                    config = _state["config"]

            gray_tensor, rgb_tensor = preprocess_image(image, image_size=config.data.image_size)
            device = next(model.parameters()).device
            gray_tensor = gray_tensor.to(device)
            rgb_tensor = rgb_tensor.to(device)

            model.eval()
            with torch.no_grad():
                output = model(gray_tensor, rgb_tensor)
                prob_ai = torch.sigmoid(output['probability']).item()
                pred_class = "AI-Generated" if prob_ai >= threshold else "Real"
                confidence = abs(prob_ai - 0.5) * 2

                gate_weights = output['gate_weights'][0].cpu().numpy()
                gate_labels = ['Semantic', 'Spatial', 'Frequency']

                prediction = {pred_class: confidence}

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(gate_labels, gate_weights)
                ax.set_ylabel('Weight')
                ax.set_title('Branch Gate Weights')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)

                status = f"Classification complete: {pred_class} (confidence: {confidence:.2%})"
                return prediction, prob_ai, fig, status

        except Exception as e:
            return {}, 0.0, plt.figure(), f"Error: {str(e)}"

    load_model_btn.click(fn=load_model, inputs=[checkpoint_path], outputs=[status_text])
    classify_btn.click(
        fn=classify_image,
        inputs=[input_image, threshold, checkpoint_path],
        outputs=[prediction_label, raw_probability, gate_weights_plot, status_text]
    )


def build_visualization_tab():
    """Build the Training Visualizations tab."""
    with gr.Tab("Training Visualizations") as tab:
        gr.Markdown("## Training Metrics Visualization")

        with gr.Row():
            metrics_file_path = gr.Textbox(label="Metrics JSON File Path", value="outputs/metrics.json")
            load_metrics_btn = gr.Button("Load Metrics from File")

        status_msg = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            loss_plot = gr.Plot(label="Loss Curves")
            auc_plot = gr.Plot(label="AUC Progress")

        gate_weights_plot = gr.Plot(label="Gate Weights Distribution")
        val_metrics_df = gr.DataFrame(label="Per-epoch Val Metrics")
        full_test_metrics = gr.JSON(label="Full Test Metrics")

    def load_metrics(filepath):
        """Load metrics from JSON file."""
        try:
            if not os.path.exists(filepath):
                return plt.figure(), plt.figure(), plt.figure(), {}, {}, f"File not found: {filepath}"

            with open(filepath, 'r') as f:
                metrics = json.load(f)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            if 'train_losses' in metrics and 'val_losses' in metrics:
                epochs_range = range(1, len(metrics['train_losses']) + 1)
                ax1.plot(epochs_range, metrics['train_losses'], 'b-', label='Train Loss')
                ax1.plot(epochs_range, metrics['val_losses'], 'r-', label='Val Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(True)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            if 'val_aucs' in metrics:
                epochs_range = range(1, len(metrics['val_aucs']) + 1)
                ax2.plot(epochs_range, metrics['val_aucs'], 'g-', label='Val AUC')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('AUC')
                ax2.set_title('Validation AUC Progress')
                ax2.legend()
                ax2.grid(True)
                ax2.set_ylim([0, 1])

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            if 'test_metrics' in metrics and 'gate_weights' in metrics['test_metrics']:
                gate_data = metrics['test_metrics']['gate_weights']
                labels = list(gate_data.keys())
                values = list(gate_data.values())
                ax3.bar(labels, values)
                ax3.set_ylabel('Weight')
                ax3.set_title('Mean Gate Weights (Test Set)')
                ax3.set_ylim([0, 1])
                ax3.grid(True, alpha=0.3)

            df_data = []
            if 'val_metrics' in metrics:
                for i, vm in enumerate(metrics['val_metrics']):
                    df_data.append({
                        'epoch': i + 1,
                        'val_loss': vm.get('loss', 0),
                        'val_accuracy': vm.get('accuracy', 0),
                        'val_auc': vm.get('auc', 0),
                        'val_f1': vm.get('f1', 0)
                    })

            return fig1, fig2, fig3, df_data, metrics.get('test_metrics', {}), f"Loaded metrics from {filepath}"

        except Exception as e:
            return plt.figure(), plt.figure(), plt.figure(), {}, {}, f"Error: {str(e)}"

    load_metrics_btn.click(
        fn=load_metrics,
        inputs=[metrics_file_path],
        outputs=[loss_plot, auc_plot, gate_weights_plot, val_metrics_df, full_test_metrics, status_msg]
    )


def create_ui():
    """Create the main Gradio UI."""
    with gr.Blocks(title="NPR AI/Real Image Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NPR AI/Real Image Detector")
        gr.Markdown("### Interactive Web Interface for Training, Inference, and Visualization")

        build_training_tab()
        build_inference_tab()
        build_visualization_tab()

    return demo


def main():
    """Main entry point for Gradio UI."""
    demo = create_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == '__main__':
    main()
