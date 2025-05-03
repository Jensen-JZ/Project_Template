from tensorboardX import SummaryWriter


class Logger(object):
    """
    Thin wrapper around tensorboardX's SummaryWriter to log training progress.

    This utility class provides a simple interface to log scalar values (e.g., loss, accuracy)
    and images during training, so they can be visualized in TensorBoard.

    Attributes:
        writer (SummaryWriter): Instance of tensorboardX's SummaryWriter for logging.
    """

    def __init__(self, log_dir):
        """
        Initializes the SummaryWriter to write logs to the specified directory.

        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.

        Returns:
            None
        """
        self.writer = SummaryWriter(log_dir)
        print(f"[Logger] TensorBoard logs will be saved to: {log_dir}")

    def scalar_summary(self, tag, scalar_value, step, prefix=""):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): Name of the metric to log (e.g., 'loss', 'accuracy').
            scalar_value (float): The scalar value to log (e.g., 'loss', 'accuracy').
            step (int): Training step or epoch number at which the value is logged.
            prefix (str): Optional prefix for the metric name (e.g., 'train/', 'val/').

        Returns:
            None
        """
        full_tag = f"{prefix}/{tag}" if prefix else tag
        self.writer.add_scalar(full_tag, scalar_value, step)

    def scalar_dict_summary(self, scalar_dict, step, prefix=""):
        """
        Logs a dictionary of scalar values to TensorBoard.

        Args:
            scalar_dict (dict): Dictionary of scalar values to log ({metric_name: value}).
            step (int): Training step or epoch number at which the values are logged.
            prefix (str): Optional prefix for the metric names (e.g., 'train/', 'val/').

        Returns:
            None
        """
        for key, value in scalar_dict.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, image_tensor, step, prefix=""):
        """
        Logs an image to TensorBoard.

        Args:
            tag (str): Name of the image to log (e.g., 'images').
            image_tensor (Tensor): A 3D tensor representing the image ([C, H, W]), normalized to [0, 1].
            step (int): Training step or epoch number at which the image is logged.
            prefix (str): Optional prefix for the image name (e.g., 'input/', 'generated/').

        Returns:
            None
        """
        full_tag = f"{prefix}/{tag}" if prefix else tag
        self.writer.add_image(full_tag, image_tensor, step)

    def image_batch_summary(self, tag, images, step, max_images=4, prefix=""):
        """
        Logs a batch of images to TensorBoard.

        Args:
            tag (str): Name of the image batch to log (e.g., 'images').
            images (Tensor): A 4D tensor representing a batch of images ([B, C, H, W]), normalized to [0, 1].
            step (int): Training step or epoch number at which the images are logged.
            max_images (int): Maximum number of images to log from the batch.
            prefix (str): Optional prefix for the image name (e.g., 'input/', 'generated/').

        Returns:
            None
        """
        full_tag = f"{prefix}/{tag}" if prefix else tag
        self.writer.add_images(full_tag, images[:max_images], step)

    def histogram_summary(self, tag, values, step, prefix=""):
        """
        Logs a histogram of values to TensorBoard.

        Args:
            tag (str): Name of the histogram to log (e.g., 'weights', 'gradients').
            values (Tensor): A 1D tensor of values to log as a histogram (e.g., weights, gradients).
            step (int): Training step or epoch number at which the histogram is logged.
            prefix (str): Optional prefix for the histogram name (e.g., 'conv1/', 'fc/').

        Returns:
            None
        """
        full_tag = f"{prefix}/{tag}" if prefix else tag
        values = values.view(-1)  # Flatten to 1D
        self.writer.add_histogram(full_tag, values, step)

    def close(self):
        """
        Closes the SummaryWriter.
        This should be called at the end of training to ensure all logs are written properly.

        Returns:
            None
        """
        self.writer.close()
        print("[Logger] SummaryWriter closed.")
