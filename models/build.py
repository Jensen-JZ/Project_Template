from munch import Munch

from models.geoclip import GeoCLIP


def build_model(args):
    """
    Constructs the GeoCLIP model and returns it in a Munch container.

    Args:
        args (Namespace or Munch): Configuration objecy with model arguments such as `from_pretrained` and `queue_size`.

    Returns:
        Munch: A Munch object containing the image encoder, location encoder, and the GeoCLIP model.
    """

    geoclip_model = GeoCLIP(
        from_pretrained=args.from_pretrained,
        queue_size=args.queue_size,
    )

    nets = Munch(
        image_encoder=geoclip_model.image_encoder,
        location_encoder=geoclip_model.location_encoder,
        geoclip=geoclip_model,
    )

    return nets
