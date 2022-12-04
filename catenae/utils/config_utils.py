# pylint: disable=unspecified-encoding
"""Config utilities.
Methods used to manipulate YAML-based configuration files.
"""

import logging
import yaml


from catenae.utils import immutables

__all__ = ['load']

logger = logging.getLogger(__name__)


def load(config_file: str) -> immutables.ImmutableConfig:
    """Load an ImmutableConfig from a YAML configuration file.

    Args:
        config_file (_type_): path to YAML configuration file

    Returns:
        immutables.ImmutableConfig: FrozenDict containing configuration
    """

    logger.info("Loading config from file %s", config_file)

    with open(config_file, 'r') as config_stream:
        config = yaml.safe_load(config_stream)

    return immutables.ImmutableConfig(config)
