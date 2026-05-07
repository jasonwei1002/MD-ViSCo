"""Experiment-tracking wrapper backed by SwanLab.

The module retains the historical ``WandBWrapper`` / ``WandBWrapperConfig``
names and the ``wandb_enabled`` / ``progress_bar.wandb`` attribute paths so
the existing Hydra yaml configs and call sites continue to work unchanged.
Internally every call now goes through SwanLab's native API.

API mapping (wandb -> swanlab):
    wandb.init(entity, project, name, config)
        -> swanlab.init(workspace, project, experiment_name, config)
    wandb.log(metrics, step=)        -> swanlab.log(metrics, step=)
    wandb.finish()                   -> swanlab.finish()
    wandb.run.get_url()              -> swanlab.get_run().public.cloud.experiment_url
    wandb.watch(model, ...)          -> not natively supported; logs an info
                                        line and returns without raising.
"""

# Standard library imports
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

logger = logging.getLogger(__name__)

try:
    import swanlab
except ImportError:
    swanlab = None
    logging.warning("swanlab not found, pip install swanlab")


@dataclass
class WandBWrapperConfig:
    """Configuration for WandBWrapper with Hydra-compatible defaults.

    Field names retain the ``wandb_*`` prefix to keep existing yaml configs
    valid; values are forwarded to SwanLab at init time.
    """

    _target_: str = "src.loggings.wandb_wrapper.WandBWrapper"
    project_name: str = MISSING
    run_name: str | None = None
    wandb_enabled: bool = False
    entity: str = MISSING


class WandBWrapper:
    """SwanLab-backed experiment tracker that keeps the WandB wrapper API."""

    def __init__(
        self,
        run_name: str | None = None,
        project_name: str | None = None,
        wandb_enabled: bool = True,
        entity: str | None = None,
    ):
        """Initialize the tracker without rank determination.

        Args:
            run_name: Optional run name (mapped to SwanLab experiment_name).
            project_name: Project name (falls back to ``WANDB_PROJECT`` /
                ``SWANLAB_PROJECT`` environment variables).
            wandb_enabled: Whether to enable tracking. Name retained for
                backward compatibility; gates SwanLab as well.
            entity: Workspace / team name (mapped to SwanLab workspace).

        Raises:
            ValueError: If parameters are invalid.
        """
        self.run_name = run_name
        self.project_name = (
            project_name
            or os.environ.get("WANDB_PROJECT")
            or os.environ.get("SWANLAB_PROJECT")
        )
        self.wandb_enabled = wandb_enabled
        self.entity = entity
        self._initialized = False

        self._validate_config()

        logger.debug(f"Created WandBWrapper: {self}")

    def __repr__(self) -> str:
        """Return string representation for logging and debugging."""
        return (
            f"<WandBWrapper(project_name='{self.project_name}', "
            f"wandb_enabled={self.wandb_enabled}, "
            f"initialized={self._initialized})>"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not isinstance(self.wandb_enabled, bool):
            raise ValueError("wandb_enabled must be a boolean")
        if self.entity and not isinstance(self.entity, str):
            raise ValueError("entity must be a string")

    def initialize(self, is_rank0: bool, config: Any):
        """Initialize SwanLab with rank boolean and config.

        Args:
            is_rank0: Whether this is the rank-0 process.
            config: Configuration object from Hydra (required).
        """
        if not self._should_initialize(is_rank0):
            return

        self._initialize(config)

    def _should_initialize(self, is_rank0: bool) -> bool:
        """Check if tracking should be initialized based on rank boolean."""
        return (
            swanlab is not None
            and is_rank0
            and self.wandb_enabled
            and self.project_name is not None
            and not self._initialized
        )

    def _initialize(self, config: Any):
        """Initialize SwanLab with rank-specific settings."""
        if swanlab is None:
            return
        try:
            from src.utils.validation_utils import flatten_config

            flat_config = flatten_config(config)

            init_kwargs: dict[str, Any] = {"config": flat_config}
            if self.project_name is not None:
                init_kwargs["project"] = self.project_name
            if self.entity is not None:
                init_kwargs["workspace"] = self.entity
            if self.run_name is not None:
                init_kwargs["experiment_name"] = self.run_name

            swanlab.init(**init_kwargs)
            self._initialized = True

            logger.info("SwanLab initialized on rank 0")
        except Exception:
            logger.exception("Failed to initialize SwanLab")

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        is_rank0: bool = False,
    ):
        """Log metrics only on rank 0."""
        if not self._initialized or not is_rank0 or swanlab is None:
            return

        try:
            if step is None:
                swanlab.log(metrics)
            else:
                swanlab.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to SwanLab: {e}")

    def _is_ready(self) -> bool:
        """Check if SwanLab is ready for logging."""
        return swanlab is not None and self._initialized

    def finish(self):
        """Finish SwanLab run and clean up environment."""
        if self._is_ready() and swanlab is not None:
            try:
                swanlab.finish()
            except Exception as e:
                logger.warning(f"Failed to finish SwanLab run: {e}")
            self._initialized = False

    @contextmanager
    def run(self):
        """Provide context manager for SwanLab run."""
        try:
            yield self
        finally:
            self.finish()

    def log_status_if_master(self, message: str, is_rank0: bool = False):
        """Log status message only if this is the master process."""
        if is_rank0:
            logger.info(message)

    def url(self) -> str | None:
        """Get SwanLab experiment URL if available."""
        if not self._is_ready() or swanlab is None:
            return None
        try:
            run = swanlab.get_run() if hasattr(swanlab, "get_run") else None
            if run is None:
                return None
            return run.public.cloud.experiment_url
        except Exception:
            return None

    def log_status_with_url(self, message: str, is_rank0: bool = False):
        """Log status message with SwanLab URL if available."""
        if is_rank0:
            url = self.url()
            if url:
                logger.info(f"{message} - SwanLab URL: {url}")
            else:
                logger.info(message)

    def log_domain_metric(self, task: str, is_rank0: bool = False, **kwargs):
        """Log domain-specific metrics with task prefix."""
        if not kwargs:
            return

        task_metrics = {f"{task}/{k}": v for k, v in kwargs.items()}
        self.log(task_metrics, is_rank0=is_rank0)

    def watch(
        self,
        model,
        is_rank0: bool = False,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """Watch model for gradients and parameters.

        SwanLab does not provide a native gradient/parameter hook equivalent
        to ``wandb.watch``. The call is accepted for backward compatibility
        but does not register hooks; an informational log is emitted on the
        first call so users are not surprised.
        """
        if not self._initialized or not is_rank0 or swanlab is None:
            return

        del model  # SwanLab has no native gradient hook equivalent.
        if not getattr(self, "_watch_warned", False):
            logger.info(
                "SwanLab has no direct equivalent of wandb.watch(); "
                "model gradient/parameter tracking is skipped (log=%s, "
                "log_freq=%d).",
                log,
                log_freq,
            )
            self._watch_warned = True


def initialize_wandb_if_enabled(obj: Any, cfg: Any, is_rank0: bool) -> None:
    """Initialize the SwanLab-backed tracker if available and enabled.

    Function name retained for backward compatibility; behaviour now targets
    SwanLab. Used by both train.py and test.py entry points.

    Args:
        obj: Object containing progress_bar with the tracker.
        cfg: Configuration object for tracker initialization.
        is_rank0: Whether this is the rank-0 process (True for single-process
            runs).
    """
    if (
        hasattr(obj, "progress_bar")
        and obj.progress_bar
        and obj.progress_bar.wandb
        and obj.progress_bar.wandb.wandb_enabled
    ):
        obj.progress_bar.wandb.initialize(is_rank0, cfg)
        logger.info(f"SwanLab initialized (is_rank0={is_rank0})")


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(group="wandb_wrapper", name="base_wandb_wrapper", node=WandBWrapperConfig)
