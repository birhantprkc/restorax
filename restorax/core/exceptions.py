class RestoraXError(Exception):
    """Base exception for all RestoraX errors."""


class RestorerNotFoundError(RestoraXError):
    """Raised when a requested restorer name is not in the registry."""


class RestorerLoadError(RestoraXError):
    """Raised when a restorer fails to load its weights."""


class VideoReadError(RestoraXError):
    """Raised on unrecoverable video decoding errors."""


class VideoWriteError(RestoraXError):
    """Raised on unrecoverable video encoding errors."""


class JobNotFoundError(RestoraXError):
    """Raised when a job ID does not exist in the database."""


class PipelineConfigError(RestoraXError):
    """Raised when a pipeline YAML config is invalid."""


class StorageError(RestoraXError):
    """Raised on storage backend read/write failures."""


class AudioReadError(RestoraXError):
    """Raised when audio extraction from a container fails."""


class AudioWriteError(RestoraXError):
    """Raised when audio encoding or muxing into a container fails."""
