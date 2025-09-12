"""Version information for SuperQuantX."""

__version__ = "0.1.3"
__version_info__ = (0, 1, 3)

# Release information
__release_date__ = "2025-09-11"
__release_name__ = "Agentic AI Genesis"

# Build information - will be updated during CI/CD
__build__ = "dev"
__commit__ = "unknown"

def get_version_string() -> str:
    """Get formatted version string with build info."""
    version_str = __version__
    if __build__ != "release":
        version_str += f"+{__build__}"
        if __commit__ != "unknown":
            version_str += f".{__commit__[:8]}"
    return version_str

def get_full_version_info() -> dict:
    """Get complete version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "release_date": __release_date__,
        "release_name": __release_name__,
        "build": __build__,
        "commit": __commit__,
        "full_version": get_version_string(),
    }
