"""Concrete adapters selected by deployment configuration."""

from .qdrant import QdrantClient

__all__ = ["QdrantClient"]
